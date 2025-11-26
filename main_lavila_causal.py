import glob
import os
import random
import time
from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.distributed import ReduceOp
import utils
from utils.causal_trainer import Trainer

from model.helpers import Logger
from utils.args import get_args
from dataloader.dataloader import ActionFuseDataset as ActionDataset

from logging import log

def reduce_tensor(tensor):
	rt = tensor.clone()
	torch.distributed.all_reduce(rt, op=ReduceOp.SUM)
	rt /= dist.get_world_size()
	return rt

def main():
	# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
	args = get_args()
	os.environ['PYTHONHASHSEED'] = str(args.seed)

	if args.verbose:
		print(args)
	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	ngpus_per_node = 1 #torch.cuda.device_count()

	if args.multiprocessing_distributed:
		args.world_size = ngpus_per_node * args.world_size
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
	args.gpu = 0
	# print('gpuid:', args.gpu)

	if args.distributed:
		if args.multiprocessing_distributed:
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(
			backend=args.dist_backend,
			init_method=args.dist_url,
			world_size=args.world_size,
			rank=args.rank,
		)
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
			args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
	
	train_dataset = ActionDataset(
		args.root,
		args=args,
		is_val=False,
	)
	# Test data loading code
	test_dataset = ActionDataset(
		args.root,
		args=args,
		is_val=True,
	)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_thread_reader,
		pin_memory=args.pin_memory,
	)
	
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=args.batch_size_val,
		shuffle=False,
		drop_last=False,
		num_workers=args.num_thread_reader,
	)

	# create trainer
	trainer = Trainer(
	                train_loader=train_loader,
		            test_loader=test_loader,
		            z_dim=args.vae_latent_dim,
	                verb_dim=args.verb_dim,
		            noun_dim=args.noun_dim,
					action_dim=args.action_dim,		            
					lags=1,
					hidden_dim=128,
					domain_embedding_dim=1024, #1024,
					lr=args.lr,
					beta=0.0025,
					gamma=0.0075,
					delta=0.05,
					correlation='Pearson',
					pretrain_vae=args.pretrain_vae,
					fusion_alpha=args.fusion_alpha,
					fusion_use_std_scale=not args.disable_fusion_std_scale,
					fusion_gate_clamp=args.fusion_gate_clamp,
					feature_norm=args.feature_norm,
					feature_norm_eps=args.feature_norm_eps,
					temporal_target_len=args.temporal_target_len,
					temporal_pooling=args.temporal_pooling,
					stride_step=args.temporal_stride_step,
			        )
	trainer.cuda()
	trainer.verb_net.cuda()
	trainer.noun_net.cuda()
	trainer.domain_enc_act.cuda()
	trainer.transition_prior_verb.cuda()
	trainer.transition_prior_noun.cuda()
	trainer.cls_net.cuda()
	trainer.verb_norm.cuda()
	trainer.noun_norm.cuda()
	trainer.act_norm.cuda()

	print('Temporal pooling:', args.temporal_pooling)
	print('Temporal target len:', args.temporal_target_len)
	print('Temporal stride step:', args.temporal_stride_step)
	
	# create optimizer
	if args.optimizer.lower() == 'adam':
		optimizer = torch.optim.Adam(trainer.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
	elif args.optimizer.lower() == 'adamw':
		optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
	else:  # default to SGD
		optimizer = torch.optim.SGD(trainer.parameters(), args.lr, momentum=0.9, weight_decay=4e-5)
	trainer.optimizer = optimizer
	
	checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint', args.checkpoint_dir)
	if args.checkpoint_dir != '' and not (os.path.isdir(checkpoint_dir)) and args.rank == 0:
		os.mkdir(checkpoint_dir)

	if args.resume:
		checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if checkpoint_path:
			log("=> loading checkpoint '{}'".format(checkpoint_path), args)
			checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(args.rank))
			args.start_epoch = checkpoint["epoch"]
			trainer.load_state_dict(checkpoint["model"])
			optimizer.load_state_dict(checkpoint["optimizer"])
			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.cuda()
			tb_logdir = checkpoint["tb_logdir"]
			if args.rank == 0:
				# creat logger
				tb_logger = Logger(tb_logdir)
				log("=> loaded checkpoint '{}' (epoch {}){}".format(checkpoint_path, checkpoint["epoch"], args.gpu), args)
		else:
			time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
			logname = args.log_root + '_' + time_pre + '_' + args.dataset
			tb_logdir = os.path.join(args.log_root, logname)

			# creat logger
			if not (os.path.exists(tb_logdir)):
				os.makedirs(tb_logdir)
			tb_logger = Logger(tb_logdir)
			tb_logger.log_info(args)
			log("=> no checkpoint found at '{}'".format(args.resume), args)

	if args.cudnn_benchmark:
		cudnn.benchmark = True
	total_batch_size = args.world_size * args.batch_size
	log(
		"Starting training loop for rank: {}, total batch size: {}".format(
			args.rank, total_batch_size
		), args
	)
	
	best_loss = 100
	max_acc = 0
	old_max_epoch = 0
	
	save_max = os.path.join(os.path.dirname(__file__), 'save_max/'+args.dataset)
	for epoch in range(args.start_epoch, args.epochs):		
		# train for one epoch
		if (epoch + 1) % 5 == 0:  # calculate on training set			
			loss, recon_loss, kld_normal, kld_laplace, class_loss, acc1, acc5 = trainer.training_step(True)			
			acc1_reduced = acc1.cuda().item()
			acc5_reduced = acc5.cuda().item()
			class_losses_reduced = class_loss.cuda().item()				
			losses_reduced = loss.cuda().item()
			
			recon_losses_reduced = recon_loss.cuda().item()
			kld_normal_losses_reduced = kld_normal.cuda().item()
			kld_laplace_losses_reduced = kld_laplace.cuda().item()

			
			logs = OrderedDict()
			logs['Train/EpochLoss'] = losses_reduced
			logs['Train/EpochAcc@1'] = acc1_reduced
			logs['Train/EpochAcc@5'] = acc5_reduced
			logs['Train/rec_losses'] = recon_losses_reduced
			logs['Train/kld_normal_losses'] = kld_normal_losses_reduced
			logs['Train/kld_laplace_losses'] = kld_laplace_losses_reduced
			logs['Train/class_losses'] = class_losses_reduced
			for key, value in logs.items():
				tb_logger.log_scalar(value, key, epoch + 1)
			tb_logger.flush()
		else:
			loss, recon_loss, kld_normal, kld_laplace, class_loss = trainer.training_step(False)
			losses_reduced = loss.cuda().item()				
			class_losses_reduced = class_loss.cuda().item()				
									
			recon_losses_reduced = recon_loss.cuda().item()
			kld_normal_losses_reduced = kld_normal.cuda().item()
			kld_laplace_losses_reduced = kld_laplace.cuda().item()
			
			print('lrs:')
			for p in trainer.optimizer.param_groups:
				print(p['lr'])					
			print('---------------------------------')

			logs = OrderedDict()
			logs['Train/EpochLoss'] = losses_reduced				
			logs['Train/rec_losses'] = recon_losses_reduced
			logs['Train/kld_normal_losses'] = kld_normal_losses_reduced
			logs['Train/kld_laplace_losses'] = kld_laplace_losses_reduced
			logs['Train/class_losses'] = class_losses_reduced
			
			for key, value in logs.items():
				tb_logger.log_scalar(value, key, epoch + 1)

			tb_logger.flush()

		if ((epoch + 1) % (epoch + 1) == 0) and args.evaluate:  # or epoch > 18
			loss, recon_loss, kld_normal, kld_laplace, class_loss, acc1, acc5 = trainer.validate()
			losses_reduced = loss.cuda().item()
			acc1_reduced = acc1.cuda().item()
			acc5_reduced = acc5.cuda().item()
			class_losses_reduced = class_loss.cuda().item()
						
			logs = OrderedDict()
			logs['Test/EpochLoss'] = losses_reduced
			logs['Test/EpochAcc@1'] = acc1_reduced
			logs['Test/EpochAcc@5'] = acc5_reduced
			logs['Test/class_losses'] = class_losses_reduced
			for key, value in logs.items():
				tb_logger.log_scalar(value, key, epoch + 1)

			tb_logger.flush()
			print(acc1_reduced, max_acc)
			if acc1_reduced >= max_acc:
					save_checkpoint2(
						{
							"epoch": epoch + 1,
							"model": trainer.state_dict(),
							"optimizer": trainer.optimizer.state_dict(),
							"tb_logdir": tb_logdir,
						}, save_max, old_max_epoch, epoch + 1, args, args.rank
					)
					max_acc = acc1_reduced
					old_max_epoch = epoch + 1

		if (epoch + 1) % args.save_freq == 0:
			save_checkpoint(
				{
					"epoch": epoch + 1,
					"model": trainer.state_dict(),
					"optimizer": trainer.optimizer.state_dict(),
					"tb_logdir": tb_logdir,
				}, checkpoint_dir, epoch + 1
			)
		



def log(output, args):
	with open(os.path.join(os.path.dirname(__file__), 'log', args.checkpoint_dir + '.txt'), "a") as f:
		f.write(output + '\n')


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=3):
	torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
	if epoch - n_ckpt >= 0:
		oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch-n_ckpt))
		if os.path.isfile(oldest_ckpt):
			os.remove(oldest_ckpt)


def save_checkpoint2(state, checkpoint_dir, old_epoch, epoch, args, rank):
		torch.save(state, os.path.join(checkpoint_dir, "T"+str(args.horizon)+"_epoch{:0>4d}".format(epoch)+".pth.tar")) 
		if old_epoch > 0:
			oldest_ckpt = os.path.join(checkpoint_dir, "T"+str(args.horizon)+"_epoch{:0>4d}".format(old_epoch)+".pth.tar")
			if os.path.isfile(oldest_ckpt):
				os.remove(oldest_ckpt)
		

def get_last_checkpoint(checkpoint_dir):
	all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'epoch*.pth.tar'))
	if all_ckpt:
		all_ckpt = sorted(all_ckpt)
		return all_ckpt[-1]
	else:
		return ''


if __name__ == "__main__":
	main()
