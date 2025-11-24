import copy
from model.helpers import AverageMeter
from .accuracy import *

class Trainer(object):
	def __init__(
			self,
			causal_model,
			train_loader,
			test_loader,
			optimizer,
			train_lr=1e-5,
	):
		super().__init__()
		self.model = causal_model	
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		# self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, diffusion_model.parameters()), lr=train_lr, weight_decay=0.0)

	# -----------------------------------------------------------------------------#
	# ------------------------------------ api ------------------------------------#
	# -----------------------------------------------------------------------------#

	def validate(self, args):
		self.model.eval()
		losses = AverageMeter()
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		for batch in self.test_loader:
			with torch.no_grad():								
				spatial_verb_feat, spatial_noun_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, verb_label, noun_label, action_label, spatial_act_logits = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda()
				#spatial_verb_feat, spatial_noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda()
				#action_label = action_label.view(-1,1)
				
				#spatial_verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(2), spatial_verb_feat), dim=2)
				#spatial_noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(2), spatial_noun_feat), dim=2)
				
				labels=action_label
				#print(spatial_verb_feat.shape)
				bs = spatial_verb_feat.shape[0]
				
				
				loss, flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss, class_loss, pred = self.model(spatial_verb_data=spatial_verb_feat, spatial_noun_data=spatial_noun_feat, labels=labels, act_logits=spatial_act_logits, is_val=True)
							
				losses.update(loss.item(), bs)
				class_losses.update(class_loss.item(), bs)
				(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
				acc_top1.update(acc1.item(), bs)
				acc_top5.update(acc5.item(), bs)
				
		return torch.tensor(losses.avg), torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)
	
	def validate_vae(self, args):
		self.model.eval()
		losses = AverageMeter()
		
		for batch in self.test_loader:
			with torch.no_grad():								
				motion_feat, spatial_verb_feat, spatial_noun_feat, verb_label, noun_label, action_label = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
				#action_label = action_label.view(-1,1)
				
				labels=verb_label
				
				bs = motion_feat.shape[0]
				loss, flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss, flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss, flow_mu, spatial_verb_mu, spatial_noun_mu = self.model(flow_data=motion_feat, spatial_verb_data=spatial_verb_feat, spatial_noun_data=spatial_noun_feat, labels=labels)
							
				losses.update(loss.item(), bs)				
		return torch.tensor(losses.avg)
		
	def validate_pred(self, args):
		self.model.eval()
		losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		for batch in self.test_loader:
			with torch.no_grad():								
				motion_feat, spatial_verb_feat, spatial_noun_feat, verb_label, noun_label, action_label = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
				#action_label = action_label.view(-1,1)
				labels=verb_label
				
				bs = motion_feat.shape[0]
				loss, flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss, flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss, class_loss, pred = self.model(spatial_verb_data=spatial_verb_feat, spatial_noun_data=spatial_noun_feat, labels=labels)
				
				#pred = self.model.class_net(torch.cat((spatial_verb_mu, spatial_verb_logvar, spatial_noun_mu, spatial_noun_logvar, flow_mu, flow_logvar), dim=-1)).squeeze(1)
				
				#class_loss = self.cls_criterion(pred, action_label)
				(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
				acc_top1.update(acc1.item(), bs)
				acc_top5.update(acc5.item(), bs)
							
				losses.update(class_loss.item(), bs)				
		return torch.tensor(losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)
				
	def train(self, if_calculate_acc, args):
		self.model.train()
		losses = AverageMeter()
		verb_noun_rec_losses = AverageMeter()
		verb_noun_sparsity_losses = AverageMeter()
		verb_noun_kld_losses = AverageMeter()
		verb_noun_structure_losses = AverageMeter()
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		
		for batch in self.train_loader:
			spatial_verb_feat, spatial_noun_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, verb_label, noun_label, action_label, spatial_act_logits = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda()
			bs = spatial_verb_feat.shape[0]
			#action_label = action_label.view(-1,1)
			#print(action_label.shape)
			labels = action_label
			
			#spatial_verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(1), spatial_verb_feat), dim=1)
			#spatial_noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(1), spatial_noun_feat), dim=1)
			
			loss, verb_noun_rec_loss, verb_noun_sparsity_loss, verb_noun_kld_loss, verb_noun_structure_loss, class_loss, pred = self.model(spatial_verb_data=spatial_verb_feat[:,:,:], spatial_noun_data=spatial_noun_feat[:,:,:], labels=labels, act_logits=spatial_act_logits, is_val=False)
			
			self.optimizer.zero_grad()			
			loss.backward()
			losses.update(loss.item(), bs)
			
			verb_noun_rec_losses.update(verb_noun_rec_loss.item(), bs)
			verb_noun_sparsity_losses.update(verb_noun_sparsity_loss.item(), bs)
			verb_noun_kld_losses.update(verb_noun_kld_loss.item(), bs)
			verb_noun_structure_losses.update(verb_noun_structure_loss.item(), bs)
			class_losses.update(class_loss.item(), bs)
			

			self.optimizer.step()
			self.optimizer.zero_grad()

			if if_calculate_acc:
				with torch.no_grad():				
					(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
					acc_top1.update(acc1.item(), bs)
					acc_top5.update(acc5.item(), bs)
					
		if if_calculate_acc:
			return torch.tensor(losses.avg), torch.tensor(verb_noun_rec_losses.avg), torch.tensor(verb_noun_sparsity_losses.avg), torch.tensor(verb_noun_kld_losses.avg), torch.tensor(verb_noun_structure_losses.avg), torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)

		else:
			return torch.tensor(losses.avg), torch.tensor(verb_noun_rec_losses.avg), torch.tensor(verb_noun_sparsity_losses.avg), torch.tensor(verb_noun_kld_losses.avg), torch.tensor(verb_noun_structure_losses.avg), torch.tensor(class_losses.avg)
			
	def train_vae(self, args):
		self.model.train()
		losses = AverageMeter()
		flow_verb_rec_losses = AverageMeter()
		flow_verb_sparsity_losses = AverageMeter()
		flow_verb_kld_losses = AverageMeter()
		flow_verb_structure_losses = AverageMeter()
		flow_noun_rec_losses = AverageMeter()
		flow_noun_sparsity_losses = AverageMeter()
		flow_noun_kld_losses = AverageMeter()
		flow_noun_structure_losses = AverageMeter()
		
		for batch in self.train_loader:
			motion_feat, spatial_verb_feat, spatial_noun_feat, verb_label, noun_label, action_label = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
			bs = motion_feat.shape[0]
			#action_label = action_label.view(-1,1)
			#print(action_label.shape)
			labels=action_label
			loss, flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss, flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss, flow_mu, spatial_verb_mu, spatial_noun_mu = self.model(flow_data=motion_feat, spatial_verb_data=spatial_verb_feat, spatial_noun_data=spatial_noun_feat, labels=labels)
			
			self.optimizer.zero_grad()			
			loss.backward()
			losses.update(loss.item(), bs)
			flow_verb_rec_losses.update(flow_verb_rec_loss.item(), bs)
			flow_verb_sparsity_losses.update(flow_verb_sparsity_loss.item(), bs)
			flow_verb_kld_losses.update(flow_verb_kld_loss.item(), bs)
			flow_verb_structure_losses.update(flow_verb_structure_loss.item(), bs)
			flow_noun_rec_losses.update(flow_noun_rec_loss.item(), bs)
			flow_noun_sparsity_losses.update(flow_noun_sparsity_loss.item(), bs)
			flow_noun_kld_losses.update(flow_noun_kld_loss.item(), bs)
			flow_noun_structure_losses.update(flow_noun_structure_loss.item(), bs)
			
			self.optimizer.step()
			self.optimizer.zero_grad()

		return torch.tensor(losses.avg), torch.tensor(flow_verb_rec_losses.avg), torch.tensor(flow_verb_sparsity_losses.avg), torch.tensor(flow_verb_kld_losses.avg), torch.tensor(flow_verb_structure_losses.avg), torch.tensor(flow_noun_rec_losses.avg), torch.tensor(flow_noun_sparsity_losses.avg), torch.tensor(flow_noun_kld_losses.avg), torch.tensor(flow_noun_structure_losses.avg)
		
	def train_pred(self, if_calculate_acc, args):
		self.model.train()
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		for batch in self.train_loader:
			motion_feat, spatial_verb_feat, spatial_noun_feat, verb_label, noun_label, action_label = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
			bs = motion_feat.shape[0]
			labels=verb_label
			loss, flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss, flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss,  class_loss, pred = self.model(flow_data=motion_feat, spatial_verb_data=spatial_verb_feat, spatial_noun_data=spatial_noun_feat, labels=labels)
			
			#pred = self.class_net(torch.cat((spatial_verb_mu, spatial_verb_logvar, spatial_noun_mu, spatial_noun_logvar, flow_mu, flow_logvar), dim=-1)).squeeze(1)
			
			#class_loss = self.cls_criterion(pred, action_label)
			class_losses.update(class_loss.item(), bs)
			
		if if_calculate_acc:
			with torch.no_grad():				
				(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
				acc_top1.update(acc1.item(), bs)
				acc_top5.update(acc5.item(), bs)
				
				return torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)
		else:
			return torch.tensor(class_losses.avg)
			
	
		
		
