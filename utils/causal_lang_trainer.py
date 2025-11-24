import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from model.causal import MLP, BetaVAE_MLP, NPTransitionPrior, NPChangeTransitionPrior
from model.latent_domain import DomainEncoder
from model.helpers import AverageMeter
from .accuracy import *

class Trainer(nn.Module):
	def __init__(
			self,
			train_loader,
			test_loader,
			z_dim,
			verb_dim=768,
			noun_dim=768,
			lags=1,
			hidden_dim=128,
			domain_embedding_dim=1024,
			lr=1e-4,
			beta=0.0025,
			gamma=0.0075,
			delta=0.05,
			correlation='Pearson',
			pretrain_vae=False,
			):
		'''Nonlinear ICA for nonparametric stationary processes'''
		super().__init__()
		#self.save_hyperparameters()
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.verb_dim = verb_dim
		self.noun_dim = noun_dim
		self.z_dim = z_dim
		self.lags = lags
		#self.n_class = n_class
		self.hidden_dim = hidden_dim
		self.domain_embedding_dim = domain_embedding_dim
		self.lr = lr
		self.beta = beta
		self.gamma = gamma
		self.delta = delta
		self.correlation = correlation
		self.verb_net = BetaVAE_MLP(input_dim=verb_dim, z_dim=z_dim, hidden_dim=hidden_dim)
		self.noun_net = BetaVAE_MLP(input_dim=noun_dim, z_dim=z_dim, hidden_dim=hidden_dim)
		self.domain_enc_act = DomainEncoder(input_size=domain_embedding_dim, hidden_size=512, n_adapters=30)
		self.domain_enc_lang = DomainEncoder(input_size=768, hidden_size=512, n_adapters=10)
		self.pretrain_vae = pretrain_vae

		# Initialize transition prior
		#self.transition_prior = NPTransitionPrior(lags=lags, latent_size=z_dim, num_layers=2, hidden_dim=hidden_dim)
		self.transition_prior_verb = NPChangeTransitionPrior(lags=lags, latent_size=z_dim, embedding_dim=512, num_layers=2, hidden_dim=hidden_dim)
		self.transition_prior_noun = NPChangeTransitionPrior(lags=lags, latent_size=z_dim, embedding_dim=512, num_layers=2, hidden_dim=hidden_dim)
		
		#self.cls_net = MLP(input_dim=8*(z_dim*2), hidden_dim=128, output_dim=106, num_layers=3)
		self.cls_net = MLP(input_dim=8*(z_dim*2)+8*768, hidden_dim=128, output_dim=106, num_layers=3)
		self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
		
		# base distribution for calculation of log prob under the model
		self.register_buffer('base_dist_mean', torch.zeros(z_dim).cuda())
		self.register_buffer('base_dist_var', torch.eye(z_dim).cuda())
		self.optimizer = None
		
	@property
	def base_dist(self):
		# Noise density function
		return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
				
	def reconstruction_loss(self, x, x_recon, distribution='gaussian'):
		batch_size = x.shape[0]
		assert batch_size != 0

		if distribution == 'bernoulli':
			x_recon = F.sigmoid(x_recon)
			recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
			
			#recon_loss = F.binary_cross_entropy(x_recon, x, size_average=True).div(batch_size)

		elif distribution == 'gaussian':
			recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

		elif distribution == 'sigmoid_gaussian':
			x_recon = F.sigmoid(x_recon)
			recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

		return recon_loss
		
	def kld(self, mus, logvars, z_est, domain_feat, transition_prior, log_qz, recon_loss, length):
		# dynamics
		# Past KLD
		p_dist = D.Normal(torch.zeros_like(mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
		log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
		log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
		if recon_loss>0:
			kld_normal = log_qz_normal - log_pz_normal
		else:
			kld_normal = log_pz_normal - log_qz_normal
		kld_normal = kld_normal.mean()
		# Future KLD
		log_qz_laplace = log_qz[:, self.lags:]
		residuals, logabsdet = transition_prior(z_est, domain_feat)

		log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1).cuda() + logabsdet.cuda().sum(dim=1)
		log_pz_laplace = log_pz_laplace.cuda()
		if recon_loss>0:
			kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (length - self.lags)
		else:
			kld_laplace = (-torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) + log_pz_laplace) / (length - self.lags)
		kld_laplace = kld_laplace.mean()
		return kld_normal, kld_laplace
	
	def training_step(self, if_calculate_acc):
		# (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
		losses = AverageMeter()
		recon_losses = AverageMeter()
		kld_normal_losses = AverageMeter()
		kld_laplace_losses = AverageMeter()
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		self.verb_net.train()
		self.noun_net.train()
		self.domain_enc_act.train()
		self.cls_net.train()
		self.transition_prior_verb.train()
		self.transition_prior_noun.train()
		self.domain_enc_lang.train()
		
		for batch in self.train_loader:
			verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, lang_embedding = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
			'''act_feat = act_feat[:,::2,:]
			verb_feat = verb_feat[:,::2,:]
			noun_feat = noun_feat[:,::2,:]'''
			# Max pooling: select max features from adjacent frames to go from 16 to 8
			act_feat = torch.max(act_feat[:, 0::2, :], act_feat[:, 1::2, :])  # [B, 8, 1024]
			lang_embedding = lang_embedding[:,2:10,:]
			'''verb_feat = verb_feat[:,:8,:]
			noun_feat = noun_feat[:,:8,:]'''
			#print(verb_feat.shape, spatial_verb_feat_cls.shape)
			
			#verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(1), verb_feat), dim=1)
			#noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(1), noun_feat), dim=1)
			#verb_feat = torch.cat((verb_feat, spatial_verb_feat_cls.unsqueeze(1)), dim=1)
			#noun_feat = torch.cat((noun_feat, spatial_noun_feat_cls.unsqueeze(1)), dim=1)
			
			batch_size, length, _ = verb_feat.shape	
			verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat)
			noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat)
			#act_u = self.domain_enc_act(act_feat)
			act_u = self.domain_enc_act(act_feat)
			#act_l = self.domain_enc_lang(lang_embedding)
			#act_u = torch.cat((act_u,act_l),dim=1)
			#act_u = act_u + act_l
			
			#print(verb_mus, verb_logvars)

			#82.15
			#pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
			#pred = pred + action_logits.squeeze(1)
			
			
			#class_loss = self.criterion(pred, labels)
			
			# recon_loss = self.reconstruction_loss(x, x_recon)
			verb_recon_loss = self.reconstruction_loss(verb_feat[:, :self.lags], verb_x_recon[:, :self.lags]) + (self.reconstruction_loss(verb_feat[:, self.lags:], verb_x_recon[:, self.lags:]))/(length-self.lags)
			noun_recon_loss = self.reconstruction_loss(noun_feat[:, :self.lags], noun_x_recon[:, :self.lags]) + (self.reconstruction_loss(noun_feat[:, self.lags:], noun_x_recon[:, self.lags:]))/(length-self.lags)

			verb_q_dist = D.Normal(verb_mus, torch.exp(verb_logvars / 2))
			verb_log_qz = verb_q_dist.log_prob(verb_z_est)
			
			noun_q_dist = D.Normal(noun_mus, torch.exp(noun_logvars / 2))
			noun_log_qz = noun_q_dist.log_prob(noun_z_est)

			kld_normal_verb, kld_laplace_verb = self.kld(verb_mus, verb_logvars, verb_z_est, act_u, self.transition_prior_verb, verb_log_qz, verb_recon_loss, length)
			kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)
			
			#print(verb_z_est, noun_z_est)
			if not self.pretrain_vae:
				#pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
				pred = self.cls_net(torch.cat((verb_z_est, noun_z_est, lang_embedding), dim=2).view(batch_size,-1))
				#pred = pred + action_logits.squeeze(1)
				class_loss = self.criterion(pred, labels)

			# VAE training
			#loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + self.delta * kld_obs
			recon_loss = verb_recon_loss + noun_recon_loss
			kld_normal = kld_normal_verb + kld_normal_noun
			#kld_laplace = kld_normal_verb + kld_normal_noun
			kld_laplace = kld_laplace_verb + kld_laplace_noun
			if not self.pretrain_vae:
				loss = self.delta * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace + class_loss
			else:
				loss = self.delta * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

			self.optimizer.zero_grad()			
			loss.backward()
			losses.update(loss.item(), batch_size)
			
			recon_losses.update(recon_loss.item(), batch_size)
			kld_normal_losses.update(kld_normal.item(), batch_size)
			kld_laplace_losses.update(kld_laplace.item(), batch_size)
			if not self.pretrain_vae:
				class_losses.update(class_loss.item(), batch_size)

			
			self.optimizer.step()
			self.optimizer.zero_grad()
		
			if if_calculate_acc:
				with torch.no_grad():				
					(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
					acc_top1.update(acc1.item(), batch_size)
					acc_top5.update(acc5.item(), batch_size)
		
		if not self.pretrain_vae:
			if if_calculate_acc:
				return torch.tensor(losses.avg), torch.tensor(recon_losses.avg), torch.tensor(kld_normal_losses.avg), torch.tensor(kld_laplace_losses.avg), torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)
			else:
				return torch.tensor(losses.avg), torch.tensor(recon_losses.avg), torch.tensor(kld_normal_losses.avg), torch.tensor(kld_laplace_losses.avg), torch.tensor(class_losses.avg)
		else:	
			return torch.tensor(losses.avg), torch.tensor(recon_losses.avg), torch.tensor(kld_normal_losses.avg), torch.tensor(kld_laplace_losses.avg),
		
	def validate(self):
		# (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
		losses = AverageMeter()
		recon_losses = AverageMeter()
		kld_normal_losses = AverageMeter()
		kld_laplace_losses = AverageMeter()
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		self.verb_net.eval()
		self.noun_net.eval()
		self.domain_enc_act.eval()
		self.cls_net.eval()
		self.transition_prior_verb.eval()
		self.transition_prior_noun.eval()
		self.domain_enc_lang.eval()
		
		for batch in self.test_loader:
			verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, lang_embedding = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
			
			'''act_feat = act_feat[:,:,::2,:]
			verb_feat = verb_feat[:,:,::2,:]
			noun_feat = noun_feat[:,:,::2,:]'''
			# Max pooling: select max features from adjacent frames to go from 16 to 8
			act_feat = torch.max(act_feat[:, :, 0::2, :], act_feat[:, :, 1::2, :])  # [B, crops, 8, 1024]
			lang_embedding = lang_embedding[:,2:10,:]
			'''verb_feat = verb_feat[:,:,:8,:]
			noun_feat = noun_feat[:,:,:8,:]'''

			#verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(2), verb_feat), dim=2)
			#noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(2), noun_feat), dim=2)
			#verb_feat = torch.cat((verb_feat, spatial_verb_feat_cls.unsqueeze(2)), dim=2)
			#noun_feat = torch.cat((noun_feat, spatial_noun_feat_cls.unsqueeze(2)), dim=2)
		
			preds = []
			for i in range(verb_feat.shape[1]):
			#for i in range(10):
				batch_size, length, _ = verb_feat[:,i,:,:].shape	
				verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat[:,i,:,:])
				noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat[:,i,:,:])
				#act_u = self.domain_enc_act(act_feat[:,i,:,:])
				act_u = self.domain_enc_act(act_feat[:,i,:,:])
				#act_l = self.domain_enc_lang(lang_embedding)
				#act_u = torch.cat((act_u,act_l),dim=1)
				#act_u = act_u + act_l

						
				# recon_loss = self.reconstruction_loss(x, x_recon)
				verb_recon_loss = self.reconstruction_loss(verb_feat[:,i,:,:][:, :self.lags], verb_x_recon[:, :self.lags]) + (self.reconstruction_loss(verb_feat[:,i,:,:][:, self.lags:], verb_x_recon[:, self.lags:]))/(length-self.lags)
				noun_recon_loss = self.reconstruction_loss(noun_feat[:,i,:,:][:, :self.lags], noun_x_recon[:, :self.lags]) + (self.reconstruction_loss(noun_feat[:,i,:,:][:, self.lags:], noun_x_recon[:, self.lags:]))/(length-self.lags)
				
				if not self.pretrain_vae:
					#pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
					pred = self.cls_net(torch.cat((verb_z_est, noun_z_est, lang_embedding), dim=2).view(batch_size,-1))
					preds.append(pred.unsqueeze(1))
				
				verb_q_dist = D.Normal(verb_mus, torch.exp(verb_logvars / 2))
				verb_log_qz = verb_q_dist.log_prob(verb_z_est)
				
				noun_q_dist = D.Normal(noun_mus, torch.exp(noun_logvars / 2))
				noun_log_qz = noun_q_dist.log_prob(noun_z_est)
				
				kld_normal_verb, kld_laplace_verb = self.kld(verb_mus, verb_logvars, verb_z_est, act_u, self.transition_prior_verb, verb_log_qz, verb_recon_loss, length)
				kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)

				recon_loss = verb_recon_loss + noun_recon_loss
				kld_normal = kld_normal_verb + kld_normal_noun
				kld_laplace = kld_laplace_verb + kld_laplace_noun
				loss = self.delta * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace
			
			if not self.pretrain_vae:
				preds = torch.cat(preds, dim=1)
				preds = torch.mean(preds, dim=1)
				#pred = preds + torch.mean(action_logits, dim=1)
				pred = preds 
			
				(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
				acc_top1.update(acc1.item(), batch_size)
				acc_top5.update(acc5.item(), batch_size)
						
				class_loss = self.criterion(pred, labels)
				loss = loss + class_loss
				
				class_losses.update(class_loss.item(), batch_size)
			
			losses.update(loss.item(), batch_size)			
			recon_losses.update(recon_loss.item(), batch_size)
			kld_normal_losses.update(kld_normal.item(), batch_size)
			kld_laplace_losses.update(kld_laplace.item(), batch_size)
			
		if not self.pretrain_vae:					
			return torch.tensor(losses.avg), torch.tensor(recon_losses.avg), torch.tensor(kld_normal_losses.avg), torch.tensor(kld_laplace_losses.avg), torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)
		else:
			return torch.tensor(losses.avg), torch.tensor(recon_losses.avg), torch.tensor(kld_normal_losses.avg), torch.tensor(kld_laplace_losses.avg)
		
	def eval(self):
		# (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
		losses = AverageMeter()
		recon_losses = AverageMeter()
		kld_normal_losses = AverageMeter()
		kld_laplace_losses = AverageMeter()
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		self.verb_net.eval()
		self.noun_net.eval()
		self.domain_enc_act.eval()
		self.cls_net.eval()
		self.transition_prior_verb.eval()
		self.transition_prior_noun.eval()
		total_z_verb = []
		total_z_noun = []
		total_pred = []
		total_act_u = []
		total_final_pred = []
		total_lavila_pred = []
		total_label = []
		with torch.no_grad():
			for batch in self.test_loader:
				verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, lang_embedding = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
				
				#act_feat = act_feat[:,:,::2,:]
				#verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(2), verb_feat), dim=2)
				#noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(2), noun_feat), dim=2)
			
				preds = []
				for i in range(verb_feat.shape[1]):
				#for i in range(10):
					batch_size, length, _ = verb_feat[:,i,:,:].shape	
					verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat[:,i,:,:])
					noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat[:,i,:,:])
					act_u = self.domain_enc_act(act_feat[:,i,:,:])
					
					total_act_u.append(act_u)
					total_z_verb.append(verb_z_est)
					total_z_noun.append(noun_z_est)

							
					# recon_loss = self.reconstruction_loss(x, x_recon)
					verb_recon_loss = self.reconstruction_loss(verb_feat[:,i,:,:][:, :self.lags], verb_x_recon[:, :self.lags]) + (self.reconstruction_loss(verb_feat[:,i,:,:][:, self.lags:], verb_x_recon[:, self.lags:]))/(length-self.lags)
					noun_recon_loss = self.reconstruction_loss(noun_feat[:,i,:,:][:, :self.lags], noun_x_recon[:, :self.lags]) + (self.reconstruction_loss(noun_feat[:,i,:,:][:, self.lags:], noun_x_recon[:, self.lags:]))/(length-self.lags)
					
					pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
					preds.append(pred.unsqueeze(1))
					total_pred.append(pred)
					
					verb_q_dist = D.Normal(verb_mus, torch.exp(verb_logvars / 2))
					verb_log_qz = verb_q_dist.log_prob(verb_z_est)
					
					noun_q_dist = D.Normal(noun_mus, torch.exp(noun_logvars / 2))
					noun_log_qz = noun_q_dist.log_prob(noun_z_est)
					
					kld_normal_verb, kld_laplace_verb = self.kld(verb_mus, verb_logvars, verb_z_est, act_u, self.transition_prior_verb, verb_log_qz, verb_recon_loss, length)
					kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)

					# VAE training
					recon_loss = verb_recon_loss + noun_recon_loss
					kld_normal = kld_normal_verb + kld_normal_noun
					kld_laplace = kld_normal_verb + kld_normal_noun
					loss = self.delta * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace
				
				preds = torch.cat(preds, dim=1)
				preds = torch.mean(preds, dim=1)
				total_lavila_pred.append(torch.mean(action_logits, dim=1))
				pred = preds + torch.mean(action_logits, dim=1)
				#pred = preds 
				total_final_pred.append(pred)
				total_label.append(labels)
				
				(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
				acc_top1.update(acc1.item(), batch_size)
				acc_top5.update(acc5.item(), batch_size)
						
				class_loss = self.criterion(pred, labels)
				loss = loss + class_loss
				
				losses.update(loss.item(), batch_size)			
				recon_losses.update(recon_loss.item(), batch_size)
				kld_normal_losses.update(kld_normal.item(), batch_size)
				kld_laplace_losses.update(kld_laplace.item(), batch_size)
				class_losses.update(class_loss.item(), batch_size)
		total_z_verb = torch.cat(total_z_verb, dim=0)
		total_z_noun = torch.cat(total_z_noun, dim=0)
		total_pred = torch.cat(total_pred, dim=0)
		total_act_u = torch.cat(total_act_u, dim=0)		
		total_final_pred = torch.cat(total_final_pred, dim=0)
		total_lavila_pred = torch.cat(total_lavila_pred, dim=0)
		total_label = torch.cat(total_label, dim=0)
		
		return torch.tensor(losses.avg), torch.tensor(recon_losses.avg), torch.tensor(kld_normal_losses.avg), torch.tensor(kld_laplace_losses.avg), torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg), total_z_verb, total_z_noun, total_pred, total_act_u, total_final_pred, total_lavila_pred, total_label
	 
	def validate_pred(self):
		# (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		
		self.cls_net.eval()

		for batch in self.test_loader:
			verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, lang_embedding = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
			
			'''act_feat = act_feat[:,:,::2,:]
			verb_feat = verb_feat[:,:,::2,:]
			noun_feat = noun_feat[:,:,::2,:]'''
			# Max pooling: select max features from adjacent frames to go from 16 to 8
			act_feat = torch.max(act_feat[:, :, 0::2, :], act_feat[:, :, 1::2, :])  # [B, crops, 8, 1024]
			'''verb_feat = verb_feat[:,:,:8,:]
			noun_feat = noun_feat[:,:,:8,:]'''

			#verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(2), verb_feat), dim=2)
			#noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(2), noun_feat), dim=2)
			#verb_feat = torch.cat((verb_feat, spatial_verb_feat_cls.unsqueeze(2)), dim=2)
			#noun_feat = torch.cat((noun_feat, spatial_noun_feat_cls.unsqueeze(2)), dim=2)
		
			preds = []
			for i in range(verb_feat.shape[1]):
			#for i in range(10):
				batch_size, length, _ = verb_feat[:,i,:,:].shape	
				verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat[:,i,:,:])
				noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat[:,i,:,:])
				act_u = self.domain_enc_act(act_feat[:,i,:,:])
						
				pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
				preds.append(pred.unsqueeze(1))
				
			preds = torch.cat(preds, dim=1)
			preds = torch.mean(preds, dim=1)
			pred = preds + torch.mean(action_logits, dim=1)
			#pred = preds 
			
			(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
			acc_top1.update(acc1.item(), batch_size)
			acc_top5.update(acc5.item(), batch_size)
						
			class_loss = self.criterion(pred, labels)				
			class_losses.update(class_loss.item(), batch_size)
						
		return torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)

	def training_pred(self, if_calculate_acc):
		# (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
		class_losses = AverageMeter()
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		self.verb_net.eval()
		self.noun_net.eval()
		self.domain_enc_act.eval()
		self.cls_net.train()
		self.transition_prior_verb.eval()
		self.transition_prior_noun.eval()
		
		for batch in self.train_loader:
			verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, lang_embedding = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
			'''act_feat = act_feat[:,::2,:]
			verb_feat = verb_feat[:,::2,:]
			noun_feat = noun_feat[:,::2,:]'''
			# Max pooling: select max features from adjacent frames to go from 16 to 8
			act_feat = torch.max(act_feat[:, 0::2, :], act_feat[:, 1::2, :])  # [B, 8, 1024]
			'''verb_feat = verb_feat[:,:8,:]
			noun_feat = noun_feat[:,:8,:]'''
			#print(verb_feat.shape, spatial_verb_feat_cls.shape)
			
			#verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(1), verb_feat), dim=1)
			#noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(1), noun_feat), dim=1)
			#verb_feat = torch.cat((verb_feat, spatial_verb_feat_cls.unsqueeze(1)), dim=1)
			#noun_feat = torch.cat((noun_feat, spatial_noun_feat_cls.unsqueeze(1)), dim=1)
			
			batch_size, length, _ = verb_feat.shape	
			verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat)
			noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat)
			act_u = self.domain_enc_act(act_feat)


			#print(verb_z_est.shape, noun_z_est.shape, act_u.shape)

			pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
			#pred = pred + action_logits.squeeze(1)
			class_loss = self.criterion(pred, labels)

			self.optimizer.zero_grad()			
			class_loss.backward()
			class_losses.update(class_loss.item(), batch_size)
	
			self.optimizer.step()
		
			if if_calculate_acc:
				with torch.no_grad():				
					(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
					acc_top1.update(acc1.item(), batch_size)
					acc_top5.update(acc5.item(), batch_size)
		
		if if_calculate_acc:
			return torch.tensor(class_losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)
		else:
			return torch.tensor(class_losses.avg)

		

	def eval_pred(self):
		# (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
		acc_top1 = AverageMeter()
		acc_top5 = AverageMeter()
		self.verb_net.eval()
		self.noun_net.eval()
		self.domain_enc_act.eval()
		self.transition_prior_verb.eval()
		self.transition_prior_noun.eval()
		self.cls_net.eval()

		for batch in self.test_loader:
			verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, lang_embedding = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
			
			'''act_feat = act_feat[:,:,::2,:]
			verb_feat = verb_feat[:,:,::2,:]
			noun_feat = noun_feat[:,:,::2,:]'''
			# Max pooling: select max features from adjacent frames to go from 16 to 8
			act_feat = torch.max(act_feat[:, :, 0::2, :], act_feat[:, :, 1::2, :])  # [B, crops, 8, 1024]
			'''verb_feat = verb_feat[:,:,:8,:]
			noun_feat = noun_feat[:,:,:8,:]'''

			#verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(2), verb_feat), dim=2)
			#noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(2), noun_feat), dim=2)
			#verb_feat = torch.cat((verb_feat, spatial_verb_feat_cls.unsqueeze(2)), dim=2)
			#noun_feat = torch.cat((noun_feat, spatial_noun_feat_cls.unsqueeze(2)), dim=2)
		
			preds = []
			for i in range(verb_feat.shape[1]):
			#for i in range(10):
				batch_size, length, _ = verb_feat[:,i,:,:].shape	
				verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat[:,i,:,:])
				noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat[:,i,:,:])
				act_u = self.domain_enc_act(act_feat[:,i,:,:])
						
				pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
				preds.append(pred.unsqueeze(1))
				
			preds = torch.cat(preds, dim=1)
			preds = torch.mean(preds, dim=1)
			pred = preds + torch.mean(action_logits, dim=1)
			pred = preds 
			
			(acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
			acc_top1.update(acc1.item(), batch_size)
			acc_top5.update(acc5.item(), batch_size)
												
		return torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)

	
   
