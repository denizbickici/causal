import torch
import math
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
            action_dim=106,
            lags=1,
            hidden_dim=256,
            domain_embedding_dim=1024,
            lr=1e-4,
            beta=0.0025,
            gamma=0.0075,
            delta=0.05,
            correlation='Pearson',
            pretrain_vae=False,
            fusion_alpha=1,
            fusion_use_std_scale=False,
            temporal_target_len=16,
            temporal_pooling='stride',  # 'adaptive_avg' | 'adaptive_max' | 'stride'
            stride_step=0,
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
        #self.domain_enc_act = DomainEncoder(input_size=domain_embedding_dim, hidden_size=512, n_adapters=30)
        self.domain_enc_act = DomainEncoder(input_size=domain_embedding_dim, hidden_size=512, n_adapters=30)
        #self.domain_enc_act2 = DomainEncoder(input_size=domain_embedding_dim, hidden_size=512, n_adapters=60)
        self.pretrain_vae = pretrain_vae

        # Initialize transition prior
        #self.transition_prior = NPTransitionPrior(lags=lags, latent_size=z_dim, num_layers=2, hidden_dim=hidden_dim)
        self.transition_prior_verb = NPChangeTransitionPrior(lags=lags, latent_size=z_dim, embedding_dim=512, num_layers=2, hidden_dim=hidden_dim)
        self.transition_prior_noun = NPChangeTransitionPrior(lags=lags, latent_size=z_dim, embedding_dim=512, num_layers=2, hidden_dim=hidden_dim)
        
        # Classifier consumes all temporal steps; align input size with configured target length
        self.cls_net = MLP(input_dim=temporal_target_len*(z_dim*2), hidden_dim=128, output_dim=action_dim, num_layers=3)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(z_dim).cuda())
        self.register_buffer('base_dist_var', torch.eye(z_dim).cuda())
        self.optimizer = None
        # fusion controls
        self.fusion_alpha = fusion_alpha
        self.fusion_use_std_scale = fusion_use_std_scale
        # temporal reduction controls
        self.temporal_target_len = temporal_target_len
        self.temporal_pooling = temporal_pooling
        self.stride_step = stride_step
        
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

    def _pool_to_steps(self, tensor, target_len=None, mode='adaptive_avg'):
        """Temporal pooling that keeps batch/crop dims intact (avg or max)."""
        if target_len is None:
            target_len = self.temporal_target_len
        if target_len is None:
            return tensor
        if tensor.shape[-2] == target_len:
            return tensor
        orig_shape = tensor.shape
        flattened = tensor.reshape(-1, orig_shape[-2], orig_shape[-1])
        x = flattened.transpose(1, 2)  # (N, F, T)
        if mode == 'adaptive_max':
            pooled = F.adaptive_max_pool1d(x, target_len)
        else:
            pooled = F.adaptive_avg_pool1d(x, target_len)
        pooled = pooled.transpose(1, 2)
        return pooled.reshape(*orig_shape[:-2], target_len, orig_shape[-1])

    def _stride_to_steps(self, tensor, target_len=None, step=None):
        """Temporal striding (subsampling) along time axis, then nearest upsample if needed."""
        if target_len is None:
            target_len = self.temporal_target_len
        if target_len is None and (step is None or step <= 1):
            return tensor
        orig_shape = tensor.shape
        flattened = tensor.reshape(-1, orig_shape[-2], orig_shape[-1])  # (N, T, F)
        T = flattened.shape[1]
        if step is None or step <= 1:
            # choose step to land near target_len
            step = max(1, math.ceil(T / max(1, target_len)))
        x = flattened[:, ::step, :]  # (N, S, F)
        # If exact target_len requested, up/downsample to match via nearest
        if target_len is not None and x.shape[1] != target_len:
            x_chw = x.transpose(1, 2)  # (N, F, S)
            x_chw = F.interpolate(x_chw, size=target_len, mode='nearest')
            x = x_chw.transpose(1, 2)
        return x.reshape(*orig_shape[:-2], x.shape[1], orig_shape[-1])

    def _shrink_time(self, tensor, target_len=None):
        """Reduce temporal length by configured strategy."""
        if target_len is None:
            target_len = self.temporal_target_len
        mode = (self.temporal_pooling or 'adaptive_avg').lower()
        if mode == 'stride':
            return self._stride_to_steps(tensor, target_len=target_len, step=self.stride_step)
        elif mode == 'adaptive_max':
            return self._pool_to_steps(tensor, target_len=target_len, mode='adaptive_max')
        else:
            return self._pool_to_steps(tensor, target_len=target_len, mode='adaptive_avg')

    def _reduce_action_logits(self, action_logits):
        """Collapse temporal/crop dimensions so logits align with [B, num_classes]."""
        if action_logits.dim() <= 2:
            return action_logits
        flattened = action_logits.reshape(action_logits.shape[0], -1, action_logits.shape[-1])
        return flattened.mean(dim=1)
        
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
        #self.domain_enc_act2.train()
        
        for batch in self.train_loader:
            verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, spatial_act_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
            '''act_feat = act_feat[:,::2,:]
            verb_feat = verb_feat[:,::2,:]
            noun_feat = noun_feat[:,::2,:]'''
            # Max pooling: select max features from adjacent frames to go from 16 to 8
            act_feat = self._shrink_time(act_feat, target_len=self.temporal_target_len)
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
            #act_u2 = self.domain_enc_act2(act_feat)
            #print(verb_mus, verb_logvars)

            #82.15
            #pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
            #pred = pred + logits
            
            
            #class_loss = self.criterion(pred, labels)
            
            # recon_loss = self.reconstruction_loss(x, x_recon)
            verb_recon_loss = self.reconstruction_loss(verb_feat[:, :self.lags], verb_x_recon[:, :self.lags]) + (self.reconstruction_loss(verb_feat[:, self.lags:], verb_x_recon[:, self.lags:]))/(length-self.lags)
            noun_recon_loss = self.reconstruction_loss(noun_feat[:, :self.lags], noun_x_recon[:, :self.lags]) + (self.reconstruction_loss(noun_feat[:, self.lags:], noun_x_recon[:, self.lags:]))/(length-self.lags)

            # Stabilize posterior construction (verb)
            verb_mus = torch.nan_to_num(verb_mus, nan=0.0, posinf=1e6, neginf=-1e6)
            verb_logvars = torch.nan_to_num(verb_logvars, nan=0.0, posinf=10.0, neginf=-10.0)
            verb_logvars = torch.clamp(verb_logvars, min=-20.0, max=10.0)
            verb_q_dist = D.Normal(verb_mus, torch.clamp(torch.exp(verb_logvars / 2), min=1e-6))
            verb_log_qz = verb_q_dist.log_prob(verb_z_est)
            
            # Stabilize posterior construction (noun)
            noun_mus = torch.nan_to_num(noun_mus, nan=0.0, posinf=1e6, neginf=-1e6)
            noun_logvars = torch.nan_to_num(noun_logvars, nan=0.0, posinf=10.0, neginf=-10.0)
            noun_logvars = torch.clamp(noun_logvars, min=-20.0, max=10.0)
            noun_q_dist = D.Normal(noun_mus, torch.clamp(torch.exp(noun_logvars / 2), min=1e-6))
            noun_log_qz = noun_q_dist.log_prob(noun_z_est)

            kld_normal_verb, kld_laplace_verb = self.kld(verb_mus, verb_logvars, verb_z_est, act_u, self.transition_prior_verb, verb_log_qz, verb_recon_loss, length)
            #kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u2, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)
            kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)
            
            #print(verb_z_est, noun_z_est)
            if not self.pretrain_vae:
                pred_cls = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
                logits = self._reduce_action_logits(action_logits)
                if self.fusion_use_std_scale:
                    pred_std = pred_cls.std(dim=1, keepdim=True)
                    logits_std = logits.std(dim=1, keepdim=True)
                    scale = (logits_std / (pred_std + 1e-6)).detach()
                else:
                    scale = 1.0
                pred = logits + self.fusion_alpha * pred_cls * scale
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
            # Clip exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
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
        #self.domain_enc_act2.eval()
        
        for batch in self.test_loader:
            verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls, spatial_act_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), batch[9].cuda()
            
            '''act_feat = act_feat[:,:,::2,:]
            verb_feat = verb_feat[:,:,::2,:]
            noun_feat = noun_feat[:,:,::2,:]'''

            if verb_feat.dim() == 3:
                verb_feat = verb_feat.unsqueeze(1)
            if noun_feat.dim() == 3:
                noun_feat = noun_feat.unsqueeze(1)
            if act_feat.dim() == 3:
                act_feat = act_feat.unsqueeze(1)
            if action_logits.dim() == 2:
                action_logits = action_logits.unsqueeze(1)
            # Restrict to first view/crop to match single TIM view
            if verb_feat.shape[1] > 1:
                verb_feat = verb_feat[:, :1, ...]
            if noun_feat.shape[1] > 1:
                noun_feat = noun_feat[:, :1, ...]
            if act_feat.shape[1] > 1:
                act_feat = act_feat[:, :1, ...]
            if action_logits.shape[1] > 1:
                action_logits = action_logits[:, :1, ...]
            # Max pooling: select max features from adjacent frames to go from 16 to 8
            act_feat = self._shrink_time(act_feat, target_len=self.temporal_target_len)
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
                #act_u2 = self.domain_enc_act2(act_feat[:,i,:,:])

                        
                # recon_loss = self.reconstruction_loss(x, x_recon)
                verb_recon_loss = self.reconstruction_loss(verb_feat[:,i,:,:][:, :self.lags], verb_x_recon[:, :self.lags]) + (self.reconstruction_loss(verb_feat[:,i,:,:][:, self.lags:], verb_x_recon[:, self.lags:]))/(length-self.lags)
                noun_recon_loss = self.reconstruction_loss(noun_feat[:,i,:,:][:, :self.lags], noun_x_recon[:, :self.lags]) + (self.reconstruction_loss(noun_feat[:,i,:,:][:, self.lags:], noun_x_recon[:, self.lags:]))/(length-self.lags)
                
                if not self.pretrain_vae:
                    pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
                    preds.append(pred.unsqueeze(1))
                
                verb_mus = torch.nan_to_num(verb_mus, nan=0.0, posinf=1e6, neginf=-1e6)
                verb_logvars = torch.nan_to_num(verb_logvars, nan=0.0, posinf=10.0, neginf=-10.0)
                verb_logvars = torch.clamp(verb_logvars, min=-20.0, max=10.0)
                verb_q_dist = D.Normal(verb_mus, torch.clamp(torch.exp(verb_logvars / 2), min=1e-6))
                verb_log_qz = verb_q_dist.log_prob(verb_z_est)
                
                noun_mus = torch.nan_to_num(noun_mus, nan=0.0, posinf=1e6, neginf=-1e6)
                noun_logvars = torch.nan_to_num(noun_logvars, nan=0.0, posinf=10.0, neginf=-10.0)
                noun_logvars = torch.clamp(noun_logvars, min=-20.0, max=10.0)
                noun_q_dist = D.Normal(noun_mus, torch.clamp(torch.exp(noun_logvars / 2), min=1e-6))
                noun_log_qz = noun_q_dist.log_prob(noun_z_est)
                
                kld_normal_verb, kld_laplace_verb = self.kld(verb_mus, verb_logvars, verb_z_est, act_u, self.transition_prior_verb, verb_log_qz, verb_recon_loss, length)
                #kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u2, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)
                kld_normal_noun, kld_laplace_noun = self.kld(noun_mus, noun_logvars, noun_z_est, act_u, self.transition_prior_noun, noun_log_qz, noun_recon_loss, length)

                recon_loss = verb_recon_loss + noun_recon_loss
                kld_normal = kld_normal_verb + kld_normal_noun
                kld_laplace = kld_laplace_verb + kld_laplace_noun
                loss = self.delta * recon_loss + self.beta * kld_normal + self.gamma * kld_laplace
            
            if not self.pretrain_vae:
                preds = torch.cat(preds, dim=1)
                preds = torch.mean(preds, dim=1)
                logits = self._reduce_action_logits(action_logits)
                if self.fusion_use_std_scale:
                    pred_std = preds.std(dim=1, keepdim=True)
                    logits_std = logits.std(dim=1, keepdim=True)
                    scale = (logits_std / (pred_std + 1e-6)).detach()
                else:
                    scale = 1.0
                pred = logits + self.fusion_alpha * preds * scale
                #pred = logits
                #pred = preds 
            
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
                verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda()
                
                #act_feat = act_feat[:,:,::2,:]
                #verb_feat = torch.cat((spatial_verb_feat_cls.unsqueeze(2), verb_feat), dim=2)
                #noun_feat = torch.cat((spatial_noun_feat_cls.unsqueeze(2), noun_feat), dim=2)

                if verb_feat.dim() == 3:
                    verb_feat = verb_feat.unsqueeze(1)
                if noun_feat.dim() == 3:
                    noun_feat = noun_feat.unsqueeze(1)
                if act_feat.dim() == 3:
                    act_feat = act_feat.unsqueeze(1)
                if action_logits.dim() == 2:
                    action_logits = action_logits.unsqueeze(1)
                # Restrict to first view/crop to match single TIM view
                if verb_feat.shape[1] > 1:
                    verb_feat = verb_feat[:, :1, ...]
                if noun_feat.shape[1] > 1:
                    noun_feat = noun_feat[:, :1, ...]
                if act_feat.shape[1] > 1:
                    act_feat = act_feat[:, :1, ...]
                if action_logits.shape[1] > 1:
                    action_logits = action_logits[:, :1, ...]
            
                preds = []
                for i in range(verb_feat.shape[1]):
                    batch_size, length, _ = verb_feat[:, i, :, :].shape    
                    verb_x_recon, verb_mus, verb_logvars, verb_z_est = self.verb_net(verb_feat[:, i, :, :])
                    noun_x_recon, noun_mus, noun_logvars, noun_z_est = self.noun_net(noun_feat[:, i, :, :])
                    act_u = self.domain_enc_act(act_feat[:, i, :, :])
                    
                    total_act_u.append(act_u)
                    total_z_verb.append(verb_z_est)
                    total_z_noun.append(noun_z_est)

                    # recon_loss = self.reconstruction_loss(x, x_recon)
                    verb_recon_loss = self.reconstruction_loss(verb_feat[:, i, :, :][:, :self.lags], verb_x_recon[:, :self.lags]) + (self.reconstruction_loss(verb_feat[:, i, :, :][:, self.lags:], verb_x_recon[:, self.lags:]))/(length-self.lags)
                    noun_recon_loss = self.reconstruction_loss(noun_feat[:, i, :, :][:, :self.lags], noun_x_recon[:, :self.lags]) + (self.reconstruction_loss(noun_feat[:, i, :, :][:, self.lags:], noun_x_recon[:, self.lags:]))/(length-self.lags)
                    
                    pred = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
                    preds.append(pred.unsqueeze(1))
                    total_pred.append(pred)
                    
                    verb_mus = torch.nan_to_num(verb_mus, nan=0.0, posinf=1e6, neginf=-1e6)
                    verb_logvars = torch.nan_to_num(verb_logvars, nan=0.0, posinf=10.0, neginf=-10.0)
                    verb_logvars = torch.clamp(verb_logvars, min=-20.0, max=10.0)
                    verb_q_dist = D.Normal(verb_mus, torch.clamp(torch.exp(verb_logvars / 2), min=1e-6))
                    verb_log_qz = verb_q_dist.log_prob(verb_z_est)
                    
                    noun_mus = torch.nan_to_num(noun_mus, nan=0.0, posinf=1e6, neginf=-1e6)
                    noun_logvars = torch.nan_to_num(noun_logvars, nan=0.0, posinf=10.0, neginf=-10.0)
                    noun_logvars = torch.clamp(noun_logvars, min=-20.0, max=10.0)
                    noun_q_dist = D.Normal(noun_mus, torch.clamp(torch.exp(noun_logvars / 2), min=1e-6))
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
                logits = self._reduce_action_logits(action_logits)
                total_lavila_pred.append(logits)
                if self.fusion_use_std_scale:
                    pred_std = preds.std(dim=1, keepdim=True)
                    logits_std = logits.std(dim=1, keepdim=True)
                    scale = (logits_std / (pred_std + 1e-6)).detach()
                else:
                    scale = 1.0
                pred = logits + self.fusion_alpha * preds * scale
                #pred = logits
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
            verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda()
            
            '''act_feat = act_feat[:,:,::2,:]
            verb_feat = verb_feat[:,:,::2,:]
            noun_feat = noun_feat[:,:,::2,:]'''

            if verb_feat.dim() == 3:
                verb_feat = verb_feat.unsqueeze(1)
            if noun_feat.dim() == 3:
                noun_feat = noun_feat.unsqueeze(1)
            if act_feat.dim() == 3:
                act_feat = act_feat.unsqueeze(1)
            if action_logits.dim() == 2:
                action_logits = action_logits.unsqueeze(1)
            # Restrict to first view/crop to match single TIM view
            if verb_feat.shape[1] > 1:
                verb_feat = verb_feat[:, :1, ...]
            if noun_feat.shape[1] > 1:
                noun_feat = noun_feat[:, :1, ...]
            if act_feat.shape[1] > 1:
                act_feat = act_feat[:, :1, ...]
            if action_logits.shape[1] > 1:
                action_logits = action_logits[:, :1, ...]
            act_feat = self._shrink_time(act_feat, target_len=self.temporal_target_len)
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
            logits = self._reduce_action_logits(action_logits)
            if self.fusion_use_std_scale:
                pred_std = preds.std(dim=1, keepdim=True)
                logits_std = logits.std(dim=1, keepdim=True)
                scale = (logits_std / (pred_std + 1e-6)).detach()
            else:
                scale = 1.0
            pred = logits + self.fusion_alpha * preds * scale
            #pred = logits
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
            verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda()
            '''act_feat = act_feat[:,::2,:]
            verb_feat = verb_feat[:,::2,:]
            noun_feat = noun_feat[:,::2,:]'''

            act_feat = self._shrink_time(act_feat, target_len=self.temporal_target_len)
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

            pred_cls = self.cls_net(torch.cat((verb_z_est, noun_z_est), dim=2).view(batch_size,-1))
            logits = self._reduce_action_logits(action_logits)
            if self.fusion_use_std_scale:
                pred_std = pred_cls.std(dim=1, keepdim=True)
                logits_std = logits.std(dim=1, keepdim=True)
                scale = (logits_std / (pred_std + 1e-6)).detach()
            else:
                scale = 1.0
            pred = logits + self.fusion_alpha * pred_cls * scale
            #pred = logits
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
            verb_feat, noun_feat, verb_label, noun_label, labels, action_logits, act_feat, spatial_verb_feat_cls, spatial_noun_feat_cls = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda()
            
            '''act_feat = act_feat[:,:,::2,:]
            verb_feat = verb_feat[:,:,::2,:]
            noun_feat = noun_feat[:,:,::2,:]'''
            if verb_feat.dim() == 3:
                verb_feat = verb_feat.unsqueeze(1)
            if noun_feat.dim() == 3:
                noun_feat = noun_feat.unsqueeze(1)
            if act_feat.dim() == 3:
                act_feat = act_feat.unsqueeze(1)
            if action_logits.dim() == 2:
                action_logits = action_logits.unsqueeze(1)
            # Restrict to first view/crop to match single TIM view
            if verb_feat.shape[1] > 1:
                verb_feat = verb_feat[:, :1, ...]
            if noun_feat.shape[1] > 1:
                noun_feat = noun_feat[:, :1, ...]
            if act_feat.shape[1] > 1:
                act_feat = act_feat[:, :1, ...]
            if action_logits.shape[1] > 1:
                action_logits = action_logits[:, :1, ...]
            act_feat = self._shrink_time(act_feat, target_len=self.temporal_target_len)  # [B, crops, 8, 1024]
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
            logits = self._reduce_action_logits(action_logits)
            if self.fusion_use_std_scale:
                pred_std = preds.std(dim=1, keepdim=True)
                logits_std = logits.std(dim=1, keepdim=True)
                scale = (logits_std / (pred_std + 1e-6)).detach()
            else:
                scale = 1.0
            pred = logits + self.fusion_alpha * preds * scale
            #pred = logits
            #pred = preds 
            
            (acc1, acc5) = accuracy(pred.cpu(), labels.cpu(), topk=(1, 5),)
            acc_top1.update(acc1.item(), batch_size)
            acc_top5.update(acc5.item(), batch_size)
                                                
        return torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg)

    
   
