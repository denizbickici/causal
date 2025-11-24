import torch
from torch import nn
import torch.distributions as D
from torch.nn import functional as F
from model.causal import BetaVAE_MLP

class MLP(nn.Module):
	def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
		super().__init__()
		if activation == "gelu":
			a_f = nn.GELU()
		elif activation == "relu":
			a_f = nn.ReLU()
		elif activation == "tanh":
			a_f = nn.Tanh()
		elif activation == "leakyReLU":
			a_f = nn.LeakyReLU()
		else:
			a_f = nn.Identity()

		if out_dim is None:
			out_dim = in_dim
		if layer_nums == 1:
			net = [nn.Linear(in_dim, out_dim)]
		else:

			net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
				nn.Linear(in_dim, hid_dim), a_f]
			for i in range(layer_norm - 2):
				net.append(nn.Linear(in_dim, hid_dim))
				net.append(a_f)
			net.append(nn.Linear(hid_dim, out_dim))
		self.net = nn.Sequential(*net)

	def forward(self, x):
		return self.net(x)


class Permute(nn.Module):
	def __init__(self, *dims):
		super(Permute, self).__init__()
		self.dims = dims

	def forward(self, x):
		return x.permute(*self.dims)


class NPTransitionPrior(nn.Module):
	def __init__(
			self,
			lags,
			latent_size,
			num_layers=3,
			hidden_dim=64):
		super().__init__()
		self.L = lags
		gs = [MLP(in_dim=lags * latent_size + 1,
				  out_dim=1,
				  layer_nums=num_layers,
				  hid_dim=hidden_dim, activation="leakyReLU", layer_norm=False) for _ in range(latent_size)]

		self.gs = nn.ModuleList(gs)

	def forward(self, x):
		# x: [BS, T, D] -> [BS, T-L, L+1, D]
		batch_size, length, input_dim = x.shape
		# prepare data

		x = x.unfold(dimension=1, size=self.L + 1, step=1)
		x = torch.swapaxes(x, 2, 3)

		x = x.reshape(-1, self.L + 1, input_dim)
		yy, xx = x[:, -1:], x[:, :-1]
		xx = xx.reshape(-1, self.L * input_dim)
		# get residuals and |J|
		residuals = []
		hist_jac = []

		sum_log_abs_det_jacobian = 0
		for i in range(input_dim):
			inputs = torch.cat([xx] + [yy[:, :, i]], dim=-1)

			residual = self.gs[i](inputs)
			with torch.enable_grad():
				pdd = torch.func.vmap(torch.func.jacfwd(self.gs[i]))(inputs)
			logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
			hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
			sum_log_abs_det_jacobian += logabsdet
			residuals.append(residual)

		residuals = torch.cat(residuals, dim=-1)
		residuals = residuals.reshape(batch_size, -1, input_dim)
		sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
		return residuals, sum_log_abs_det_jacobian, hist_jac


class Net(nn.Module):
	def __init__(self, args, flow_input_dim, spatial_verb_input_dim, spatial_noun_input_dim, z_dim, hidden_dim_flow, hidden_dim_spatial_verb, hidden_dim_spatial_noun, lags, threshold):
		super(Net, self).__init__()	   
		self.z_net_flow = BetaVAE_MLP(flow_input_dim, z_dim, hidden_dim_flow)
		self.z_net_spatial_verb = BetaVAE_MLP(spatial_verb_input_dim, z_dim, hidden_dim_spatial_verb)
		self.z_net_spatial_noun = BetaVAE_MLP(spatial_noun_input_dim, z_dim, hidden_dim_spatial_noun)
		self.class_net = Base_Net(input_len=8, out_len=1, input_dim=z_dim*2,
								  out_dim=args.action_dim, layer_nums=3, c_type="type2",
								  activation='leakyReLU', drop_out=0.1,
								  is_mean_std=False, hidden_dim=512, layer_norm=False)

		self.rec_criterion = nn.MSELoss()
		#self.rec_criterion = nn.BCEWithLogitsLoss()
		self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
		
		self.register_buffer('base_dist_mean', torch.zeros(z_dim))
		self.register_buffer('base_dist_var', torch.eye(z_dim))
		self.lags = lags
		self.threshold = threshold
		self.transition_prior_fix_verb_noun = NPTransitionPrior(lags=lags,
													  latent_size=z_dim,
													  num_layers=2,
													  hidden_dim=512)
		self.transition_prior_fix_verb_verb = NPTransitionPrior(lags=lags,
													  latent_size=z_dim,
													  num_layers=2,
													  hidden_dim=512)
													  
		self.rec_weight = args.alpha
		self.sparsity_weight = args.gamma
		self.z_kl_weight = args.beta
		self.structure_weight = args.delta
		self.z_dim = z_dim
		self.args = args

	def forward(self, flow_data, spatial_verb_data, spatial_noun_data, labels, act_logits, is_val):
		if is_val:
			#print(spatial_verb_data.shape)
			preds = []
			for i in range(spatial_verb_data.shape[1]):
				spatial_verb = spatial_verb_data[:,i,:,:]
				spatial_noun = spatial_noun_data[:,i,:,:]
				flow = flow_data[:,i,:,:]
				#print(spatial_verb_data.shape, spatial_noun_data.shape, flow_data.shape)
	
				# source data is obs, target data is act
				flow_x_recon, flow_mu, flow_logvar, flow_z = self.z_net_flow(flow)
				spatial_verb_x_recon, spatial_verb_mu, spatial_verb_logvar, spatial_verb_z = self.z_net_spatial_verb(spatial_verb)
				spatial_noun_x_recon, spatial_noun_mu, spatial_noun_logvar, spatial_noun_z = self.z_net_spatial_noun(spatial_noun)
								 
				flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss \
					= self.__loss_function(spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb, spatial_verb_x_recon,
										   flow_mu, flow_logvar, flow_z, flow, flow_x_recon, self.transition_prior_fix_verb_verb,
										   )
				flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss \
					= self.__loss_function(spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb, spatial_verb_x_recon,
										    spatial_noun_mu, spatial_noun_logvar, spatial_noun_z, spatial_noun, spatial_noun_x_recon, self.transition_prior_fix_verb_noun,
										   #flow_mu, flow_logvar, flow_z, flow, flow_x_recon,
										   #spatial_noun_mu, spatial_noun_logvar, spatial_noun_z, spatial_noun, spatial_noun_x_recon,  self.transition_prior_fix_verb_noun,
										   )
				flow_verb_loss = flow_verb_rec_loss * self.rec_weight + flow_verb_sparsity_loss * self.sparsity_weight + flow_verb_kld_loss * self.z_kl_weight + flow_verb_structure_loss * self.structure_weight
				
				flow_noun_loss = flow_noun_rec_loss * self.rec_weight + flow_noun_sparsity_loss * self.sparsity_weight + flow_noun_kld_loss * self.z_kl_weight + flow_noun_structure_loss * self.structure_weight
				
				pred = self.class_net(torch.cat((spatial_verb_z, spatial_noun_z), dim=-1)).squeeze(1)
				preds.append(pred.unsqueeze(1))
			preds = torch.cat(preds, dim=1)
			#print(preds.shape)
			preds = torch.mean(preds, dim=1)
			pred = preds + torch.mean(act_logits, dim=1)
			#pred = torch.nn.functional.softmax(preds, dim=1) + torch.nn.functional.softmax(torch.mean(act_logits, dim=1), dim=1)
			#pred = preds + act_logits.squeeze(1)

		else:
			flow_x_recon, flow_mu, flow_logvar, flow_z = self.z_net_flow(flow_data)
			spatial_verb_x_recon, spatial_verb_mu, spatial_verb_logvar, spatial_verb_z = self.z_net_spatial_verb(spatial_verb_data)
			spatial_noun_x_recon, spatial_noun_mu, spatial_noun_logvar, spatial_noun_z = self.z_net_spatial_noun(spatial_noun_data)
							 
			flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss \
				= self.__loss_function(spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb_data, spatial_verb_x_recon,
									   flow_mu, flow_logvar, flow_z, flow_data, flow_x_recon, self.transition_prior_fix_verb_verb,
									   )
			flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss \
				= self.__loss_function(spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb_data, spatial_verb_x_recon,
										spatial_noun_mu, spatial_noun_logvar, spatial_noun_z, spatial_noun_data, spatial_noun_x_recon, self.transition_prior_fix_verb_noun,
									   #flow_mu, flow_logvar, flow_z, flow_data, flow_x_recon,
									   #spatial_noun_mu, spatial_noun_logvar, spatial_noun_z, spatial_noun_data, spatial_noun_x_recon,  self.transition_prior_fix_verb_noun,
									   )
			flow_verb_loss = flow_verb_rec_loss * self.rec_weight + flow_verb_sparsity_loss * self.sparsity_weight + flow_verb_kld_loss * self.z_kl_weight + flow_verb_structure_loss * self.structure_weight
			
			flow_noun_loss = flow_noun_rec_loss * self.rec_weight + flow_noun_sparsity_loss * self.sparsity_weight + flow_noun_kld_loss * self.z_kl_weight + flow_noun_structure_loss * self.structure_weight
			
			pred = self.class_net(torch.cat((spatial_verb_z, spatial_noun_z), dim=-1)).squeeze(1)
			#print(pred, act_logits)
			pred = pred + act_logits.squeeze(1)
	
		
		if not self.args.pretrain_vae:
			#pred = self.class_net(torch.cat((spatial_verb_z, spatial_noun_z), dim=-1)).squeeze(1)
			#print(pred.shape, labels.shape)
			class_loss = self.cls_criterion(pred, labels)
			loss = flow_verb_loss + flow_noun_loss + class_loss
			return loss, flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss,  flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss, class_loss, pred
		else:
			loss = flow_verb_loss + flow_noun_loss
			return loss, flow_verb_rec_loss, flow_verb_sparsity_loss, flow_verb_kld_loss, flow_verb_structure_loss,  flow_noun_rec_loss, flow_noun_sparsity_loss, flow_noun_kld_loss, flow_noun_structure_loss, flow_z, spatial_verb_z, spatial_noun_z
		
	def __loss_function(self, src_z_mean, src_z_std, src_z, src_x, src_rec, tgt_z_mean, tgt_z_std, tgt_z, tgt_x, tgt_rec, transition_prior_fix):			  
		obs_rec_loss = self.rec_criterion(src_x, src_rec)
		act_rec_loss = self.rec_criterion(tgt_x, tgt_rec)
						   
		rec_loss = obs_rec_loss + act_rec_loss
		z_mean = torch.cat((src_z_mean, tgt_z_mean), dim=0)
		z_std = torch.cat((src_z_std, tgt_z_std), dim=0)
		z = torch.cat((src_z, tgt_z), dim=0)

		b, length, _ = z_mean.shape
		q_dist = D.Normal(z_mean, torch.exp(z_std / 2))

		log_qz = q_dist.log_prob(z)

		p_dist = D.Normal(torch.zeros_like(z_mean[:, :self.lags]),
						  torch.ones_like(z_std[:, :self.lags]))
		log_pz_normal = torch.sum(p_dist.log_prob(z[:, :self.lags]), dim=[-2, -1])
		log_qz_normal = torch.sum(log_qz[:, :self.lags], dim=[-2, -1])

		kld_normal = (torch.abs(log_qz_normal - log_pz_normal) / self.lags).sum()
		log_qz_laplace = log_qz[:, self.lags:]

		residuals, logabsdet, hist_jac = transition_prior_fix.forward(z)

		log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
		kld_future = ((torch.sum(log_qz_laplace, dim=[-2, -1]) - log_pz_laplace) / (
				length - self.lags)).mean()

		kld_loss = (kld_normal + kld_future) / self.z_dim

		structure_loss = torch.tensor(0, device=z_mean.device)
		sparsity_loss = torch.tensor(0, device=z_mean.device)
		for jac in hist_jac:
			sparsity_loss = sparsity_loss + F.l1_loss(jac[:, 0, :self.lags * self.z_dim],
													  torch.zeros_like(jac[:, 0, :self.lags * self.z_dim]),
													  reduction='sum')
			src_jac, trg_jac = torch.chunk(jac, dim=0, chunks=2)
			threshold = torch.quantile(src_jac, self.threshold)
			#print(torch.max(src_jac), torch.min(src_jac), torch.mean(src_jac))
			#print(torch.max(trg_jac), torch.min(trg_jac), torch.mean(trg_jac))
			I_J1_src = (src_jac > threshold).bool()
			I_J1_trg = (trg_jac > threshold).bool()

			mask = torch.bitwise_xor(I_J1_src, I_J1_trg)
			structure_loss = structure_loss + torch.sum((src_jac[mask].detach() - trg_jac[mask]) ** 2)

		sparsity_loss = sparsity_loss / b
		structure_loss = structure_loss
		return rec_loss, sparsity_loss, kld_loss, structure_loss

	@property
	def base_dist(self):
		# Noise density function
		return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
		
class Base_Net(nn.Module):
	def __init__(self, input_len, out_len, input_dim, out_dim, hidden_dim, is_mean_std=True, activation="gelu",
				 layer_norm=True, c_type="None", drop_out=0, layer_nums=2) -> None:
		super().__init__()

		self.input_dim = input_dim
		self.out_len = out_len
		self.c_type = c_type
		self.out_dim = out_dim
		self.c_type = "type1" if out_dim != input_dim and c_type == "None" else c_type
		self.radio = 2 if is_mean_std else 1

		if self.c_type == "None":
			self.net = MLP(layer_nums, in_dim=input_len, out_dim=out_len * self.radio, hid_dim=hidden_dim,
						   activation=activation,
						   layer_norm=layer_norm)
		elif self.c_type == "type1":
			self.net = MLP(layer_nums, in_dim=self.input_dim, hid_dim=hidden_dim,
						   out_dim=self.out_dim * self.radio,
						   layer_norm=layer_norm, activation=activation)
		elif self.c_type == "type2":
			self.net = MLP(layer_nums, in_dim=self.input_dim * input_len, hid_dim=hidden_dim * 2 * out_len,
						   activation=activation,
						   out_dim=self.out_dim * out_len * self.radio, layer_norm=layer_norm)

		self.dropout_net = nn.Dropout(drop_out)

	def forward(self, x):
		if self.c_type == "type1":
			x = self.net(x)
		elif self.c_type == "type2":
			x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim * self.radio)

		elif self.c_type == "None":
			x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
		#x = self.dropout_net(x)
		if self.radio == 2:
			dim = 2 if self.c_type == "type1" or self.c_type == "type2" else 1
			x = torch.chunk(x, dim=dim, chunks=2)
		return x

class HierarchicalActionPredictor(nn.Module):
	def __init__(
		self,
		args,
		feat_dim,
		hidden_dim,
	):
		super().__init__()

		# Dimensions
		self.hidden_dim = hidden_dim
		#self.num_verb_classes = num_verb_classes
		#self.num_noun_classes = num_noun_classes

		# Query vectors for attention (one per action class)
		self.action_queries = nn.Parameter(
			torch.randn(args.action_dim, self.hidden_dim)
		)
		nn.init.xavier_uniform_(self.action_queries)

		# Projections for verb and noun features
		self.verb_projection = nn.Sequential(
			nn.Linear(feat_dim, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.3),
		)

		self.noun_projection = nn.Sequential(
			nn.Linear(feat_dim, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.3),
		)

		# Cross-attention layers
		self.verb_attention = nn.MultiheadAttention(
			embed_dim=self.hidden_dim, num_heads=16, dropout=0.1, batch_first=True
		)

		self.noun_attention = nn.MultiheadAttention(
			embed_dim=self.hidden_dim, num_heads=16, dropout=0.1, batch_first=True
		)

		# Final classifier
		self.classifier = nn.Sequential(
			nn.Linear(self.hidden_dim * 2, self.hidden_dim),
			nn.LayerNorm(self.hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(self.hidden_dim, args.action_dim),
		)

		# Initialize weights
		self._init_weights()

	def _init_weights(self):
		"""Initialize trainable weights"""
		for module in [self.verb_projection, self.noun_projection, self.classifier]:
			for m in module.modules():
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
					if m.bias is not None:
						nn.init.zeros_(m.bias)

	def forward(
		self,
		verb_feat,
		noun_feat,
		#dist,
		#attn_mask,
	):
		batch_size = verb_feat.size(0)

		# Project verb and noun features to hidden dimension
		verb_features = self.verb_projection(verb_feat)  # [batch_size, hidden_dim]
		noun_features = self.noun_projection(noun_feat)  # [batch_size, hidden_dim]

		# Expand action queries for batch processing
		query = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
		# [batch_size, num_action_classes, hidden_dim]

		# Apply cross-attention between action queries and verb/noun features
		#verb_features = verb_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
		#noun_features = noun_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]

		attended_verb, _ = self.verb_attention(
			query, verb_features, verb_features
		)  # [batch_size, num_action_classes, hidden_dim]

		attended_noun, _ = self.noun_attention(
			query, noun_features, noun_features
		)  # [batch_size, num_action_classes, hidden_dim]

		# Concatenate attended features
		combined_features = torch.cat(
			[attended_verb, attended_noun], dim=-1
		)  # [batch_size, num_action_classes, hidden_dim*2]

		# Get logits for each action class
		action_logits = self.classifier(
			combined_features
		)  # [batch_size, num_action_classes, num_action_classes]

		# Average across action queries to get final predictions
		#print('action_logits', action_logits.shape)
		action_logits = action_logits.mean(dim=1)  # [batch_size, num_action_classes]
		return action_logits

	def get_trainable_parameters(self):
		"""Get only trainable parameters"""
		return filter(lambda p: p.requires_grad, self.parameters())

	

class Net_Lavila(nn.Module):
	def __init__(self, args, spatial_verb_input_dim, spatial_noun_input_dim, z_dim, hidden_dim_spatial_verb, hidden_dim_spatial_noun, lags, threshold):
		super(Net_Lavila, self).__init__()	   
		#self.z_net_flow = BetaVAE_MLP(flow_input_dim, z_dim, hidden_dim_flow)
		self.z_net_spatial_verb = BetaVAE_MLP(spatial_verb_input_dim, z_dim, hidden_dim_spatial_verb)
		self.z_net_spatial_noun = BetaVAE_MLP(spatial_noun_input_dim, z_dim, hidden_dim_spatial_noun)
		self.class_net = Base_Net(input_len=8, out_len=1, input_dim=z_dim*2,
								  out_dim=args.action_dim, layer_nums=2, c_type="type2",
								  activation='leakyReLU', drop_out=0.1,
								  is_mean_std=False, hidden_dim=512, layer_norm=False)
		#self.class_net = HierarchicalActionPredictor(args=args, feat_dim=z_dim, hidden_dim=1024)

		self.rec_criterion = nn.MSELoss()
		#self.rec_criterion = nn.BCEWithLogitsLoss()
		self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
		
		self.register_buffer('base_dist_mean', torch.zeros(z_dim))
		self.register_buffer('base_dist_var', torch.eye(z_dim))
		self.lags = lags
		self.threshold = threshold
		self.transition_prior_fix_verb_noun = NPTransitionPrior(lags=lags,
													  latent_size=z_dim,
													  num_layers=3,
													  hidden_dim=256)
		self.transition_prior_fix_verb_verb = NPTransitionPrior(lags=lags,
													  latent_size=z_dim,
													  num_layers=3,
													  hidden_dim=256)
													  
		self.rec_weight = args.alpha
		self.sparsity_weight = args.gamma
		self.z_kl_weight = args.beta
		self.structure_weight = args.delta
		self.z_dim = z_dim
		self.args = args
		

	def forward(self, spatial_verb_data, spatial_noun_data, labels, act_logits, is_val):
		# source data is obs, target data is act
		#print(spatial_verb_data.shape)
		if is_val:
			#print(spatial_verb_data.shape)
			all_spatial_verb_z = []
			all_spatial_noun_z = []
			preds = []
			for i in range(spatial_verb_data.shape[1]):
			#for i in range(10):
				spatial_verb = spatial_verb_data[:,i,:,:]
				spatial_noun = spatial_noun_data[:,i,:,:]
				spatial_verb_x_recon, spatial_verb_mu, spatial_verb_logvar, spatial_verb_z = self.z_net_spatial_verb(spatial_verb)
				spatial_noun_x_recon, spatial_noun_mu, spatial_noun_logvar, spatial_noun_z = self.z_net_spatial_noun(spatial_noun)
				
				verb_noun_rec_loss, verb_noun_sparsity_loss, verb_noun_kld_loss, verb_noun_structure_loss \
				= self.__loss_function(spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb, spatial_verb_x_recon,
									spatial_noun_mu, spatial_noun_logvar, spatial_noun_z, spatial_noun, spatial_noun_x_recon, self.transition_prior_fix_verb_noun,
								   )
				verb_noun_loss = verb_noun_rec_loss * self.rec_weight + verb_noun_sparsity_loss * self.sparsity_weight + verb_noun_kld_loss * self.z_kl_weight + verb_noun_structure_loss * self.structure_weight
				
				#print(spatial_verb_z.shape)
				pred = self.class_net(torch.cat((spatial_verb_z, spatial_noun_z), dim=-1)).squeeze(1)
				#pred = self.class_net(spatial_verb_z, spatial_noun_z)
				#print(pred.shape)
				preds.append(pred.unsqueeze(1))
			preds = torch.cat(preds, dim=1)
			#print(preds.shape)
			preds = torch.mean(preds, dim=1)
			pred = preds + torch.mean(act_logits, dim=1)
			#pred = pred + act_logits.squeeze(1)
				
		else:		
			spatial_verb_x_recon, spatial_verb_mu, spatial_verb_logvar, spatial_verb_z = self.z_net_spatial_verb(spatial_verb_data)
			spatial_noun_x_recon, spatial_noun_mu, spatial_noun_logvar, spatial_noun_z = self.z_net_spatial_noun(spatial_noun_data)
									 
			verb_noun_rec_loss, verb_noun_sparsity_loss, verb_noun_kld_loss, verb_noun_structure_loss \
				= self.__loss_function(spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb_data, spatial_verb_x_recon,
										spatial_noun_mu, spatial_noun_logvar, spatial_noun_z, spatial_noun_data, spatial_noun_x_recon, self.transition_prior_fix_verb_noun,
									   #flow_mu, flow_logvar, flow_z, flow_data, flow_x_recon, self.transition_prior_fix_verb_noun,
									   #spatial_verb_mu, spatial_verb_logvar, spatial_verb_z, spatial_verb_data, spatial_verb_x_recon,  self.transition_prior_fix_verb_noun,
									   )
						
			verb_noun_loss = verb_noun_rec_loss * self.rec_weight + verb_noun_sparsity_loss * self.sparsity_weight + verb_noun_kld_loss * self.z_kl_weight + verb_noun_structure_loss * self.structure_weight
			
			pred = self.class_net(torch.cat((spatial_verb_z, spatial_noun_z), dim=-1)).squeeze(1)
			#pred = self.class_net(spatial_verb_z, spatial_noun_z)
			pred = pred + act_logits.squeeze(1)
		
		if not self.args.pretrain_vae:
			'''if is_val:
				preds = []
				print(spatial_verb_z.shape)
				for i in range(spatial_verb_z.shape[1]):
					pred = self.class_net(torch.cat((spatial_verb_z[:,i,:,:], spatial_noun_z[:,i,:,:]), dim=-1)).squeeze(1)
					preds.append(pred)
				preds = torch.cat(preds)
				preds = torch.mean(preds, dim=1)
			else:
				pred = self.class_net(torch.cat((spatial_verb_z, spatial_noun_z), dim=-1)).squeeze(1)
		#print(pred.shape, labels.shape)
			#print(pred.shape, act_logits.shape)
			if is_val:
				pred = preds + torch.mean(act_logits, dim=1)
			else:
				pred = pred + act_logits.squeeze(1)'''
			class_loss = self.cls_criterion(pred, labels)
			loss = verb_noun_loss + class_loss
			return loss, verb_noun_rec_loss, verb_noun_sparsity_loss, verb_noun_kld_loss, verb_noun_structure_loss, class_loss, pred
		else:
			loss = verb_noun_loss
			return loss, verb_noun_rec_loss, verb_noun_sparsity_loss, verb_noun_kld_loss, verb_noun_structure_loss, spatial_verb_z, spatial_noun_z

	def __loss_function(self, src_z_mean, src_z_std, src_z, src_x, src_rec, tgt_z_mean, tgt_z_std, tgt_z, tgt_x, tgt_rec, transition_prior_fix):			  
		obs_rec_loss = self.rec_criterion(src_x, src_rec)
		act_rec_loss = self.rec_criterion(tgt_x, tgt_rec)
						   
		rec_loss = obs_rec_loss + act_rec_loss
		z_mean = torch.cat((src_z_mean, tgt_z_mean), dim=0)
		z_std = torch.cat((src_z_std, tgt_z_std), dim=0)
		z = torch.cat((src_z, tgt_z), dim=0)

		b, length, _ = z_mean.shape
		q_dist = D.Normal(z_mean, torch.exp(z_std / 2))

		log_qz = q_dist.log_prob(z)

		p_dist = D.Normal(torch.zeros_like(z_mean[:, :self.lags]),
						  torch.ones_like(z_std[:, :self.lags]))
		log_pz_normal = torch.sum(p_dist.log_prob(z[:, :self.lags]), dim=[-2, -1])
		log_qz_normal = torch.sum(log_qz[:, :self.lags], dim=[-2, -1])

		kld_normal = (torch.abs(log_qz_normal - log_pz_normal) / self.lags).sum()
		log_qz_laplace = log_qz[:, self.lags:]

		residuals, logabsdet, hist_jac = transition_prior_fix.forward(z)

		log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
		kld_future = ((torch.sum(log_qz_laplace, dim=[-2, -1]) - log_pz_laplace) / (
				length - self.lags)).mean()

		kld_loss = (kld_normal + kld_future) / self.z_dim

		structure_loss = torch.tensor(0, device=z_mean.device)
		sparsity_loss = torch.tensor(0, device=z_mean.device)
		for jac in hist_jac:
			sparsity_loss = sparsity_loss + F.l1_loss(jac[:, 0, :self.lags * self.z_dim],
													  torch.zeros_like(jac[:, 0, :self.lags * self.z_dim]),
													  reduction='sum')
			src_jac, trg_jac = torch.chunk(jac, dim=0, chunks=2)
			threshold = torch.quantile(src_jac, self.threshold)
			#print(torch.max(src_jac), torch.min(src_jac), torch.mean(src_jac))
			#print(torch.max(trg_jac), torch.min(trg_jac), torch.mean(trg_jac))
			I_J1_src = (src_jac > threshold).bool()
			I_J1_trg = (trg_jac > threshold).bool()

			mask = torch.bitwise_xor(I_J1_src, I_J1_trg)
			structure_loss = structure_loss + torch.sum((src_jac[mask].detach() - trg_jac[mask]) ** 2)

		sparsity_loss = sparsity_loss / b
		structure_loss = structure_loss
		return rec_loss, sparsity_loss, kld_loss, structure_loss

	@property
	def base_dist(self):
		# Noise density function
		return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
