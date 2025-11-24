import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functorch import vmap, jacfwd, grad
from torch.autograd.functional import jacobian

import torch.distributions as tD

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def reparametrize(mu, logvar):
    # Sanitize inputs and clamp for numerical stability
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    logvar = torch.nan_to_num(logvar, nan=0.0, posinf=10.0, neginf=-10.0)
    logvar = torch.clamp(logvar, min=-20.0, max=10.0)
    std = torch.exp(logvar * 0.5).clamp_min(1e-6)
    eps = torch.randn_like(std)
    return mu + std * eps

class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MLPEncoder(nn.Module):

    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        # TODO: Do not use ground-truth decoder architecture 
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, z):
        return self.net(z)

class Inference(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, lag, z_dim, num_layers=4, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.lag = lag
        self.f1 = nn.Linear(lag*z_dim, z_dim*2)
        self.f2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.net = NLayerLeakyMLP(in_features=hidden_dim, 
                                  out_features=z_dim*2, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)

    def forward(self, x):
        zs = x[:,:self.lag*self.z_dim]
        distributions = self.f1(zs)
        enc = self.f2(x[:,self.lag*self.z_dim:])
        distributions = distributions + self.net(enc)
        return distributions

class MLP(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=1536, z_dim=10, hidden_dim=256, leaky_relu_slope=0.2):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, 2*z_dim)
        )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Linear(hidden_dim, input_dim)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

        distributions = self._encode(x)
        mu = distributions[:,:, :self.z_dim]
        logvar = distributions[:,:, self.z_dim:]
        # Sanitize encoder outputs to avoid NaNs/Infs downstream
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        mu = torch.clamp(mu, min=-100.0, max=100.0)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=10.0, neginf=-10.0)
        logvar = torch.clamp(logvar, min=-20.0, max=10.0)
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class NPTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        self.gs = nn.ModuleList([MLP(input_dim=lags*latent_size + 1, hidden_dim=hidden_dim,
                                output_dim=1,  num_layers=num_layers) for _ in range(latent_size)])
        # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags + 1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags+1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)
        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)
            
            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags*mask[i], batch_x_t[:, i:i+1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                (batch_x_lags, batch_x_t[:, i:i+1]), dim=-1)
            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.clamp(torch.abs(data_J[:, -1]), min=1e-6))
            logabsdet = torch.nan_to_num(logabsdet, nan=0.0, posinf=50.0, neginf=-50.0)

            sum_log_abs_det_jacobian += logabsdet
            residual = torch.nan_to_num(residual)
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian

class NPChangeTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            embedding_dim,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.latent_size = latent_size
        self.lags = lags
        self.gs = nn.ModuleList([MLP(input_dim=hidden_dim+lags*latent_size + 1, hidden_dim=hidden_dim,
                                output_dim=1,  num_layers=num_layers) for _ in range(latent_size)])
        self.fc = MLP(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x, embeddings):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags + 1, step=1).transpose(2, 3)
        # (batch_size, lags+length, hidden_dim)
        #print(embeddings.shape)
        embeddings = self.fc(embeddings)
        # batch_embeddings: (batch_size, lags+length, hidden_dim) -> (batch_size, length, lags+1, hidden_dim) -> (batch_size*length, hidden_dim)
        # batch_embeddings = embeddings.unfold(
        #     dimension=1, size=self.lags+1, step=1).transpose(2, 3)[:, :, -1].reshape(batch_size * length, -1)
        batch_embeddings = embeddings[:, -length:].expand(batch_size,length,-1).reshape(batch_size*length,-1)
        batch_x = batch_x.reshape(-1, self.lags+1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1:]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)
        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)
            batch_inputs = torch.cat((batch_embeddings, batch_x_lags, batch_x_t[:, :, i]), dim=-1)
            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.clamp(torch.abs(data_J[:, -1]), min=1e-6))
            logabsdet = torch.nan_to_num(logabsdet, nan=0.0, posinf=50.0, neginf=-50.0)

            sum_log_abs_det_jacobian += logabsdet
            residual = torch.nan_to_num(residual)
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian
