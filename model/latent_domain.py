import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functorch import vmap, jacfwd, grad
from torch.autograd.functional import jacobian
import torch.distributions as tD

def conv1x1(c_in, c_out, stride=1):
    """1x1 convolution w/o padding"""
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        """Initialize sparsemax activation.
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim


    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device="cuda", dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output


    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
      
class SparseBattery(nn.Module):
    def __init__(self, num_adapters, c_in, c_out, stride3x3):
        super(SparseBattery, self).__init__()
        self.gate = nn.Sequential(nn.Linear(c_in, num_adapters), Sparsemax(dim=1))
        self.adapters = nn.ModuleList([conv1x1(c_in, c_out, stride3x3) for _ in range(num_adapters)])

    def forward(self, x):
        # Contract batch over height and width
        g = self.gate(x.mean(dim=(2, 3)))
        #g = self.gate(x.mean(dim=(2)))
        h = []
        for k in range(len(self.adapters)):
            h.append(g[:, k].view(-1, *3*[1]) * self.adapters[k](x))

        out = sum(h)
        return out
        
class DomainEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_adapters):
        super(DomainEncoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.latent_domain = SparseBattery(n_adapters, 1, 1, 1)

    def forward(self, x):        
        x = self.latent_domain(x.unsqueeze(1))
        x, _ = self.gru(x.squeeze(1))
        return x
        
'''model = DomainEncoder(1536, 12, 32).cuda()
x = torch.rand(64, 3, 1536).cuda()
out = model(x)
print(out.shape)'''
        

