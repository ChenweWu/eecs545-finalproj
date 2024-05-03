
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F

def constant_init(value):
    def _init(tensor):
        with torch.no_grad():
            nn.init.constant_(tensor, value)
    return _init

class DenseMonotone(nn.Module):
    def __init__(self, in_features, out_features, kernel_init=None, bias_init=None, use_bias=True):
        super(DenseMonotone, self).__init__()
        seed_everything(42)
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)


        nn.init.normal_(self.weight)

        # Bias initialization
        if use_bias and bias_init is not None:
            bias_init(self.bias)

    def forward(self, x):
        # Ensure weights are non-negative
        weight = torch.abs(self.weight)
        output = F.linear(x, weight, self.bias)
        return output


class MonotonicMLP(nn.Module):
    def __init__(self, in_features, n_features=1024, nonlinear=True):
        super(MonotonicMLP, self).__init__()
        self.n_features = n_features
        self.nonlinear = nonlinear
        self.gamma_min = 0.0001
        self.gamma_max = 1
        n_out = in_features
        init_bias = self.gamma_min
        init_scale = self.gamma_max - init_bias

        self.l1 = DenseMonotone(in_features, n_out,
                                kernel_init=None,
                                bias_init=constant_init(init_bias))
        if self.nonlinear:
            self.l2 = DenseMonotone(in_features, n_features)
            self.l3 = DenseMonotone(n_features, n_out, use_bias=False)

    def forward(self, t, det_min_max=False):
        

        if t.dim() == 0:
            t = t * torch.ones(1, 1, device=t.device)
        else:
            t = t.view(-1, 1)

        h = self.l1(t)
        
        if self.nonlinear:
            _h = 2. * (t - 0.5)  # scale input to [-1, +1]
            _h = self.l2(_h)
            
            _h = 2 * (torch.sigmoid(_h) - 0.5)  # more stable than torch.tanh(h)
            _h = self.l3(_h) / self.n_features
            
            h += _h
        # scale-up
        h = torch.sigmoid(h) * 4 - 4
        return torch.squeeze(h, -1)
