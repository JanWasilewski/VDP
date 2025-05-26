from utils_vdp import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VDPFirstLinear(nn.Module):
    """y = w.x + b"""
    def __init__(self, units, bias=True, sigma_init=-4.6):
        super(VDPFirstLinear, self).__init__()
        self.units = units
        self.first_run = True
        self.ini_sigma = -4.6
        self.bias = bias
        self.sigma_init = sigma_init

    def forward(self, inputs):  # inputs shape: [batch_size, seq_len, input_dim]
        if self.first_run:
            self.w_mu = nn.Parameter(torch.empty(self.units, inputs.size()[-1]))
            nn.init.xavier_normal_(self.w_mu)            
            self.w_sigma =  nn.Parameter(torch.full((self.units,), self.sigma_init)) # nn.Parameter(torch.empty(self.units).uniform_(self.sigma_init, -2.25))
            self.first_run = False
            if self.bias:
                self.b_mu = nn.Parameter(torch.zeros(self.units))
                self.b_sigma = nn.Parameter(torch.full((self.units,), self.sigma_init))
        
        w_sigma = F.softplus(self.w_sigma)
        kl_fc = kl_regularizer(self.w_mu, w_sigma)

        mu_out = torch.matmul(inputs, self.w_mu.t())
        sigma_out = x_Sigma_w_x_T(inputs, w_sigma) 
        
        if self.bias:
            b_sigma = F.softplus(self.b_sigma)
            kl_fc += kl_regularizer(self.b_mu, b_sigma, is_conv=False)
            mu_out += self.b_mu.view(1, -1)
            sigma_out += b_sigma.view(1, -1)
        
        sigma_out = sigma_out.clamp(min=1e-6)

        # sigma_out = F.softplus(sigma_out)
        # sigma_out = torch.nan_to_num(sigma_out, nan=1e-5, posinf=1.0, neginf=1.0)
        return mu_out, sigma_out, kl_fc

class VDPIntermediateLinear(nn.Module):
    """y = w.x + b"""
    def __init__(self, units, bias=True, sigma_init=-4.6):
        super(VDPIntermediateLinear, self).__init__()
        self.first_run = True
        self.units = units
        self.sigma_init = sigma_init
        self.bias = bias
    
    def forward(self, mu_in, sigma_in):
        if self.first_run:
            self.w_mu = nn.Parameter(torch.empty(self.units, mu_in.shape[-1])) #nn.Parameter(torch.ones(mu_in.shape[-1], self.units)*0.005)
            nn.init.xavier_normal_(self.w_mu)
            self.w_sigma = nn.Parameter(torch.full((self.units,), self.sigma_init)) # nn.Parameter(torch.empty(self.units).uniform_(self.sigma_init, -2.25))
            if self.bias:
                self.b_mu = nn.Parameter(torch.zeros(self.units))
                self.b_sigma = nn.Parameter(torch.full((self.units,), self.sigma_init))
            self.first_run = False
        
        w_sigma = F.softplus(self.w_sigma)
        
        kl_fc = kl_regularizer(self.w_mu.t(), w_sigma)
        mu_out = torch.matmul(mu_in, self.w_mu.t())

        sigma_1 = w_t_Sigma_i_w(self.w_mu.t(), sigma_in)
        sigma_2 = x_Sigma_w_x_T(mu_in, w_sigma)
        sigma_3 = tr_Sigma_w_Sigma_in(sigma_in, w_sigma)
        sigma_out = sigma_1 + sigma_2 + sigma_3

        if self.bias:
            b_sigma  = F.softplus(self.b_sigma)
            kl_fc += kl_regularizer(self.b_mu, b_sigma, is_conv=False)
            mu_out += self.b_mu.view(1, -1)
            sigma_out += b_sigma.view(1, -1)

        sigma_out = sigma_out.clamp(min=1e-6)

        # sigma_out = F.softplus(sigma_out)
        # sigma_out = torch.nan_to_num(sigma_out, nan=1e-5, posinf=1.0, neginf=1.0)

        return mu_out, sigma_out, kl_fc

class VDPFirstConv(nn.Module):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding=1, bias=True, ini_sigma=-4.6):
        
        super(VDPFirstConv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        self.first_run=True
        self.bias = bias
        self.ini_sigma = ini_sigma

    def forward(self, mu_in):
        batch_size = mu_in.size(0)
        num_channel = mu_in.size(1)
        if self.first_run:
               #self.w_mu = nn.Parameter(torch.ones(kernel_num, 1, kernel_size, kernel_size) *0.02)
            init_from_numpy = torch.tensor(np.random.normal(scale=0.05, size=(self.kernel_num, num_channel, self.kernel_size, self.kernel_size)), dtype=torch.float32)
            self.w_mu = nn.Parameter(init_from_numpy)
            self.w_sigma = nn.Parameter(torch.full((self.kernel_num,), self.ini_sigma))
            if self.bias:
                self.b_mu = nn.Parameter(torch.zeros(self.kernel_num))
                self.b_sigma = nn.Parameter(torch.full((self.kernel_num,), self.ini_sigma))     
            self.first_run=False

        w_sigma = F.softplus(self.w_sigma)
        kl_conv = kl_regularizer(self.w_mu, w_sigma, True) 

        mu_in = F.pad(mu_in, (self.padding, self.padding, self.padding, self.padding))
        mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)

        # Extract patches
        x_train_patches = F.unfold(mu_in, kernel_size=self.kernel_size, stride=self.kernel_stride)
        x_train_matrix = x_train_patches.view(batch_size, num_channel * self.kernel_size * self.kernel_size, -1)
        x_dim = x_train_matrix.size(1)
        x_train_matrix = torch.sum(x_train_matrix ** 2, dim=1) / x_dim
        X_XTranspose = x_train_matrix.unsqueeze(-1).repeat(1, 1, self.kernel_num)
        sigma_out = (w_sigma * X_XTranspose).view(batch_size, mu_out.shape[-1], mu_out.shape[-1], self.kernel_num).permute(0,3,1,2)
        
        if self.bias:
            b_sigma = F.softplus(self.b_sigma)
            kl_conv += kl_regularizer(self.b_mu, b_sigma, is_conv=False)
            mu_out += self.b_mu.view(1, -1, 1, 1)
            sigma_out += b_sigma.view(1, -1, 1, 1)

        sigma_out = sigma_out.clamp(min=1e-6)

        # sigma_out = F.softplus(sigma_out.view_as(mu_out))
        # sigma_out = torch.nan_to_num(sigma_out, nan=1e-5, posinf=1.0, neginf=1.0)
 
        return mu_out, sigma_out , kl_conv # mu_out: [batch_size, out_channels, H_out, W_out], sigma_out: [batch_size, L, kernel_num]


class VDPIntermediateConv(nn.Module):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding=1, bias=True):
        super(VDPIntermediateConv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        self.ini_sigma = -2.25
        self.min_sigma = -4.6
        self.first_run = True
        self.bias = bias

    def forward(self, mu_in, sigma_in):
        if self.first_run:
            self.w_mu = nn.Parameter(torch.empty(self.kernel_num, mu_in.size(1), self.kernel_size, self.kernel_size))
            nn.init.xavier_normal_(self.w_mu)  # Glorot initialization
            self.w_sigma = nn.Parameter(torch.full((self.kernel_num,), self.ini_sigma)) #nn.Parameter(torch.empty(self.kernel_num).uniform_(self.min_sigma, self.ini_sigma)) 
            if self.bias:
                self.b_mu = nn.Parameter(torch.zeros(self.kernel_num))
                self.b_sigma = nn.Parameter(torch.full((self.kernel_num,), self.ini_sigma))
            self.first_run = False


        w_sigma = F.softplus(self.w_sigma)
        kl_conv = kl_regularizer(self.w_mu, w_sigma, True)

        mu_in = F.pad(mu_in, (self.padding, self.padding, self.padding, self.padding))
        sigma_in = F.pad(sigma_in, (self.padding, self.padding, self.padding, self.padding))

        B, C, H, W = mu_in.shape
        K = self.w_mu.size(0)
        D = self.kernel_size * self.kernel_size * C

        mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)

        # Unfold patches
        sigma_patches = F.unfold(sigma_in, kernel_size=self.kernel_size, stride=self.kernel_stride).transpose(1, 2)  # [B, L, D]
        mu_patches = F.unfold(mu_in, kernel_size=self.kernel_size, stride=self.kernel_stride).transpose(1, 2)       # [B, L, D]

        mu_cov_sq = self.w_mu.view(K, -1).pow(2) / D           # [K, D]
        mu_sigma = sigma_patches @ mu_cov_sq.t()          # [B, L, K]
        trace_term = sigma_patches.sum(-1, keepdim=True) * w_sigma.view(1, 1, -1) / D
        mu_term = mu_patches.pow(2).sum(-1, keepdim=True) * w_sigma.view(1, 1, -1) / D

        sigma_out =(mu_sigma + trace_term + mu_term).permute(0, 2, 1).view(B, K, mu_out.shape[2], mu_out.shape[3])
        
        if self.bias:
            b_sigma = F.softplus(self.b_sigma)
            kl_conv += kl_regularizer(self.b_mu, b_sigma, is_conv=False)
            mu_out += self.b_mu.view(1, -1, 1, 1)
            sigma_out += b_sigma.view(1, -1, 1, 1)
            
        sigma_out = sigma_out.clamp(min=1e-6)
        return mu_out, sigma_out, kl_conv
    
class VDPMaxPooling(nn.Module):
    """VDP_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad=0):
        super(VDPMaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad # TODO - NOT USED

    def forward(self, mu_in, Sigma_in):
        batch_size, num_channel, hw_in, _ = mu_in.shape
        mu_out, indices = F.max_pool2d(mu_in, kernel_size=self.pooling_size, stride=self.pooling_stride, padding=self.pooling_pad, return_indices=True)
        
        # Get output dimensions
        hw_out = mu_out.shape[2]
        indices_flat = indices.view(batch_size, num_channel, -1)  # [batch_size, num_channel, new_size * new_size]
        
        # Compute x_index and y_index
        x_index = indices_flat % hw_in
        y_index = indices_flat // hw_in

        # Combine x_index and y_index to create a final index
        index = y_index * hw_in + x_index  # [batch_size, num_channel, new_size * new_size]
        Sigma_in_reshaped = Sigma_in.view(batch_size, num_channel, -1)  # [batch_size, num_channel, im_size * im_size]
        Sigma_out = Sigma_in_reshaped.gather(2, index)  # [batch_size, num_channel, new_size * new_size]
        Sigma_out = Sigma_out.view(batch_size, num_channel, hw_out, hw_out)  # [batch_size, num_channel, new_size, new_size]

        return mu_out, Sigma_out

class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_dim = mu_in.size(-1)
        mu_out = F.softmax(mu_in, dim=-1)
        
        grad = (mu_out - mu_out**2)**2
        sigma_out = grad * Sigma_in / mu_dim
        #sigma_out = F.softplus(sigma_out)
        sigma_out = torch.nan_to_num(sigma_out, nan=1e-5, posinf=1.0, neginf=1.0)

        return mu_out, sigma_out

class VDP_ReLU(nn.Module):
    def forward(self, mu_in, sigma_in):
        mu_out = F.relu(mu_in)
        mu_in.requires_grad_()
        out = F.relu(mu_in)
        gradi = torch.autograd.grad(out, mu_in, torch.ones_like(out), retain_graph=True)[0]
        sigma_out = activation_Sigma(gradi, sigma_in) 
        sigma_out = torch.nan_to_num(sigma_out, nan=1e-5, posinf=1.0, neginf=1.0)
        return mu_out, sigma_out

class VDP_GAP(nn.Module):
    def forward(self, mu_in, Sigma_in):
        """
        Args:
            mu_in: [B, C, H, W] - mean
            Sigma_in: [B, C, H, W] - variance map per pixel
        Returns:
            mu_out: [B, C]
            Sigma_out: [B, C, C] - diagonal covariance per sample
        """
        B, C, H, W = mu_in.shape
        n = H * W
        mu_out = mu_in.mean(dim=[2, 3])  # [B, C]
        sigma_out = Sigma_in.mean(dim=[2, 3]) / n  # [B, C]
        # Sigma_out = torch.stack([torch.diag(s) for s in sigma_out], dim=0)  # [B, C, C]
        return mu_out, sigma_out

class VDP_BatchNorm2D(nn.Module):
    def __init__(self, var_epsilon=1e-5, num_features=16, momentum=0.1):
        super().__init__()
        self.var_epsilon = var_epsilon
        self.momentum    = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, mu_in, sigma_in):
        if self.training:
            dims       = [0,2,3]
            batch_mean = mu_in.mean(dim=dims, keepdim=True)
            batch_var  = mu_in.var(dim=dims, unbiased=False, keepdim=True)

            # flatten to [C]
            new_mean = batch_mean.view(-1)
            new_var  = batch_var .view(-1)

            # correct in-place update: add_(other, alpha)
            self.running_mean.mul_(1 - self.momentum) \
                             .add_(new_mean,      alpha=self.momentum)
            self.running_var .mul_(1 - self.momentum) \
                             .add_(new_var,       alpha=self.momentum)

            mean, var = batch_mean, batch_var
        else:
            mean = self.running_mean.view(1,-1,1,1)
            var  = self.running_var .view(1,-1,1,1)

        w = self.weight.view(1,-1,1,1)
        b = self.bias  .view(1,-1,1,1)

        mu_out    = w * (mu_in - mean) / torch.sqrt(var + self.var_epsilon) + b
        scale     = w.pow(2) / (var + self.var_epsilon)
        sigma_out = sigma_in * scale

        return mu_out, sigma_out
    
class VDP_Dropout(nn.Module):
    def __init__(self, drop_prob):
        super(VDP_Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, mu_in, Sigma_in, training=True):
        if training:
            # Apply dropout to the mean
            mu_out = F.dropout(mu_in, p=self.drop_prob, training=training)

            # Identify non-zero elements (non-dropped positions)
            non_zero_mask = mu_out != 0  # Boolean mask
            non_zero_sigma_mask = Sigma_in[non_zero_mask]

            # Gather indices where the mask is True
            idx_sigma = non_zero_mask.nonzero(as_tuple=True)

            # Compute Sigma_out using scatter
            Sigma_out = torch.zeros_like(Sigma_in)
            Sigma_out[idx_sigma] = non_zero_sigma_mask / mu_out.size(-1)
        else:
            mu_out = mu_in
            Sigma_out = Sigma_in


class VDPResnetBasicBlock(nn.Module):
    def __init__(self, out_channels, stride1=1, stride2=1, previous_channels=None, bias=True):
        super(VDPResnetBasicBlock, self).__init__()
        self.conv1 = VDPIntermediateConv(kernel_size=3, kernel_num=out_channels, kernel_stride=stride1, padding=1, bias=bias)
        self.bn1 = VDP_BatchNorm2D(num_features=out_channels)
        self.relu = VDP_ReLU()
        self.conv2 = VDPIntermediateConv(kernel_size=3, kernel_num=out_channels, kernel_stride=stride2, padding=1, bias=bias)
        self.bn2 = VDP_BatchNorm2D(num_features=out_channels)
        
        # Skip connection
        self.isShortcut = out_channels != previous_channels
        if self.isShortcut:
            self.shortcut_conv = VDPIntermediateConv(kernel_size=1, kernel_num=out_channels, kernel_stride=stride1, padding=0, bias=bias)
            self.shortcut_batchnorm = VDP_BatchNorm2D(num_features=out_channels)
                
    def forward(self, mu, sigma):
        mu_in, sigma_in = mu, sigma
        mu_out, sigma_out, kl_conv1 = self.conv1(mu_in, sigma_in)
        mu_out, sigma_out = self.bn1(mu_out, sigma_out)
        mu_out, sigma_out = self.relu(mu_out, sigma_out)
        mu_out, sigma_out, kl_conv2 = self.conv2(mu_out, sigma_out)
        mu_out, sigma_out = self.bn2(mu_out, sigma_out)
        
        kl_shortcut = 0.0
        if self.isShortcut:
            mu_in, sigma_in, kl_shortcut = self.shortcut_conv(mu_in, sigma_in)
            mu_in, sigma_in = self.shortcut_batchnorm(mu_in, sigma_in)
        mu_out += mu_in
        sigma_out += sigma_in # torch.sqrt(sigma_in **2 + sigma_out **2)
        
        mu_out, sigma_out = self.relu(mu_out, sigma_out)

        kl = kl_conv1 + kl_conv2 + kl_shortcut
        return mu_out, sigma_out, kl


