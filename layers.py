from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VDPFirstLinear(nn.Module): #NOT CHECKED
    """y = w.x + b"""
    def __init__(self, units):
        super(VDPFirstLinear, self).__init__()
        self.units = units
        self.first_run = True
        self.ini_sigma = -4.6


    def forward(self, inputs):  # inputs shape: [batch_size, seq_len, input_dim]
        if self.first_run:
            self.w_mu = nn.Parameter(torch.empty(self.units, inputs.size()[-1]))
            nn.init.xavier_normal_(self.w_mu)            
            self.w_sigma = nn.Parameter(torch.empty(self.units).uniform_(-4.6, -2.25)) # nn.Parameter(torch.full((self.units,), -4.6))
            self.first_run = False

        # KL Divergence regularization (you will need to define this function)
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)

        # Mean output
        mu_out = torch.matmul(inputs, self.w_mu.t())  # Shape: [50, 17, 64]

        # Variance
        W_Sigma = torch.log(1 + torch.exp(self.w_sigma))  # Shape: [64]
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma)  # Define this function similarly to how it was done in Keras
        Sigma_out = F.softplus(Sigma_out)
        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out, kl_fc

class VDPIntermediateLinear(nn.Module):
    """y = w.x + b"""
    def __init__(self, units):
        super(VDPIntermediateLinear, self).__init__()
        self.first_run = True
        self.units = units
        
    def forward(self, mu_in, Sigma_in):
        if self.first_run:
            self.w_mu = nn.Parameter(torch.empty(self.units, mu_in.shape[-1])) #nn.Parameter(torch.ones(mu_in.shape[-1], self.units)*0.005)
            nn.init.xavier_normal_(self.w_mu)
            self.w_sigma = nn.Parameter(torch.empty(self.units).uniform_(-4.6, -2.25)) #nn.Parameter(torch.full((self.units,), -4.6))
            self.first_run = False

        kl_fc = kl_regularizer(self.w_mu.t(), self.w_sigma)

        mu_out = torch.matmul(mu_in, self.w_mu.t())
        W_Sigma = torch.log(1 + torch.exp(self.w_sigma))

        # Compute covariances
        Sigma_1 = w_t_Sigma_i_w(self.w_mu.t(), Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)
        Sigma_3 = tr_Sigma_w_Sigma_in(Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3
        Sigma_out = F.softplus(Sigma_out)

        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out, kl_fc

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

        kl_conv = kl_regularizer(self.w_mu, self.w_sigma, True) 
        w_sigma_2 = torch.log1p(torch.exp(self.w_sigma))

                
        mu_in = F.pad(mu_in, (self.padding, self.padding, self.padding, self.padding))
        mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)

        # Extract patches
        x_train_patches = F.unfold(mu_in, kernel_size=self.kernel_size, stride=self.kernel_stride)
        x_train_matrix = x_train_patches.view(batch_size, num_channel * self.kernel_size * self.kernel_size, -1)
        x_dim = x_train_matrix.size(1)
        x_train_matrix = torch.sum(x_train_matrix ** 2, dim=1) / x_dim
        X_XTranspose = x_train_matrix.unsqueeze(-1).repeat(1, 1, self.kernel_num)
        Sigma_out = (w_sigma_2 * X_XTranspose).view(batch_size, mu_out.shape[-1], mu_out.shape[-1], self.kernel_num).permute(0,3,1,2)
        Sigma_out = F.softplus(Sigma_out.view_as(mu_out))
        
        if self.bias:
            kl_conv += kl_regularizer(self.b_mu, self.b_sigma, is_conv=False)
            mu_out += self.b_mu.view(1, -1, 1, 1)
            Sigma_out += F.softplus(torch.log1p(torch.exp(self.b_sigma))).view(1, -1, 1, 1)
            
        
        return mu_out, Sigma_out , kl_conv # mu_out: [batch_size, out_channels, H_out, W_out], Sigma_out: [batch_size, L, kernel_num]


class VDPIntermediateConv(nn.Module):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding=1):
        super(VDPIntermediateConv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        self.ini_sigma = -2.25
        self.min_sigma = -4.6
        self.first_run = True

    def extract_patches(self, input, kernel_size, stride):
        """
        Extracts patches from the input tensor using PyTorch's unfold.
        """
        return input.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

    def forward(self, mu_in, sigma_in):
        if self.first_run:
            self.w_mu = nn.Parameter(torch.empty(self.kernel_num, mu_in.size(1), self.kernel_size, self.kernel_size))
            nn.init.xavier_normal_(self.w_mu)  # Glorot initialization
            self.w_sigma = nn.Parameter(torch.empty(self.kernel_num).uniform_(self.min_sigma, self.ini_sigma)) 
            self.first_run = False


        kl_conv = kl_regularizer(self.w_mu, self.w_sigma, True)
        w_sigma_2 = torch.log1p(torch.exp(self.w_sigma))
        # Apply padding to mu_in and sigma_in
        mu_in = F.pad(mu_in, (self.padding, self.padding, self.padding, self.padding))
        sigma_in = F.pad(sigma_in, (self.padding, self.padding, self.padding, self.padding))

        # ORIGINAL   
        # batch_size = mu_in.size(0)
        # num_channels = mu_in.size(1)     
        # mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)

        # # Extract patches from sigma_in
        # patches_sigma = self.extract_patches(sigma_in, self.kernel_size, self.kernel_stride).reshape(batch_size, -1, self.kernel_size * self.kernel_size * num_channels)
        # mu_cov_square = self.w_mu.view(self.kernel_num, -1).pow(2)
        # mu_dim = mu_cov_square.size(1)
        # mu_wT_sigma_gs_mu_w = torch.matmul(patches_sigma, mu_cov_square.t() / mu_dim)  # Shape: [batch_size, new_im_size * new_im_size, kernel_num]
        # trace = torch.sum(patches_sigma, dim=-1, keepdim=True)
        # trace = trace.expand(-1, -1, self.kernel_num) * w_sigma_2 / mu_dim
        
        # patches_mu = self.extract_patches(mu_in, self.kernel_size, self.kernel_stride).reshape(batch_size, -1, self.kernel_size * self.kernel_size * num_channels)
        # mu_gT_mu_g = torch.sum(patches_mu.pow(2), dim=-1, keepdim=True)
        # sigmaw_mu_gT_mu_g = (w_sigma_2 * mu_gT_mu_g.expand(-1, -1, self.kernel_num) / mu_dim)
        # Sigma_out = trace + mu_wT_sigma_gs_mu_w + sigmaw_mu_gT_mu_g
        # Sigma_out = F.softplus(Sigma_out.view_as(mu_out))
        
        # CLAUDE
        
        # B, C, H, W = mu_in.shape
        # K = self.w_mu.size(0)
        # D = self.kernel_size * self.kernel_size * C
        
        # # Compute the main convolution output
        # mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)
        # out_H, out_W = mu_out.shape[2], mu_out.shape[3]
        # L = out_H * out_W
        
        # # Precompute once
        # w_sigma_2 = torch.log1p(torch.exp(self.w_sigma))
        # mu_cov_sq = self.w_mu.view(K, -1).pow(2) / D
        
        # # Initialize output tensor
        # Sigma_out = torch.zeros_like(mu_out)
        
        # # Process sample by sample (extreme memory reduction)
        # for i in range(B):
        #     # Process one sample at a time
        #     for h_idx in range(0, out_H, 1):
        #         h_end = min(h_idx + 1, out_H)
                
        #         # Calculate the corresponding input region
        #         h_start_in = h_idx * self.kernel_stride
        #         h_end_in = h_start_in + self.kernel_size
                
        #         for w_idx in range(0, out_W, 1):
        #             w_end = min(w_idx + 1, out_W)
                    
        #             # Calculate the corresponding input region
        #             w_start_in = w_idx * self.kernel_stride
        #             w_end_in = w_start_in + self.kernel_size
                    
        #             # Extract the patches directly without unfold
        #             sigma_patch = sigma_in[i:i+1, :, h_start_in:h_end_in, w_start_in:w_end_in]
        #             mu_patch = mu_in[i:i+1, :, h_start_in:h_end_in, w_start_in:w_end_in]
                    
        #             # Reshape patches to vectors
        #             sigma_patch_flat = sigma_patch.reshape(1, -1)  # [1, D]
        #             mu_patch_flat = mu_patch.reshape(1, -1)        # [1, D]
                    
        #             # Compute the three terms directly
        #             # 1. Trace term
        #             trace_term = sigma_patch_flat.sum() * w_sigma_2 / D  # [K]
                    
        #             # 2. Mu-sigma term
        #             mu_sigma_term = torch.matmul(sigma_patch_flat, mu_cov_sq.t())  # [1, K]
                    
        #             # 3. Mu term
        #             mu_term = (mu_patch_flat.pow(2).sum()) * w_sigma_2 / D  # [K]
                    
        #             # Combine terms
        #             patch_result = F.softplus(mu_sigma_term + trace_term + mu_term)
                    
        #             # Update the output tensor at the correct position
        #             Sigma_out[i, :, h_idx:h_end, w_idx:w_end] = patch_result.view(-1, 1, 1)
                    
        #             # Free memory
        #             del sigma_patch, mu_patch, sigma_patch_flat, mu_patch_flat, trace_term, mu_sigma_term, mu_term, patch_result
            
        #     # Clear cache after each sample
        #     if hasattr(torch, 'cuda') and torch.cuda.is_available():
        #         torch.cuda.empty_cache()

        # GPT - equivalent to original
        B, C, H, W = mu_in.shape
        K = self.w_mu.size(0)
        D = self.kernel_size * self.kernel_size * C
        w_sigma_2 = torch.log1p(torch.exp(self.w_sigma))

        mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)

        # Unfold patches
        sigma_patches = F.unfold(sigma_in, kernel_size=self.kernel_size, stride=self.kernel_stride).transpose(1, 2)  # [B, L, D]
        mu_patches = F.unfold(mu_in, kernel_size=self.kernel_size, stride=self.kernel_stride).transpose(1, 2)       # [B, L, D]

        mu_cov_sq = self.w_mu.view(K, -1).pow(2) / D           # [K, D]
        mu_sigma = sigma_patches @ mu_cov_sq.t()          # [B, L, K]
        trace_term = sigma_patches.sum(-1, keepdim=True) * w_sigma_2.view(1, 1, -1) / D
        #mu_sq = mu_patches.pow(2).sum(-1, keepdim=True)
        mu_term = mu_patches.pow(2).sum(-1, keepdim=True) * w_sigma_2.view(1, 1, -1) / D

        Sigma_out = F.softplus((mu_sigma + trace_term + mu_term).permute(0, 2, 1).view(B, K, mu_out.shape[2], mu_out.shape[3]))
        
        return mu_out, Sigma_out, kl_conv
    
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
        Sigma_out = grad * Sigma_in / mu_dim
        Sigma_out = F.softplus(Sigma_out)
        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out

class VDP_ReLU(nn.Module):
    """ReLU"""
    def __init__(self):
        super(VDP_ReLU, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_out = F.relu(mu_in)

        # Compute gradients
        mu_in.requires_grad_()
        out = F.relu(mu_in)
        gradi = torch.autograd.grad(out, mu_in, torch.ones_like(out), retain_graph=True)[0]

        # Compute new Sigma
        Sigma_out = activation_Sigma(gradi, Sigma_in) 
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out

class VDP_GAP(nn.Module):
    def __init__(self):
        super(VDP_GAP, self).__init__()

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
    def __init__(self, var_epsilon=1e-4):
        super(VDP_BatchNorm2D, self).__init__()
        self.var_epsilon = var_epsilon
        self.first_run = True

    def forward(self, mu_in, Sigma_in):
        """
        Args:
            mu_in: [B, C, H, W] - mean of the activations
            Sigma_in: [B, C, H, W] - diagonal variance (same shape)
        Returns:
            mu_out: [B, C, H, W]
            Sigma_out: [B, C, H, W]
        """
        if self.first_run:
            self.gamma = nn.Parameter(torch.ones(mu_in.size(1))).view(1, -1, 1, 1)
            self.beta = nn.Parameter(torch.zeros(mu_in.size(1))).view(1, -1, 1, 1)
            self.first_run = False
        # Compute batch mean and variance over batch and spatial dims
        
        dims = [0, 2, 3]
        mean = mu_in.mean(dim=dims, keepdim=True)
        var = mu_in.var(dim=dims, unbiased=False, keepdim=True)
        mu_out = self.gamma * (mu_in - mean) / torch.sqrt(var + self.var_epsilon) + self.beta
        scaling = self.gamma**2 / (var + self.var_epsilon)
        Sigma_out = Sigma_in * scaling  # same shape as input

        return mu_out, Sigma_out
    
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

class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_dim = mu_in.size(-1)
        mu_out = F.softmax(mu_in, dim=-1)
        
        grad = (mu_out - mu_out**2)**2
        Sigma_out = grad * Sigma_in / mu_dim
        Sigma_out = F.softplus(Sigma_out)
        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out

class VDPResnetBasicBlock(nn.Module):
    def __init__(self, out_channels, stride1=1, stride2=1, previous_channels=None):
        super(VDPResnetBasicBlock, self).__init__()
        self.conv1 = VDPIntermediateConv(kernel_size=3, kernel_num=out_channels, kernel_stride=stride1, padding=1)
        self.bn1 = VDP_BatchNorm2D()
        self.relu = VDP_ReLU()
        self.conv2 = VDPIntermediateConv(kernel_size=3, kernel_num=out_channels, kernel_stride=stride2, padding=1)
        self.bn2 = VDP_BatchNorm2D()
        
        # Skip connection
        self.isShortcut = (stride1 != 1) or (stride2 != 1) or (out_channels != previous_channels)
        if self.isShortcut:
            self.shortcut_conv = VDPIntermediateConv(kernel_size=1, kernel_num=out_channels, kernel_stride=stride1, padding=0)
            self.shortcut_batchnorm = VDP_BatchNorm2D()
                
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


