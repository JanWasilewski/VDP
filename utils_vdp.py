import torch
import torch.nn.functional as F
# -------------------- LOSS FUNCTIONS -------------------

def gaussian_loss(mu, sigma, targets):
    targets = F.one_hot(targets.long().view(-1), num_classes= mu.shape[1]).float()
    sigma = torch.clamp(sigma, min=1e-6)  
    precision = 1.0 / sigma
    loss1 = ((targets - mu) ** 2 * precision)
    loss2 = torch.log(sigma)
    loss = 0.5 * (loss1 + loss2).sum(dim=1)
    return loss.mean()


def mc_nll(mu, sigma, y_gt, num_samples=10):
    samples = mu.unsqueeze(1) + torch.sqrt(sigma.unsqueeze(1)) * torch.randn(mu.size(0), num_samples, mu.size(1), device=mu.device)
    probs = F.softmax(samples, dim=-1)  # [B, S, C]
    y_onehot = F.one_hot(y_gt, num_classes=mu.size(1)).float()  # [B, C]
    y_onehot = y_onehot.unsqueeze(1)  # [B, 1, C]
    
    log_probs = torch.log((probs * y_onehot).sum(dim=-1) + 1e-8)  # [B, S]
    nll = -log_probs.mean()  
    return nll

def dirichlet_loss(mu, sigma, targets, eps=1e-8, lambda_reg=0.01):
    # 1) build strictly‐positive evidence
    evidence    = F.softplus(mu) / (sigma + eps)      # [B, C]
    alpha       = evidence + 1                        # [B, C]
    alpha_0     = alpha.sum(dim=1)                    # [B]

    # 2) NLL term for a single one‐hot target
    target_oh   = F.one_hot(targets, mu.size(1)).float()
    alpha_t     = (alpha * target_oh).sum(dim=1)      # [B]
    main_loss   = torch.log(alpha_0 + eps) - torch.log(alpha_t + eps)

    # 3) regularize only the uncertainty on the true class
    sigma_t     = (sigma * target_oh).sum(dim=1)      # [B]
    reg_loss    = lambda_reg * sigma_t

    return (main_loss + reg_loss).mean()


def dirichlet_loss_old2(mu, sigma, targets, eps=1e-8, lambda_reg=0.01):
    """
    mu: [B, C] - predicted mean logits
    sigma: [B, C] - predicted variance logits (uncertainty)
    targets: [B] - ground truth labels
    lambda_reg: regularization weight
    """

    # Better evidence calculation - inversely related to uncertainty
    evidence = mu / (sigma + eps)  # [B, C]
    
    # Dirichlet concentration parameters 
    alpha = evidence + 1  # avoid zero concentration
    
    # Compute alpha_0 (sum of all alpha parameters)
    alpha_0 = torch.sum(alpha, dim=1)  # [B]
    
    # Get target alphas
    target_oh = F.one_hot(targets, num_classes=mu.size(1)).float()  # [B, C]
    alpha_target = torch.sum(alpha * target_oh, dim=1)  # [B]
    
    # Main loss term - simplified approximation of Dirichlet negative log-likelihood
    main_loss = torch.log(alpha_0) - torch.log(alpha_target)
    
    # Regularization term to encourage low uncertainty for predicted class
    reg_loss = lambda_reg * torch.sum(mu * sigma, dim=1)

    # Total loss
    loss = main_loss + reg_loss
    
    return loss.mean()


def dirichlet_loss_old(targets, mu, sigma, eps=1e-8):
    """
    mu: [B, C] - predicted mean logits
    sigma: [B, C] - predicted variance logits (uncertainty)
    targets: [B] - ground truth labels
    """
    p = F.softmax(mu, dim=1)  # [B, C]
    evidence = p**2 / (sigma + eps)  # [B, C]

    # Dirichlet concentration parameters
    alpha = evidence + 1  # avoid zero concentration

    # Dirichlet negative log likelihood
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]
    # Gather ground truth alpha
    idx = targets.unsqueeze(1)
    alpha_y = torch.gather(alpha, 1, idx).squeeze(1)  # [B]
    loss = -torch.log(alpha_y / (alpha_0.squeeze(1) + eps))  # [B]
    return loss.mean()


# ------------------- REGULARIZERS -------------------
def kl_regularizer(mu, logvar, is_conv=False):
    if is_conv:
        # Reshape for the convolution case
        k = mu.shape[0]
        mu = mu.view(-1, k)  # Reshape the tensor for convolution case
    
    n = mu.shape[0]
    prior_var = 0.01 #1.0
    kl = -torch.mean(
        (1 + logvar - torch.log(1 + torch.exp(logvar)) / prior_var)
        - (torch.sum(mu ** 2, dim=0) / (prior_var))
    
    #    - (torch.sum(mu ** 2, dim=0) / (n * prior_var))
    )
    
    # Replace NaN and Inf values with a small constant
    kl = torch.where(torch.isnan(kl), torch.tensor(1.0e-5, device=kl.device), kl)
    kl = torch.where(torch.isinf(kl), torch.tensor(1.0e5, device=kl.device), kl)
    
    return kl



# --------------------- HELPERS ------------------------
def x_Sigma_w_x_T(x, W_Sigma):
    dim = x.shape[-1]
    x = x / dim
    xx_t = torch.sum(x * x, dim=-1, keepdim=True)
    return xx_t * W_Sigma

def w_t_Sigma_i_w(w_mu, in_Sigma):
    dim = torch.sqrt(torch.tensor(w_mu.shape[0], dtype=torch.float32))
    w_mu = w_mu / dim
    Sigma_1 = torch.matmul(in_Sigma, w_mu * w_mu)
    return Sigma_1

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    dim = W_Sigma.shape[-1]
    Sigma = torch.sum(in_Sigma, dim=-1, keepdim=True)
    return Sigma * W_Sigma / dim

def activation_Sigma(gradi, Sigma_in):
    grad1 = gradi * gradi  
    dim = grad1.shape[1]
    return Sigma_in * grad1 / dim 

def get_parameters(model):
    params = list(model.parameters())
    return [params[i:i+2] for i in range(0, len(params), 2)]

