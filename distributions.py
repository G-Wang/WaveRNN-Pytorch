import torch
from torch.distributions import Beta

def sample_from_beta_dist(y_hat):
    """
    y_hat (batch_size x seq_len x 2):
    
    """
    # take exponentional to ensure positive
    loc_y = y_hat.exp()
    alpha = loc_y[:,:,0].unsqueeze(-1)
    beta = loc_y[:,:,1].unsqueeze(-1)
    dist = Beta(alpha, beta)
    sample = dist.sample()
    # rescale sample from [0,1] to [-1, 1]
    sample = 2.0*sample-1.0
    return sample


def Beta_MLE_Loss(y_hat, y, reduce=True):
    """y_hat (batch_size x seq_len x 2)
        y (batch_size x seq_len x 1)
        
    """
    # take exponentional to ensure positive
    loc_y = y_hat.exp()
    alpha = loc_y[:,:,0].unsqueeze(-1)
    beta = loc_y[:,:,1].unsqueeze(-1)
    dist = Beta(alpha, beta)
    # rescale y to be between 
    y = (y + 1.0)/2.0
    # note that we will get inf loss if y == 0 or 1.0 exactly, so we will clip it slightly just in case
    y = torch.clamp(y, 1e-5, 0.99999)
    # compute logprob
    loss = -dist.log_prob(y).squeeze(-1)
    if reduce:
        return loss.mean()
    else:
        return loss