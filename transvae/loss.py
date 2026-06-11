import torch
import torch.nn.functional as F


def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, beta=1, lambda_prop=0.01):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence + Mask Length Prediction + Property Prediction"
    x = x.long()[:,1:] - 1 # shift by 1 for start token, and -1 for 0 indexing
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        # PREDLOSS = F.mse_loss(pred_prop.squeeze(-1), true_prop)*0.01
        PREDLOSS = torch.nn.HuberLoss(reduction='mean', delta=10.0)(pred_prop.squeeze(-1), true_prop)
    else:
        PREDLOSS = torch.tensor(0.)
    if torch.isnan(KLD).any().item() or torch.isinf(KLD).any().item():
        KLD = torch.tensor(0.)
    PREDLOSS = PREDLOSS * lambda_prop
    return BCEmol + BCEmask + KLD + PREDLOSS, BCEmol, BCEmask, KLD, PREDLOSS