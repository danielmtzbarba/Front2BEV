import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import balanced_binary_cross_entropy, kl_divergence_loss, \
     focal_loss, prior_offset_loss, prior_uncertainty_loss

def calc_weights(priors, weight_mode="inverse"):
    if weight_mode == 'inverse':
        class_weights = 1 / priors
    elif weight_mode == 'sqrt_inverse':
        class_weights = torch.sqrt(1 / priors)
    elif weight_mode == 'equal':
        class_weights = torch.ones_like(priors)
    else:
        raise ValueError('Unknown weight mode option: ' + weight_mode)
    print(weight_mode, class_weights)
    return class_weights

class OccupancyCriterion(nn.Module):

    def __init__(self, priors, xent_weight=1., uncert_weight=0., 
                 weight_mode='sqrt_inverse'):
        super().__init__()

        self.xent_weight = xent_weight
        self.uncert_weight = uncert_weight

        self.priors = torch.tensor(priors)
        self.class_weights = calc_weights(self.priors)
    

    def forward(self, logits, labels, mask, *args):

        # Compute binary cross entropy loss
        self.class_weights = self.class_weights.to(logits)
        bce_loss = balanced_binary_cross_entropy(logits, labels,
                                                 mask, self.class_weights)
        
        # Compute uncertainty loss for unknown image regions
        self.priors = self.priors.to(logits)
        uncert_loss = prior_uncertainty_loss(logits, mask, self.priors)

        return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight


class VaeOccupancyCriterion(OccupancyCriterion):

    def __init__(self, priors, xent_weight=0.9, uncert_weight=0., weight_mode='sqrt_inverse',  kld_weight=0.1):
        super().__init__(priors, xent_weight, uncert_weight, weight_mode)
        self.kld_weight = kld_weight

    def forward(self, logits, labels, mask, mu, logvar):
        kld_loss = kl_divergence_loss(mu, logvar)
        occ_loss = super().forward(logits, labels, mask)
        return occ_loss + kld_loss * self.kld_weight

class FocalLossCriterion(nn.Module):

    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels, mask, *args):
        return focal_loss(logits, labels, mask, self.alpha, self.gamma)


class PriorOffsetCriterion(nn.Module):

    def __init__(self, priors):
        super().__init__()
        self.priors = priors
    
    def forward(self, logits, labels, mask, *args):
        return prior_offset_loss(logits, labels, mask, self.priors)


class VedCriterion(nn.Module):
    def __init__(self, num_class, class_weights, gpu_id,
            xent_weight=0.9, kld_weight=0.1):
        super().__init__()
        self._num_class = num_class
        self._kld_w = kld_weight
        self._xent_w = xent_weight
        self._class_weights = torch.Tensor(class_weights).to(gpu_id)
        
    def forward(self, logits, labels, mu, logvar):
        CE = F.cross_entropy(logits, labels.view(-1, 64, 64),
                              weight=self._class_weights, ignore_index=self._num_class)

        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return self._xent_w*CE + self._kld_w*KLD
