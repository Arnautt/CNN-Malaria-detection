import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    """Focal loss introduced by Lin et al., in 2017"""
    def __init__(self, alpha=4, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_probs, y_true):
        """
        Forward pass given probabilities for each class and true labels

        Parameters
        ----------
        y_probs: torch.Tensor
            Probabilities of each class, tensor of shape batch_sie * n_class
        y_true: torch.Tensor
            True labels, of shape n_class
        """
        ce = F.cross_entropy(y_probs, y_true, reduction='none')
        pt = torch.exp(-ce)
        loss = (self.alpha * (1 - pt)**self.gamma * ce).mean()
        return loss
