import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    Focuses training on hard, misclassified examples by modulating the cross-entropy loss.
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = 'mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
        """
        log_softmax = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_softmax, targets, reduction='none')

        prod_probabilities = torch.exp(-ce_loss)
        focal_weight = (1.0 - prod_probabilities) ** self.gamma

        if self.alpha is not None:
            at = self.alpha[targets]
            focal_loss = at * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss