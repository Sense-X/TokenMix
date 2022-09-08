"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from timm.loss import BinaryCrossEntropy

def ce_loss(logit_p, logit_q):
    p = torch.softmax(logit_p, dim=1)
    log_q = torch.log_softmax(logit_q, dim=1)
    loss = (-p * log_q).sum(dim=1).mean()
    return loss

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, use_ce=False, distill_token=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.use_ce = use_ce
        self.distill_token = distill_token

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if self.distill_token:
            if not isinstance(outputs, torch.Tensor):
                # assume that the model outputs a tuple of [outputs, outputs_kd]
                outputs, outputs_kd = outputs
        else:
            outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            if self.use_ce:
                # distillation_loss = BinaryCrossEntropy(smoothing=0)(outputs_kd, teacher_outputs)
                T = self.tau
                distillation_loss = ce_loss(teacher_outputs / T, outputs_kd / T) * T * T
            else:
                T = self.tau
                # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
                # with slight modifications
                distillation_loss = F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class BCELossSmooth(torch.nn.Module):
    def __init__(self, base_criterion, smooth=0):
        super(BCELossSmooth, self).__init__()
        self.base_criterion = base_criterion
        self.smooth = smooth

    def forward(self, inputs, outputs, labels):
        batch_size, num_classes = outputs.shape
        labels = labels.unsqueeze(1)
        if self.smooth <= 0.0:
            targets = torch.zeros(batch_size, num_classes).cuda().scatter_(1, labels, 1)
            loss = self.base_criterion(outputs, targets)
        else:
            targets = torch.zeros(batch_size, num_classes).cuda().scatter_(1, labels, 1)
            targets = (targets + self.smooth).clamp(0, 1)
            loss = self.base_criterion(outputs, targets)
        return loss