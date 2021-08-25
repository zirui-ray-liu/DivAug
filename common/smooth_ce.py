import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

LOGSOFTMAX = torch.nn.LogSoftmax(dim=1)


class SoftCrossEntropyLoss(torch.nn.Module):
    """Calculate the CrossEntropyLoss with soft targets.

    :param weight: Weight to assign to each of the classes. Default: None
    :type weight: list of float
    :param reduction: The way to reduce the losses: 'none' | 'mean' | 'sum'.
        'none': no reduction,
        'mean': the mean of the losses,
        'sum': the sum of the losses.
    :type reduction: str
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        # n, k = input.shape
        # losses = input.new_zeros(n)
        #
        # for i in range(k):
        #     cls_idx = input.new_full((n,), i, dtype=torch.long)
        #     loss = F.cross_entropy(input, cls_idx, reduction="none")
        #     losses += target[:, i].float() * loss
        losses = torch.sum(-target * LOGSOFTMAX(input), dim=1)
        if self.reduction == "mean":
            losses = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum()
        elif self.reduction != "none":
            raise ValueError(f"Unrecognized reduction: {self.reduction}")
        return losses


# class SmoothCrossEntropyLoss(Module):
#     def __init__(self, label_smoothing=0.0, size_average=True):
#         super().__init__()
#         self.label_smoothing = label_smoothing
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if len(target.size()) == 1:
#             target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
#             target = target.float().cuda()
#         if self.label_smoothing > 0.0:
#             s_by_c = self.label_smoothing / len(input[0])
#             smooth = torch.zeros_like(target)
#             smooth = smooth + s_by_c
#             target = target * (1. - s_by_c) + smooth
#
#         return cross_entropy(input, target, self.size_average)
#
#
# def cross_entropy(input, target, size_average=True):
#     """ Cross entropy that accepts soft targets
#     Args:
#          pred: predictions for neural network
#          targets: targets, can be soft
#          size_average: if false, sum is returned instead of mean
#     Examples::
#         input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
#         input = torch.autograd.Variable(out, requires_grad=True)
#         target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
#         target = torch.autograd.Variable(y1)
#         loss = cross_entropy(input, target)
#         loss.backward()
#     """
#     logsoftmax = torch.nn.LogSoftmax(dim=1)
#     if size_average:
#         return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
#     else:
#         return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
