"""USEANet's weighted dual-branch (foreground/background) structure loss.

Ported from the original USEANet repo (models/loss.py + lib/utils/one_hot.py).
For binary segmentation (num_classes == 1) the prediction and mask share the
same shape, so the one-hot branch is skipped; the loss reduces to a
boundary-weighted wBCE + wIoU on the foreground plus a weighted wBCE on the
background.
"""

import torch
import torch.nn.functional as F


def expand_as_one_hot(input_tensor, num_classes):
    """(N, H, W) or (N, 1, H, W) label -> (N, num_classes, H, W) one-hot float."""
    if input_tensor.dim() == 4 and input_tensor.size(1) == 1:
        input_tensor = input_tensor.squeeze(1)
    input_tensor = input_tensor.long()
    one_hot = F.one_hot(input_tensor, num_classes=num_classes)
    return one_hot.permute(0, 3, 1, 2).float()


def structure_loss(pred, pred_bg, mask_fg, mask_bg, num_classes):
    if pred.shape != mask_fg.shape:
        mask_fg = expand_as_one_hot(mask_fg.long(), num_classes)
        mask_bg = expand_as_one_hot(mask_bg.long(), num_classes)

    # Boundary-emphasising weight: amplify the penalty near object edges.
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_fg, kernel_size=31, stride=1, padding=15) - mask_fg)

    # Weighted BCE (foreground)
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted BCE (background)
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)

    # Weighted IoU loss
    inter = ((pred * mask_fg) * weit).sum(dim=(2, 3))
    union = ((pred + mask_fg) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou + 0.8 * wbce2).mean()
