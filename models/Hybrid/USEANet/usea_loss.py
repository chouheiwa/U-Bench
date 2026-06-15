"""USEANet's weighted dual-branch (foreground/background) structure loss.

(原 docstring 保留;区域项现可经 USEANET_LOSS_REGION 在 iou/dice/focal_tversky 间切换。)
"""

import os
import warnings

import torch
import torch.nn.functional as F


def expand_as_one_hot(input_tensor, num_classes):
    """(N, H, W) or (N, 1, H, W) label -> (N, num_classes, H, W) one-hot float."""
    if input_tensor.dim() == 4 and input_tensor.size(1) == 1:
        input_tensor = input_tensor.squeeze(1)
    input_tensor = input_tensor.long()
    one_hot = F.one_hot(input_tensor, num_classes=num_classes)
    return one_hot.permute(0, 3, 1, 2).float()


def _region_iou(p, mask_fg, weit):
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    union = ((p + mask_fg) * weit).sum(dim=(2, 3))
    return 1 - (inter + 1) / (union - inter + 1)


def _region_dice(p, mask_fg, weit):
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    psum = (p * weit).sum(dim=(2, 3))
    msum = (mask_fg * weit).sum(dim=(2, 3))
    return 1 - (2 * inter + 1) / (psum + msum + 1)


def _region_focal_tversky(p, mask_fg, weit, alpha, beta, gamma):
    inter = ((p * mask_fg) * weit).sum(dim=(2, 3))
    fp = ((p * (1 - mask_fg)) * weit).sum(dim=(2, 3))
    fn = (((1 - p) * mask_fg) * weit).sum(dim=(2, 3))
    ti = (inter + 1) / (inter + alpha * fp + beta * fn + 1)
    return (1 - ti).clamp(min=1e-7) ** (1.0 / gamma)


def _resolve_region():
    mode = os.environ.get("USEANET_LOSS_REGION", "iou").lower()
    if mode not in ("iou", "dice", "focal_tversky"):
        warnings.warn(
            f"USEANET_LOSS_REGION={mode!r} invalid; falling back to 'iou'"
        )
        mode = "iou"
    return mode


def _ft_params():
    alpha = float(os.environ.get("USEANET_FT_ALPHA", "0.3"))
    beta = float(os.environ.get("USEANET_FT_BETA", "0.7"))
    gamma = float(os.environ.get("USEANET_FT_GAMMA", str(4.0 / 3.0)))
    return alpha, beta, gamma


def structure_loss(pred, pred_bg, mask_fg, mask_bg, num_classes):
    if pred.shape != mask_fg.shape:
        mask_fg = expand_as_one_hot(mask_fg.long(), num_classes)
        mask_bg = expand_as_one_hot(mask_bg.long(), num_classes)

    # Boundary-emphasising weight: amplify the penalty near object edges.
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_fg, kernel_size=31, stride=1, padding=15) - mask_fg)

    # Weighted BCE (foreground) — classification anchor, kept across all region modes.
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted BCE (background)
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit * wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)

    # Switchable region term (B1): iou (default) / dice / focal_tversky.
    mode = _resolve_region()
    if mode == "dice":
        region = _region_dice(pred, mask_fg, weit)
    elif mode == "focal_tversky":
        alpha, beta, gamma = _ft_params()
        region = _region_focal_tversky(pred, mask_fg, weit, alpha, beta, gamma)
    else:
        region = _region_iou(pred, mask_fg, weit)

    return (wbce + region + 0.8 * wbce2).mean()
