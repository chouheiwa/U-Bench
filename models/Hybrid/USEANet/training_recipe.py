"""USEANet-only training recipe levers (discriminative LR / warmup / weight EMA /
AdamW). Every lever is gated by a ``USEANET_*`` environment variable and is
default-off; with all switches off the helpers reproduce ``main.py``'s original
SGD + poly schedule exactly. This module imports only torch + stdlib so it never
pulls in the heavy USEANet backbone."""

import os
import warnings

import torch
from torch import optim

# Adapter (models/Hybrid/USEANet/__init__.py) holds self.net -> core.self.backbone,
# so the pretrained PVT-B0 params are prefixed "net.backbone." in the adapter's
# named_parameters().
BACKBONE_PARAM_PREFIXES = ("net.backbone.",)


def lr_at(group_base_lr, iter_num, max_iterations, warmup_iters):
    """Per-group LR. warmup_iters==0 reduces to main.py's original poly schedule:
    group_base_lr * (1 - iter_num/max_iterations) ** 0.9."""
    if warmup_iters > 0 and iter_num < warmup_iters:
        return group_base_lr * (iter_num + 1) / warmup_iters
    denom = max_iterations - warmup_iters
    return group_base_lr * (1.0 - (iter_num - warmup_iters) / denom) ** 0.9


def warmup_iters(iters_per_epoch):
    """USEANET_WARMUP_EPOCHS (default 0) * iters_per_epoch."""
    epochs = int(os.environ.get("USEANET_WARMUP_EPOCHS", "0"))
    return epochs * iters_per_epoch


def _split_params(model):
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pref) for pref in BACKBONE_PARAM_PREFIXES):
            backbone.append(p)
        else:
            rest.append(p)
    return backbone, rest


def build_optimizer(model, base_lr):
    """Return (optimizer, group_base_lrs). With USEANET_DISC_LR and USEANET_ADAMW
    both off this is byte-identical to main.py's original:
    optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)."""
    disc = os.environ.get("USEANET_DISC_LR") == "1"
    adamw = os.environ.get("USEANET_ADAMW") == "1"

    if not disc and not adamw:
        opt = optim.SGD(model.parameters(), lr=base_lr,
                        momentum=0.9, weight_decay=0.0001)
        return opt, [base_lr]

    if disc:
        mult = float(os.environ.get("USEANET_BACKBONE_LR_MULT", "0.1"))
        backbone, rest = _split_params(model)
        if not backbone:
            warnings.warn(
                "USEANET_DISC_LR: no params matched %s; falling back to a single "
                "group" % (BACKBONE_PARAM_PREFIXES,))
            groups = [{"params": rest, "lr": base_lr}]
            group_base_lrs = [base_lr]
        else:
            backbone_lr = base_lr * mult
            groups = [{"params": backbone, "lr": backbone_lr},
                      {"params": rest, "lr": base_lr}]
            group_base_lrs = [backbone_lr, base_lr]
    else:  # adamw only, single group
        groups = [{"params": list(model.parameters()), "lr": base_lr}]
        group_base_lrs = [base_lr]

    if adamw:
        opt = optim.AdamW(groups, lr=base_lr, weight_decay=0.0001)
    else:
        opt = optim.SGD(groups, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    return opt, group_base_lrs


class ModelEMA:
    """Exponential moving average of model weights. best ckpt is saved from the
    EMA shadow; checkpoint_final keeps the real training weights (resume-correct)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self._backup = None

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                s = self.shadow[k]
                if torch.is_floating_point(v):
                    s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    s.copy_(v)  # buffers (e.g. counters) track the latest value

    def store(self, model):
        self._backup = {k: v.detach().clone()
                        for k, v in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

    def state_dict(self):
        return self.shadow
