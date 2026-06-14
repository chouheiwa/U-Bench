"""MoE auxiliary losses and utilization metric (design §3.4, §6).

router_supervision_loss: KL(proxy || gate) anchoring the gate to the label-free
  physical proxy (or physical GT in Track C).
load_balance_loss: switch-transformer style importance/load balancing.
effective_experts: exp(entropy(mean gate)) in [1, E]; a monitoring metric.
"""
import torch


def router_supervision_loss(gate, proxy, eps=1e-8):
    """KL(proxy || gate) averaged over positions. Both [B,E,H,W], sum-1 over E."""
    g = gate.clamp_min(eps)
    p = proxy.clamp_min(eps)
    kl = (p * (p.log() - g.log())).sum(dim=1)   # [B,H,W]
    return kl.mean()


def load_balance_loss(gate):
    """Switch loss: E * sum_e (f_e * P_e), minimised at uniform usage.

    f_e (load) is treated as a stop-gradient estimate of routing fraction,
    consistent with the Switch Transformer formulation where gradient flows
    only through P_e (importance). For softmax gates all values are > 0,
    so f_e = 1 everywhere; instead we use the detached gate mean as a soft
    load proxy so the metric remains informative across gate distributions.
    """
    e = gate.shape[1]
    importance = gate.mean(dim=(0, 2, 3))                 # P_e  (differentiable)
    load = gate.detach().mean(dim=(0, 2, 3))              # f_e  (stop-gradient)
    return e * torch.sum(importance * load)


def effective_experts(gate, eps=1e-8):
    """exp(entropy(mean gate over all positions)); float in [1, E]."""
    mean_gate = gate.mean(dim=(0, 2, 3))                  # [E]
    mean_gate = mean_gate / mean_gate.sum().clamp_min(eps)
    entropy = -(mean_gate * (mean_gate + eps).log()).sum()
    return float(torch.exp(entropy).item())
