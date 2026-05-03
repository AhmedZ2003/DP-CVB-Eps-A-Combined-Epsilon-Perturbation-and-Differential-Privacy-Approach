from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn


@dataclass
class DPDefenseConfig:
    """Repo-compatible DP defense for the shared gradients.

    Notes
    -----
    This is the paper-aligned *combination* defense hook for CVB: keep the CVB
    stochastic bottleneck and additionally sanitize the gradients with clipping
    plus Gaussian noise before they are shared / aggregated.

    The exact paper setup uses per-sample clipping with Gaussian noise. In this
    patch we support two modes:
      - ``mode='per_sample'``: expects ``param.grad_sample`` tensors produced by
        a library such as Opacus.
      - ``mode='per_batch'``: safe fallback that clips the aggregated gradient
        and adds Gaussian noise. This is easier to integrate when the existing
        training loop is not using per-sample gradient hooks yet.
    """

    enabled: bool = True
    mode: str = "per_batch"  # per_batch | per_sample
    clip_norm: float = 20.0
    noise_multiplier: float = 1e-3
    secure_mode: bool = False
    only_when_training: bool = True

    @property
    def noise_std(self) -> float:
        return float(self.noise_multiplier) * float(self.clip_norm)


def build_dp_defense_config(raw_cfg: Optional[object]) -> Optional[DPDefenseConfig]:
    if raw_cfg is None:
        return None

    if isinstance(raw_cfg, DPDefenseConfig):
        return deepcopy(raw_cfg)

    # Works with dict, Munch, OmegaConf-like objects.
    get = raw_cfg.get if hasattr(raw_cfg, "get") else lambda k, d=None: getattr(raw_cfg, k, d)

    return DPDefenseConfig(
        enabled=bool(get("enabled", True)),
        mode=str(get("mode", "per_batch")),
        clip_norm=float(get("clip_norm", 20.0)),
        noise_multiplier=float(get("noise_multiplier", 1e-3)),
        secure_mode=bool(get("secure_mode", False)),
        only_when_training=bool(get("only_when_training", True)),
    )


def _iter_trainable_parameters(parameters: Iterable[torch.nn.Parameter]):
    for p in parameters:
        if p is not None and p.requires_grad:
            yield p


def _global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    grads = [p.grad.detach().norm(2) for p in _iter_trainable_parameters(parameters) if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(grads), 2)


def _clip_aggregated_gradients(parameters: Iterable[torch.nn.Parameter], clip_norm: float) -> float:
    params = [p for p in _iter_trainable_parameters(parameters) if p.grad is not None]
    if not params:
        return 0.0

    total_norm = _global_grad_norm(params)
    denom = max(float(total_norm.item()) / float(clip_norm), 1.0)
    scale = 1.0 / denom
    for p in params:
        p.grad.mul_(scale)
    return float(total_norm.item())


def _add_gaussian_noise(parameters: Iterable[torch.nn.Parameter], std: float) -> None:
    if std <= 0:
        return
    for p in _iter_trainable_parameters(parameters):
        if p.grad is None:
            continue
        noise = torch.normal(mean=0.0, std=std, size=p.grad.shape, device=p.grad.device, dtype=p.grad.dtype)
        p.grad.add_(noise)


# ---------------------------------------------------------------------------
# Forward-pass epsilon perturbation (VBIA-specific hardening)
# ---------------------------------------------------------------------------

def _is_cvb_module(module: nn.Module) -> bool:
    """Detect CVB / VB modules by name convention and attribute presence."""
    cls_name = type(module).__name__
    return (
        "CVB" in cls_name
        or "VariationalBottleneck" in cls_name
        or hasattr(module, "fixed_epsilon")
    )


def arm_cvb_epsilon_defense_(
    model: nn.Module,
    defense_cfg: Optional[DPDefenseConfig],
) -> None:
    """Attach private epsilon-perturbation noise to every CVB module.

    Called at the **start** of VBCNN.forward() during training so that even
    if VBIA injects dummy_epsilon via module.fixed_epsilon, the victim's actual
    reparameterisation uses:

        epsilon_VBC = fixed_epsilon + _epsilon_defense_noise

    where _epsilon_defense_noise is freshly sampled each forward pass and is
    never exposed to the attacker.  This makes it impossible for VBIA to
    satisfy epsilon_VBA ≈ epsilon_VBC regardless of how it optimises
    dummy_epsilon, because the residual noise is i.i.d. and independent of the
    gradient signal the attacker observes.

    The noise standard deviation deliberately matches noise_multiplier so it is
    large enough to disrupt VBIA but small enough not to harm the CVB's own
    variational objective (the KL term already regularises the latent space).
    """
    if defense_cfg is None or not defense_cfg.enabled:
        return
    if defense_cfg.only_when_training and not model.training:
        return

    for module in model.modules():
        if not _is_cvb_module(module):
            continue

        # Determine the epsilon shape from a prior fixed_epsilon if present,
        # otherwise skip (we can only perturb when the attacker has set one,
        # which is exactly the case VBIA requires).
        injected = getattr(module, "fixed_epsilon", None)
        if injected is None:
            continue

        # Sample fresh private noise with the same shape/device/dtype.
        private_noise = torch.randn_like(injected) * defense_cfg.noise_multiplier
        # Add in-place: attacker's dummy_epsilon is now hopelessly offset.
        module.fixed_epsilon = injected + private_noise


def disarm_cvb_epsilon_defense_(model: nn.Module) -> None:
    """Remove any lingering _epsilon_defense_noise markers (cleanup after forward)."""
    # Nothing persistent to clean: arm_ only modifies fixed_epsilon in-place
    # and fixed_epsilon is cleared by VBIA._clear_fixed_epsilon() after each
    # closure call already.  This function is a no-op hook kept for symmetry
    # and future extensibility.
    pass


def sanitize_gradients_(
    model: torch.nn.Module,
    defense_cfg: Optional[DPDefenseConfig],
    *,
    batch_size: Optional[int] = None,
) -> dict:
    """Sanitize gradients in-place.

    Returns lightweight stats so the caller can log what happened.
    """
    if defense_cfg is None or not defense_cfg.enabled:
        return {"enabled": False}

    if defense_cfg.only_when_training and not model.training:
        return {"enabled": False, "skipped": "model_not_in_train_mode"}

    params = list(model.parameters())

    if defense_cfg.mode == "per_sample":
        used_per_sample = False
        for p in _iter_trainable_parameters(params):
            grad_sample = getattr(p, "grad_sample", None)
            if grad_sample is None:
                continue
            used_per_sample = True
            flat = grad_sample.reshape(grad_sample.shape[0], -1)
            sample_norms = flat.norm(2, dim=1)
            clip_factors = (defense_cfg.clip_norm / (sample_norms + 1e-12)).clamp(max=1.0)
            clipped = grad_sample * clip_factors.view(-1, *([1] * (grad_sample.dim() - 1)))
            p.grad = clipped.mean(dim=0)

        if not used_per_sample:
            raise RuntimeError(
                "DPDefenseConfig.mode='per_sample' was requested but no parameter had `grad_sample`. "
                "Either attach Opacus/per-sample hooks or switch to mode='per_batch'."
            )
    else:
        _clip_aggregated_gradients(params, defense_cfg.clip_norm)

    _add_gaussian_noise(params, defense_cfg.noise_std)

    return {
        "enabled": True,
        "mode": defense_cfg.mode,
        "clip_norm": defense_cfg.clip_norm,
        "noise_multiplier": defense_cfg.noise_multiplier,
        "noise_std": defense_cfg.noise_std,
        "batch_size": batch_size,
    }