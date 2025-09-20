from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from contextlib import nullcontext


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    clip_eps_v: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    aux_mine_weight: float = 0.0
    aux_mine_calib_weight: float = 0.0
    max_grad_norm: float = 0.5
    beta_l2: float = 0.0
    adv_guess_weight: float = 0.0


def ppo_update(model, optimizer, batch, cfg: PPOConfig, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
    use_cuda = batch.obs.is_cuda
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else nullcontext()
    with autocast_ctx:
        need_mine_head = cfg.aux_mine_weight > 0 or cfg.aux_mine_calib_weight > 0
        if need_mine_head:
            logits, value, mine_logits = model(batch.obs, return_mine=True)
        else:
            logits, value = model(batch.obs, return_mine=False)
            mine_logits = None
        neg_inf = -1e9
        if logits.dtype in (torch.float16, torch.bfloat16):
            neg_inf = -1e4
        masked_logits = logits.masked_fill(~batch.action_mask, neg_inf)

        logp = F.log_softmax(masked_logits, dim=-1)
        logp_act = logp.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        ratio = (logp_act - batch.old_logp).exp()
        s1 = ratio * batch.advantages
        s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * batch.advantages
        policy_loss = -torch.min(s1, s2).mean()

        value_pred = value.view(-1)
        value_clipped = batch.values + (value_pred - batch.values).clamp(-cfg.clip_eps_v, cfg.clip_eps_v)
        v1 = (value_pred - batch.returns).pow(2)
        v2 = (value_clipped - batch.returns).pow(2)
        value_loss = 0.5 * torch.max(v1, v2).mean()

        ent = -(torch.softmax(masked_logits, -1) * logp).sum(-1).mean()

        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent

        aux_bce = None
        calib_loss = None
        if need_mine_head and hasattr(batch, "mine_labels") and mine_logits is not None:
            logits_flat = mine_logits.squeeze(1)
            labels = batch.mine_labels
            mask = getattr(batch, "mine_valid", None)
            if mask is None:
                mask = torch.ones_like(labels, dtype=torch.bool)
            valid_logits = logits_flat[mask]
            valid_labels = labels[mask]
            if valid_labels.numel() > 0:
                pos = valid_labels.sum()
                neg = valid_labels.numel() - pos
                if cfg.aux_mine_weight > 0:
                    pos_weight = (neg + 1e-6) / (pos + 1e-6)
                    pos_weight_tensor = torch.full((), float(pos_weight), dtype=valid_logits.dtype, device=valid_logits.device)
                    aux_bce = F.binary_cross_entropy_with_logits(
                        valid_logits,
                        valid_labels,
                        pos_weight=pos_weight_tensor,
                    )
                    loss = loss + cfg.aux_mine_weight * aux_bce
                if cfg.aux_mine_calib_weight > 0:
                    probs = torch.sigmoid(valid_logits)
                    calib_loss = (probs - valid_labels).pow(2).mean()
                    loss = loss + cfg.aux_mine_calib_weight * calib_loss
            else:
                zero = torch.zeros((), dtype=logits_flat.dtype, device=logits_flat.device)
                if cfg.aux_mine_weight > 0:
                    aux_bce = zero
                if cfg.aux_mine_calib_weight > 0:
                    calib_loss = zero

        if cfg.beta_l2 > 0 and hasattr(model, "beta_regularizer"):
            beta_pen = model.beta_regularizer()
            loss = loss + cfg.beta_l2 * beta_pen

        adv_penalty_mean = None
        if cfg.adv_guess_weight > 0 and mine_logits is not None:
            mine_logits_flat = mine_logits.view(mine_logits.shape[0], -1)
            gather_idx = torch.arange(mine_logits_flat.shape[0], device=mine_logits_flat.device)
            chosen_logits = mine_logits_flat[gather_idx, batch.actions]
            if hasattr(batch, "mine_valid") and batch.mine_valid is not None:
                mine_valid_flat = batch.mine_valid.view(batch.mine_valid.shape[0], -1)
                chosen_valid = mine_valid_flat[gather_idx, batch.actions].float()
            else:
                chosen_valid = torch.ones_like(chosen_logits)
            adv_penalty_mean = (torch.sigmoid(chosen_logits) * chosen_valid).mean()
            loss = loss + cfg.adv_guess_weight * adv_penalty_mean

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and batch.obs.is_cuda:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

    stats = {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(ent.item()),
    }
    if aux_bce is not None:
        stats["aux_bce"] = float(aux_bce.item())
    if calib_loss is not None:
        stats["aux_calib"] = float(calib_loss.item())
    if cfg.adv_guess_weight > 0:
        stats["adv_weight"] = float(cfg.adv_guess_weight)
    if adv_penalty_mean is not None:
        stats["adv_penalty"] = float(adv_penalty_mean.item())
    return stats
