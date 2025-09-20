from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rel_pos_index: Optional[torch.Tensor] = None,
        rel_pos_bins: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.rel_pos_index = rel_pos_index
        if rel_pos_index is not None:
            if rel_pos_bins is None:
                raise ValueError("rel_pos_bins must be provided with rel_pos_index")
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, rel_pos_bins))
        else:
            self.rel_pos_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if self.rel_pos_bias is not None and self.rel_pos_index is not None:
            bias = self.rel_pos_bias[:, self.rel_pos_index]
            attn_mask = bias.unsqueeze(0).to(dtype=q.dtype, device=q.device)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _TransformerMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        rel_pos_index: Optional[torch.Tensor] = None,
        rel_pos_bins: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, rel_pos_index, rel_pos_bins)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _TransformerMLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _build_rel_pos_index(H: int, W: int, radius: Optional[int]) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    if radius is None or radius <= 0:
        return None, None
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)
    coords = coords.reshape(-1, 2)
    rel = coords[:, None, :] - coords[None, :, :]
    far_mask = (rel.abs() > radius).any(dim=-1)
    rel = rel.clamp(-radius, radius)
    bins_per_axis = 2 * radius + 1
    rel = rel + radius
    rel_index = rel[..., 0] * bins_per_axis + rel[..., 1]
    far_bin = bins_per_axis * bins_per_axis
    rel_index = torch.where(far_mask, torch.full_like(rel_index, far_bin), rel_index)
    rel_pos_bins = far_bin + 1
    return rel_index, rel_pos_bins


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        board_shape: tuple[int, int],
        *,
        include_flags_channel: bool = False,
        include_frontier_channel: bool = False,
        include_remaining_mines_channel: bool = False,
        include_progress_channel: bool = False,
        d_model: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        rel_pos_radius: Optional[int] = 4,
        tie_reveal_to_belief: bool = False,
        num_global_tokens: int = 2,
        mlp_ratio_last: float | None = None,
    ) -> None:
        super().__init__()
        self.H, self.W = board_shape
        self.N = self.H * self.W
        self.include_flags_channel = include_flags_channel
        self.include_frontier_channel = include_frontier_channel
        self.include_remaining_mines_channel = include_remaining_mines_channel
        self.include_progress_channel = include_progress_channel
        self.tie_reveal_to_belief = bool(tie_reveal_to_belief)
        self.num_global_tokens = max(0, int(num_global_tokens))
        self.gradient_checkpointing = False
        self._last_beta_reg: Optional[torch.Tensor] = None

        token_dim = 1 + 9
        if include_flags_channel:
            token_dim += 1
        if include_frontier_channel:
            token_dim += 1
        if include_remaining_mines_channel:
            token_dim += 1
        if include_progress_channel:
            token_dim += 1

        self.input_proj = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if self.num_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.zeros(1, self.num_global_tokens, d_model))
        else:
            self.register_parameter("global_tokens", None)

        self.row_embed = nn.Parameter(torch.randn(self.H, d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(self.W, d_model) * 0.02)

        rel_pos_index, rel_pos_bins = _build_rel_pos_index(self.H, self.W, rel_pos_radius)
        total_tokens = self.N + 1 + self.num_global_tokens
        if rel_pos_index is not None and rel_pos_bins is not None:
            far_bin = rel_pos_bins - 1
            rel_full = torch.full((total_tokens, total_tokens), far_bin, dtype=torch.long)
            rel_full[1 : self.N + 1, 1 : self.N + 1] = rel_pos_index
            self.register_buffer("rel_pos_index", rel_full, persistent=False)
            self.rel_pos_bins = rel_pos_bins
        else:
            self.register_buffer("rel_pos_index", None, persistent=False)
            self.rel_pos_bins = None

        blocks = []
        for i in range(depth):
            rel_idx = self.rel_pos_index if rel_pos_index is not None else None
            rel_bins = self.rel_pos_bins if rel_pos_index is not None else None
            blocks.append(
                _TransformerBlock(
                    d_model,
                    num_heads,
                    mlp_ratio if (mlp_ratio_last is None or i < depth - 1) else mlp_ratio_last,
                    rel_idx,
                    rel_bins,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

        if self.tie_reveal_to_belief:
            self.belief_policy_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 2),
            )
        else:
            self.belief_policy_proj = None

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.mine_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        rows = torch.arange(self.H)
        cols = torch.arange(self.W)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("row_ids", rr.reshape(-1), persistent=False)
        self.register_buffer("col_ids", cc.reshape(-1), persistent=False)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _tokens_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = obs.shape
        assert H == self.H and W == self.W, "Observation spatial dims mismatch"
        offset = 0
        revealed = obs[:, offset : offset + 1]
        offset += 1

        flag_feat = None
        if self.include_flags_channel:
            flag_feat = obs[:, offset : offset + 1]
            offset += 1

        counts = obs[:, offset : offset + 9]
        offset += 9

        features = [revealed]
        if flag_feat is not None:
            features.append(flag_feat)
        features.append(counts)

        if self.include_frontier_channel:
            frontier = obs[:, offset : offset + 1]
            features.append(frontier)
            offset += 1
        if self.include_remaining_mines_channel:
            remaining = obs[:, offset : offset + 1]
            features.append(remaining)
            offset += 1
        if self.include_progress_channel:
            progress = obs[:, offset : offset + 1]
            features.append(progress)
            offset += 1

        tokens = torch.cat(features, dim=1)
        tokens = tokens.view(B, tokens.shape[1], -1).permute(0, 2, 1).contiguous()
        return tokens

    def forward(self, obs: torch.Tensor, return_mine: bool = False):
        from torch.utils.checkpoint import checkpoint

        B = obs.shape[0]
        tokens = self._tokens_from_obs(obs)
        x = self.input_proj(tokens)

        pos = self.row_embed[self.row_ids] + self.col_embed[self.col_ids]
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        cls = self.cls_token.expand(B, -1, -1)
        token_embed = x + pos
        if self.num_global_tokens > 0:
            global_tokens = self.global_tokens.expand(B, -1, -1)
            x = torch.cat([cls, global_tokens, token_embed], dim=1)
        else:
            global_tokens = None
            x = torch.cat([cls, token_embed], dim=1)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                try:
                    x = checkpoint(blk, x, use_reentrant=False)
                except TypeError:  # pragma: no cover
                    x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)

        cls_token, token_states = x[:, :1], x[:, 1 : self.N + 1]
        mine_logits_map = None
        need_mine = self.tie_reveal_to_belief or return_mine
        if need_mine:
            mine_logits_flat = self.mine_head(token_states).squeeze(-1)
            mine_logits_map = mine_logits_flat.view(B, 1, self.H, self.W)
        else:
            mine_logits_flat = None

        if self.tie_reveal_to_belief:
            assert self.belief_policy_proj is not None
            if global_tokens is not None:
                pooled = torch.cat([cls_token, global_tokens], dim=1).mean(dim=1)
            else:
                pooled = cls_token.squeeze(1)
            params = self.belief_policy_proj(pooled)
            beta_raw, bias = params.split(1, dim=-1)
            beta = F.softplus(beta_raw) + 1e-3
            self._last_beta_reg = (beta ** 2).mean()
            policy_logits_flat = (-beta * mine_logits_flat + bias).view(B, self.N)
        else:
            self._last_beta_reg = None
            logits_tokens = self.policy_head(token_states).squeeze(-1)
            policy_logits_flat = logits_tokens.view(B, self.N)

        value = self.value_head(cls_token).squeeze(-1).squeeze(-1)

        if return_mine:
            if mine_logits_map is None:
                mine_logits_flat = self.mine_head(token_states).squeeze(-1)
                mine_logits_map = mine_logits_flat.view(B, 1, self.H, self.W)
            return policy_logits_flat, value, mine_logits_map
        return policy_logits_flat, value

    def beta_regularizer(self) -> torch.Tensor:
        if self._last_beta_reg is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_beta_reg
