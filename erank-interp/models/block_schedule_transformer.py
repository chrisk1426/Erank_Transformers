"""
models/block_schedule_transformer.py

Custom transformer that supports arbitrary per-layer block schedules:
  "AM" : standard transformer block (attention then MLP, both with residual)
  "A"  : attention-only block (residual + attention)
  "M"  : MLP-only block (residual + MLP)

Hidden behaviour matches the HookedTransformer modular-addition baseline:
  - learned token + positional embeddings
  - causal self-attention with split heads
  - GELU MLP, hidden dim = d_mlp
  - normalization_type = None (no LayerNorm)
  - linear unembed

After each forward pass, every block stashes its pre/mid/post residual stream
on attributes so that extract_residual_activations() can produce a dict in the
exact format that analysis/compute_erank_profiles.py expects:

    {
      "blocks.{i}.hook_resid_pre":  Tensor (n_examples, d_model),
      "blocks.{i}.hook_resid_mid":  Tensor (n_examples, d_model),
      "blocks.{i}.hook_resid_post": Tensor (n_examples, d_model),
    }

For "A"-only blocks, mid == post; for "M"-only blocks, mid == pre.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

VALID_BLOCK_KINDS: tuple[str, ...] = ("AM", "A", "M")


class _Attention(nn.Module):
    """Multi-head causal self-attention. No layer norm (matches HookedTransformer modadd config)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_Q = nn.Linear(d_model, inner, bias=True)
        self.W_K = nn.Linear(d_model, inner, bias=True)
        self.W_V = nn.Linear(d_model, inner, bias=True)
        self.W_O = nn.Linear(inner, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, h, T, T)
        # causal mask
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, h, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.W_O(out)


class _MLP(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp, bias=True)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    """Single transformer block of kind 'AM', 'A', or 'M'."""

    def __init__(self, kind: str, d_model: int, n_heads: int, d_head: int, d_mlp: int):
        super().__init__()
        if kind not in VALID_BLOCK_KINDS:
            raise ValueError(f"Invalid block kind {kind!r}; must be one of {VALID_BLOCK_KINDS}")
        self.kind = kind
        self.has_attn = "A" in kind
        self.has_mlp = "M" in kind
        if self.has_attn:
            self.attn = _Attention(d_model, n_heads, d_head)
        if self.has_mlp:
            self.mlp = _MLP(d_model, d_mlp)
        # Slots populated each forward pass for activation capture.
        self._resid_pre: torch.Tensor | None = None
        self._resid_mid: torch.Tensor | None = None
        self._resid_post: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the residual stream entering this block.
        self._resid_pre = x
        if self.has_attn:
            mid = x + self.attn(x)
        else:
            mid = x
        self._resid_mid = mid
        if self.has_mlp:
            post = mid + self.mlp(mid)
        else:
            post = mid
        self._resid_post = post
        return post


class BlockScheduleTransformer(nn.Module):
    """
    Transformer with per-layer block schedule.

    Attributes:
        cfg: SimpleNamespace exposing n_layers, d_model, n_heads, d_head, d_mlp,
             d_vocab, d_vocab_out, n_ctx, block_schedule. Mimics the subset of
             HookedTransformer's cfg that downstream code reads.
    """

    def __init__(
        self,
        block_schedule: list[str],
        d_model: int,
        n_heads: int,
        d_head: int,
        d_mlp: int,
        d_vocab: int,
        d_vocab_out: int,
        n_ctx: int,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        for kind in block_schedule:
            if kind not in VALID_BLOCK_KINDS:
                raise ValueError(f"Invalid block kind {kind!r}; valid: {VALID_BLOCK_KINDS}")

        self.cfg = SimpleNamespace(
            n_layers=len(block_schedule),
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            d_mlp=d_mlp,
            d_vocab=d_vocab,
            d_vocab_out=d_vocab_out,
            n_ctx=n_ctx,
            block_schedule=list(block_schedule),
            attn_only=False,  # not used; left for compatibility
        )

        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = nn.Embedding(n_ctx, d_model)
        self.blocks = nn.ModuleList(
            [_Block(kind, d_model, n_heads, d_head, d_mlp) for kind in block_schedule]
        )
        self.unembed = nn.Linear(d_model, d_vocab_out, bias=False)

        # HookedTransformer-style initialization: small Gaussian on embeddings + linears.
        # We rely on PyTorch defaults except for explicit small init on embeddings.
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T)
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(positions).unsqueeze(0)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)  # (B, T, d_vocab_out)

    @torch.no_grad()
    def extract_residual_activations(
        self,
        dataloader: DataLoader,
        pred_position: int,
        n_examples: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Run examples through the model and collect resid_pre/mid/post at the
        prediction position for every block. Returns a dict in the exact format
        expected by analysis/compute_erank_profiles.compute_erank_profile().
        """
        was_training = self.training
        self.eval()

        n_layers = self.cfg.n_layers
        accum: dict[str, list[torch.Tensor]] = {}
        for i in range(n_layers):
            for hook in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post"):
                accum[f"blocks.{i}.{hook}"] = []

        collected = 0
        for inputs, _labels in dataloader:
            if collected >= n_examples:
                break
            inputs = inputs.to(device)
            B = inputs.size(0)
            remaining = n_examples - collected
            if B > remaining:
                inputs = inputs[:remaining]
                B = remaining

            self.forward(inputs)  # populates each block's _resid_{pre,mid,post}
            for i, block in enumerate(self.blocks):
                pre = block._resid_pre[:, pred_position, :].detach().cpu()
                mid = block._resid_mid[:, pred_position, :].detach().cpu()
                post = block._resid_post[:, pred_position, :].detach().cpu()
                accum[f"blocks.{i}.hook_resid_pre"].append(pre)
                accum[f"blocks.{i}.hook_resid_mid"].append(mid)
                accum[f"blocks.{i}.hook_resid_post"].append(post)
            collected += B

        result: dict[str, torch.Tensor] = {}
        for name, parts in accum.items():
            result[name] = torch.cat(parts, dim=0)[:n_examples]

        if was_training:
            self.train()
        return result


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = 113
    schedules = [
        ["AM", "AM", "AM", "AM"],
        ["A", "A", "A", "A"],
        ["M", "M", "M", "M"],
        ["A", "M", "M", "M"],
        ["M", "M", "M", "A"],
        ["A", "A", "M", "M"],
        ["M", "M", "A", "A"],
    ]
    for sched in schedules:
        m = BlockScheduleTransformer(
            block_schedule=sched,
            d_model=128,
            n_heads=4,
            d_head=32,
            d_mlp=512,
            d_vocab=p + 1,
            d_vocab_out=p,
            n_ctx=3,
            seed=0,
        )
        n_params = sum(t.numel() for t in m.parameters())
        dummy = torch.zeros(4, 3, dtype=torch.long)
        out = m(dummy)
        assert out.shape == (4, 3, p), f"bad shape for {sched}: {out.shape}"
        # confirm pre/mid/post slots populated
        for i, blk in enumerate(m.blocks):
            assert blk._resid_pre is not None and blk._resid_post is not None
            if not blk.has_attn:
                assert torch.equal(blk._resid_mid, blk._resid_pre), (
                    f"M-only block {i} should have mid == pre"
                )
            if not blk.has_mlp:
                assert torch.equal(blk._resid_post, blk._resid_mid), (
                    f"A-only block {i} should have post == mid"
                )
        print(f"  {'-'.join(sched):<20s} params={n_params:>8d}  forward+hooks OK")
    print("\nAll BlockScheduleTransformer smoke tests passed.")
