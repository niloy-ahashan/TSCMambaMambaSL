"""MambaSL branch for the TSCMamba + MambaSL fused architecture.

Replicates the MambaSL block from
`thuml/Time-Series-Library/models/MambaSingleLayer.py` (paper: "MambaSL:
Exploring Single-Layer Mamba for Time Series Classification", ICLR 2026):

    Conv1d token embedding + sinusoidal PE + Dropout
    -> single Mamba_TimeVariant (with its in_proj / conv1d / SSM / [out_proj])
    -> LayerNorm -> SiLU

Two variants are supported, selected via `skip_out_proj`:

* `skip_out_proj=False` (default, "variant 1"):
    The full MambaBlock is used. Mamba_TimeVariant.forward returns
    out_proj(gated_ssm) of shape (B, L, d_model_sl). The shape adapter then
    maps that down to (B, enc_in, projected_space*3).

* `skip_out_proj=True`  ("variant 2", drops the top Linear of the MambaBlock):
    Mamba_TimeVariant.out_proj is replaced with nn.Identity, so the block
    output is the post-gate hidden state of shape (B, L, d_inner) where
    d_inner = expand * d_model_sl. LayerNorm and the shape adapter widen to
    that d_inner channel count accordingly. The MambaBlock's final "Linear"
    in the paper's Figure 2 (the output projection) is skipped entirely.

The MambaSL classifier head (per-time logit + per-time softmax weight + Sum)
is intentionally NOT applied here: the user's fused architecture reuses
TSCMamba's classifier downstream. We only expose the contextualized hidden
states and a shape adapter that maps them into (B, enc_in, projected_space*3)
so the result can be summed with TSCMamba's x1 and x2 tensors before pooling.
"""

import torch
import torch.nn as nn

from layers.Embed import PositionalEmbedding
from layers.MambaBlock_tsl import Mamba_TimeVariant


class TokenEmbedding_cls(nn.Module):
    """Conv1d token embedding with configurable kernel size, matches TSLib MambaSL."""

    def __init__(self, c_in: int, d_model: int, d_kernel: int = 3):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=d_kernel,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L) -> conv -> (B, d_model, L) -> (B, L, d_model)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding_cls(nn.Module):
    """Token embedding + sinusoidal positional encoding + dropout, MambaSL-style."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1,
                 d_kernel: int = 3, seq_len: int = 5000):
        super().__init__()
        self.value_embedding = TokenEmbedding_cls(c_in=c_in, d_model=d_model, d_kernel=d_kernel)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class MambaSLBranch(nn.Module):
    """MambaSL feature extractor + shape adapter for fusion with TSCMamba.

    Parameters
    ----------
    enc_in : int
        Number of input channels (D / num_variates).
    seq_len : int
        Maximum input sequence length (for the PE max_len).
    d_model_sl : int
        Hidden dimension of the MambaSL block.
    d_state, d_conv, expand : Mamba hyperparameters.
    d_kernel : int
        Conv1d kernel size in the token embedding.
    dropout : float
    timevariant_dt / timevariant_B / timevariant_C / use_D : bool
        Selective-SSM modular toggles (MambaSL ablation flags).
    out_channels : int
        Target channel count of the adapted output (= TSCMamba's enc_in).
    out_features : int
        Target feature dimension of the adapted output (= projected_space * 3).
    skip_out_proj : bool, default False
        If True, the MambaBlock's top Linear (out_proj) is replaced with
        nn.Identity so the branch surfaces the post-gate hidden state of
        shape (B, L, expand * d_model_sl). LayerNorm and the shape adapter
        widen to that d_inner channel count automatically.
    """

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        d_model_sl: int,
        d_state: int,
        d_conv: int,
        expand: int,
        d_kernel: int,
        dropout: float,
        timevariant_dt: bool,
        timevariant_B: bool,
        timevariant_C: bool,
        use_D: bool,
        out_channels: int,
        out_features: int,
        skip_out_proj: bool = False,
    ):
        super().__init__()
        self.skip_out_proj = bool(skip_out_proj)

        self.embedding = DataEmbedding_cls(
            c_in=enc_in,
            d_model=d_model_sl,
            dropout=dropout,
            d_kernel=d_kernel,
            seq_len=seq_len,
        )

        # Single Modular-Selective-SSM block (the "single layer" of MambaSL).
        # We hold the block as its own attribute so the optional out_proj can be
        # bypassed below; the LayerNorm + SiLU follow, matching
        # thuml/Time-Series-Library/models/MambaSingleLayer.py::Model.__init__.
        self.mamba_block = Mamba_TimeVariant(
            d_model=d_model_sl,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            timevariant_dt=timevariant_dt,
            timevariant_B=timevariant_B,
            timevariant_C=timevariant_C,
            use_D=use_D,
        )
        if self.skip_out_proj:
            # Drop the top "Linear" of the MambaBlock so the block surfaces the
            # post-gate hidden state (B, L, d_inner). The MambaSL paper uses
            # out_proj to feed its own classifier head; in the fused model
            # TSCMamba's classifier is used instead, so out_proj is vestigial.
            self.mamba_block.out_proj = nn.Identity()
            feat_dim = self.mamba_block.d_inner   # expand * d_model_sl
        else:
            feat_dim = d_model_sl

        self.norm = nn.LayerNorm(feat_dim)
        self.act = nn.SiLU()

        # Shape adapter: (B, L, feat_dim) -> (B, enc_in, projected_space*3)
        # Time axis L is compressed to enc_in via adaptive average pooling,
        # then a Linear projects feat_dim onto the target feature width.
        self.pool_L = nn.AdaptiveAvgPool1d(out_channels)
        self.proj = nn.Linear(feat_dim, out_features)

    def forward(self, x_mts: torch.Tensor) -> torch.Tensor:
        """x_mts: (B, L, D). Returns (B, enc_in, projected_space*3)."""
        h = self.embedding(x_mts)        # (B, L, d_model_sl)
        h = self.mamba_block(h)          # (B, L, feat_dim)
        h = self.norm(h)
        h = self.act(h)
        h = h.permute(0, 2, 1)           # (B, feat_dim, L)
        h = self.pool_L(h)               # (B, feat_dim, enc_in)
        h = h.permute(0, 2, 1)           # (B, enc_in, feat_dim)
        h = self.proj(h)                 # (B, enc_in, projected_space*3)
        return h
