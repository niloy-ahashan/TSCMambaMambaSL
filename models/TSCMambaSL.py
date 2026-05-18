"""TSCMamba + MambaSL fused architecture.

Pipeline
--------
1. TSCMamba pre-encoder (verbatim from models/TSCMamba.py):
   CWT image -> ConversionLayer -> x_patched
   Raw MTS / ROCKET features -> projector -> x_projected
   Multi-view fusion (additive or multiplicative, with learnable_focus) -> x_fused
   concatenated_x = cat([x_patched, x_fused, x_projected], dim=2) -> LayerNorm

2. TSCMamba dual-axis Mamba (verbatim):
   - mamba1 on token axis (forward + optional reversed scan) -> x1
   - mamba2 on channel axis (forward + optional reversed scan) -> x2

3. MambaSL branch on the raw MTS (this file's addition):
   TokenConv1d + sinusoidal PE + Dropout
   -> Mamba_TimeVariant (with output projection) + LayerNorm + SiLU
   -> AdaptiveAvgPool1d(enc_in) + Linear(d_model_sl, projected_space*3)
   -> xsl  with shape (B, enc_in, projected_space*3)

4. Fusion at the X3 point:  x3 = x1 + x2 + xsl

5. Downstream Processing (DP) - identical to TSCMamba:
   pooling (mean / max) -> flatten -> 2-layer classifier MLP.

Module versions
---------------
- TSCMamba's two scanning blocks import `Mamba` from `mamba_ssm` (the same
  high-level wrapper used by the upstream Atik-Ahamed/TSCMamba repository).
- The MambaSL branch reuses `layers/MambaBlock_tsl.py::Mamba_TimeVariant`, which
  is byte-for-byte equivalent to the `Mamba_TimeVariant` class in
  thuml/Time-Series-Library/layers/MambaBlock.py (the MambaSL paper's official
  building block). Both rely on `mamba_ssm.ops.selective_scan_interface`.

Data contract
-------------
Forward signature is `forward(x_cwt, x_features, x_raw_mts)`. This model
requires the data loader to be set up with `--add_raw_mts 1`, which makes
data_provider's collate_fn emit a 4-tuple `(X_cwt, X_features, raw_MTS, label)`
where `raw_MTS` has shape `(B, D, L)`. The MambaSL branch consumes
`raw_MTS.permute(0, 2, 1)` so it sees `(B, L, D)`.
"""

import copy

import torch
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba  # same package/import path used by upstream TSCMamba

from models.MambaSL_branch import MambaSLBranch


class ConversionLayer(torch.nn.Module):
    """CWT image -> patch tokens + projector. Verbatim from models/TSCMamba.py."""

    def __init__(self, d_in, l_in, d_out, l_out, im_size, patch_size):
        super(ConversionLayer, self).__init__()
        self.d_in = d_in
        self.l_in = l_in
        self.d_out = d_out
        self.l_out = l_out
        self.im_size = im_size
        self.patch_size = patch_size

        self.patch_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.d_in,
                out_channels=self.d_out,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size),
                padding=0,
            ),
            Rearrange("b c h w -> b c (h w)"),
        )
        self.projector = torch.nn.Linear(
            (self.im_size // self.patch_size) * (self.im_size // self.patch_size),
            self.l_out,
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.projector(x)
        return x


class Model(torch.nn.Module):
    """TSCMamba pipeline with an extra MambaSL branch fused at X3."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.configs_copy = copy.deepcopy(self.configs)

        if self.configs.task_name != "classification":
            raise NotImplementedError(
                "TSCMambaSL currently supports only the 'classification' task."
            )

        # ---- TSCMamba components (verbatim from models/TSCMamba.py) ----
        self.patcher = ConversionLayer(
            d_in=self.configs.enc_in,
            l_in=self.configs.seq_len,
            l_out=self.configs.projected_space,
            d_out=self.configs.enc_in,
            im_size=self.configs.rescale_size,
            patch_size=self.configs.patch_size,
        )
        if self.configs.no_rocket == 1:
            self.projector = torch.nn.Linear(self.configs.seq_len, self.configs.projected_space)
        elif self.configs.half_rocket == 1:
            self.projector = torch.nn.Linear(self.configs.seq_len, self.configs.projected_space // 2)

        self.ln = torch.nn.LayerNorm(self.configs.projected_space * 3)
        self.dropout = torch.nn.Dropout(self.configs.dropout)
        self.learnable_focus = torch.nn.Parameter(torch.tensor([self.configs.initial_focus]))
        self.gelu = torch.nn.GELU()

        self.mamba1 = torch.nn.ModuleList([
            Mamba(
                d_model=self.configs.projected_space * 3,
                d_state=self.configs.d_state,
                d_conv=self.configs.dconv,
                expand=self.configs.e_fact,
            )
            for _ in range(self.configs.num_mambas)
        ])
        self.mamba2 = torch.nn.ModuleList([
            Mamba(
                d_model=self.configs.enc_in,
                d_state=self.configs.d_state,
                d_conv=self.configs.dconv,
                expand=self.configs.e_fact,
            )
            for _ in range(self.configs.num_mambas)
        ])

        # ---- MambaSL branch ----
        # The shape adapter outputs (B, enc_in, projected_space*3), matching x1/x2.
        # When --mambasl_skip_out_proj 1, the top Linear of the MambaBlock is
        # bypassed and the branch returns the post-gate hidden state widened to
        # d_inner = expand * mambasl_d_model.
        self.mambasl_branch = MambaSLBranch(
            enc_in=self.configs.enc_in,
            seq_len=self.configs.seq_len,
            d_model_sl=self.configs.mambasl_d_model,
            d_state=self.configs.mambasl_d_state,
            d_conv=self.configs.mambasl_d_conv,
            expand=self.configs.mambasl_expand,
            d_kernel=self.configs.mambasl_kernel,
            dropout=self.configs.dropout,
            timevariant_dt=bool(self.configs.tv_dt),
            timevariant_B=bool(self.configs.tv_B),
            timevariant_C=bool(self.configs.tv_C),
            use_D=bool(self.configs.use_D),
            out_channels=self.configs.enc_in,
            out_features=self.configs.projected_space * 3,
            skip_out_proj=bool(int(getattr(self.configs, "mambasl_skip_out_proj", 0))),
        )

        # Optional learnable scalar gate on the MambaSL contribution. When
        # mambasl_alpha_learnable=0 the branch is summed in with weight 1.0.
        if int(getattr(self.configs, "mambasl_alpha_learnable", 0)) == 1:
            init_alpha = float(getattr(self.configs, "mambasl_alpha_init", 1.0))
            self.mambasl_alpha = torch.nn.Parameter(torch.tensor([init_alpha]))
        else:
            self.register_buffer(
                "mambasl_alpha",
                torch.tensor([float(getattr(self.configs, "mambasl_alpha_init", 1.0))]),
            )

        # ---- Downstream Processing (same as TSCMamba) ----
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.configs.projected_space * 3, (self.configs.projected_space * 3) // 2),
            torch.nn.Dropout(self.configs.dropout),
            torch.nn.Linear((self.configs.projected_space * 3) // 2, self.configs.num_class),
        )

    # ------------------------------------------------------------------ #
    def classification(self, x_cwt, batch_x_features, x_raw_mts):
        # ----- TSCMamba pre-encoder (verbatim) -----
        x_patched = self.patcher(x_cwt)

        if self.configs.no_rocket == 1:
            x_projected = self.projector(batch_x_features)
        else:
            if self.configs.half_rocket == 0:
                x_projected = batch_x_features
            else:
                x_projected = torch.cat(
                    [
                        batch_x_features[:, :, : self.configs.projected_space // 2],
                        self.projector(batch_x_features[:, :, self.configs.projected_space:]),
                    ],
                    dim=2,
                )

        if self.configs.additive_fusion == 1:
            x_fused = (self.learnable_focus * x_projected) + (2.0 - self.learnable_focus) * x_patched
        else:
            x_fused = (self.learnable_focus * x_projected) * (2.0 - self.learnable_focus) * x_patched

        x_fused = self.gelu(x_fused)
        concatenated_x = torch.cat([x_patched, x_fused, x_projected], dim=2)
        concatenated_x = self.ln(concatenated_x)

        # ----- TSCMamba dual-axis Mamba (verbatim) -----
        if self.configs.num_mambas != 0:
            x1 = concatenated_x.clone()
            for i in range(self.configs.num_mambas):
                x1 = self.mamba1[i](x1) + x1.clone()

            if self.configs.only_forward_scan == 0:
                concatenated_x_flipped = torch.flip(concatenated_x, dims=[self.configs.flip_dir])
                x1_flipped = concatenated_x_flipped.clone()
                for i in range(self.configs.num_mambas):
                    x1_flipped = self.mamba1[i](x1_flipped) + x1_flipped.clone()
                if self.configs.reverse_flip == 0:
                    x1 = x1 + x1_flipped
                elif self.configs.reverse_flip == 1:
                    x1 = x1 + torch.flip(x1_flipped, dims=[self.configs.flip_dir])

            concatenated_x_perm = torch.permute(concatenated_x, (0, 2, 1))
            x2 = concatenated_x_perm.clone()
            for i in range(self.configs.num_mambas):
                x2 = self.mamba2[i](x2) + x2.clone()
            x2 = torch.permute(x2, (0, 2, 1))

            if self.configs.only_forward_scan == 0:
                x2_flipped = torch.flip(concatenated_x_perm.clone(), dims=[self.configs.flip_dir])
                for i in range(self.configs.num_mambas):
                    x2_flipped = self.mamba2[i](x2_flipped) + x2_flipped.clone()
                x2_flipped = torch.permute(x2_flipped, (0, 2, 1))
                if self.configs.reverse_flip == 0:
                    x2 = x2 + x2_flipped
                elif self.configs.reverse_flip == 1:
                    x2 = x2 + torch.flip(x2_flipped, dims=[self.configs.flip_dir])
        else:
            x1 = torch.zeros_like(concatenated_x)
            x2 = torch.zeros_like(concatenated_x)

        # ----- MambaSL branch on the raw multivariate time series -----
        # x_raw_mts arrives as (B, D, L) from the collate_fn; MambaSL expects (B, L, D).
        xsl = self.mambasl_branch(x_raw_mts.permute(0, 2, 1))  # (B, enc_in, projected_space*3)

        # ----- Tri-source fusion at X3 -----
        x3 = x1 + x2 + self.mambasl_alpha * xsl

        # ----- DP: pooling + classifier (same as TSCMamba) -----
        if self.configs.max_pooling == 0:
            x3 = x3.mean(1)
        else:
            x3, _ = x3.max(1)
        x3 = self.flatten(x3)
        x_logits = self.classifier(x3)
        return x_logits

    # ------------------------------------------------------------------ #
    def forward(self, x_cwt, x_features, x_raw_mts):
        if self.configs.task_name == "classification":
            return self.classification(x_cwt, x_features, x_raw_mts)
