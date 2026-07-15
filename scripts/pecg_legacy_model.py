"""PECG 체크포인트(paper_full)와 호환되는 레거시 ResUMambaNet — export 전용."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import AdaIN1D, CrossAttentionFusion, MambaLikeBlock, ResUBlock


class LegacyResUMambaNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        patient_dim: int = 128,
        base_channels: int = 64,
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        clinical_dim: int = 32,
        num_attn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_fusion = base_channels + mamba_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.resu1 = ResUBlock(base_channels, depth=3, dropout=dropout)
        self.resu2 = ResUBlock(base_channels, depth=2, dropout=dropout)
        self.mamba_in = nn.Sequential(
            nn.Conv1d(base_channels, mamba_d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(mamba_d_model),
            nn.GELU(),
        )
        self.mamba_blocks = nn.ModuleList([
            MambaLikeBlock(mamba_d_model, dropout=dropout) for _ in range(mamba_layers)
        ])
        self.fusion_proj = nn.Sequential(
            nn.Conv1d(d_fusion, d_fusion, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_fusion),
            nn.GELU(),
        )
        self.adain = AdaIN1D(d_fusion, patient_dim)
        self.cross_attn = CrossAttentionFusion(d_fusion, clinical_dim, num_attn_heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_fusion, d_fusion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, num_classes),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        x: torch.Tensor,
        patient_vec: torch.Tensor,
        clinical_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.stem(x)
        y_u = self.resu2(self.resu1(h))
        y_m = self.mamba_in(h)
        for blk in self.mamba_blocks:
            y_m = blk(y_m)
        y = self.fusion_proj(torch.cat([y_u, y_m], dim=1))
        y = self.adain(y, patient_vec)
        if clinical_feats is not None:
            z = self.cross_attn(y, clinical_feats)
        else:
            z = self.global_pool(y).squeeze(-1)
        return self.head(z)


def patch_pecg_for_legacy_ckpt() -> None:
    """paper_full ckpt: resu1+resu2. MambaLikeBlock -> MambaAtenChunked (blocks.py)."""
    import src.models.resumamba as rum

    rum.ResUMambaNet = LegacyResUMambaNet
