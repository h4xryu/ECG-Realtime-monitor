# models/MambaResU.py
# Reconstructed from checkpoint: proj_in → Mamba → proj_out → ResU → pool → fc
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super().__init__()
        K = 9
        P = (K - 1) // 2
        self.conv1 = nn.Conv1d(out_ch, out_ch, kernel_size=K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for idx in range(layers):
            in_ch = out_ch if idx == 0 else mid_ch
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, kernel_size=K, stride=2, padding=P, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.LeakyReLU(),
            ))
            out_ch_dec = out_ch if idx == layers - 1 else mid_ch
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(mid_ch * 2, out_ch_dec, kernel_size=K, stride=2,
                                   padding=P, output_padding=1, bias=False),
                nn.BatchNorm1d(out_ch_dec),
                nn.LeakyReLU(),
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU(),
        )
        self.downsample = downsampling
        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))
        out = x_in
        enc_outs = []
        for enc in self.encoders:
            out = enc(out)
            enc_outs.append(out)
        out = self.bottleneck(out)
        for idx, dec in enumerate(self.decoders):
            skip = enc_outs[-1 - idx]
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(out, size=skip.size(-1), mode='linear', align_corners=False)
            out = dec(torch.cat([out, skip], dim=1))
        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]
        out = out + x_in
        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)
        return out


class MambaResU(nn.Module):
    """
    Mamba → ResU architecture.
    proj_in(1→mamba_dim) → LayerNorm → Mamba×N → LayerNorm → proj_out(mamba_dim→out_ch)
    → ResidualUBlock → AdaptiveAvgPool1d → fc(out_ch→nOUT)

    mamba_cls : Mamba 블록 클래스. None 이면 mamba_ssm.Mamba (지연 임포트).
                MambaAten / MambaAtenChunked 를 넘기면 CUDA 익스텐션 로드 없이 생성.
    mamba_kw  : mamba_cls 에 추가로 전달할 키워드 인자 (예: chunk_size).
    """
    def __init__(
        self,
        nOUT: int = 5,
        in_channels: int = 1,
        out_ch: int = 180,
        mid_ch: int = 30,
        rub_layers: int = 2,
        mamba_dim: int = 8,
        mamba_layers: int = 1,
        dropout_rate: float = 0.2,
        mamba_cls=None,
        mamba_kw: dict | None = None,
    ):
        super().__init__()
        if mamba_cls is None:
            from mamba_ssm import Mamba as mamba_cls  # lazy — GPU 미사용 시 CUDA ext 안 로드
        _kw = mamba_kw or {}
        self.proj_in  = nn.Conv1d(in_channels, mamba_dim, kernel_size=1, bias=False)
        self.ln_in    = nn.LayerNorm(mamba_dim)
        self.mambas   = nn.ModuleList([
            mamba_cls(d_model=mamba_dim, d_state=16, d_conv=4, expand=2, **_kw)
            for _ in range(mamba_layers)
        ])
        self.ln_out   = nn.LayerNorm(mamba_dim)
        self.proj_out = nn.Conv1d(mamba_dim, out_ch, kernel_size=1, bias=False)
        self.rub      = ResidualUBlock(out_ch, mid_ch, layers=rub_layers, downsampling=True)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.fc       = nn.Sequential(
            nn.Linear(out_ch, out_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_ch // 2, nOUT),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, in_channels, T]
        z = self.proj_in(x)          # [B, mamba_dim, T]
        z = z.transpose(1, 2)        # [B, T, mamba_dim]
        z = self.ln_in(z)
        for blk in self.mambas:
            z = blk(z)
        z = self.ln_out(z)
        z = z.transpose(1, 2)        # [B, mamba_dim, T]
        x = self.proj_out(z)         # [B, out_ch, T]
        x = self.rub(x)              # [B, out_ch, T//4]
        feat = self.pool(x).squeeze(-1)  # [B, out_ch]
        logits = self.fc(feat)       # [B, nOUT]
        return logits, feat
