"""Conv1d(Cin=Cout) → DW-Sep 근사 (rub.conv1 등 square conv 전용)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def svd_rank1_dwsep_from_conv(conv: nn.Conv1d) -> tuple[nn.Conv1d, nn.Conv1d, dict]:
    """rank=1 특수 경우 — 하위 호환."""
    module, meta = svd_multirank_dwsep_from_conv(conv, ranks=1)
    dw = module.blocks[0]["dw"]
    pw = module.blocks[0]["pw"]
    return dw, pw, meta


def svd_energy_stats(conv: nn.Conv1d) -> dict:
    w = conv.weight.data
    c = w.shape[1]
    ratios: list[float] = []
    rel_err: list[float] = []
    for ch in range(c):
        s = torch.linalg.svdvals(w[:, ch, :])
        ratios.append(float((s[0] ** 2 / s.pow(2).sum()).item()))
        u, sv, vh = torch.linalg.svd(w[:, ch, :], full_matrices=False)
        approx = torch.outer(u[:, 0], vh[0, :]) * sv[0]
        rel_err.append(float(torch.norm(w[:, ch, :] - approx) / torch.norm(w[:, ch, :])))
    return {
        "mean_sigma1_energy_ratio": float(sum(ratios) / c),
        "median_sigma1_energy_ratio": float(sorted(ratios)[c // 2]),
        "pct_channels_sigma1_gt_80": float(sum(1 for r in ratios if r > 0.8) / c),
        "mean_rank1_rel_frobenius_error": float(sum(rel_err) / c),
    }


def svd_multirank_dwsep_from_conv(conv: nn.Conv1d, ranks: int = 1) -> tuple[nn.Module, dict]:
    """
    입력 채널 c마다 W[:,c,:]의 상위 `ranks`개 특이값을 각각 DW+PW 쌍으로 분해해 합산.

    ranks=1 이면 사용자가 제시한 SVD rank-1 per-input-channel 과 동일.
    """
    if conv.in_channels != conv.out_channels:
        raise ValueError("SVD multi-rank DW-Sep은 Cin=Cout 일 때만 지원")
    c = conv.in_channels
    k = conv.kernel_size[0]
    pad = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    ranks = max(1, min(ranks, k, c))
    w = conv.weight.data

    blocks: list[tuple[nn.Conv1d, nn.Conv1d]] = []
    energy_captured: list[float] = []

    with torch.no_grad():
        for r in range(ranks):
            dw = nn.Conv1d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)
            pw = nn.Conv1d(c, c, kernel_size=1, bias=False)
            dw.weight.zero_()
            pw.weight.zero_()
            for ch in range(c):
                slice_mat = w[:, ch, :]
                u, s, vh = torch.linalg.svd(slice_mat, full_matrices=False)
                if r >= s.numel():
                    continue
                sigma = s[r].clamp(min=0)
                sqrt_s = torch.sqrt(sigma)
                pw.weight[:, ch, 0].copy_(sqrt_s * u[:, r])
                dw.weight[ch, 0, :].copy_(sqrt_s * vh[r, :])
            blocks.append((dw, pw))

        for ch in range(c):
            s = torch.linalg.svdvals(w[:, ch, :])
            energy_captured.append(float(s[:ranks].pow(2).sum() / s.pow(2).sum()))

    class MultiRankDSConv1d(nn.Module):
        def __init__(self, pairs: list[tuple[nn.Conv1d, nn.Conv1d]]):
            super().__init__()
            self.blocks = nn.ModuleList([nn.ModuleDict({"dw": d, "pw": p}) for d, p in pairs])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = 0
            for block in self.blocks:
                y = y + block["pw"](block["dw"](x))
            return y

    meta = {
        "svd_ranks": ranks,
        "mean_weight_energy_captured": float(sum(energy_captured) / c),
        **svd_energy_stats(conv),
    }
    return MultiRankDSConv1d(blocks), meta


def replace_conv_with_svd_dwsep(conv: nn.Conv1d, ranks: int = 1) -> tuple[nn.Module, dict]:
    return svd_multirank_dwsep_from_conv(conv, ranks=ranks)


def diagonal_depthwise_from_conv(conv: nn.Conv1d) -> nn.Conv1d:
    """(구버전) 대각 spatial 필터만 추출 — 성능 손실 큼."""
    if conv.in_channels != conv.out_channels:
        raise ValueError("diagonal DW 근사는 Cin=Cout 일 때만 지원")
    c = conv.in_channels
    k = conv.kernel_size[0]
    pad = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    dw = nn.Conv1d(c, c, kernel_size=k, padding=pad, groups=c, bias=False)
    with torch.no_grad():
        dw.weight.zero_()
        w = conv.weight.data
        for ch in range(c):
            dw.weight[ch, 0].copy_(w[ch, ch])
    return dw


def fit_pointwise_after_dw(
    conv: nn.Conv1d,
    dw: nn.Conv1d,
    *,
    batch: int = 16,
    length: int = 180,
    ridge: float = 1e-4,
) -> nn.Conv1d:
    """(구버전) DW 출력에 1×1 PW ridge 회귀."""
    c = conv.in_channels
    pw = nn.Conv1d(c, c, kernel_size=1, bias=False)
    conv.eval()
    dw.eval()

    with torch.no_grad():
        x = torch.randn(batch, c, length)
        y_tgt = conv(x)
        z = dw(x)
        z_flat = z.permute(0, 2, 1).reshape(-1, c)
        y_flat = y_tgt.permute(0, 2, 1).reshape(-1, c)
        reg = ridge * torch.eye(c, device=z.device, dtype=z.dtype)
        ztz = z_flat.T @ z_flat + reg
        for out_c in range(c):
            rhs = z_flat.T @ y_flat[:, out_c]
            pw.weight.data[out_c, :, 0] = torch.linalg.solve(ztz, rhs)
    return pw


def replace_conv_with_dsconv(conv: nn.Conv1d, *, ridge: float = 1e-4) -> tuple[nn.Conv1d, nn.Conv1d]:
    dw = diagonal_depthwise_from_conv(conv)
    pw = fit_pointwise_after_dw(conv, dw, ridge=ridge)
    return dw, pw


class DSConv1d(nn.Module):
    """Depthwise k + Pointwise 1×1."""

    def __init__(self, dw: nn.Conv1d, pw: nn.Conv1d):
        super().__init__()
        self.dw = dw
        self.pw = pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


def conv_flops_per_sample(conv: nn.Conv1d, length: int) -> int:
    """MACs ≈ FLOPs/2 for conv."""
    k = conv.kernel_size[0]
    cin = conv.in_channels
    cout = conv.out_channels
    groups = conv.groups
    return 2 * k * (cin // groups) * cout * length


def dwsep_flops_per_sample(channels: int, kernel: int, length: int) -> int:
    dw = 2 * kernel * channels * length
    pw = 2 * channels * channels * length
    return dw + pw


def conv_params(conv: nn.Conv1d) -> int:
    return int(conv.weight.numel())


def dwsep_params(channels: int, kernel: int) -> int:
    return channels * kernel + channels * channels


def reduction_ratio(channels: int, kernel: int) -> float:
    full = channels * channels * kernel
    sep = channels * kernel + channels * channels
    return 1.0 - sep / full


def conv_output_mse(conv: nn.Conv1d, approx: nn.Module, length: int = 180, batch: int = 8) -> float:
    conv.eval()
    approx.eval()
    with torch.no_grad():
        x = torch.randn(batch, conv.in_channels, length)
        y = conv(x)
        y_hat = approx(x)
        return float(F.mse_loss(y_hat, y).item())
