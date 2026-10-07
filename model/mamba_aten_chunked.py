# Colab: paste this whole cell and run.
# Local: python model/mamba_aten_chunked.py

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaAten(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = F.pad(x, (self.d_conv - 1, 0))
        x = F.conv1d(x, self.conv1d.weight, self.conv1d.bias, groups=self.d_inner)
        x = F.silu(x)
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x.permute(0, 2, 1).reshape(B * L, self.d_inner))
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(
            self.dt_proj(dt_raw).reshape(B, L, self.d_inner).permute(0, 2, 1)
        )
        B_ssm = B_ssm.reshape(B, L, self.d_state).permute(0, 2, 1)
        C_ssm = C_ssm.reshape(B, L, self.d_state).permute(0, 2, 1)
        y = self._selective_scan(x, delta, A, B_ssm, C_ssm, z)
        return self.out_proj(y.permute(0, 2, 1))

    def _selective_scan(self, u, delta, A, B, C, z):
        L = int(u.shape[2])
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        h = deltaA.new_zeros(u.shape[0], A.shape[0], A.shape[1])
        ys = []
        for i in range(L):
            h = deltaA[:, :, i] * h + deltaB_u[:, :, i]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, i]))
        y = torch.stack(ys, dim=2)
        y = y + self.D.float().unsqueeze(0).unsqueeze(-1) * u
        y = y * F.silu(z)
        return y.to(u.dtype)


def selective_scan_chunked_exact(u, delta, A, B, C, z, D, chunk_size=24):
    Bsz, Dd, L = u.shape
    N = A.shape[1]
    c = int(chunk_size)
    pad = (-L) % c
    if pad:
        u_p = F.pad(u, (0, pad))
        delta_p = F.pad(delta, (0, pad))
        B_p = F.pad(B, (0, pad))
        C_p = F.pad(C, (0, pad))
    else:
        u_p, delta_p, B_p, C_p = u, delta, B, C
    Lp = L + pad
    K = Lp // c
    a = torch.exp(torch.einsum("bdl,dn->bdln", delta_p, A)).reshape(Bsz, Dd, K, c, N)
    b = torch.einsum("bdl,bnl,bdl->bdln", delta_p, B_p, u_p).reshape(Bsz, Dd, K, c, N)
    acc = torch.ones(Bsz, Dd, K, N, dtype=a.dtype, device=a.device)
    hh = torch.zeros(Bsz, Dd, K, N, dtype=a.dtype, device=a.device)
    acum_list, hloc_list = [], []
    for i in range(c):
        ai = a[:, :, :, i, :]
        acc = ai * acc
        hh = ai * hh + b[:, :, :, i, :]
        acum_list.append(acc)
        hloc_list.append(hh)
    acum = torch.stack(acum_list, dim=3)
    hloc = torch.stack(hloc_list, dim=3)
    A_chunk = acum[:, :, :, -1, :]
    h_end = hloc[:, :, :, -1, :]
    Hcarry = torch.zeros(Bsz, Dd, N, dtype=a.dtype, device=a.device)
    Hprev_list = []
    for k in range(K):
        Hprev_list.append(Hcarry)
        Hcarry = A_chunk[:, :, k, :] * Hcarry + h_end[:, :, k, :]
    Hprev = torch.stack(Hprev_list, dim=2)
    h_full = (hloc + acum * Hprev[:, :, :, None, :]).reshape(Bsz, Dd, Lp, N)
    y_core = torch.einsum("bdln,bnl->bdl", h_full, C_p)
    if pad:
        y_core = y_core[:, :, :L]
    y = y_core + D.to(y_core.dtype).unsqueeze(0).unsqueeze(-1) * u
    return y * F.silu(z)


class MambaAtenChunked(MambaAten):
    def __init__(self, *args, chunk_size=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def _selective_scan(self, u, delta, A, B, C, z):
        return selective_scan_chunked_exact(
            u, delta, A, B, C, z, self.D, chunk_size=self.chunk_size
        )


def selective_scan_chunked_simd(u, delta, A, B, C, z, D, chunk_size=24):
    """Chunked scan with a SIMD-friendly layout.

    The loop version stores (B, D, K, N) with innermost N=16, so a NEON
    vector (4 floats) barely fills.  Here the working slice is (B, K, D*N)
    — 4096 contiguous floats at d_inner=256 — which XNNPACK can vectorize.
    Einsum is replaced by mul/add so the partitioner can delegate.
    """
    Bsz, Dd, L = u.shape
    N = A.shape[1]
    c = int(chunk_size)
    pad = (-L) % c
    if pad:
        u_p = F.pad(u, (0, pad))
        delta_p = F.pad(delta, (0, pad))
        B_p = F.pad(B, (0, pad))
        C_p = F.pad(C, (0, pad))
    else:
        u_p, delta_p, B_p, C_p = u, delta, B, C
    Lp = L + pad
    K = Lp // c
    DN = Dd * N

    a = torch.exp(delta_p.unsqueeze(-1) * A.view(1, Dd, 1, N))
    b = (delta_p * u_p).unsqueeze(-1) * B_p.permute(0, 2, 1).unsqueeze(1)

    # (B, D, K, Q, N) -> (B, Q, K, D*N) so a[:, i] is a contiguous (B, K, DN)
    a = (
        a.reshape(Bsz, Dd, K, c, N)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(Bsz, c, K, DN)
    )
    b = (
        b.reshape(Bsz, Dd, K, c, N)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(Bsz, c, K, DN)
    )

    acc = a.new_ones(Bsz, K, DN)
    hh = a.new_zeros(Bsz, K, DN)
    acum = a.new_empty(Bsz, c, K, DN)
    hloc = a.new_empty(Bsz, c, K, DN)
    for i in range(c):
        ai = a[:, i]
        acc = ai * acc
        hh = ai * hh + b[:, i]
        acum[:, i] = acc
        hloc[:, i] = hh

    A_chunk = acum[:, -1]
    h_end = hloc[:, -1]
    Hcarry = a.new_zeros(Bsz, DN)
    Hprev = a.new_empty(Bsz, K, DN)
    for k in range(K):
        Hprev[:, k] = Hcarry
        Hcarry = A_chunk[:, k] * Hcarry + h_end[:, k]

    h_full = hloc + acum * Hprev.unsqueeze(1)
    h_full = (
        h_full.reshape(Bsz, c, K, Dd, N)
        .permute(0, 3, 2, 1, 4)
        .reshape(Bsz, Dd, Lp, N)
    )
    y_core = (h_full * C_p.permute(0, 2, 1).unsqueeze(1)).sum(-1)
    if pad:
        y_core = y_core[:, :, :L]
    y = y_core + D.to(y_core.dtype).view(1, -1, 1) * u
    return y * F.silu(z)


class MambaAtenChunkedSimd(MambaAten):
    def __init__(self, *args, chunk_size=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def _selective_scan(self, u, delta, A, B, C, z):
        return selective_scan_chunked_simd(
            u, delta, A, B, C, z, self.D, chunk_size=self.chunk_size
        )


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return float("nan")


def reproduce_q_cliff(L=720, d_model=128, qs=(1, 4, 9, 24)):
    from torch.export import export

    x = torch.randn(1, L, d_model)
    print(f"L={L} d_model={d_model} d_inner={2 * d_model}")
    print(f"{'Q':>4} {'c+K':>6} {'fx_nodes':>9} {'rss0':>8} {'rss1':>8} {'drss':>8}")
    print("-" * 52)
    rows = {}
    for Q in qs:
        pad = (-L) % Q
        K = (L + pad) // Q
        trips = Q + K
        m = MambaAtenChunked(d_model=d_model, chunk_size=Q).eval()
        rss0 = _rss_mb()
        with torch.no_grad():
            ep = export(m, (x,), strict=False)
        rss1 = _rss_mb()
        n = sum(1 for node in ep.graph_module.graph.nodes if node.op == "call_function")
        print(f"{Q:>4} {trips:>6} {n:>9} {rss0:>8.1f} {rss1:>8.1f} {rss1 - rss0:>8.1f}")
        rows[Q] = (trips, n)
        del ep, m
    if 1 in rows and 9 in rows:
        t1, n1 = rows[1]
        t9, n9 = rows[9]
        print("-" * 52)
        print(f"Q1/Q9 trips={t1 / t9:.2f}x nodes={n1 / n9:.2f}x")


if __name__ == "__main__":
    reproduce_q_cliff()
