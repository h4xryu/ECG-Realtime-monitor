"""
models/mamba_aten.py
====================
ATen-only Mamba SSM block — torch.export / ExecuTorch / ONNX compatible.

Design goals
------------
* Parameter names and shapes are **identical** to mamba_ssm.Mamba so that
  any checkpoint produced with mamba_ssm loads via
  ``model.load_state_dict(state, strict=False)`` without key remapping.
* No CUDA extensions: causal_conv1d_cuda and selective_scan_cuda are
  replaced with pure ATen ops (F.pad, F.conv1d, einsum, sequential loop).
* The selective scan runs a static-L for-loop; torch.export unrolls it
  into individual ATen nodes — the resulting graph is flat, verifiable, and
  deployable via ExecuTorch on edge devices.
  (Chunked variant: ``mamba_aten_chunked.MambaAtenChunked``.)

Causal conv1d
-------------
mamba_ssm's causal_conv1d_cuda implements a strictly causal depthwise conv:
output[t] depends only on input[0 .. t].  Pure-ATen equivalent:

    x = F.pad(x, (kernel_size - 1, 0))   # left-pad only
    x = F.conv1d(x, w, b, groups=D)      # no extra padding → len stays L

Selective scan
--------------
Discrete-time linear recurrence (ZOH):
    A_bar[b,d,l,n] = exp( delta[b,d,l] · A[d,n] )
    h[t] = A_bar[t] · h[t-1]  +  delta[t] · B[t] · u[t]
    y[t] = C[t] · h[t]  +  D · u[t]
Then gated by SiLU(z).  Numerically identical to selective_scan_ref.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


class MambaAten(nn.Module):
    """
    ATen-compatible drop-in for mamba_ssm.Mamba.

    Constructor kwargs match mamba_ssm.Mamba.__init__ (d_model, d_state,
    d_conv, expand) plus **kwargs to silently absorb extras (dt_rank,
    dt_min, etc.) that some mamba_ssm versions accept.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
        **kwargs,           # absorb mamba_ssm-specific kwargs (dt_rank, chunk_size, …)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        # ── parameter layout mirrors mamba_ssm.Mamba exactly ──────────────
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # padding=d_conv-1 matches the checkpoint key; actual causal padding
        # is done manually in _causal_conv1d so F.conv1d sees padding=0.
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size = d_conv,
            groups      = self.d_inner,
            padding     = d_conv - 1,
            bias        = True,
        )

        self.x_proj   = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log    = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.profiling_mode: bool = False   # profile_components() 에서만 True

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        B, L, _ = hidden_states.shape

        # ── in_proj ───────────────────────────────────────────────────────
        xz = self.in_proj(hidden_states).transpose(1, 2)   # [B, 2*D, L]
        x, z = xz.chunk(2, dim=1)                          # [B, D, L] each

        # ── causal depthwise conv1d ────────────────────────────────────────
        x = self._causal_conv1d(x)                         # [B, D, L]
        x = F.silu(x)

        # ── project to SSM inputs ─────────────────────────────────────────
        A = -torch.exp(self.A_log.float())                 # [D, N]  (always < 0)

        # flatten time: [B*L, D] → [B*L, dt_rank + 2*N]
        x_dbl = self.x_proj(
            x.permute(0, 2, 1).reshape(B * L, self.d_inner)
        )
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # delta: [B, D, L]  (dt_proj bias + softplus = ZOH discretisation)
        delta = F.softplus(
            self.dt_proj(dt_raw)
            .reshape(B, L, self.d_inner)
            .permute(0, 2, 1)                              # [B, D, L]
        )

        # B_ssm, C_ssm: [B, N, L]
        B_ssm = B_ssm.reshape(B, L, self.d_state).permute(0, 2, 1)
        C_ssm = C_ssm.reshape(B, L, self.d_state).permute(0, 2, 1)

        # ── selective scan ────────────────────────────────────────────────
        with record_function("ssm_scan"):
            y = self._selective_scan(x, delta, A, B_ssm, C_ssm, z)  # [B, D, L]

        # ── out_proj ──────────────────────────────────────────────────────
        return self.out_proj(y.permute(0, 2, 1))           # [B, L, d_model]

    def _causal_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Strictly causal depthwise conv1d — pure ATen, no CUDA extension.

        Left-pad by (kernel_size - 1) zeros; apply F.conv1d with no extra
        padding so output length equals L and output[t] uses only x[0..t].
        """
        x = F.pad(x, (self.d_conv - 1, 0))
        if self.profiling_mode:
            # module.forward() 를 경유해 forward hook 이 발화하도록 분기.
            # conv1d.padding=(d_conv-1) 은 양방향 패딩이므로 0 으로 임시 교체.
            orig = self.conv1d.padding
            self.conv1d.padding = (0,)
            out = self.conv1d(x)
            self.conv1d.padding = orig
            return out
        return F.conv1d(
            x, self.conv1d.weight, self.conv1d.bias, groups=self.d_inner
        )

    def _selective_scan(
        self,
        u:     torch.Tensor,   # [B, D, L]
        delta: torch.Tensor,   # [B, D, L]
        A:     torch.Tensor,   # [D, N]
        B:     torch.Tensor,   # [B, N, L]
        C:     torch.Tensor,   # [B, N, L]
        z:     torch.Tensor,   # [B, D, L]  gate
    ) -> torch.Tensor:
        """
        Sequential linear recurrence using ATen ops.

        Numerically equivalent to mamba_ssm.ops.selective_scan_ref.
        torch.export sees a static loop (L is a Python int for fixed input
        shapes) and unrolls it into a flat sequence of ATen ops — compatible
        with ExecuTorch and ONNX.

        Step t:
            h[t] = A_bar[t] * h[t-1] + deltaB_u[t]
            y[t] = C[:,t] · h[t]
        """
        # concrete int so torch.export unrolls range(L) statically
        L = int(u.shape[2])

        # discrete matrices  [B, D, L, N]
        deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

        # sequential state updates
        h  = deltaA.new_zeros(u.shape[0], A.shape[0], A.shape[1])  # [B,D,N]
        ys = []
        for i in range(L):
            h = deltaA[:, :, i] * h + deltaB_u[:, :, i]
            ys.append(torch.einsum('bdn,bn->bd', h, C[:, :, i]))   # [B,D]

        y = torch.stack(ys, dim=2)                                   # [B,D,L]

        # D skip connection + SiLU gate
        y = y + self.D.float().unsqueeze(0).unsqueeze(-1) * u
        y = y * F.silu(z)
        return y.to(u.dtype)
