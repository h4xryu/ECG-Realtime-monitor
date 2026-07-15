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
  replaced with pure ATen ops (F.pad, F.conv1d, einsum, parallel scan).

Causal conv1d
-------------
mamba_ssm's causal_conv1d_cuda implements a strictly causal depthwise conv:
output[t] depends only on input[0 .. t].  Pure-ATen equivalent:

    x = F.pad(x, (kernel_size - 1, 0))   # left-pad only
    x = F.conv1d(x, w, b, groups=D)      # no extra padding → len stays L

Selective scan — chunked parallel scan
---------------------------------------
The sequential for-loop (O(L) Python iterations) is replaced with a
chunked Hillis-Steele parallel prefix scan:

  1. Split L into K = ceil(L / chunk_size) chunks of size C (power of 2).
  2. Within each chunk: log2(C) vectorised ATen steps compute the full
     prefix scan simultaneously for all K chunks — no Python loop.
  3. Across chunks: K sequential Python steps propagate the carry state
     (K = L/C << L, e.g. K=12 for L=720, C=64).

The (a, b) monoid composition: (a1,b1)∘(a0,b0) = (a1·a0, a1·b0 + b1)
gives h[t] = A_scan[t]·h_init + B_scan[t] which recovers the exact
sequential recurrence for any h_init, without sequential Python iteration.
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
        d_model:    int,
        d_state:    int = 16,
        d_conv:     int = 4,
        expand:     int = 2,
        chunk_size: int = 64,   # must be a power of 2
        **kwargs,               # absorb mamba_ssm-specific kwargs we don't need
    ):
        super().__init__()
        assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, \
            f"chunk_size must be a power of 2, got {chunk_size}"
        self.d_model    = d_model
        self.d_state    = d_state
        self.d_conv     = d_conv
        self.expand     = expand
        self.d_inner    = int(expand * d_model)
        self.dt_rank    = math.ceil(d_model / 16)
        self.chunk_size = chunk_size

        # ── parameter layout mirrors mamba_ssm.Mamba exactly ──────────────
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # padding=d_conv-1 keeps the checkpoint key; actual causal padding is
        # applied manually in _causal_conv1d so F.conv1d sees padding=0.
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
        """Left-pad only → strictly causal (output[t] uses x[0..t] only)."""
        x = F.pad(x, (self.d_conv - 1, 0))
        if self.profiling_mode:
            # module.forward() 를 경유해 forward hook 이 발화하도록 분기.
            # conv1d.padding=(d_conv-1) 은 양방향 패딩이므로 0 으로 임시 교체.
            orig = self.conv1d.padding
            self.conv1d.padding = (0,)
            out = self.conv1d(x)
            self.conv1d.padding = orig
            return out
        return F.conv1d(x, self.conv1d.weight, self.conv1d.bias, groups=self.d_inner)
    
    def _selective_scan(self):
        pass

    # def _selective_scan(
    #     self,
    #     u:     torch.Tensor,   # [B, D, L]
    #     delta: torch.Tensor,   # [B, D, L]
    #     A:     torch.Tensor,   # [D, N]
    #     B:     torch.Tensor,   # [B, N, L]
    #     C:     torch.Tensor,   # [B, N, L]
    #     z:     torch.Tensor,   # [B, D, L]  gate
    # ) -> torch.Tensor:
    #     """
    #     Chunked parallel selective scan (ATen-only).

    #     Splits the sequence into K = ceil(L / chunk_size) chunks.
    #     Within each chunk: Hillis-Steele prefix scan in log2(chunk_size)
    #     vectorised ATen steps (all K chunks processed simultaneously).
    #     Across chunks: K sequential Python steps to propagate the carry.
    #     For L=720, chunk_size=64: 720 → 6 ATen steps + 12 carry steps.
    #     """
    #     B_batch, D, L = u.shape
    #     N             = A.shape[1]
    #     Cz            = self.chunk_size
    #     log2Cz        = int(math.log2(Cz))

    #     # Discrete-time matrices  [B, D, L, N]
    #     dA  = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    #     dBu = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    #     # Pad L to the nearest multiple of Cz
    #     pad   = (-L) % Cz
    #     L_pad = L + pad
    #     K     = L_pad // Cz
    #     if pad:
    #         dA  = F.pad(dA,  (0, 0, 0, pad), value=1.0)  # identity for A
    #         dBu = F.pad(dBu, (0, 0, 0, pad), value=0.0)  # zero input

    #     # [B, D, K, Cz, N]
    #     dA  = dA.reshape(B_batch, D, K, Cz, N)
    #     dBu = dBu.reshape(B_batch, D, K, Cz, N)

    #     # ── parallel prefix scan within each chunk ────────────────────────
    #     # After scan: dA[k,t] = ∏_{s≤t} orig_dA[k,s]
    #     #             dBu[k,t] = Σ_{s≤t} (∏_{r=s+1}^{t} dA[k,r]) · orig_dBu[k,s]
    #     # Gives h[k,t] = dA[k,t] · h_init[k] + dBu[k,t]
    #     dA, dBu = self._chunk_scan(dA, dBu, log2Cz)

    #     # ── inter-chunk carry propagation (K steps, K << L) ───────────────
    #     chunk_A = dA[:, :, :, -1, :]    # [B, D, K, N]  total A for each chunk
    #     chunk_B = dBu[:, :, :, -1, :]   # [B, D, K, N]  h at chunk end (h_init=0)

    #     h = dA.new_zeros(B_batch, D, N)
    #     h_inits: list[torch.Tensor] = []
    #     for k in range(K):
    #         h_inits.append(h)
    #         h = chunk_A[:, :, k] * h + chunk_B[:, :, k]
    #     h_inits_t = torch.stack(h_inits, dim=2)   # [B, D, K, N]

    #     # ── combine per-chunk carry with within-chunk scan ─────────────────
    #     # h_all[b,d,k,t,n] = dA[k,t] · h_init[k] + dBu[k,t]
    #     h_all = dA * h_inits_t.unsqueeze(3) + dBu          # [B, D, K, Cz, N]
    #     h_all = h_all.reshape(B_batch, D, L_pad, N)[:, :, :L, :]  # [B, D, L, N]

    #     # Output: y[b,d,t] = Σ_n h[b,d,t,n] · C[b,n,t]
    #     y = torch.einsum('bdln,bnl->bdl', h_all, C)        # [B, D, L]
    #     y = y + self.D.float().unsqueeze(0).unsqueeze(-1) * u
    #     y_bar = y * F.silu(z)
    #     return y_bar.to(u.dtype)

    # def _chunk_scan(
    #     self,
    #     A:          torch.Tensor,   # [B, D, K, C, N]
    #     B:          torch.Tensor,   # [B, D, K, C, N]
    #     log2_chunk: int,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Hillis-Steele parallel prefix scan for the (A, B) linear-recurrence monoid.

    #     Monoid composition: (a1, b1) ∘ (a0, b0) = (a1·a0, a1·b0 + b1)
    #     Identity element: (1, 0)

    #     After log2_chunk passes each position t holds the inclusive prefix:
    #         A[t] = ∏_{s=0}^{t} a[s]
    #         B[t] = Σ_{s=0}^{t} (∏_{r=s+1}^{t} a[r]) · b[s]

    #     So h[t] = A[t] · h_init + B[t] recovers the exact sequential recurrence
    #     for any initial state h_init, with zero Python-level time loop.
    #     """
    #     for d in range(log2_chunk):
    #         stride = 1 << d  # 2^d
    #         # Shift right by stride; fill with identity on the left
    #         A_prev = F.pad(A[:, :, :, :-stride, :], (0, 0, stride, 0), value=1.0)
    #         B_prev = F.pad(B[:, :, :, :-stride, :], (0, 0, stride, 0), value=0.0)
    #         # Compose: (A[t], B[t]) ∘ (A_prev[t], B_prev[t])
    #         B = A * B_prev + B   # new B uses old A — evaluated before A update
    #         A = A * A_prev
    #     return A, B
