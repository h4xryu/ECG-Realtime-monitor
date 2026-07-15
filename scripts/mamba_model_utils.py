"""MambaResU + baseline_best.ckpt 로드 유틸 (mamba_ssm 없이)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DEFAULT_CKPT = MODEL_DIR / "baseline_best.ckpt"


def _install_mamba_ssm_stub() -> None:
    if "mamba_ssm.modules.mamba_simple" in sys.modules:
        return
    mamba_ssm = types.ModuleType("mamba_ssm")
    modules = types.ModuleType("mamba_ssm.modules")
    ms = types.ModuleType("mamba_ssm.modules.mamba_simple")

    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    ms.Mamba = Mamba
    modules.mamba_simple = ms
    mamba_ssm.modules = modules
    for name, mod in (
        ("mamba_ssm", mamba_ssm),
        ("mamba_ssm.modules", modules),
        ("mamba_ssm.modules.mamba_simple", ms),
    ):
        sys.modules[name] = mod


def _load_model_modules():
    _install_mamba_ssm_stub()
    models = types.ModuleType("models")
    sys.modules["models"] = models
    for fname, modname in (
        ("mamba_aten.py", "mamba_aten"),
        ("mamba_aten_chunked.py", "mamba_aten_chunked"),
        ("MambaResU.py", "MambaResU"),
    ):
        spec = importlib.util.spec_from_file_location(f"models.{modname}", MODEL_DIR / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"models.{modname}"] = mod
        spec.loader.exec_module(mod)
        setattr(models, modname, mod)
    from models.mamba_aten_chunked import MambaAtenChunked
    from models.MambaResU import MambaResU

    return MambaAtenChunked, MambaResU


def encoder_state_dict(ckpt_path: Path = DEFAULT_CKPT) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder.")
    }


def load_mamba_resu(
    ckpt_path: Path = DEFAULT_CKPT,
    *,
    chunk_size: int = 24,
    eval_mode: bool = True,
) -> nn.Module:
    MambaAtenChunked, MambaResU = _load_model_modules()
    model = MambaResU(
        nOUT=5,
        mamba_cls=MambaAtenChunked,
        mamba_kw={"chunk_size": chunk_size},
    )
    model.load_state_dict(encoder_state_dict(ckpt_path), strict=True)
    if eval_mode:
        model.eval()
    return model
