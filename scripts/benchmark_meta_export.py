#!/usr/bin/env python3
"""
PC(PyTorch) 구조 지표 → assets/benchmark_meta.json
Android 벤치마크는 이 파일의 theoretical FLOPs/memory + on-device wall latency 를 합친다.

Usage:
  py -3.13 scripts/benchmark_meta_export.py --pecg "G:/내 드라이브/PECG"
  py -3.13 scripts/benchmark_meta_export.py --evaluator "C:/Users/.../evaluator/evaluator"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

WINDOW_DEMO = 720
NUM_CALIB = 16
FS_HZ = 360.0

WINDOW_HICARDI = 501
FS_HICARDI = 360.0


def _tensor_mib(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024 ** 2)


def _build_demo_module(pecg: Path, ckpt: Path | None):
    scripts = Path(__file__).resolve().parent
    import os
    os.chdir(pecg)
    for p in (pecg / "src", pecg, scripts):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from pecg_legacy_model import LegacyResUMambaNet, patch_pecg_for_legacy_ckpt
    from src.models.patient_encoder_cpc import CPCPatientEncoder

    patch_pecg_for_legacy_ckpt()

    class PecgDemoModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = CPCPatientEncoder(
                input_channels=1, base_channels=32,
                embedding_dim=128, context_dim=128, dropout=0.1,
            )
            self.model = LegacyResUMambaNet(
                in_channels=1, num_classes=5, patient_dim=128,
                base_channels=64, mamba_d_model=128, mamba_layers=2,
                clinical_dim=32, num_attn_heads=4, dropout=0.1,
            )

        def forward(self, x: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
            return self.model(x, self.encoder(calib), None)

    wrapper = PecgDemoModule()
    if ckpt and ckpt.is_file():
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        enc_sd = {k[len("patient_encoder."):]: v for k, v in sd.items() if k.startswith("patient_encoder.")}
        model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        wrapper.encoder.load_state_dict(enc_sd, strict=False)
        wrapper.model.load_state_dict(model_sd, strict=False)
    wrapper.eval()
    return wrapper


def _count_with_evaluator(model: nn.Module, sample, evaluator_dir: Path) -> tuple[int, str]:
    sys.path.insert(0, str(evaluator_dir.parent))
    from evaluator.FLOPs import count_flops as ev_flops  # type: ignore
    if isinstance(sample, tuple):
        return ev_flops(model, sample[0]), "evaluator.FLOPs (hook)"
    return ev_flops(model, sample), "evaluator.FLOPs (hook)"


def _demo_meta(pecg: Path, ckpt: Path | None, evaluator_dir: Path | None) -> dict:
    from mac_count import activation_mib, count_flops, param_count, param_mib

    model = _build_demo_module(pecg, ckpt)
    x = torch.randn(1, 1, WINDOW_DEMO)
    calib = torch.randn(1, NUM_CALIB, 1, WINDOW_DEMO)
    sample = (x, calib)

    if evaluator_dir:
        try:
            flops, src = _count_with_evaluator(model, sample, evaluator_dir)
        except Exception as e:
            print(f"  evaluator FLOPs failed ({e}), fallback mac_count")
            flops = count_flops(model, sample)
            src = "mac_count hook (Mamba SSM 일부 미포함 가능)"
    else:
        flops = count_flops(model, sample)
        src = "mac_count hook (Mamba SSM 일부 미포함 가능)"

    with torch.no_grad():
        logits = model(x, calib)

    mp = param_mib(model)
    bx = _tensor_mib(x)
    bc = _tensor_mib(calib)
    by = _tensor_mib(logits)
    act_sum = activation_mib(model, sample)
    fwd_oh = max(act_sum - by, 0.0)
    total = mp + bx + bc + by + fwd_oh

    return {
        "label": "Demo PECG",
        "pte_file": "demo_model.pte",
        "source": src,
        "window_size": WINDOW_DEMO,
        "num_calib": NUM_CALIB,
        "num_classes": 5,
        "fs_hz": FS_HZ,
        "signal_ms": WINDOW_DEMO / FS_HZ * 1000.0,
        "flops_m": round(flops / 1e6, 3),
        "params_k": round(param_count(model) / 1e3, 2),
        "memory_mib": {
            "M_param": round(mp, 4),
            "Batch_x": round(bx, 4),
            "Batch_calib": round(bc, 4),
            "Batch_y_hat": round(by, 4),
            "Forward_OH_infer": round(fwd_oh, 4),
            "Total_infer": round(total, 4),
        },
    }


def _hicardi_meta(assets: Path, evaluator_dir: Path | None, hicardi_ckpt: Path | None) -> dict:
    pte = assets / "model.pte"
    pte_bytes = pte.stat().st_size if pte.is_file() else 0
    # float32 weight 하한 추정 (실제는 int8/mixed 가능)
    params_est = int(pte_bytes / 4) if pte_bytes else 0
    mp_est = pte_bytes / (1024 ** 2) * 0.85

    bx = _tensor_mib(torch.zeros(1, 1, WINDOW_HICARDI))
    by = _tensor_mib(torch.zeros(1, 7))
    fwd_oh = 0.5
    total = mp_est + bx + by + fwd_oh

    meta = {
        "label": "HiCardi 7-class",
        "pte_file": "model.pte",
        "source": "pte_size_estimate",
        "window_size": WINDOW_HICARDI,
        "num_calib": 0,
        "num_classes": 7,
        "fs_hz": FS_HICARDI,
        "signal_ms": WINDOW_HICARDI / FS_HICARDI * 1000.0,
        "flops_m": None,
        "params_k": round(params_est / 1e3, 2),
        "memory_mib": {
            "M_param": round(mp_est, 4),
            "Batch_x": round(bx, 4),
            "Batch_calib": 0.0,
            "Batch_y_hat": round(by, 4),
            "Forward_OH_infer": round(fwd_oh, 4),
            "Total_infer": round(total, 4),
        },
    }

    if hicardi_ckpt and evaluator_dir:
        # optional: load HiCardi pytorch when checkpoint provided
        pass

    return meta


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pecg", type=Path, default=Path(r"G:/내 드라이브/PECG"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--evaluator", type=Path, default=None,
                        help="evaluator/evaluator 폴더 (FLOPs.py 있는 곳의 부모)")
    parser.add_argument("--out-dir", type=Path, default=root / "app" / "src" / "main" / "assets")
    args = parser.parse_args()

    assets = args.out_dir
    ckpt = args.ckpt
    if ckpt is None and args.pecg.is_dir():
        for pat in ("paper_full/best.ckpt", "checkpoints/best.ckpt", "best.ckpt"):
            p = args.pecg / pat
            if p.is_file():
                ckpt = p
                break

    evaluator_dir = None
    if args.evaluator:
        ev = args.evaluator
        if (ev / "FLOPs.py").is_file():
            evaluator_dir = ev
        elif (ev / "evaluator" / "FLOPs.py").is_file():
            evaluator_dir = ev / "evaluator"

    models: dict[str, dict] = {}

    if args.pecg.is_dir():
        print("Demo PECG meta (PyTorch)...")
        demo = _demo_meta(args.pecg, ckpt, evaluator_dir)
        models[demo["pte_file"]] = demo
        print(f"  FLOPs={demo['flops_m']} MFLOPs  params={demo['params_k']} K")
    else:
        print(f"PECG not found: {args.pecg} — skip demo meta")

    print("HiCardi meta (pte estimate)...")
    hicardi = _hicardi_meta(assets, evaluator_dir, None)
    models[hicardi["pte_file"]] = hicardi

    out = {
        "version": 1,
        "note": "theoretical — PC PyTorch. Android wall latency is measured separately.",
        "models": models,
    }
    out_path = assets / "benchmark_meta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
