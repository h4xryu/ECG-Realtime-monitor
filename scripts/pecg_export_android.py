#!/usr/bin/env python3
"""
PECG -> Android Demo assets exporter
====================================
MIT-BIH demo_record.json + ExecuTorch demo_model.pte (MambaAtenChunked)

Usage (ExecuTorch export requires Python 3.10-3.13 + executorch):
  py -3.13 -m pip install executorch wfdb numpy torch
  py -3.13 scripts/pecg_export_android.py --pecg "G:/내 드라이브/PECG"

Android: app uses org.pytorch:executorch-android (same major as pip executorch).
구형 app/libs/executorch.aar 는 Mamba demo_model.pte 에서 forward 0x14(operator missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_PECG = Path(r"G:/내 드라이브/PECG")
WINDOW_SIZE = 720
FS = 360
NUM_CALIB = 16
CLASS_NAMES = ["N", "S", "V", "F"]
CALIBRATION_SECONDS = 60.0
GAP_SECONDS = 300.0

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}


def zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-8)
    return ((x - mean) / std).astype(np.float32)


def find_ckpt(pecg: Path) -> Path | None:
    candidates = [
        pecg / "results" / "paper_full" / "best.ckpt",
        pecg / "results" / "rerun_weighted" / "best.ckpt",
        pecg / "results" / "main_inter_pipeline" / "classifier" / "best.ckpt",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 5_000_000:
            return p
    return None


def load_record_beats(data_dir: Path, record_name: str, class_names: list[str]) -> tuple[np.ndarray, list[dict]]:
    import wfdb

    half = WINDOW_SIZE // 2
    record_path = data_dir / record_name
    record = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float32)
    signal = zscore(signal)

    beats: list[dict] = []
    for beat_idx, (sample_idx, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        label = AAMI_MAP.get(symbol)
        if label is None or label not in class_names:
            continue
        start = int(sample_idx) - half
        end = int(sample_idx) + half
        if start < 0 or end > len(signal):
            continue
        beats.append({
            "beat_index": beat_idx,
            "sample_index": int(sample_idx),
            "label": label,
            "start": start,
            "end": end,
        })
    beats.sort(key=lambda b: b["sample_index"])
    return signal, beats


def select_eval_beats(beats: list[dict], rec_start: int) -> list[dict]:
    calib_end = rec_start + int(CALIBRATION_SECONDS * FS)
    eval_start = calib_end + int(GAP_SECONDS * FS)
    calib_beats = [b for b in beats if b["sample_index"] <= calib_end]
    if not calib_beats:
        calib_beats = beats[:NUM_CALIB]
    eval_beats = [b for b in beats if b["sample_index"] >= eval_start]
    return calib_beats, eval_beats


def make_calibration(signal: np.ndarray, calib_beats: list[dict]) -> np.ndarray:
    cal = calib_beats[:NUM_CALIB]
    if len(cal) < NUM_CALIB:
        if cal:
            cal = cal + [cal[-1]] * (NUM_CALIB - len(cal))
        else:
            return np.zeros((NUM_CALIB, WINDOW_SIZE), dtype=np.float32)
    return np.stack([signal[b["start"]:b["end"]] for b in cal], axis=0).astype(np.float32)


def export_demo_record(pecg: Path, out_json: Path, record_name: str, max_beats: int) -> dict:
    data_dir = pecg / "mitdb"
    if not (data_dir / f"{record_name}.hea").exists():
        raise FileNotFoundError(f"MIT-BIH record not found: {data_dir / record_name}")

    signal, beats = load_record_beats(data_dir, record_name, CLASS_NAMES)
    if not beats:
        raise RuntimeError(f"No beats in record {record_name}")

    rec_start = beats[0]["sample_index"]
    calib_beats, eval_beats = select_eval_beats(beats, rec_start)
    if not eval_beats:
        eval_beats = beats[NUM_CALIB:]

    buckets: dict[str, list] = {k: [] for k in CLASS_NAMES}
    for b in eval_beats:
        buckets[b["label"]].append(b)
    per_class = max(4, max_beats // 4)
    picked: list[dict] = []
    for label in CLASS_NAMES:
        picked.extend(buckets[label][:per_class])
    if len(picked) < max_beats:
        seen = {id(b) for b in picked}
        for b in eval_beats:
            if id(b) not in seen:
                picked.append(b)
                seen.add(id(b))
            if len(picked) >= max_beats:
                break
    eval_beats = picked[:max_beats]

    calibration = make_calibration(signal, calib_beats)
    calib_out = [[float(v) for v in row.tolist()] for row in calibration]

    beat_out = []
    for i, b in enumerate(eval_beats):
        wave = signal[b["start"]:b["end"]]
        beat_out.append({
            "label": b["label"],
            "sample_index": b["sample_index"],
            "beat_index": i,
            "waveform": [float(v) for v in wave.tolist()],
        })

    payload = {
        "source": "PECG/mitdb",
        "record": record_name,
        "fs": FS,
        "window_size": WINDOW_SIZE,
        "num_calibration_beats": NUM_CALIB,
        "class_names": CLASS_NAMES,
        "protocol": "de_chazal_ds2_60s_calib_5min_gap",
        "calibration_beats": calib_out,
        "beats": beat_out,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return {
        "record": record_name,
        "calibration_beats": len(calib_out),
        "eval_beats": len(beat_out),
        "labels": {k: sum(1 for b in beat_out if b["label"] == k) for k in CLASS_NAMES},
    }


def _setup_pecg_paths(pecg: Path, scripts: Path) -> None:
    import os
    os.chdir(pecg)
    for p in (pecg / "src", pecg, scripts):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def export_pte(pecg: Path, ckpt: Path, out_pte: Path) -> None:
    scripts = Path(__file__).resolve().parent
    _setup_pecg_paths(pecg, scripts)

    from pecg_legacy_model import LegacyResUMambaNet, patch_pecg_for_legacy_ckpt
    patch_pecg_for_legacy_ckpt()

    import torch
    import torch.nn as nn
    from executorch.exir import to_edge
    from src.models.patient_encoder_cpc import CPCPatientEncoder
    from torch.export import export

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
            vec = self.encoder(calib)
            return self.model(x, vec, None)

    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    wrapper = PecgDemoModule()
    enc_sd = {k[len("patient_encoder."):]: v for k, v in sd.items() if k.startswith("patient_encoder.")}
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    wrapper.encoder.load_state_dict(enc_sd, strict=False)
    wrapper.model.load_state_dict(model_sd, strict=False)
    wrapper.eval()
    print(f"  loaded checkpoint weights: {ckpt.name}")

    x = torch.randn(1, 1, WINDOW_SIZE)
    calib = torch.randn(1, NUM_CALIB, 1, WINDOW_SIZE)
    with torch.no_grad():
        out = wrapper(x, calib)
    print(f"  smoke test logits shape={tuple(out.shape)}")

    print("  torch.export ...")
    ep = export(wrapper, (x, calib), strict=False)
    print("  to_edge ...")
    edge = to_edge(ep)
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        et_prog = edge.to_executorch(partitioner=[XnnpackPartitioner()])
    except TypeError:
        et_prog = edge.to_executorch()
    out_pte.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pte, "wb") as f:
        f.write(et_prog.buffer)
    print(f"  wrote ExecuTorch: {out_pte} ({out_pte.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pecg", type=Path, default=DEFAULT_PECG)
    parser.add_argument("--record", default="100")
    parser.add_argument("--max-beats", type=int, default=40)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=root / "app" / "src" / "main" / "assets")
    args = parser.parse_args()

    pecg: Path = args.pecg
    if not pecg.is_dir():
        print(f"PECG not found: {pecg}")
        sys.exit(1)

    out_json = args.out_dir / "demo_record.json"
    out_pte = args.out_dir / "demo_model.pte"

    print(f"Export demo_record.json from {pecg / 'mitdb'}/{args.record} ...")
    info = export_demo_record(pecg, out_json, args.record, args.max_beats)
    print(f"  -> {out_json}")
    print(f"  beats={info['eval_beats']} labels={info['labels']}")

    ckpt = args.ckpt or find_ckpt(pecg)
    if ckpt is None:
        print("No .ckpt found - skip demo_model.pte")
        sys.exit(1)

    print(f"Export demo_model.pte (ExecuTorch) from {ckpt} ...")
    export_pte(pecg, ckpt, out_pte)

    print("Export benchmark_meta.json (PC theoretical) ...")
    try:
        import subprocess
        script = Path(__file__).resolve().parent / "benchmark_meta_export.py"
        subprocess.run(
            [sys.executable, str(script), "--pecg", str(pecg), "--ckpt", str(ckpt), "--out-dir", str(args.out_dir)],
            check=False,
        )
    except Exception as e:
        print(f"  benchmark_meta skip: {e}")


if __name__ == "__main__":
    main()
