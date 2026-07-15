#!/usr/bin/env python3
"""
MambaResU demo assets — demo_record.json + demo_model.pte 복사

Usage:
  py -3.13 scripts/mamba_demo_assets.py
  py -3.13 scripts/mamba_demo_assets.py --record 100 --max-beats 80
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
ASSETS = ROOT / "app" / "src" / "main" / "assets"
PTE_SRC = ROOT / "model" / "mamba_chunked_xnnpack.pte"

WINDOW = 720
FS = 360
CLASS_NAMES = ["N", "S", "V", "F"]
DEFAULT_MITDB = Path(r"D:\workspace\Classification_workflow\mit-bih-arrhythmia-database-1.0.0")

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)).astype(np.float32)


def export_demo_record(mitdb: Path, record: str, max_beats: int) -> dict:
    import wfdb

    half = WINDOW // 2
    path = mitdb / record
    rec = wfdb.rdrecord(str(path))
    ann = wfdb.rdann(str(path), "atr")
    signal = zscore(np.asarray(rec.p_signal[:, 0], dtype=np.float32))

    beats = []
    for beat_idx, (sample_idx, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        label = AAMI_MAP.get(symbol)
        if label is None or label not in CLASS_NAMES:
            continue
        start = int(sample_idx) - half
        end = int(sample_idx) + half
        if start < 0 or end > len(signal):
            continue
        wf = signal[start:end].tolist()
        beats.append({
            "beat_index": beat_idx,
            "sample_index": int(sample_idx),
            "label": label,
            "start": start,
            "end": end,
            "waveform": wf,
        })
        if max_beats > 0 and len(beats) >= max_beats:
            break

    return {
        "source": "MIT-BIH MambaResU",
        "record": record,
        "fs": FS,
        "window_size": WINDOW,
        "num_calibration_beats": 0,
        "class_names": CLASS_NAMES,
        "protocol": "beat_centered_zscore_single_input",
        "beats": beats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitdb", type=Path, default=DEFAULT_MITDB)
    parser.add_argument("--record", default="100")
    parser.add_argument("--max-beats", type=int, default=60)
    parser.add_argument("--skip-pte", action="store_true")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)

    if not args.mitdb.is_dir():
        print(f"MIT-BIH not found: {args.mitdb}")
        return 1

    info = export_demo_record(args.mitdb, args.record, args.max_beats)
    out_json = ASSETS / "demo_record.json"
    out_json.write_text(json.dumps(info), encoding="utf-8")
    print(f"demo_record.json: {len(info['beats'])} beats from {args.record}")

    if not args.skip_pte:
        if not PTE_SRC.is_file():
            print(f"PTE missing: {PTE_SRC}")
            return 1
        out_pte = ASSETS / "demo_model.pte"
        shutil.copy2(PTE_SRC, out_pte)
        print(f"demo_model.pte: copied ({out_pte.stat().st_size // 1024} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
