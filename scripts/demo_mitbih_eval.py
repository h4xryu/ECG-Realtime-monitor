#!/usr/bin/env python3
"""
MIT-BIH DS2 소규모 평가 — model/baseline_best.ckpt (MambaResU)

Usage:
  py -3.13 scripts/demo_mitbih_eval.py
  py -3.13 scripts/demo_mitbih_eval.py --records 100 103 200 --max-beats 200
  py -3.13 scripts/demo_mitbih_eval.py --variant dsconv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from dsconv_approx import (
    DSConv1d,
    conv_flops_per_sample,
    conv_output_mse,
    conv_params,
    diagonal_depthwise_from_conv,
    dwsep_flops_per_sample,
    dwsep_params,
    reduction_ratio,
    replace_conv_with_dsconv,
    replace_conv_with_svd_dwsep,
)
from mamba_model_utils import load_mamba_resu

WINDOW = 720
FS = 360
CLASS_NAMES_5 = ["N", "S", "V", "F", "Q"]
CLASS_NAMES_4 = ["N", "S", "V", "F"]

DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
]

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

DEFAULT_MITDB = Path(r"D:\workspace\Classification_workflow\mit-bih-arrhythmia-database-1.0.0")


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)).astype(np.float32)


def map_to_demo4(label5: str) -> str:
    return label5 if label5 in CLASS_NAMES_4 else "N"


def load_beats(mitdb: Path, record: str) -> list[dict]:
    import wfdb

    half = WINDOW // 2
    path = mitdb / record
    rec = wfdb.rdrecord(str(path))
    ann = wfdb.rdann(str(path), "atr")
    signal = zscore(np.asarray(rec.p_signal[:, 0], dtype=np.float32))

    beats: list[dict] = []
    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        label = AAMI_MAP.get(symbol)
        if label is None:
            continue
        start = int(sample_idx) - half
        end = int(sample_idx) + half
        if start < 0 or end > len(signal):
            continue
        beats.append({
            "record": record,
            "label5": label,
            "label4": map_to_demo4(label),
            "waveform": signal[start:end],
        })
    return beats


def macro_f1(y_true: list[str], y_pred: list[str], names: list[str]) -> float:
    f1s: list[float] = []
    for cls in names:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def apply_variant(model: nn.Module, variant: str, svd_ranks: int = 1) -> dict:
    rub = model.rub
    conv = rub.conv1
    c = conv.in_channels
    k = conv.kernel_size[0]
    seq_len = 180  # rub.conv1 직후 길이 (720 → pool 전)
    info: dict = {
        "variant": variant,
        "conv_channels": c,
        "conv_kernel": k,
        "conv_flops_per_sample": conv_flops_per_sample(conv, seq_len),
        "conv_params": conv_params(conv),
    }

    if variant == "baseline":
        return info

    if variant == "dw_only":
        dw = diagonal_depthwise_from_conv(conv)
        rub.conv1 = dw
        info["conv_mse"] = conv_output_mse(conv, dw)
        return info

    if variant == "dsconv":
        dw, pw = replace_conv_with_dsconv(conv)
        ds = DSConv1d(dw, pw)
        info["conv_mse"] = conv_output_mse(conv, ds)
        rub.conv1 = ds
        return info

    if variant in ("svd_dwsep", "dwsep"):
        ds, svd_meta = replace_conv_with_svd_dwsep(conv, ranks=svd_ranks)
        info["conv_mse"] = conv_output_mse(conv, ds)
        info.update(svd_meta)
        info["dwsep_flops_per_sample"] = dwsep_flops_per_sample(c, k, seq_len) * svd_ranks
        info["dwsep_params"] = dwsep_params(c, k) * svd_ranks
        info["flops_reduction"] = 1.0 - (dwsep_flops_per_sample(c, k, seq_len) * svd_ranks) / info["conv_flops_per_sample"]
        info["params_reduction"] = 1.0 - (dwsep_params(c, k) * svd_ranks) / info["conv_params"]
        rub.conv1 = ds
        return info

    raise ValueError(f"unknown variant: {variant}")


@torch.no_grad()
def predict(model: nn.Module, waveforms: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    preds: list[int] = []
    for i in range(0, len(waveforms), batch_size):
        batch = waveforms[i : i + batch_size]
        x = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1)
        logits = model(x)[0]
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(preds, dtype=np.int64)


def evaluate_preds(y_true: list[str], y_pred_idx: np.ndarray, names: list[str]) -> dict:
    assert len(y_true) == len(y_pred_idx)
    y_pred = [names[i] for i in y_pred_idx]
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    per_class: dict[str, dict] = {}
    for cls in names:
        idxs = [i for i, t in enumerate(y_true) if t == cls]
        if not idxs:
            continue
        hit = sum(1 for i in idxs if y_pred[i] == cls)
        per_class[cls] = {
            "support": len(idxs),
            "recall": hit / len(idxs),
            "correct": hit,
        }
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "confusion": {
            t: dict(Counter(y_pred[i] for i, yt in enumerate(y_true) if yt == t))
            for t in names
        },
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MIT-BIH DS2 quick eval for MambaResU demo")
    parser.add_argument("--mitdb", type=Path, default=DEFAULT_MITDB)
    parser.add_argument("--records", nargs="+", default=["100", "103", "200", "202", "213"])
    parser.add_argument("--max-beats", type=int, default=0, help="0 = all beats per record")
    parser.add_argument(
        "--variant",
        choices=["baseline", "dw_only", "dsconv", "svd_dwsep", "all", "compare"],
        default="compare",
        help="compare=baseline+svd_dwsep (권장), all=구버전 포함 전체",
    )
    parser.add_argument("--svd-ranks", type=int, default=1, help="svd_dwsep 변형 시 rank 수 (1=제시한 공식)")
    parser.add_argument("--out", type=Path, default=ROOT / "results_dwsep_decompose.json")
    args = parser.parse_args()

    if not args.mitdb.is_dir():
        print(f"MIT-BIH not found: {args.mitdb}")
        return 1

    beats: list[dict] = []
    for rec in args.records:
        hea = args.mitdb / f"{rec}.hea"
        if not hea.exists():
            print(f"skip missing record {rec}")
            continue
        rec_beats = load_beats(args.mitdb, rec)
        if args.max_beats > 0:
            rec_beats = rec_beats[: args.max_beats]
        beats.extend(rec_beats)
        print(f"  {rec}: {len(rec_beats)} beats")

    if not beats:
        print("no beats loaded")
        return 1

    waves = [b["waveform"] for b in beats]
    y4 = [b["label4"] for b in beats]
    y5 = [b["label5"] for b in beats]
    print(f"total beats: {len(beats)}  dist4={dict(Counter(y4))}")

    if args.variant == "all":
        variants = ["baseline", "dw_only", "dsconv", "svd_dwsep"]
    elif args.variant == "compare":
        variants = ["baseline", "svd_dwsep"]
    else:
        variants = [args.variant]
    report: dict = {
        "mitdb": str(args.mitdb),
        "records": args.records,
        "n_beats": len(beats),
        "label_dist_4": dict(Counter(y4)),
        "variants": {},
    }

    for variant in variants:
        model = load_mamba_resu()
        orig_conv = model.rub.conv1
        meta = apply_variant(model, variant, svd_ranks=args.svd_ranks)
        if variant != "baseline":
            print(f"\n[{variant}] rub.conv1 conv-output MSE vs original: {meta.get('conv_mse', 0):.6f}")

        pred5 = predict(model, waves)
        pred4 = np.array([CLASS_NAMES_4.index(map_to_demo4(CLASS_NAMES_5[i])) for i in pred5])

        res5 = evaluate_preds(y5, pred5, CLASS_NAMES_5)
        res4 = evaluate_preds(y4, pred4, CLASS_NAMES_4)
        pred4_names = [CLASS_NAMES_4[i] for i in pred4]
        mf1 = macro_f1(y4, pred4_names, CLASS_NAMES_4)
        report["variants"][variant] = {
            **meta,
            "acc5": res5["accuracy"],
            "acc4": res4["accuracy"],
            "macro_f1_4": mf1,
            "per_class4": res4["per_class"],
            "confusion4": res4["confusion"],
        }
        extra = ""
        if "conv_mse" in meta:
            extra += f"  conv_mse={meta['conv_mse']:.6f}"
        if "flops_reduction" in meta:
            extra += f"  flops↓={meta['flops_reduction']*100:.1f}%"
        if "mean_sigma1_energy_ratio" in meta:
            extra += f"  σ₁에너지={meta['mean_sigma1_energy_ratio']*100:.1f}%"
        print(
            f"[{variant}] acc4={res4['accuracy']*100:.2f}%  "
            f"macro-F1={mf1:.4f}  "
            f"(N={res4['per_class'].get('N',{}).get('recall',0):.2f} "
            f"S={res4['per_class'].get('S',{}).get('recall',0):.2f} "
            f"V={res4['per_class'].get('V',{}).get('recall',0):.2f} "
            f"F={res4['per_class'].get('F',{}).get('recall',0):.2f})"
            f"{extra}"
        )
        del model

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
