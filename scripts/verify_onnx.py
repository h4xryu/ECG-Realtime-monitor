#!/usr/bin/env python3
"""PyTorch vs ONNX 동일 입력 비교"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

pecg = Path(r"G:/내 드라이브/PECG")
root = Path(__file__).resolve().parent
assets = root.parent / "app/src/main/assets"
os.chdir(pecg)
sys.path.insert(0, str(pecg / "src"))
sys.path.insert(0, str(pecg))
sys.path.insert(0, str(root))
from pecg_legacy_model import patch_pecg_for_legacy_ckpt

patch_pecg_for_legacy_ckpt()
from src.lightning.arrhythmia_module import ArrhythmiaLightningModule

demo = json.load(open(assets / "demo_record.json"))
lit = ArrhythmiaLightningModule.load_from_checkpoint(
    str(pecg / "results/paper_full/best.ckpt"),
    map_location="cpu",
    strict=False,
    patient_encoder_checkpoint=None,
)
lit.eval()
cal = torch.tensor(demo["calibration_beats"], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
sess = ort.InferenceSession(str(assets / "demo_model.onnx"), providers=["CPUExecutionProvider"])
names = ["N", "S", "V", "F", "Q"]

for i in [0, 10, 20, 30]:
    b = demo["beats"][i]
    x = torch.tensor(b["waveform"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pt = lit(x, cal, None).numpy()[0]
    ox = sess.run(None, {"x": x.numpy(), "calib": cal.numpy()})[0][0]
    print(f"beat{i} gt={b['label']} pt={names[pt.argmax()]} onnx={names[ox.argmax()]} match={np.allclose(pt, ox, atol=1e-2)}")
