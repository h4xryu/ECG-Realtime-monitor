"""Lightweight dense MAC counter (evaluator.MACs 호환 subset)."""

from __future__ import annotations

import torch
import torch.nn as nn

_HANDLERS: dict[type, object] = {}


def _reg(layer_type):
    def deco(fn):
        _HANDLERS[layer_type] = fn
        return fn
    return deco


@_reg(nn.Conv1d)
def _conv1d(m: nn.Conv1d, _inp, out) -> int:
    return (
        m.kernel_size[0]
        * (m.in_channels // m.groups)
        * m.out_channels
        * int(out.shape[-1])
    )


@_reg(nn.Conv2d)
def _conv2d(m: nn.Conv2d, _inp, out) -> int:
    kH, kW = m.kernel_size
    return (
        kH * kW
        * (m.in_channels // m.groups)
        * m.out_channels
        * int(out.shape[-2])
        * int(out.shape[-1])
    )


@_reg(nn.Linear)
def _linear(m: nn.Linear, _inp, out) -> int:
    return int(m.in_features * out.numel())


@_reg(nn.ConvTranspose1d)
def _conv_t1d(m: nn.ConvTranspose1d, _inp, out) -> int:
    return (
        m.kernel_size[0]
        * m.in_channels
        * (m.out_channels // m.groups)
        * int(out.shape[-1])
    )


@_reg(nn.BatchNorm1d)
def _bn1d(_m, _inp, _out) -> int:
    return 0


@_reg(nn.LayerNorm)
def _ln(_m, _inp, out) -> int:
    return int(out.numel())


def count_macs(model: nn.Module, sample: tuple[torch.Tensor, ...] | torch.Tensor) -> int:
    total = [0]
    handles = []

    def make_hook(fn):
        def hook(module, inp, out):
            total[0] += int(fn(module, inp, out))
        return hook

    for m in model.modules():
        for layer_type, fn in _HANDLERS.items():
            if type(m) is layer_type:
                handles.append(m.register_forward_hook(make_hook(fn)))
                break

    model.eval()
    with torch.no_grad():
        if isinstance(sample, tuple):
            model(*sample)
        else:
            model(sample)

    for h in handles:
        h.remove()
    return total[0]


def count_flops(model: nn.Module, sample: tuple[torch.Tensor, ...] | torch.Tensor) -> int:
    return 2 * count_macs(model, sample)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_mib(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def activation_mib(model: nn.Module, sample: tuple[torch.Tensor, ...] | torch.Tensor) -> float:
    """Forward 출력 텐서 합 (MiB) — CPU activation 하한."""
    acc = [0.0]
    handles = []

    def hook(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            acc[0] += out.numel() * out.element_size() / (1024 ** 2)
        elif isinstance(out, tuple):
            for t in out:
                if isinstance(t, torch.Tensor):
                    acc[0] += t.numel() * t.element_size() / (1024 ** 2)

    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        if isinstance(sample, tuple):
            model(*sample)
        else:
            model(sample)

    for h in handles:
        h.remove()
    return acc[0]
