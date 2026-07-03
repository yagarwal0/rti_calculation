"""
prune_utils.py — shared helpers used by main.py to make every pruning
algorithm interchangeable.

Different algorithms return masks in different forms; these helpers normalise
everything to a single convention:

    A "mask list" is one binary tensor per prunable layer, ordered exactly as
    `prunable_modules(model)` yields them — i.e. nn.Conv2d / nn.Linear in
    module-iteration order. mask == 1 keeps the weight, 0 prunes it.

This lets us compute masks on a model at initialisation, then apply them to a
fresh copy with the same initialisation, regardless of which algorithm produced
them.
"""

import torch
import torch.nn as nn


def prunable_modules(model):
    """Yield the prunable layers (Conv2d / Linear) in module-iteration order.

    Note: the masked-layer subclasses used by SynFlow / YOPO subclass nn.Conv2d
    and nn.Linear, so they are matched here too — ordering is preserved.
    """
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def last_linear(model):
    """Return the final nn.Linear module (the classifier head)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return None
    return linears[-1]


def prunable_sparsity(model):
    """Fraction of zero weights among prunable (Conv2d / Linear) layers."""
    zeros, total = 0, 0
    for m in prunable_modules(model):
        w = m.weight.data
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / total if total else 0.0


def apply_masks_aligned(model, masks):
    """Zero out weights according to `masks` (a mask list aligned to
    prunable_modules(model)). Modifies `model` in place."""
    mods = prunable_modules(model)
    if len(masks) != len(mods):
        raise ValueError(
            f"mask/layer count mismatch: {len(masks)} masks vs {len(mods)} prunable layers")
    for m, mk in zip(mods, masks):
        mk = mk.to(m.weight.device).reshape(m.weight.shape).float()
        m.weight.data.mul_(mk)
    return model


def masks_from_weights(model):
    """Recover a mask list from a model whose weights have already been zeroed
    (used for algorithms that prune by zeroing weights directly, e.g. YOPO)."""
    return [(m.weight.data != 0).float() for m in prunable_modules(model)]


def transfer_mask(source_masks, target_model):
    """Transfer a mask list computed on one model onto `target_model`'s prunable
    layers, by position. Layers whose weight shape matches the source mask keep
    the (transferred) mask; layers that don't match — typically the classifier
    head when the source and target datasets have a different number of classes —
    stay dense (all-ones).

    Returns a mask list aligned to prunable_modules(target_model). The two models
    must have the same prunable-layer count (same architecture, possibly differing
    only in the classifier width).
    """
    tmods = prunable_modules(target_model)
    if len(source_masks) != len(tmods):
        raise ValueError(
            f"layer-count mismatch: {len(source_masks)} source masks vs "
            f"{len(tmods)} target prunable layers (need the same architecture)")
    out = []
    for mk, mod in zip(source_masks, tmods):
        if tuple(mk.shape) == tuple(mod.weight.shape):
            out.append(mk.detach().clone())
        else:                                   # e.g. classifier with a new #classes
            out.append(torch.ones_like(mod.weight).detach().cpu())
    return out


def to_synflow_model(model):
    """Recursively replace standard nn.Conv2d / nn.Linear / nn.BatchNorm2d
    layers with the SynFlow masked-layer equivalents (which carry a
    `weight_mask` buffer), copying weights/stats across. Returns `model`
    (modified in place).

    SynFlow scores via these mask buffers, so a stock model must be converted
    before `synflow_prune_100` can prune it.
    """
    from pruning import synflow as sfl  # lazy import (needs numpy, tqdm)

    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d) and not isinstance(child, sfl.Conv2d):
            new = sfl.Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                             stride=child.stride, padding=child.padding,
                             dilation=child.dilation, groups=child.groups,
                             bias=child.bias is not None, padding_mode=child.padding_mode)
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data)
            setattr(model, name, new)

        elif isinstance(child, nn.Linear) and not isinstance(child, sfl.Linear):
            new = sfl.Linear(child.in_features, child.out_features,
                             bias=child.bias is not None)
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data)
            setattr(model, name, new)

        elif isinstance(child, nn.BatchNorm2d) and not isinstance(child, sfl.BatchNorm2d):
            new = sfl.BatchNorm2d(child.num_features, eps=child.eps,
                                  momentum=child.momentum, affine=child.affine,
                                  track_running_stats=child.track_running_stats)
            if child.affine:
                new.weight.data.copy_(child.weight.data)
                new.bias.data.copy_(child.bias.data)
            if child.track_running_stats:
                new.running_mean.copy_(child.running_mean)
                new.running_var.copy_(child.running_var)
                new.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(model, name, new)

        else:
            to_synflow_model(child)  # recurse into containers / blocks

    return model
