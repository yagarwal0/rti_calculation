"""
main.py — prune every model with every pruning-at-initialization algorithm.

Run:
    python main.py                                   # all models x all pruners, defaults
    python main.py --dataset cifar100
    python main.py --models resnet20 vgg19 --pruners snip synflow --sparsity 0.9 0.95
    python main.py --dataset tinyimagenet --tinyimagenet-dir ./tiny-imagenet-200

For each (model, pruner, sparsity) it:
    1. builds the model at a fixed random initialisation,
    2. computes a pruning mask with the chosen algorithm,
    3. applies that mask to a fresh copy at the SAME initialisation,
    4. reports achieved sparsity and saves the pruned model + mask to --output-dir.

Algorithms (in pruning/):
    snip     SNIP            (data:  one batch of labelled data)
    grasp    GraSP           (data:  a few samples per class)
    synflow  SynFlow         (data-free; 100-step iterative schedule)
    dpai     DPaI / NPB      (data-free; effective-path optimisation)
    yopo     YOPO (NMF)      (data-free; NMF reconstruction-error scoring)

Convention: `sparsity` = fraction of prunable weights REMOVED (0.9 => keep 10%).
"""

import os
import sys
import argparse
import copy
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sibling packages importable when run as `python main.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model, MODELS
from dataloader import get_loaders
from utils import (prunable_modules, prunable_sparsity, apply_masks_aligned,
                   masks_from_weights, to_synflow_model, last_linear,
                   set_seed, parse_args_with_config)

ALL_PRUNERS = ['snip', 'grasp', 'synflow', 'dpai', 'yopo']


def _wrap_norm(model, args, device):
    """Wrap a model so it consumes [0,1] inputs and normalizes INSIDE the forward
    pass (matching the FGSM/PGD/AutoAttack convention). Used for the data-driven
    pruners (SNIP/GraSP) so their saliency is scored on the same inputs the model
    sees during training/eval. NormalizedModel reuses the wrapped model's layer
    objects, so masks stay aligned to `prunable_modules(model)`."""
    from attacks import NormalizedModel, NORM_STATS
    mean, std = NORM_STATS[args.dataset]
    return NormalizedModel(model, mean, std).to(device)


# ===========================================================================
# Per-algorithm mask computation.
# Each returns a "mask list" aligned to prunable_modules(base): one binary
# tensor per Conv2d/Linear layer, computed on `base` at its initialisation.
# ===========================================================================
def masks_snip(base, loader, device, sparsity, num_classes, args):
    from pruning.snip import SNIP
    keep_ratio = 1.0 - sparsity
    net = _wrap_norm(base, args, device)   # score on [0,1] + internal normalization
    return SNIP(net, keep_ratio, loader, device, loss_fn=F.cross_entropy)


def masks_grasp(base, loader, device, sparsity, num_classes, args):
    from pruning.grasp import GraSP
    spc = args.grasp_samples_per_class
    net = _wrap_norm(base, args, device)   # score on [0,1] + internal normalization
    keep = GraSP(net, sparsity, loader, device,
                 num_classes=num_classes, samples_per_class=spc)
    # keep is keyed by base's prunable module objects (NormalizedModel reuses them)
    return [keep[m] for m in prunable_modules(base)]


def masks_dpai(base, loader, device, sparsity, num_classes, args):
    from pruning.dpai import dpai_prune
    sample, _ = next(iter(loader))                 # infer input shape (no hardcoding)
    input_shape = (1,) + tuple(sample.shape[1:])
    masks = dpai_prune(base, sparsity, device, num_steps=args.dpai_steps,
                       input_shape=input_shape)
    return [mk.to(device) for mk in masks]


def masks_synflow(base, loader, device, sparsity, num_classes, args):
    from pruning.synflow import synflow_prune_100
    density = 1.0 - sparsity              # synflow's `sparsity` arg = fraction KEPT
    sf_model = to_synflow_model(base).to(device)
    try:
        synflow_prune_100(sf_model, loader, device, sparsity=density)
    except SystemExit:
        # prune.py calls quit() if it can't hit the target within ~5 params;
        # masks were already written before that check, so we proceed.
        print("  [synflow] note: sparsity tolerance check tripped; using masks as-is.")
    masks = []
    for m in sf_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            masks.append(m.weight_mask.detach().clone())
    return masks


def masks_yopo(base, loader, device, sparsity, num_classes, args):
    from pruning import yopo
    head = last_linear(base)
    masked = yopo.convert_to_masked(base, head).to(device)
    score_cache = yopo.compute_score(masked)
    sf = yopo.find_optimal_std_factor_global(
        masked, score_cache, sparsity_goal=sparsity * 100.0,
        threshold_type='mad', num_steps=args.yopo_search_steps)
    yopo.apply_nmf_masks_global(masked, score_cache, 'mad', std_factor=sf)
    # apply_nmf_masks_global zeroes the weights directly -> read masks back off them
    return masks_from_weights(masked)


MASK_FUNCS = {
    'snip': masks_snip,
    'grasp': masks_grasp,
    'synflow': masks_synflow,
    'dpai': masks_dpai,
    'yopo': masks_yopo,
}


# ===========================================================================
# Drivers
# ===========================================================================
def prune_one(model_name, pruner, sparsity, num_classes, loader, device, args):
    """Build -> prune -> apply -> measure. Returns (pruned_model, mask_list, achieved)."""
    set_seed(args.seed, deterministic=getattr(args, 'deterministic', False))
    base = build_model(model_name, num_classes).to(device)
    init_state = copy.deepcopy(base.state_dict())

    masks = MASK_FUNCS[pruner](base, loader, device, sparsity, num_classes, args)

    # Apply the computed masks to a fresh copy at the identical initialisation.
    pruned = build_model(model_name, num_classes).to(device)
    pruned.load_state_dict(init_state)
    apply_masks_aligned(pruned, masks)

    achieved = prunable_sparsity(pruned)
    cpu_masks = [mk.detach().cpu() for mk in masks]
    return pruned, cpu_masks, achieved


def main():
    p = argparse.ArgumentParser(description="Prune all models with all pruning algorithms.")
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'tinyimagenet'])
    p.add_argument('--models', nargs='+', default=list(MODELS),
                   choices=list(MODELS), help='models to prune (default: all)')
    p.add_argument('--pruners', nargs='+', default=ALL_PRUNERS,
                   choices=ALL_PRUNERS, help='algorithms to run (default: all)')
    p.add_argument('--sparsity', nargs='+', type=float, default=[0.9],
                   help='fraction(s) of weights to remove (default: 0.9)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', default='./data', help='CIFAR download/cache dir')
    p.add_argument('--tinyimagenet-dir', default='./tiny-imagenet-200')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--deterministic', action='store_true',
                   help='force deterministic cuDNN/algorithms (slower, reproducible)')
    p.add_argument('--output-dir', default='./output')
    # algorithm-specific knobs
    p.add_argument('--dpai-steps', type=int, default=100,
                   help='DPaI score-optimisation steps (default: 100)')
    p.add_argument('--grasp-samples-per-class', type=int, default=10)
    p.add_argument('--yopo-search-steps', type=int, default=30,
                   help='std-factor search resolution for YOPO (default: 30)')
    p.add_argument('--no-save', action='store_true', help='do not write pruned checkpoints')
    p.add_argument('--config', default=None, help='YAML/JSON config file (CLI args override it)')
    args = parse_args_with_config(p)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    print(f"Device: {device} | seed: {args.seed}"
          + (" | deterministic" if args.deterministic else ""))

    # Data (used for scoring by SNIP/GraSP; SynFlow reads only the input shape;
    # DPaI/YOPO are data-free but receive the loader for a uniform interface).
    train_loader, test_loader, num_classes = get_loaders(
        args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, tinyimagenet_dir=args.tinyimagenet_dir)
    print(f"Dataset: {args.dataset}  (num_classes={num_classes})")

    os.makedirs(args.output_dir, exist_ok=True)
    out_ds_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_ds_dir, exist_ok=True)

    results = []  # (model, pruner, target, achieved, status)
    for model_name in args.models:
        for pruner in args.pruners:
            for sparsity in args.sparsity:
                tag = f"{model_name} | {pruner} | target sparsity {sparsity:.4g}"
                print("\n" + "=" * 78)
                print(f"PRUNING  {tag}")
                print("=" * 78)
                try:
                    pruned, cpu_masks, achieved = prune_one(
                        model_name, pruner, sparsity, num_classes,
                        train_loader, device, args)
                    print(f"--> achieved prunable sparsity: {achieved * 100:.2f}%")
                    results.append((model_name, pruner, sparsity, achieved, "ok"))

                    if not args.no_save:
                        fname = f"{model_name}_{pruner}_s{sparsity:.4g}.pth".replace('.', '_', 1)
                        fpath = os.path.join(out_ds_dir, fname)
                        torch.save({
                            'state_dict': pruned.state_dict(),
                            'masks': cpu_masks,
                            'model': model_name,
                            'pruner': pruner,
                            'dataset': args.dataset,
                            'num_classes': num_classes,
                            'target_sparsity': sparsity,
                            'achieved_sparsity': achieved,
                            'seed': args.seed,
                        }, fpath)
                        print(f"--> saved: {fpath}")
                except Exception as e:  # keep the sweep going on any single failure
                    print(f"!! FAILED: {tag}\n{e}")
                    traceback.print_exc()
                    results.append((model_name, pruner, sparsity, float('nan'), f"FAIL: {e}"))

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'model':<10}{'pruner':<10}{'target':<10}{'achieved':<12}{'status'}")
    for model_name, pruner, target, achieved, status in results:
        ach = "-" if achieved != achieved else f"{achieved * 100:.2f}%"   # NaN check
        print(f"{model_name:<10}{pruner:<10}{target:<10.4g}{ach:<12}{status}")


if __name__ == '__main__':
    main()
