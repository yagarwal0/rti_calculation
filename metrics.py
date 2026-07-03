"""
metrics.py — clean and robust accuracy metrics.

Computes, for a (pruned / trained) model:
    * Clean accuracy   (top-1; optional top-5)
    * Robust accuracy  under FGSM, PGD-10, PGD-20, PGD-100, AutoAttack
    * Attack success rate per attack
    * Worst-case robust accuracy (AutoAttack — the certified-strength lower bound)
    * Prunable sparsity of the model

The attack objects come from `attacks.py` (torchattacks); this module is the
reporting layer that ties everything into one structured result + table.

Examples
--------
    # Full metrics for a pruned/trained checkpoint
    python metrics.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --n-samples 1000

    # Programmatic use
    from metrics import compute_metrics
    m = compute_metrics(wrapped_model, loader, device, num_classes=10)
"""

import os
import sys
import json
import argparse
import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from utils import prunable_sparsity, set_seed, parse_args_with_config
# Reuse the attack suite + loaders + the core accuracy loops (no duplication).
from attacks import (NormalizedModel, get_clean_test_loader, build_attacks,
                     clean_accuracy, robust_accuracy, EPS, ALPHA, AA_EPS,
                     NUM_CLASSES)

# Canonical attack ordering for reports.
ATTACK_ORDER = ['FGSM', 'PGD-10', 'PGD-20', 'PGD-100', 'AutoAttack']


@torch.no_grad()
def top_k_accuracy(model, loader, device, k=5):
    """Top-k clean accuracy (%)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        topk = model(x).topk(k, dim=1).indices
        correct += topk.eq(y.view(-1, 1)).any(dim=1).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def _asr(clean, robust):
    """Attack success rate (%) = drop from clean to robust accuracy."""
    return round(100.0 * (clean - robust) / clean, 4) if clean > 0 else 0.0


def compute_metrics(model, loader, device, num_classes, eps=EPS, alpha=ALPHA,
                    pgd_steps=(10, 20, 100), include_autoattack=True,
                    aa_version='standard', report_top5=False, verbose=True, seed=0,
                    aa_eps=AA_EPS):
    """Compute clean + robust accuracy for every attack.

    `model` must accept [0, 1] inputs (i.e. a NormalizedModel). Returns a dict:
        {
          'clean_accuracy': float,
          'clean_top5_accuracy': float (optional),
          'sparsity': float (optional),
          'attacks': {name: {'robust_accuracy': float, 'attack_success_rate': float}},
          'worst_case_robust_accuracy': float,
          'threat_model': {...},
        }
    """
    model = model.to(device).eval()

    clean = clean_accuracy(model, loader, device)
    metrics = {'clean_accuracy': round(clean, 4), 'attacks': {}}
    if verbose:
        print(f"  {'clean':<12}: {clean:6.2f}%")

    if report_top5 and num_classes >= 5:
        top5 = top_k_accuracy(model, loader, device, k=5)
        metrics['clean_top5_accuracy'] = round(top5, 4)
        if verbose:
            print(f"  {'clean top-5':<12}: {top5:6.2f}%")

    attacks = build_attacks(model, num_classes, eps, alpha, pgd_steps,
                            include_autoattack, aa_version, seed=seed,
                            aa_eps=aa_eps)
    robusts = []
    for name, atk in attacks.items():
        racc = robust_accuracy(model, loader, atk, device)
        metrics['attacks'][name] = {'robust_accuracy': round(racc, 4),
                                    'attack_success_rate': _asr(clean, racc)}
        robusts.append((name, racc))
        if verbose:
            print(f"  {name:<12}: {racc:6.2f}%   (ASR {_asr(clean, racc):6.2f}%)")

    # Worst-case robustness: AutoAttack if present, else the minimum over attacks.
    if 'AutoAttack' in metrics['attacks']:
        metrics['worst_case_robust_accuracy'] = metrics['attacks']['AutoAttack']['robust_accuracy']
    elif robusts:
        metrics['worst_case_robust_accuracy'] = round(min(r for _, r in robusts), 4)

    metrics['threat_model'] = {
        'norm': 'Linf', 'eps': eps, 'eps_over_255': round(eps * 255, 4),
        'alpha': alpha, 'pgd_steps': list(pgd_steps),
        'autoattack': aa_version if include_autoattack else None,
        'aa_eps': aa_eps if include_autoattack else None,
        'aa_eps_over_255': round(aa_eps * 255, 4) if include_autoattack else None,
    }
    return metrics


def format_table(metrics, title=None):
    """Render a metrics dict as a readable text table."""
    lines = []
    if title:
        lines.append(title)
    lines.append("-" * 46)
    lines.append(f"  {'metric':<26}{'accuracy':>10}")
    lines.append("-" * 46)
    lines.append(f"  {'Clean':<26}{metrics['clean_accuracy']:>9.2f}%")
    if 'clean_top5_accuracy' in metrics:
        lines.append(f"  {'Clean (top-5)':<26}{metrics['clean_top5_accuracy']:>9.2f}%")
    # attacks in canonical order, then any extras
    names = [n for n in ATTACK_ORDER if n in metrics['attacks']]
    names += [n for n in metrics['attacks'] if n not in names]
    for n in names:
        racc = metrics['attacks'][n]['robust_accuracy']
        lines.append(f"  {'Robust: ' + n:<26}{racc:>9.2f}%")
    if 'worst_case_robust_accuracy' in metrics:
        lines.append("-" * 46)
        lines.append(f"  {'Worst-case (AutoAttack)':<26}"
                     f"{metrics['worst_case_robust_accuracy']:>9.2f}%")
    lines.append("-" * 46)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_model(args):
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        dataset = ckpt.get('dataset', args.dataset)
        model_name = ckpt.get('model', args.model)
        num_classes = ckpt.get('num_classes', NUM_CLASSES[dataset])
        model = build_model(model_name, num_classes)
        model.load_state_dict(ckpt['state_dict'])
        meta = {k: ckpt.get(k) for k in
                ('pruner', 'target_sparsity', 'achieved_sparsity', 'test_acc')}
        return model, dataset, model_name, num_classes, meta

    dataset = args.dataset
    model_name = args.model
    num_classes = NUM_CLASSES[dataset]
    model = build_model(model_name, num_classes)
    if args.weights:
        sd = torch.load(args.weights, map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd)
    return model, dataset, model_name, num_classes, {}


def main():
    p = argparse.ArgumentParser(description="Compute clean + robust accuracy metrics.")
    p.add_argument('--checkpoint', default=None, help='checkpoint from main.py / train.py')
    p.add_argument('--model', default='resnet20')
    p.add_argument('--weights', default=None, help='bare state_dict to load into --model')
    p.add_argument('--dataset', default='cifar10', choices=list(NUM_CLASSES))
    p.add_argument('--eps', type=float, default=EPS, help='L_inf budget for FGSM/PGD (default 8/255)')
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument('--aa-eps', type=float, default=AA_EPS,
                   help='L_inf budget for AutoAttack (default 4/255)')
    p.add_argument('--pgd-steps', nargs='+', type=int, default=[10, 20, 100])
    p.add_argument('--no-autoattack', action='store_true')
    p.add_argument('--aa-version', default='standard', choices=['standard', 'plus', 'rand'])
    p.add_argument('--top5', action='store_true', help='also report clean top-5 accuracy')
    p.add_argument('--n-samples', type=int, default=1000, help='test images (0 = full set)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--tinyimagenet-dir', default='./tiny-imagenet-200')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--results-file', default=None, help='optional JSON path to append metrics')
    p.add_argument('--config', default=None, help='YAML/JSON config file (CLI args override it)')
    args = parse_args_with_config(p)

    set_seed(args.seed)
    device = torch.device(args.device)
    model, dataset, model_name, num_classes, meta = _load_model(args)
    sparsity = prunable_sparsity(model)

    loader, mean, std, num_classes = get_clean_test_loader(
        dataset, batch_size=args.batch_size, n_samples=args.n_samples,
        num_workers=args.num_workers, data_root=args.data_root,
        tinyimagenet_dir=args.tinyimagenet_dir)
    wrapped = NormalizedModel(model, mean, std).to(device).eval()

    print("=" * 60)
    print(f"METRICS | {model_name} | {dataset} | sparsity {sparsity*100:.2f}% | "
          f"n={args.n_samples or 'full'}")
    print("=" * 60)

    metrics = compute_metrics(
        wrapped, loader, device, num_classes,
        eps=args.eps, alpha=args.alpha, pgd_steps=tuple(args.pgd_steps),
        include_autoattack=not args.no_autoattack, aa_version=args.aa_version,
        report_top5=args.top5, seed=args.seed, aa_eps=args.aa_eps)
    metrics['sparsity'] = round(sparsity, 6)

    print("\n" + format_table(metrics, title=f"{model_name} / {dataset}"))

    if args.results_file:
        payload = {'model': model_name, 'dataset': dataset, 'sparsity': round(sparsity, 6),
                   'n_samples': args.n_samples, 'checkpoint': args.checkpoint,
                   'checkpoint_meta': meta,
                   'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                   **metrics}
        existing = []
        if os.path.isfile(args.results_file):
            try:
                with open(args.results_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(payload)
        os.makedirs(os.path.dirname(args.results_file) or '.', exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nMetrics appended to {args.results_file}")


if __name__ == '__main__':
    main()
