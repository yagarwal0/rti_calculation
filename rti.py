"""
rti.py — Robust Transfer Index (RTI) and Normalized RTI (nRTI).

For a model / pruning configuration `m` evaluated on a source dataset D1 and a
target dataset D2 (transferable-pruning setting), with robust accuracies
A_rob(m, D1) and A_rob(m, D2) expressed as *fractions* in [0, 1]:

    RTI(m)  = | A_rob(m, D1) - A_rob(m, D2) | * 100                            (absolute gap, %)
    nRTI(m) = | A_rob(m, D1) - A_rob(m, D2) | * 100 / max(A_rob(m, D1), eps)   (relative gap, %)

Lower is better: the robustness of `m` is preserved when transferred D1 -> D2.
Both indices are scaled by 100 so they read as percentages (matching the thesis
Eq. 4.3). nRTI normalises by the *source* robustness so configurations with
different baseline robustness can be compared fairly. The denominator is guarded
with a small `eps` (the floor tau) so nRTI stays finite even when A_rob(m, D1)
is 0; `eps` defaults to 0.01 (i.e. the tau = 1% floor).

`A_rob` can be any attack's robust accuracy (FGSM / PGD-* / AutoAttack) or the
clean accuracy; the helpers below compute the indices per metric using the
result dicts produced by `metrics.compute_metrics`.

Examples
--------
    from rti import rti, nrti
    rti(0.72, 0.60)     # -> 12.0     (12 percentage-point gap)
    nrti(0.72, 0.60)    # -> 16.6667  (16.67% relative robustness drop)

    # From two metrics-JSON files written by metrics.py
    python rti.py --d1-metrics results/cifar10.json --d2-metrics results/cifar100.json

    # Or evaluate two checkpoints directly and compute the indices
    python rti.py --d1-checkpoint ckpt_cifar10.pth --d2-checkpoint ckpt_cifar100.pth \
                  --n-samples 1000
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default denominator floor (tau) for nRTI. Per the thesis Eq. 4.3 the
# denominator is max(A_rob(D1), tau) with tau = 0.01 (i.e. 1%); robust
# accuracies are fractions in [0, 1], so the floor is in the same units. This
# keeps nRTI finite (and bounded) when the source robust accuracy is ~0.
# Override with the `eps` argument / --nrti-eps.
EPS_DEFAULT = 0.01


# ---------------------------------------------------------------------------
# Core scalar definitions
# ---------------------------------------------------------------------------
def rti(a_rob_d1, a_rob_d2):
    """Robust Transfer Index: |A_rob(D1) - A_rob(D2)| * 100.

    `a_rob_d1` / `a_rob_d2` are robust accuracies as fractions in [0, 1]; the
    * 100 scaling makes RTI a percentage-point gap."""
    return abs(float(a_rob_d1) - float(a_rob_d2)) * 100.0


def nrti(a_rob_d1, a_rob_d2, eps=EPS_DEFAULT):
    """Normalized RTI: |A_rob(D1) - A_rob(D2)| * 100 / max(A_rob(D1), eps).

    `a_rob_d1` / `a_rob_d2` are robust accuracies as fractions in [0, 1]. The
    denominator is max(A_rob(D1), eps): `eps` (the floor tau) defaults to 0.01
    (1%) so the index stays finite (and bounded) even when the source robust
    accuracy A_rob(D1) is 0 (e.g. a model with no robustness)."""
    a1 = float(a_rob_d1)
    return abs(a1 - float(a_rob_d2)) * 100.0 / max(a1, eps)


# ---------------------------------------------------------------------------
# Index over full metrics dicts (per attack + clean)
# ---------------------------------------------------------------------------
def _robust_value(metrics, key):
    """Pull a robust/clean accuracy out of a metrics dict by key."""
    if key == 'clean':
        return metrics.get('clean_accuracy')
    return metrics.get('attacks', {}).get(key, {}).get('robust_accuracy')


def transfer_index(metrics_d1, metrics_d2, include_clean=True, nrti_eps=EPS_DEFAULT):
    """Compute RTI and nRTI for every metric shared by two metrics dicts.

    Each input is a dict as returned by `metrics.compute_metrics` (has
    'clean_accuracy' and 'attacks': {name: {'robust_accuracy': ...}}); those
    accuracies are stored as percentages and are converted to fractions before
    the indices are computed. `nrti_eps` is the denominator floor (tau) for nRTI
    as a fraction (default: 0.01, i.e. the 1% floor).

    Returns {metric_name: {'A_rob_D1', 'A_rob_D2', 'RTI', 'nRTI'}} with A_rob
    values in percent and RTI / nRTI scaled by 100 (per the thesis Eq. 4.3).
    """
    keys = []
    if include_clean:
        keys.append('clean')
    keys += list(metrics_d1.get('attacks', {}).keys())

    out = {}
    for key in keys:
        a1 = _robust_value(metrics_d1, key)
        a2 = _robust_value(metrics_d2, key)
        if a1 is None or a2 is None:
            continue
        # metrics store accuracies as percentages; the RTI / nRTI formulas take
        # fractions in [0, 1] (and re-scale by 100 internally).
        f1, f2 = float(a1) / 100.0, float(a2) / 100.0
        out[key] = {
            'A_rob_D1': round(float(a1), 4),
            'A_rob_D2': round(float(a2), 4),
            'RTI': round(rti(f1, f2), 4),
            # denominator = max(A_rob(D1), nrti_eps): finite even when A_rob(D1)=0
            'nRTI': round(nrti(f1, f2, eps=nrti_eps), 4),
        }
    return out


def format_table(indices, d1_name='D1', d2_name='D2', title=None):
    """Render a transfer_index() result as a text table."""
    cm, ca, cr, cn = 12, 20, 10, 12          # column widths
    width = 2 + cm + ca + ca + cr + cn
    lines = []
    if title:
        lines.append(title)
    lines.append("-" * width)
    lines.append(f"  {'metric':<{cm}}{f'A_rob({d1_name})':>{ca}}{f'A_rob({d2_name})':>{ca}}"
                 f"{'RTI':>{cr}}{'nRTI':>{cn}}")
    lines.append("-" * width)
    # canonical order: clean, FGSM, PGD-*, AutoAttack, then extras
    order = ['clean', 'FGSM', 'PGD-10', 'PGD-20', 'PGD-100', 'AutoAttack']
    names = [k for k in order if k in indices] + [k for k in indices if k not in order]
    for k in names:
        r = indices[k]
        a1 = f"{r['A_rob_D1']:.2f}%"
        a2 = f"{r['A_rob_D2']:.2f}%"
        rti_str = f"{r['RTI']:.2f}"
        nrti_str = f"{r['nRTI']:.2f}"
        lines.append(f"  {k:<{cm}}{a1:>{ca}}{a2:>{ca}}{rti_str:>{cr}}{nrti_str:>{cn}}")
    lines.append("-" * width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading metrics JSON written by metrics.py / attacks.py
# ---------------------------------------------------------------------------
def load_metrics_record(path, index=-1):
    """Load a metrics dict from a JSON file. metrics.py appends a list of
    records; `index` selects which one (default: the most recent)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"No records in {path}")
        return data[index]
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _metrics_from_checkpoint(ckpt_path, dataset, args):
    """Evaluate a checkpoint and return its metrics dict (needs torchattacks)."""
    import torch
    from models import build_model
    from attacks import NormalizedModel, get_clean_test_loader, NUM_CLASSES
    from metrics import compute_metrics

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ds = ckpt.get('dataset', dataset)
    model_name = ckpt.get('model', args.model)
    num_classes = ckpt.get('num_classes', NUM_CLASSES[ds])
    model = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['state_dict'])

    loader, mean, std, num_classes = get_clean_test_loader(
        ds, batch_size=args.batch_size, n_samples=args.n_samples,
        num_workers=args.num_workers, data_root=args.data_root,
        tinyimagenet_dir=args.tinyimagenet_dir)
    wrapped = NormalizedModel(model, mean, std).to(device).eval()

    method = (ckpt.get('pruner') if isinstance(ckpt, dict) else None) or 'unknown'
    print(f"Evaluating {model_name} on {ds} ...")
    metrics = compute_metrics(
        wrapped, loader, device, num_classes,
        eps=args.eps, alpha=args.alpha, pgd_steps=tuple(args.pgd_steps),
        include_autoattack=not args.no_autoattack, aa_version=args.aa_version,
        verbose=True, aa_eps=args.aa_eps)
    return metrics, ds, model_name, method


def main():
    from attacks import EPS, ALPHA, AA_EPS  # defaults shared with the attack module

    p = argparse.ArgumentParser(description="Compute Robust Transfer Index (RTI) and nRTI.")
    # Mode A: two precomputed metrics JSON files
    p.add_argument('--d1-metrics', help='metrics JSON for source dataset D1')
    p.add_argument('--d2-metrics', help='metrics JSON for target dataset D2')
    p.add_argument('--d1-index', type=int, default=-1, help='record index in --d1-metrics')
    p.add_argument('--d2-index', type=int, default=-1, help='record index in --d2-metrics')
    # Mode B: evaluate two checkpoints
    p.add_argument('--d1-checkpoint', help='checkpoint evaluated on its dataset (source)')
    p.add_argument('--d2-checkpoint', help='checkpoint evaluated on its dataset (target)')
    p.add_argument('--d1-dataset', default='cifar10')
    p.add_argument('--d2-dataset', default='cifar100')
    p.add_argument('--model', default='resnet20', help='fallback model name for bare checkpoints')
    # shared eval knobs (Mode B)
    p.add_argument('--eps', type=float, default=EPS,
                   help='L_inf budget for FGSM/PGD (default 8/255)')
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument('--aa-eps', type=float, default=AA_EPS,
                   help='L_inf budget for AutoAttack (default 4/255)')
    p.add_argument('--pgd-steps', nargs='+', type=int, default=[10, 20, 100])
    p.add_argument('--no-autoattack', action='store_true')
    p.add_argument('--aa-version', default='standard', choices=['standard', 'plus', 'rand'])
    p.add_argument('--n-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--tinyimagenet-dir', default='./tiny-imagenet-200')
    try:
        import torch
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        default_device = 'cpu'
    p.add_argument('--device', default=default_device)
    p.add_argument('--method', default=None, help='pruning method name (for Table 2)')
    p.add_argument('--arch', default=None, help='architecture name (for Table 2)')
    p.add_argument('--nrti-eps', type=float, default=None,
                   help='nRTI denominator floor (tau) max(A_rob(D1), eps), as a '
                        'fraction; default = 0.01 (the tau = 1%% floor)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--results-file', default=None, help='optional JSON to write the indices')
    p.add_argument('--config', default=None, help='YAML/JSON config file (CLI args override it)')
    from utils import set_seed, parse_args_with_config
    args = parse_args_with_config(p)
    set_seed(args.seed)

    method, arch = args.method, args.arch
    if args.d1_metrics and args.d2_metrics:
        m1 = load_metrics_record(args.d1_metrics, args.d1_index)
        m2 = load_metrics_record(args.d2_metrics, args.d2_index)
        d1_name = m1.get('dataset', 'D1')
        d2_name = m2.get('dataset', 'D2')
        meta1 = m1.get('checkpoint_meta') or {}
        method = method or meta1.get('pruner') or m1.get('pruner') or m1.get('method')
        arch = arch or m1.get('model')
    elif args.d1_checkpoint and args.d2_checkpoint:
        m1, d1_name, arch1, method1 = _metrics_from_checkpoint(args.d1_checkpoint, args.d1_dataset, args)
        m2, d2_name, _, _ = _metrics_from_checkpoint(args.d2_checkpoint, args.d2_dataset, args)
        method = method or method1
        arch = arch or arch1
    else:
        p.error("provide either (--d1-metrics and --d2-metrics) "
                "or (--d1-checkpoint and --d2-checkpoint)")

    # nRTI denominator floor (tau): explicit --nrti-eps, else the 0.01 (1%) default.
    nrti_eps = args.nrti_eps if args.nrti_eps is not None else EPS_DEFAULT
    indices = transfer_index(m1, m2, include_clean=True, nrti_eps=nrti_eps)
    title = (f"Robust Transfer Index  |  D1={d1_name} (source)  ->  D2={d2_name} (target)"
             f"{f'  |  {arch}/{method}' if (arch or method) else ''}")
    print("\n" + format_table(indices, d1_name, d2_name, title=title))

    if args.results_file:
        payload = {'method': method, 'arch': arch, 'D1': d1_name, 'D2': d2_name,
                   'indices': indices}
        os.makedirs(os.path.dirname(args.results_file) or '.', exist_ok=True)
        with open(args.results_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nIndices written to {args.results_file}")


if __name__ == '__main__':
    main()
