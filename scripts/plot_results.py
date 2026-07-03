"""
scripts/plot_results.py — figures from the experiment JSONs.

Reads `experiments_out/exp{1..5}_*.json` and produces:
    exp1_robust_accuracy.png        bar chart: clean / PGD-100 / AutoAttack per method
    exp2_rti_c10_to_c100.png        bar chart: RTI (AutoAttack) and nRTI per method
    exp3_rti_c100_to_c10.png        same, symmetric direction
    exp4_rti_vs_sparsity.png        line plot: RTI / nRTI vs sparsity
    exp4_robust_acc_vs_sparsity.png line plot: robust acc (original vs transferred)
    exp5_efficiency.png             grouped bars: robust acc / time / FLOPs

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-dir experiments_out --out-dir plots
    python scripts/plot_results.py --metric AutoAttack   # which attack to plot in 2/3/4

Missing JSONs are skipped silently.
"""

import os
import json
import argparse
import glob

import matplotlib
matplotlib.use('Agg')              # headless
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _find(results_dir, prefix):
    """Find a JSON whose name starts with `prefix` (handles dataset suffixes)."""
    cands = sorted(glob.glob(os.path.join(results_dir, prefix + '*.json')))
    if not cands:
        return None
    return _load(cands[0])


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


def _accuracy(rec, key):
    """Robust accuracy under attack `key`, or clean if key=='clean'."""
    m = rec.get('metrics', {})
    if key == 'clean':
        return m.get('clean_accuracy')
    return m.get('attacks', {}).get(key, {}).get('robust_accuracy')


# ---------------------------------------------------------------------------
# exp1: per-method robust accuracy bar chart
# ---------------------------------------------------------------------------
def plot_exp1(records, out_dir):
    if not records:
        return
    # group by method (average over models/sparsities if multiple)
    methods = sorted({r['method'] for r in records})
    keys = ['clean', 'PGD-100', 'AutoAttack']

    data = {k: [] for k in keys}
    for m in methods:
        rs = [r for r in records if r['method'] == m]
        for k in keys:
            vals = [_accuracy(r, k) for r in rs]
            vals = [v for v in vals if v is not None]
            data[k].append(np.mean(vals) if vals else 0.0)

    x = np.arange(len(methods))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(keys):
        ax.bar(x + (i - 1) * w, data[k], w, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Experiment 1 — Robust Accuracy on CIFAR-10 (per method)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'exp1_robust_accuracy.png')


# ---------------------------------------------------------------------------
# exp2 / exp3: RTI / nRTI per method (mask transfer original vs transferred)
# ---------------------------------------------------------------------------
def plot_transfer(records, out_dir, name, title, metric='AutoAttack'):
    if not records:
        return
    methods = sorted({r['method'] for r in records})
    rti_vals, nrti_vals = [], []
    for m in methods:
        rs = [r for r in records if r['method'] == m]
        rti = [r['rti'].get(metric, {}).get('RTI') for r in rs if 'rti' in r]
        nrti = [r['rti'].get(metric, {}).get('nRTI') for r in rs if 'rti' in r]
        rti = [v for v in rti if v is not None]
        nrti = [v for v in nrti if v is not None]
        rti_vals.append(np.mean(rti) if rti else 0.0)
        nrti_vals.append(np.mean(nrti) if nrti else 0.0)

    x = np.arange(len(methods))
    w = 0.4
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - w / 2, rti_vals, w, color='tab:blue', label=f'RTI ({metric})')
    ax1.set_xticks(x); ax1.set_xticklabels(methods)
    ax1.set_ylabel(f'RTI ({metric})', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, nrti_vals, w, color='tab:red', label=f'nRTI ({metric})')
    ax2.set_ylabel(f'nRTI ({metric})', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title(title)
    ax1.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# exp4: RTI / nRTI / robust acc vs sparsity (line plots)
# ---------------------------------------------------------------------------
def plot_exp4(records, out_dir, metric='AutoAttack'):
    if not records:
        return
    records = sorted(records, key=lambda r: r['target_sparsity'])
    xs = [r['target_sparsity'] * 100 for r in records]
    rti = [r['rti'].get(metric, {}).get('RTI') for r in records]
    nrti = [r['rti'].get(metric, {}).get('nRTI') for r in records]
    rob_orig = [_accuracy(r['original'], metric) for r in records]
    rob_trans = [_accuracy(r['transferred'], metric) for r in records]

    # RTI / nRTI vs sparsity (twin axes)
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(xs, rti, 'o-', color='tab:blue', label=f'RTI ({metric})')
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel(f'RTI ({metric})', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, nrti, 's--', color='tab:red', label=f'nRTI ({metric})')
    ax2.set_ylabel(f'nRTI ({metric})', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    method = records[0].get('method', '')
    ax1.set_title(f'Experiment 4 — RTI vs Sparsity ({method})')
    fig.tight_layout()
    _save(fig, out_dir, 'exp4_rti_vs_sparsity.png')

    # robust accuracy original vs transferred
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, rob_orig, 'o-', label='original')
    ax.plot(xs, rob_trans, 's--', label='transferred')
    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel(f'Robust accuracy ({metric}, %)')
    ax.set_title(f'Experiment 4 — Robust accuracy vs Sparsity ({method})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, 'exp4_robust_acc_vs_sparsity.png')


# ---------------------------------------------------------------------------
# exp5: dense vs YOPO — grouped bars (acc / time / FLOPs)
# ---------------------------------------------------------------------------
def plot_exp5(records, out_dir):
    if not records:
        return
    rec = records[0]                              # one record per dataset/model
    rows = rec.get('summary', [])
    if not rows:
        return
    cfgs = [r['config'] for r in rows]
    acc = [r.get('robust_accuracy_AA') or 0 for r in rows]
    tim = [r.get('train_time_per_epoch_s') or 0 for r in rows]
    gfl = [(r.get('flops_effective') or 0) / 1e9 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.arange(len(cfgs))
    axes[0].bar(x, acc, color=['tab:gray', 'tab:green'])
    axes[0].set_xticks(x); axes[0].set_xticklabels(cfgs)
    axes[0].set_ylabel('Robust accuracy (AA, %)')
    axes[0].set_title('Robust accuracy')
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(x, tim, color=['tab:gray', 'tab:orange'])
    axes[1].set_xticks(x); axes[1].set_xticklabels(cfgs)
    axes[1].set_ylabel('Training time / epoch (s)')
    axes[1].set_title('Train time per epoch')
    axes[1].grid(True, axis='y', alpha=0.3)

    axes[2].bar(x, gfl, color=['tab:gray', 'tab:purple'])
    axes[2].set_xticks(x); axes[2].set_xticklabels(cfgs)
    axes[2].set_ylabel('Effective FLOPs (G)')
    axes[2].set_title('Total FLOPs')
    axes[2].grid(True, axis='y', alpha=0.3)

    ds = rec.get('dataset', '?'); model = rec.get('model', '?')
    fig.suptitle(f'Experiment 5 — Adv-Training Efficiency ({model}, {ds})')
    fig.tight_layout()
    _save(fig, out_dir, 'exp5_efficiency.png')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default='./experiments_out')
    p.add_argument('--out-dir', default='./plots')
    p.add_argument('--metric', default='AutoAttack',
                   help="attack used in transfer plots (default AutoAttack; "
                        "if your runs skipped AA, try PGD-100)")
    args = p.parse_args()

    exp1 = _find(args.results_dir, 'exp1_robustness')
    exp2 = _find(args.results_dir, 'exp2_transfer')
    exp3 = _find(args.results_dir, 'exp3_transfer')
    exp4 = _find(args.results_dir, 'exp4_sparsity')
    exp5 = _find(args.results_dir, 'exp5_adv')

    plot_exp1(exp1, args.out_dir)
    plot_transfer(exp2, args.out_dir, 'exp2_rti_c10_to_c100.png',
                  'Experiment 2 — RTI / nRTI (CIFAR-10 → CIFAR-100)', args.metric)
    plot_transfer(exp3, args.out_dir, 'exp3_rti_c100_to_c10.png',
                  'Experiment 3 — RTI / nRTI (CIFAR-100 → CIFAR-10)', args.metric)
    plot_exp4(exp4, args.out_dir, args.metric)
    plot_exp5(exp5, args.out_dir)

    print(f"\nDone. Plots written to {args.out_dir}/")


if __name__ == '__main__':
    main()
