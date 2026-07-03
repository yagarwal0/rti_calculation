"""
tables.py — generate publication-ready tables (Markdown + LaTeX/booktabs).

Table 1 — Robust Accuracy
    | Method | Sparsity | Clean | PGD-100 | AutoAttack |
  built from metrics JSON records written by `metrics.py --results-file`.

Table 2 — RTI
    | Method | Arch | RTI | nRTI |
  built from RTI JSON records written by `rti.py --results-file`
  (RTI/nRTI taken from a chosen metric, default AutoAttack).

Table 3 — Training Efficiency
    | Method | FLOPs | Time | Robust Acc |
  built from the Experiment-5 JSON written by `exp5_adv_efficiency.py`
  (dense vs YOPO-sparse: effective GFLOPs, training time per epoch, robust acc).

Examples
--------
    # Table 1 from one or more metrics files
    python tables.py --metrics-files results/*.json --out-dir tables

    # Table 2 from RTI files, using AutoAttack robustness for the index
    python tables.py --rti-files results/rti_*.json --metric AutoAttack --out-dir tables

    # Table 3 from the Experiment-5 efficiency JSON
    python tables.py --efficiency-files experiments_out/exp5_adv_efficiency.json --out-dir tables

    # All at once, print to stdout and save .md + .tex
    python tables.py --metrics-files results/metrics.json --rti-files results/rti.json \
                     --efficiency-files experiments_out/exp5_adv_efficiency.json
"""

import os
import glob
import json
import argparse


# ---------------------------------------------------------------------------
# Loading / field extraction (tolerant to the schemas of metrics.py / rti.py)
# ---------------------------------------------------------------------------
def load_records(paths):
    """Load and flatten JSON records from files/globs. Each file may hold a
    single dict or a list of dicts."""
    records = []
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) or ([p] if os.path.isfile(p) else []))
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        records.extend(data if isinstance(data, list) else [data])
    return records


def _method(rec):
    meta = rec.get('checkpoint_meta') or {}
    return rec.get('method') or meta.get('pruner') or rec.get('pruner') or '-'


def _sparsity(rec):
    s = rec.get('sparsity')
    if s is None:
        s = (rec.get('checkpoint_meta') or {}).get('target_sparsity')
    return s


def _robust(rec, attack):
    return rec.get('attacks', {}).get(attack, {}).get('robust_accuracy')


def _fmt(v, nd=2, pct=False, percent_scale=False):
    """Format a number for a cell; '-' if missing."""
    if v is None:
        return '-'
    val = float(v) * (100.0 if percent_scale else 1.0)
    s = f"{val:.{nd}f}"
    return s + ('%' if pct else '')


# ---------------------------------------------------------------------------
# Table builders -> (headers, rows)  with rows as lists of strings
# ---------------------------------------------------------------------------
def build_table1(records, sparsity_percent=True, md=True):
    """Table 1: Method | Sparsity | Clean | PGD-100 | AutoAttack."""
    headers = ['Method', 'Sparsity (%)' if sparsity_percent else 'Sparsity',
               'Clean', 'PGD-100', 'AutoAttack']
    rows = []
    for rec in records:
        sp = _sparsity(rec)
        sp_cell = _fmt(sp, nd=2, percent_scale=sparsity_percent) if sp is not None else '-'
        rows.append([
            _method(rec),
            sp_cell,
            _fmt(rec.get('clean_accuracy'), pct=md),
            _fmt(_robust(rec, 'PGD-100'), pct=md),
            _fmt(_robust(rec, 'AutoAttack'), pct=md),
        ])
    # sort by method then sparsity
    rows.sort(key=lambda r: (r[0], _sortnum(r[1])))
    return headers, rows


def build_table2(rti_records, metric='AutoAttack', md=True):
    """Table 2: Method | Arch | RTI | nRTI (from a chosen metric)."""
    headers = ['Method', 'Arch', f'RTI ({metric})', f'nRTI ({metric})']
    rows = []
    for rec in rti_records:
        idx = rec.get('indices', {})
        cell = idx.get(metric, {})
        rows.append([
            rec.get('method') or '-',
            rec.get('arch') or '-',
            _fmt(cell.get('RTI'), nd=2),
            _fmt(cell.get('nRTI'), nd=4),
        ])
    rows.sort(key=lambda r: (r[1], r[0]))   # arch, method
    return headers, rows


def build_table3(eff_records, md=True):
    """Table 3: Method | FLOPs (G) | Time (s/epoch) | Robust Acc.

    Reads the Experiment-5 records written by `exp5_adv_efficiency.py`. Each
    record carries a 'summary' list with one row per configuration (dense /
    yopo_s<sparsity>) holding effective FLOPs, per-epoch training time and the
    worst-case (AutoAttack) robust accuracy.
    """
    headers = ['Method', 'FLOPs (G)', 'Time (s/epoch)', 'Robust Acc']
    rows = []
    for rec in eff_records:
        summary = rec.get('summary')
        srows = summary if isinstance(summary, list) else [rec]
        model = rec.get('model')
        for s in srows:
            method = s.get('config') or s.get('method') or '-'
            if model:
                method = f"{model}/{method}"
            flops = s.get('flops_effective')
            gflops = '-' if flops is None else _fmt(float(flops) / 1e9, nd=4)
            rows.append([
                method,
                gflops,
                _fmt(s.get('train_time_per_epoch_s'), nd=2),
                _fmt(s.get('robust_accuracy_AA'), pct=md),
            ])
    return headers, rows


def _sortnum(s):
    try:
        return float(str(s).replace('%', ''))
    except ValueError:
        return float('inf')


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_markdown(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def render_latex(headers, rows, caption='', label='', align=None):
    align = align or ('l' + 'c' * (len(headers) - 1))
    esc = lambda s: str(s).replace('%', r'\%').replace('_', r'\_')
    lines = [r'\begin{table}[t]', r'  \centering']
    if caption:
        lines.append(f'  \\caption{{{caption}}}')
    if label:
        lines.append(f'  \\label{{{label}}}')
    lines.append(f'  \\begin{{tabular}}{{{align}}}')
    lines.append(r'    \toprule')
    lines.append('    ' + ' & '.join(esc(h) for h in headers) + r' \\')
    lines.append(r'    \midrule')
    for r in rows:
        lines.append('    ' + ' & '.join(esc(c) for c in r) + r' \\')
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def _emit(name, headers, rows, fmt, out_dir, caption, label):
    print(f"\n===== {name} =====")
    if fmt in ('markdown', 'both'):
        md = render_markdown(headers, rows)
        print('\n[Markdown]\n' + md)
    if fmt in ('latex', 'both'):
        tex = render_latex(headers, rows, caption=caption, label=label)
        print('\n[LaTeX]\n' + tex)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = name.lower().replace(' ', '_').replace(':', '')
        if fmt in ('markdown', 'both'):
            with open(os.path.join(out_dir, base + '.md'), 'w') as f:
                f.write(render_markdown(headers, rows) + '\n')
        if fmt in ('latex', 'both'):
            with open(os.path.join(out_dir, base + '.tex'), 'w') as f:
                f.write(render_latex(headers, rows, caption=caption, label=label) + '\n')
        print(f"  (saved to {out_dir}/{base}.md / .tex)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate publication-ready tables (Markdown + LaTeX).")
    p.add_argument('--metrics-files', nargs='+', default=[],
                   help='metrics JSON files/globs (for Table 1)')
    p.add_argument('--rti-files', nargs='+', default=[],
                   help='RTI JSON files/globs (for Table 2)')
    p.add_argument('--efficiency-files', nargs='+', default=[],
                   help='Experiment-5 JSON files/globs (for Table 3)')
    p.add_argument('--metric', default='AutoAttack',
                   help='which attack RTI/nRTI to tabulate in Table 2 (default: AutoAttack)')
    p.add_argument('--fmt', default='both', choices=['markdown', 'latex', 'both'])
    p.add_argument('--sparsity-fraction', action='store_true',
                   help='show sparsity as a fraction (0-1) instead of percent')
    p.add_argument('--out-dir', default=None, help='directory to save .md/.tex files')
    args = p.parse_args()

    if not args.metrics_files and not args.rti_files and not args.efficiency_files:
        p.error("provide --metrics-files (Table 1), --rti-files (Table 2) "
                "and/or --efficiency-files (Table 3)")

    md_flag = args.fmt in ('markdown', 'both')

    if args.metrics_files:
        recs = load_records(args.metrics_files)
        headers, rows = build_table1(recs, sparsity_percent=not args.sparsity_fraction, md=md_flag)
        _emit("Table 1 Robust Accuracy", headers, rows, args.fmt, args.out_dir,
              caption="Robust accuracy of pruned models (clean, PGD-100, AutoAttack).",
              label="tab:robust_accuracy")

    if args.rti_files:
        recs = load_records(args.rti_files)
        headers, rows = build_table2(recs, metric=args.metric, md=md_flag)
        _emit("Table 2 RTI", headers, rows, args.fmt, args.out_dir,
              caption=f"Robust Transfer Index (RTI) and normalized RTI (nRTI), {args.metric}.",
              label="tab:rti")

    if args.efficiency_files:
        recs = load_records(args.efficiency_files)
        headers, rows = build_table3(recs, md=md_flag)
        _emit("Table 3 Training Efficiency", headers, rows, args.fmt, args.out_dir,
              caption="Adversarial-training efficiency: effective FLOPs, training "
                      "time per epoch, and robust accuracy (dense vs YOPO-sparse).",
              label="tab:efficiency")


if __name__ == '__main__':
    main()
