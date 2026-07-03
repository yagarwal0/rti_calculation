# Running the full RTI / nRTI sweep

Reproduce the **complete Robust Transfer Index matrix**:

- **Attacks**:  FGSM, PGD-10, PGD-20, PGD-100, AutoAttack
- **Pruning methods**:  SNIP, GraSP, SynFlow, DPaI, YOPO
- **Sparsities**:  20, 30, 50, 68.37, 90, 95, 99 %
- **Models**:  resnet20, resnet18, vgg19
- Transfer direction:  CIFAR-10  →  CIFAR-100  (default; any source/target via `--source` / `--target`, see [CIFAR-10 → Tiny ImageNet](#cifar-10--tiny-imagenet))

For every `(model, method, sparsity)` combo, `exp4_sparsity_sweep.py` trains
two models on CIFAR-100 — one with a mask computed natively on CIFAR-100
(*original*) and one with a mask transferred from CIFAR-10 (*transferred*) —
evaluates clean + all five attacks on CIFAR-100, and computes RTI / nRTI per
attack. (No code change is needed; `exp4` already does this.)

---

## 1. Install

```cmd
cd /d C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models
pip install -r requirements.txt
```

## 2. Run the full sweep

`exp4` already iterates the 7 sparsities (`20, 30, 50, 68.37, 90, 95, 99 %`)
and computes per-attack RTI/nRTI. We just loop over models and methods.

**Windows cmd (single line):**
```cmd
cd /d C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models && for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --results-dir experiments_out\full\%M\%P
```

**PowerShell (single line):**
```powershell
cd C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models; foreach ($M in 'resnet20','resnet18','vgg19') { foreach ($P in 'snip','grasp','synflow','dpai','yopo') { python experiments\exp4_sparsity_sweep.py --models $M --methods $P --results-dir "experiments_out\full\$M\$P" } }
```

**Bash / Git Bash (single line):**
```bash
cd /c/Users/yashi/OneDrive/Desktop/final_thesis_24052026/prune_models && for M in resnet20 resnet18 vgg19; do for P in snip grasp synflow dpai yopo; do python experiments/exp4_sparsity_sweep.py --models $M --methods $P --results-dir experiments_out/full/$M/$P; done; done
```

Defaults used by `exp4`:
- epochs `200`, batch size `128`, eval subset `1000` images
- AutoAttack `standard` ensemble (APGD-CE + APGD-T + FAB-T + Square)
- PGD-{10, 20, 100} with eps = 8/255, alpha = 2/255
- 7 sparsities = the standard `[0.20, 0.30, 0.50, 0.6837, 0.90, 0.95, 0.99]`

Override any of them on the CLI, e.g. `--epochs 50 --n-samples 500 --no-autoattack`.

## 3. Where the results land

After the sweep finishes you'll have **15 JSON files** (3 models × 5 methods):

```
experiments_out/full/<model>/<method>/exp4_sparsity_sweep.json
```

Each JSON is a list of 7 records (one per sparsity). Per record:

```jsonc
{
  "experiment": 4,
  "source": "cifar10", "target": "cifar100",
  "model": "resnet20", "method": "snip",
  "target_sparsity": 0.9,
  "original":    { "metrics": { "clean_accuracy": ..., "attacks": {...} },
                   "sparsity": ..., "avg_epoch_time_s": ..., "flops_effective": {...} },
  "transferred": { ... },
  "rti": {
    "clean":     { "A_rob_D1": ..., "A_rob_D2": ..., "RTI": ..., "nRTI": ... },
    "FGSM":      { "A_rob_D1": ..., "A_rob_D2": ..., "RTI": ..., "nRTI": ... },
    "PGD-10":    { ... },
    "PGD-20":    { ... },
    "PGD-100":   { ... },
    "AutoAttack":{ ... }
  }
}
```

So the **full RTI/nRTI matrix** is right there: open the JSON, walk through
`records[i]['rti'][<attack>]` for each sparsity `i` and each attack.

Best trained checkpoints (self-describing) are saved alongside, in
`experiments_out/checkpoints/exp4_*.pth`.

## 4. Render plots (one folder per attack)

```cmd
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do for %A in (FGSM PGD-10 PGD-20 PGD-100 AutoAttack) do python scripts\plot_results.py --results-dir experiments_out\full\%M\%P --out-dir plots\full\%M\%P\%A --metric %A
```

You'll get, per `(model, method, attack)`:
- `exp4_rti_vs_sparsity.png` — RTI (left axis) and nRTI (right axis) vs sparsity
- `exp4_robust_acc_vs_sparsity.png` — original vs transferred robust accuracy

## 5. (Optional) publication tables

`tables.py` renders the Table-2 RTI summary in **Markdown + LaTeX/booktabs**.
First, distil one RTI record per `(model, method)` from each exp4 JSON into the
shape `tables.py` expects, then call it. The fastest way is to write tiny per-
combo RTI JSON files via `rti.py --d1-metrics ... --d2-metrics ...` from the
checkpoints exp4 already produced, then:

```cmd
python tables.py --rti-files "rti_out\*.json" --metric AutoAttack --out-dir tables
```

(If you only need raw numbers, the JSONs from step 3 are already enough — the
tables step is purely for paper output.)

---

## Compute scale — read this before launching

The full sweep is **3 × 5 × 7 = 105 (model, method, sparsity) combos**, and
each combo trains **two** models (original + transferred) = **210 training
runs**, each followed by FGSM + PGD-10/20/100 + AutoAttack evaluation. With
the defaults (200 epochs, full AutoAttack) this is **days** of compute even on
a single GPU; AutoAttack dominates.

Sensible knobs while you iterate:

| flag | recommended | effect |
|------|-------------|--------|
| `--epochs 50` | yes | 4× faster training; numbers are still informative |
| `--n-samples 500` | yes | 2× faster attack eval |
| `--no-autoattack` | yes (initial pass) | removes the slowest attack; switch to AA for the final run |
| `--num-workers 4` | yes | DataLoader parallelism |
| `--logger none` | optional | skips TensorBoard I/O |

Fast first pass (still produces the full RTI/nRTI matrix for FGSM + PGD-*):
```cmd
cd /d C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models && for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --epochs 50 --n-samples 500 --no-autoattack --num-workers 4 --logger none --results-dir experiments_out\full\%M\%P
```

When you're ready for the final publication numbers, drop the overrides and
re-run — the JSONs will be overwritten with full-fidelity values including
AutoAttack.

---

## CIFAR-10 → Tiny ImageNet

The same `exp4` script supports any source/target pair via `--source` /
`--target` (added so this protocol works without code edits). Tiny ImageNet
is treated identically to the CIFAR datasets — every image is resized to
**32 × 32** in the loader, so the same CIFAR-sized models apply unchanged.

### 0. One-time setup
Tiny ImageNet is not on torchvision, so download and unzip it once:

1. Download <http://cs231n.stanford.edu/tiny-imagenet-200.zip>
2. Unzip so the layout is:
   ```
   prune_models\tiny-imagenet-200\
     train\<wnid>\images\*.JPEG
     val\images\*.JPEG
     val\val_annotations.txt
   ```

(Or unzip anywhere and pass `--tinyimagenet-dir <path>` to every command.)

### 1. Run the sweep (single line, cmd)
```cmd
cd /d C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models && for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --source cifar10 --target tinyimagenet --results-dir experiments_out\full_c10_to_tin\%M\%P
```

PowerShell:
```powershell
cd C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models; foreach ($M in 'resnet20','resnet18','vgg19') { foreach ($P in 'snip','grasp','synflow','dpai','yopo') { python experiments\exp4_sparsity_sweep.py --models $M --methods $P --source cifar10 --target tinyimagenet --results-dir "experiments_out\full_c10_to_tin\$M\$P" } }
```

Bash / Git Bash:
```bash
cd /c/Users/yashi/OneDrive/Desktop/final_thesis_24052026/prune_models && for M in resnet20 resnet18 vgg19; do for P in snip grasp synflow dpai yopo; do python experiments/exp4_sparsity_sweep.py --models $M --methods $P --source cifar10 --target tinyimagenet --results-dir experiments_out/full_c10_to_tin/$M/$P; done; done
```

The schema of every output JSON
(`experiments_out\full_c10_to_tin\<model>\<method>\exp4_sparsity_sweep.json`)
is identical to the CIFAR-10 → CIFAR-100 case (see §3): one record per
sparsity, each record with `rti.{FGSM, PGD-10, PGD-20, PGD-100, AutoAttack, clean}`.

### 2. Plots
```cmd
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do for %A in (FGSM PGD-10 PGD-20 PGD-100 AutoAttack) do python scripts\plot_results.py --results-dir experiments_out\full_c10_to_tin\%M\%P --out-dir plots\full_c10_to_tin\%M\%P\%A --metric %A
```

### Notes for Tiny ImageNet
- **Target dataset size**: Tiny ImageNet train = 100 000 images (CIFAR-100 is 50 000), so each epoch is ~2 × slower than the CIFAR-100 sweep.
- **Classifier mismatch is handled automatically**: a mask computed on CIFAR-10 (10-class classifier) is applied to a Tiny ImageNet model (200-class classifier) via `utils.transfer_mask`, which keeps the mismatched classifier layer dense (mask = ones). Conv masks transfer as-is.
- **Normalization**: the loader uses Tiny ImageNet's own channel stats by default (`tinyimagenet_norm='tinyimagenet'`); use the dataset's `--tinyimagenet-dir` flag if your unzipped folder isn't at `./tiny-imagenet-200`.
- A practical fast first pass:
  ```cmd
  cd /d C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models && for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --source cifar10 --target tinyimagenet --epochs 30 --n-samples 500 --no-autoattack --num-workers 4 --logger none --results-dir experiments_out\full_c10_to_tin\%M\%P
  ```

### Symmetric direction (Tiny ImageNet → CIFAR-10)
Just swap the flags:
```cmd
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --source tinyimagenet --target cifar10 --results-dir experiments_out\full_tin_to_c10\%M\%P
```

---

## Smoke test (1 epoch, no AA) — verify wiring first

```cmd
cd /d C:\Users\yashi\OneDrive\Desktop\final_thesis_24052026\prune_models && for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --epochs 1 --no-autoattack --n-samples 256 --num-workers 0 --batch-size 128 --logger none --results-dir experiments_out\smoke\%M\%P
```

This produces one JSON per `(model, method)` very fast; check that
`experiments_out\smoke\<model>\<method>\exp4_sparsity_sweep.json` exists for
all 15 combos, then launch the real sweep.
