# prune_models — Pruning + Adversarial-Robustness Transfer Workspace

A complete, self-contained codebase for studying **transferable pruning under
adversarial attack**:

> Prune a network *at initialization* (SNIP / GraSP / SynFlow / DPaI / YOPO),
> train the sparse sub-network (optionally with PGD adversarial training),
> evaluate clean and robust accuracy (FGSM, PGD-10/20/100, AutoAttack), and
> measure how well the pruning mask's robustness **transfers** from one dataset
> to another via the **Robust Transfer Index (RTI / nRTI)**.

Everything lives inside this one folder. Results are logged to TensorBoard/W&B
and exported as publication-ready tables (Markdown + LaTeX) and figures.

---

## Table of contents
1. [Installation](#1-installation)
2. [Repository layout](#2-repository-layout)
3. [Core components](#3-core-components) — datasets, models, pruning methods, threat model
4. [Quickstart](#4-quickstart)
5. [Pipeline stages](#5-pipeline-stages) — prune → train → attack → metrics → RTI → tables
6. [Experimental protocol](#6-experimental-protocol) — Experiments 1–5
7. [Transfer for any source → target pair](#7-transfer-for-any-source--target-pair)
8. [Full sweep over all models & methods](#8-full-sweep-over-all-models--methods)
9. [Final deliverables](#9-final-deliverables)
10. [Reproducibility](#10-reproducibility)
11. [CLI reference](#11-cli-reference)
12. [Notes & caveats](#12-notes--caveats)

---

## 1. Installation
```bash
pip install -r requirements.txt
pip install git+https://github.com/fra31/auto-attack   # only for AutoAttack (FGSM/PGD are built-in)
```

Then download the datasets into the locations the code expects:
```bash
python download_dataset.py                          # CIFAR-10 + CIFAR-100 + TinyImageNet
python download_dataset.py --datasets cifar10 cifar100   # only the CIFAR pair
```
`download_dataset.py` puts CIFAR-10/100 under `./data` and TinyImageNet under
`./tiny-imagenet-200` (the defaults used everywhere else), so every later command
works without extra flags. It is idempotent — already-present datasets are
skipped unless you pass `--force`.

- **GPU** is auto-detected (`--device` defaults to CUDA when available).
- **YOPO** additionally needs `scikit-learn` (for the NMF factorization).
- **TinyImageNet** is a ~240 MB download (not on torchvision) — see [§12](#12-notes--caveats).

---

## 2. Repository layout
```
prune_models/
├── main.py            # orchestrator: build → prune → apply mask → measure → save
├── train.py           # training pipeline (SGD+momentum, cosine, 200 ep; optional PGD-AT)
├── adv_train.py       # PGD adversarial-example generator used during training
├── attacks.py         # adversarial eval: FGSM/PGD (built-in), AutoAttack (fra31/auto-attack)
├── metrics.py         # clean + robust accuracy report (text table + JSON)
├── rti.py             # Robust Transfer Index (RTI) and Normalized RTI (nRTI)
├── tables.py          # publication tables (Markdown + LaTeX): Table 1 / 2 / 3
├── logger.py          # TensorBoard / W&B: accuracy, robustness, RTI-vs-sparsity
├── dataloader.py      # CIFAR-10 / CIFAR-100 / TinyImageNet (→32×32) loaders
├── download_dataset.py # fetch CIFAR-10/100 + TinyImageNet into ./data and ./tiny-imagenet-200
├── requirements.txt
├── configs/           # per-experiment YAML configs (cifar10, cifar100, tinyimagenet, smoke)
├── experiments/       # one focused script per protocol item (exp1..exp5) + _common.py
├── scripts/           # run_all_experiments.sh + plot_results.py (final deliverables)
├── models/            # network builders (plain nn.Modules, 3×32×32 input)
│   ├── resnet20.py    #   resnet20
│   ├── resnet_8x.py   #   resnet18 (ResNet18_8x)
│   └── vgg.py         #   vgg19
├── pruning/           # pruning-at-initialization algorithms
│   ├── snip.py        #   SNIP   (connection sensitivity)
│   ├── grasp.py       #   GraSP  (gradient signal preservation)
│   ├── synflow.py     #   SynFlow (data-free, 100-step iterative schedule)
│   ├── dpai.py        #   DPaI / NPB (data-free, effective-path optimisation)
│   └── yopo.py        #   YOPO   (NMF reconstruction-error — the core method)
├── utils/
│   ├── prune_utils.py #   mask application, sparsity, SynFlow conversion, transfer_mask
│   ├── seed.py        #   set_seed(): python/numpy/torch(+cuda), optional determinism
│   ├── config.py      #   load YAML/JSON configs and fold them into argparse
│   └── flops.py       #   sparsity-aware FLOPs/MACs counter (used by exp5)
├── output/            # pruned checkpoints from main.py (per dataset)
├── trained/           # best trained checkpoints from train.py
└── experiments_out/   # experiment JSONs, checkpoints, logs
```

---

## 3. Core components

### Datasets (`dataloader.py`)
| Dataset | Classes | Native | Used as | Auto-download |
|---|---|---|---|---|
| CIFAR-10 | 10 | 32×32 | source / target | ✅ |
| CIFAR-100 | 100 | 32×32 | source / target | ✅ |
| TinyImageNet | 200 | 64×64 → **resized to 32×32** | source / target | ❌ (manual) |

All datasets feed the network **3×32×32** inputs, so the same CIFAR-sized models
apply to all three (this is what makes cross-dataset mask transfer possible).

### Models (`models/`)
`resnet20`, `resnet18` (ResNet18_8x), `vgg19`. Selected with `--models`.

### Pruning methods (`pruning/`)
All prune **at initialization** (before training); selected with `--pruners` /
`--methods`. `sparsity` = the fraction of prunable (Conv2d/Linear) weights
**removed** (`0.9` ⇒ keep 10%).

| Key | Method | Data | Notes |
|---|---|---|---|
| `snip` | SNIP | 1 batch | connection sensitivity `|g·θ|` |
| `grasp` | GraSP | few/class | gradient-signal preservation |
| `synflow` | SynFlow | data-free | 100-step iterative schedule |
| `dpai` | DPaI / NPB | data-free | effective-path optimisation |
| `yopo` | **YOPO** | data-free | NMF reconstruction-error (core method) |

**YOPO (core method):** each prunable weight matrix `W` is factorized by NMF as
`W ≈ V·H`; the per-weight saliency is the reconstruction error `S = |W − V·H|`.
Weights the low-rank model *fails* to reconstruct (high `S`) are kept; well-
reconstructed (redundant) weights are pruned by thresholding `S` to hit the
target sparsity. The mask is computed **once at init** and **frozen** during
training (gradients are masked each step and exact zeros re-asserted after the
optimizer step), so pruned weights never revive. Needs `scikit-learn`.

### Threat model (`attacks.py`)
L∞, **eps = 8/255**, PGD **alpha = 2/255**. Attacks run in raw `[0,1]` pixel
space; the model is wrapped so normalization happens *inside* the forward pass,
keeping the eps-budget true to pixel space.

### RTI / nRTI (`rti.py`)
For a configuration `m` evaluated on a source `D1` and target `D2`, with robust
accuracies `A_rob(m, D1)` and `A_rob(m, D2)` as fractions in `[0, 1]`:

```
RTI(m)  = | A_rob(m, D1) − A_rob(m, D2) | × 100                          (absolute gap, %)
nRTI(m) = | A_rob(m, D1) − A_rob(m, D2) | × 100 / max(A_rob(m, D1), ε)   (relative gap, %)
```

Lower = robustness better preserved across the transfer. Both indices are scaled
by 100 so they read as percentages (thesis Eq. 4.3). **nRTI** normalizes by the
source robustness so configurations with different baselines compare fairly. The
denominator is `max(A_rob(D1), ε)` where **ε is the floor τ, default 0.01 (1%)**
(override with `--nrti-eps`), so nRTI stays finite and bounded even when source
robust accuracy is 0. Computed per metric: `clean, FGSM, PGD-10, PGD-20, PGD-100,
AutoAttack`.

---

## 4. Quickstart
```bash
# 1. Install
pip install -r requirements.txt && pip install git+https://github.com/fra31/auto-attack

# 2. Run all five experiments end-to-end (uses configs/cifar10.yaml by default)
bash scripts/run_all_experiments.sh

# 3. Render figures from the JSON results
python scripts/plot_results.py --results-dir experiments_out --out-dir plots

# Fast CPU sanity check (tiny epochs, no AutoAttack)
EXTRA="--no-autoattack" bash scripts/run_all_experiments.sh configs/smoke.yaml
```

Step-by-step (one command per stage):
```bash
python main.py    --config configs/cifar10.yaml                                            # prune sweep
python train.py   --config configs/cifar10.yaml --checkpoint output/cifar10/resnet20_snip_s0_9.pth
python metrics.py --config configs/cifar10.yaml --checkpoint trained/resnet20_snip_cifar10_best.pth --results-file results/metrics.json
python rti.py     --d1-metrics results/metrics_c10.json --d2-metrics results/metrics_c100.json --results-file results/rti.json
python tables.py  --metrics-files "results/*.json" --rti-files "results/rti*.json" --out-dir tables
```

---

## 5. Pipeline stages

### 5.1 Prune — `main.py`
For each `(model, pruner, sparsity)` it: (1) builds the model at a fixed init
(`--seed`), (2) computes a mask with the chosen algorithm, (3) applies it to a
fresh copy at the **same** init, (4) reports achieved sparsity and saves
`output/<dataset>/<model>_<pruner>_s<sparsity>.pth` (containing `state_dict`,
`masks`, and metadata).
```bash
python main.py --dataset cifar100 --sparsity 0.5 0.9 0.95
python main.py --models resnet20 --pruners synflow --sparsity 0.98
python main.py --dataset tinyimagenet --tinyimagenet-dir ./tiny-imagenet-200
python main.py --pruners snip grasp dpai            # skip the slower ones
```

### 5.2 Train — `train.py`
Standard training to fit the (pruned) network:
- **SGD + momentum** — `lr=0.1, momentum=0.9, weight_decay=5e-4`, Nesterov on
- **Cosine LR** — `CosineAnnealingLR(T_max=epochs)`
- **200 epochs** (configurable), cross-entropy loss, AMP auto-on on CUDA

Given a pruned checkpoint, only the surviving sub-network is trained (mask kept
fixed). Best checkpoint → `--output-dir` (default `./trained`).
```bash
# Train the sparse sub-network of a pruned checkpoint (mask kept fixed)
python train.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --epochs 200

# Train a fresh dense model from scratch
python train.py --model resnet20 --dataset cifar10 --epochs 200
```

**PGD adversarial training** (Madry et al.) — enable with `--adv-train`:
- `--adv-eps 8/255` (`0.0314`), `--adv-alpha 2/255` (`0.0078`), **`--adv-steps 10`** (default)
```bash
python train.py --checkpoint output/cifar10/resnet20_yopo_s0_9.pth --adv-train --adv-steps 10
```

### 5.3 Attack — `attacks.py`
L∞ adversarial robustness, all in pixel space `[0,1]` (model normalizes internally):
- **FGSM** (single-step), **PGD-10 / PGD-20 / PGD-100** (random start) — **built-in**, no extra dependency
- **AutoAttack** (mandatory) — the **official** library ([fra31/auto-attack](https://github.com/fra31/auto-attack)) via `run_standard_evaluation`; standard ensemble APGD-CE + APGD-T + FAB-T + Square
```bash
python attacks.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --n-samples 1000
python attacks.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --no-autoattack   # FGSM/PGD only, no AutoAttack install needed
```
AutoAttack is **on by default**; `--no-autoattack` skips it (and avoids the AutoAttack
install). Install AutoAttack with `pip install git+https://github.com/fra31/auto-attack`.
`--results-file` appends JSON.

### 5.4 Metrics — `metrics.py`
One-shot clean + robust report:
```bash
python metrics.py --checkpoint trained/resnet20_snip_cifar10_best.pth --n-samples 1000 \
                  --results-file results/metrics.json
```
```
  metric                      accuracy
  Clean                         91.20%
  Robust: FGSM                  52.10%
  Robust: PGD-10                48.00%
  Robust: PGD-20                47.30%
  Robust: PGD-100               46.90%
  Robust: AutoAttack            44.50%
  Worst-case (AutoAttack)       44.50%
```

### 5.5 RTI / nRTI — `rti.py`
```bash
# From two metrics JSON files written by metrics.py
python rti.py --d1-metrics results/cifar10.json --d2-metrics results/cifar100.json

# Or evaluate two checkpoints directly, then compute the indices
python rti.py --d1-checkpoint ck_c10.pth --d2-checkpoint ck_c100.pth --n-samples 1000

# Override the nRTI denominator floor τ (default = 0.01, i.e. 1%)
python rti.py --d1-metrics a.json --d2-metrics b.json --nrti-eps 1e-8
```
Programmatic: `from rti import rti, nrti, transfer_index`.

### 5.6 Logging — `logger.py`
TensorBoard (default) or W&B. Logs **accuracy curves**, **robustness curves**,
and **RTI-vs-sparsity** plots. `train.py` logs automatically
(`--logger {tensorboard,wandb,none}`).
```bash
tensorboard --logdir runs
```

### 5.7 Tables — `tables.py`
Publication tables in **Markdown** and **LaTeX (booktabs)**:

| Table | Columns | Source JSON |
|---|---|---|
| **Table 1 — Robust Accuracy** | `Method \| Sparsity \| Clean \| PGD-100 \| AutoAttack` | `metrics.py` |
| **Table 2 — RTI** | `Method \| Arch \| RTI \| nRTI` | `rti.py` |
| **Table 3 — Training Efficiency** | `Method \| FLOPs \| Time \| Robust Acc` | `exp5_adv_efficiency.py` |

```bash
python tables.py --metrics-files "results/*.json" \
                 --rti-files "results/rti*.json" \
                 --efficiency-files experiments_out/exp5_adv_efficiency.json \
                 --metric AutoAttack --out-dir tables
```
Writes `tables/table_1_robust_accuracy.{md,tex}`, `tables/table_2_rti.{md,tex}`
and `tables/table_3_training_efficiency.{md,tex}`.

---

## 6. Experimental protocol
One focused script per protocol item — see [experiments/README.md](experiments/README.md).

| # | Script | What it does |
|---|--------|--------------|
| 1 | `exp1_robustness.py` | Per method: prune & train on CIFAR-10, evaluate robustness on CIFAR-10 |
| 2 | `exp2_transfer.py` | **CORE:** *original* (target→target) vs *transferred* (source→target) mask; RTI + nRTI |
| 3 | `exp3_symmetric.py` | Mirror of 2 with source/target swapped |
| 4 | `exp4_sparsity_sweep.py` | The transfer pipeline across 20/30/50/68.37/90/95/99 % sparsity; logs RTI- and robustness-vs-sparsity plots |
| 5 | `exp5_adv_efficiency.py` | Dense vs YOPO-sparse, **adversarially trained**; robust accuracy, training time per epoch, total FLOPs |

```bash
python experiments/exp1_robustness.py
python experiments/exp2_transfer.py --models resnet20 --methods snip
python experiments/exp4_sparsity_sweep.py --models resnet20 --methods snip
python experiments/exp5_adv_efficiency.py --exp5-dataset cifar10
python experiments/exp1_robustness.py --config configs/smoke.yaml --no-autoattack   # fast CPU check
```

### What "the transfer pipeline" is (Experiments 2–4)
For each `(model, method, sparsity)` it builds and compares **two** models — both
trained and evaluated on the **target** dataset:

| Arm | Mask computed on | Trained + evaluated on | Meaning |
|---|---|---|---|
| **original** (native) | target | target | the fair baseline — mask born on the same data |
| **transferred** | **source** | target | the mask is *imported* from the source dataset |

It then computes RTI/nRTI between the two (D1 = original, D2 = transferred). A
**small RTI/nRTI** means the imported mask kept its robustness → it *transfers
well*. **Experiment 4** repeats this across the whole sparsity ladder and plots
RTI-vs-sparsity, answering *at which sparsities the mask still transfers*.

---

## 7. Transfer for any source → target pair
Set `--source dataset1` and `--target dataset2`, where each of `dataset1` /
`dataset2` ∈ `cifar10 | cifar100 | tinyimagenet`.

- **`--source`** = the dataset the pruning mask is **computed** on (D1).
- **`--target`** = the dataset the masked model is **trained and evaluated** on (D2).
- Add **`--tinyimagenet-dir ./tiny-imagenet-200`** only when either side is `tinyimagenet`.

```bash
# Transfer pipeline (Experiment 2):  source dataset1 → target dataset2
python experiments/exp2_transfer.py --source dataset1 --target dataset2 \
    --tinyimagenet-dir ./tiny-imagenet-200 --models resnet20 --methods yopo \
    --sparsity 0.9 --epochs 200 --device cuda

# Sparsity sweep (Experiment 4):  source dataset1 → target dataset2
python experiments/exp4_sparsity_sweep.py --source dataset1 --target dataset2 \
    --tinyimagenet-dir ./tiny-imagenet-200 --models resnet20 --methods yopo --device cuda
```

Concrete example — **CIFAR-10 → TinyImageNet** sparsity sweep:
```bash
python experiments/exp4_sparsity_sweep.py --source cifar10 --target tinyimagenet \
    --tinyimagenet-dir ./tiny-imagenet-200 --models resnet20 --methods yopo --device cuda
```

**Outputs:** `experiments_out/exp2_transfer_<source>_to_<target>.json` and
`experiments_out/exp4_sparsity_sweep.json`.
`--models` ∈ `resnet20 resnet18 vgg19`, `--methods` ∈ `snip grasp synflow dpai yopo`
(one or more each).

> **Cross-class-count transfer is handled automatically.** When `dataset1` and
> `dataset2` have a different number of classes (e.g. cifar10 ↔ tinyimagenet,
> cifar10 ↔ cifar100), the transferred mask keeps the **classifier head dense**
> (`utils.transfer_mask`, mask = ones) while the conv masks transfer as-is.
> Cross-dataset transfer is supported **only** through these experiment scripts,
> not the manual `main.py → train.py` path.

---

## 8. Full sweep over all models & methods
`exp4` consumes only the **first** `--models`/`--methods` entry, so to sweep
**every** model × method we loop over the pairs externally and give each its own
`--results-dir` (one JSON per combo, no overwrite). This produces 3 models × 5
methods = **15** runs, each sweeping the full 7-sparsity ladder.

> All commands below assume your shell's working directory is the
> `prune_models/` folder (`cd` into it first).

### 8.1 Run the sweep (CIFAR-10 → TinyImageNet)

**cmd** (run interactively; inside a `.bat` file double the loop vars — `%%M`, `%%P`):
```bat
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --source cifar10 --target tinyimagenet --results-dir experiments_out\full_c10_to_tin\%M\%P
```
**PowerShell:**
```powershell
foreach ($M in 'resnet20','resnet18','vgg19') { foreach ($P in 'snip','grasp','synflow','dpai','yopo') { python experiments\exp4_sparsity_sweep.py --models $M --methods $P --source cifar10 --target tinyimagenet --results-dir "experiments_out\full_c10_to_tin\$M\$P" } }
```
**Bash / Git Bash:**
```bash
for M in resnet20 resnet18 vgg19; do for P in snip grasp synflow dpai yopo; do python experiments/exp4_sparsity_sweep.py --models $M --methods $P --source cifar10 --target tinyimagenet --results-dir experiments_out/full_c10_to_tin/$M/$P; done; done
```
Each output JSON (`experiments_out\full_c10_to_tin\<model>\<method>\exp4_sparsity_sweep.json`)
has the same schema as the CIFAR-10 → CIFAR-100 case (see
[RUN_RTI_SWEEP.md](RUN_RTI_SWEEP.md) for the schema): one record per sparsity,
each with `rti.{clean, FGSM, PGD-10, PGD-20, PGD-100, AutoAttack}`.

### 8.2 Plots (one figure dir per model × method × attack)
```bat
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do for %A in (FGSM PGD-10 PGD-20 PGD-100 AutoAttack) do python scripts\plot_results.py --results-dir experiments_out\full_c10_to_tin\%M\%P --out-dir plots\full_c10_to_tin\%M\%P\%A --metric %A
```

### 8.3 Practical fast first pass (30 epochs, no AutoAttack)
```bat
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --source cifar10 --target tinyimagenet --epochs 30 --n-samples 500 --no-autoattack --num-workers 4 --logger none --results-dir experiments_out\full_c10_to_tin\%M\%P
```

### 8.4 Symmetric direction (TinyImageNet → CIFAR-10)
Just swap `--source` / `--target`:
```bat
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --source tinyimagenet --target cifar10 --results-dir experiments_out\full_tin_to_c10\%M\%P
```

### 8.5 Smoke test (1 epoch, no AA) — verify wiring first
Uses exp4's default datasets (CIFAR-10 → CIFAR-100):
```bat
for %M in (resnet20 resnet18 vgg19) do for %P in (snip grasp synflow dpai yopo) do python experiments\exp4_sparsity_sweep.py --models %M --methods %P --epochs 1 --no-autoattack --n-samples 256 --num-workers 0 --batch-size 128 --logger none --results-dir experiments_out\smoke\%M\%P
```
Produces one JSON per (model, method) very fast; confirm
`experiments_out\smoke\<model>\<method>\exp4_sparsity_sweep.json` exists for all
15 combos, then launch the real sweep.

### TinyImageNet notes
- **Target size:** TinyImageNet train = 100,000 images (CIFAR-100 = 50,000), so each epoch is ~2× slower than the CIFAR-100 sweep.
- **Classifier mismatch handled automatically:** a CIFAR-10 (10-class head) mask applied to a TinyImageNet model (200-class head) goes through `utils.transfer_mask`, which keeps the mismatched classifier layer **dense**; conv masks transfer as-is.
- **Normalization:** the loader uses TinyImageNet's own channel stats by default (`tinyimagenet_norm='tinyimagenet'` in `get_loaders`). Point `--tinyimagenet-dir` at your unzipped folder if it isn't at `./tiny-imagenet-200`.

### Full RTI / nRTI matrix
For the exact recipe to reproduce the **complete RTI / nRTI matrix** (5 attacks ×
5 pruning methods × 7 sparsities × 3 models, CIFAR-10 → CIFAR-100), see
[RUN_RTI_SWEEP.md](RUN_RTI_SWEEP.md) — single-line cmd / PowerShell / bash
commands, output schema, plotting, and a smoke variant.

---

## 9. Final deliverables
| File | What it does |
|------|--------------|
| `scripts/run_all_experiments.sh` | Runs exp1..exp5 sequentially against a config (default `configs/cifar10.yaml`); writes JSONs to `experiments_out/` and per-experiment logs to `experiments_out/logs/`. Failures don't stop the sweep — there's a SUMMARY at the end. |
| `scripts/plot_results.py` | Reads `experiments_out/exp{1..5}_*.json` and writes publication-quality PNGs to `plots/`: per-method robust accuracy (exp1), RTI/nRTI bars (exp2/exp3), RTI- and robust-acc-vs-sparsity lines (exp4), grouped efficiency bars (exp5). |

```bash
bash scripts/run_all_experiments.sh                          # all experiments, cifar10
bash scripts/run_all_experiments.sh configs/cifar100.yaml
EXPS="1 4" bash scripts/run_all_experiments.sh               # only experiments 1 and 4
EXTRA="--no-autoattack --epochs 50" bash scripts/run_all_experiments.sh
python scripts/plot_results.py --metric AutoAttack           # writes plots/*.png
```

---

## 10. Reproducibility
- **Fixed seeds** — every entry script calls `set_seed(seed)` (Python/NumPy/Torch
  + CUDA); `--seed` sets it, `--deterministic` forces deterministic cuDNN/algorithms.
- **Masks & checkpoints saved** — `main.py` writes `state_dict` + `masks` + metadata
  (`model`, `pruner`, `dataset`, `target/achieved sparsity`, `seed`); `train.py`
  writes the best checkpoint with the same provenance, so it is self-describing for
  `attacks.py` / `metrics.py`.
- **Config files per experiment** — `configs/*.yaml` (see `configs/README.md`). Every
  script takes `--config`; precedence is **defaults < config < CLI flags**.
```bash
python main.py    --config configs/cifar10.yaml
python train.py   --config configs/cifar10.yaml --checkpoint output/cifar10/resnet20_snip_s0_9.pth
python metrics.py --config configs/cifar10.yaml --checkpoint trained/resnet20_snip_cifar10_best.pth
```

---

## 11. CLI reference

### Common flags
| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `cifar10` | `cifar10` \| `cifar100` \| `tinyimagenet` |
| `--models` | all | subset of `resnet20 resnet18 vgg19` |
| `--pruners` / `--methods` | all | subset of `snip grasp synflow dpai yopo` |
| `--sparsity` | `0.9` | one or more target sparsities (fraction removed) |
| `--device` | cuda if available | `cpu` / `cuda` |
| `--seed` | `1` | RNG seed (`--deterministic` for strict reproducibility) |
| `--data-root` | `./data` | CIFAR download/cache dir |
| `--tinyimagenet-dir` | `./tiny-imagenet-200` | unzipped TinyImageNet root |

### Training (`train.py`)
| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs` | `200` | training epochs |
| `--lr` / `--momentum` / `--weight-decay` | `0.1` / `0.9` / `5e-4` | SGD hyperparameters |
| `--no-nesterov` | off | disable Nesterov momentum |
| `--amp` | `auto` | mixed precision: `auto` (on iff CUDA) / `on` / `off` |
| `--adv-train` | off | enable PGD adversarial training |
| `--adv-eps` / `--adv-alpha` / `--adv-steps` | `8/255` / `2/255` / **`10`** | PGD-AT threat model |

### Attacks / metrics (`attacks.py`, `metrics.py`)
| Flag | Default | Meaning |
|------|---------|---------|
| `--eps` / `--alpha` | `8/255` / `2/255` | L∞ budget / PGD step |
| `--pgd-steps` | `10 20 100` | PGD step counts to evaluate |
| `--no-autoattack` | off | skip AutoAttack (slow) |
| `--aa-version` | `standard` | `standard` / `plus` / `rand` |
| `--n-samples` | `1000` | test images (0 = full set) |
| `--top5` | off | also report clean top-5 (metrics.py) |

### RTI (`rti.py`)
| Flag | Default | Meaning |
|------|---------|---------|
| `--nrti-eps` | `0.01` (τ = 1%) | nRTI denominator floor `max(A_rob(D1), ε)` (fraction) |

### Pruning knobs
| Flag | Default | Meaning |
|------|---------|---------|
| `--dpai-steps` | `100` | DPaI score-optimisation iterations |
| `--grasp-samples-per-class` | `10` | samples GraSP draws per class |
| `--yopo-search-steps` | `30` | std-factor search resolution for YOPO |

---

## 12. Notes & caveats
- **TinyImageNet** is not on torchvision. The easiest way to get it is
  `python download_dataset.py --datasets tinyimagenet`, which downloads
  `http://cs231n.stanford.edu/tiny-imagenet-200.zip` (~240 MB) and unzips it to
  `./tiny-imagenet-200`. Resulting layout:
  ```
  tiny-imagenet-200/
  ├── train/<wnid>/images/*.JPEG
  ├── val/images/*.JPEG
  └── val/val_annotations.txt
  ```
  Images are resized to 32×32 so the CIFAR-sized models apply unchanged. Point
  `--tinyimagenet-dir` at the folder if it lives somewhere else.
- **SynFlow** is data-free and runs its standard 100-step exponential schedule;
  models are auto-converted to SynFlow's masked layers internally.
- **DPaI / YOPO** are data-free; **YOPO needs `scikit-learn`** for the NMF.
- **Achieved sparsity** can differ slightly from the target for **YOPO**
  (threshold search is approximate) and for SynFlow at extreme sparsities.
- **`exp4` is single-combo:** it uses only the first `--models`/`--methods` entry
  and always writes `exp4_sparsity_sweep.json` (overwrites on re-run). Use a
  per-combo `--results-dir` when sweeping many pairs ([§8](#8-full-sweep-over-all-models--methods)).
- **Robust failures are isolated:** in `main.py` and `run_all_experiments.sh`,
  a failing `(model, pruner, sparsity)` cell doesn't abort the whole sweep — see
  the final SUMMARY for per-cell status.
- **Mask transfer across class counts** keeps the classifier head dense
  ([§7](#7-transfer-for-any-source--target-pair)); the manual `main.py → train.py`
  path only supports same-class-count cases.
