# Experiment configs

Versioned config files for reproducible runs. Every script accepts `--config`:

```bash
python main.py    --config configs/cifar10.yaml          # prune sweep
python train.py   --config configs/cifar10.yaml --checkpoint output/cifar10/resnet20_snip_s0_9.pth
python attacks.py --config configs/cifar10.yaml --checkpoint trained/resnet20_snip_cifar10_best.pth
python metrics.py --config configs/cifar10.yaml --checkpoint trained/resnet20_snip_cifar10_best.pth
```

Precedence: **argparse defaults < config file < command-line flags**, so any key
can be overridden on the CLI (e.g. `--sparsity 0.99`).

## How it works
- Keys are argparse `dest` names (underscores), e.g. `batch_size`, `pgd_steps`.
- One shared config drives all scripts; a script silently ignores keys it
  doesn't define (it prints which keys it skipped).
- `eps`/`alpha` are intentionally omitted so the exact code defaults (8/255,
  2/255) are used; add them if you want a different budget.
- `device` is omitted so it auto-selects CUDA when available, else CPU.

## Files
| Config | Purpose |
|--------|---------|
| `cifar10.yaml` | CIFAR-10 full experiment |
| `cifar100.yaml` | CIFAR-100 full experiment |
| `tinyimagenet.yaml` | TinyImageNet (→32×32) full experiment |
| `smoke.yaml` | tiny/fast CPU sanity run |
