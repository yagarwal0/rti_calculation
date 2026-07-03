"""
attacks.py — adversarial robustness evaluation (L_inf).

Attacks:
    * FGSM                       single-step FGSM  (hand-written, pixel space [0,1])
    * PGD-10 / PGD-20 / PGD-100  L_inf PGD, random start (hand-written, pixel space)
    * AutoAttack (MANDATORY)     official AutoAttack library (fra31/auto-attack):
                                 standard ensemble APGD-CE + APGD-T + FAB-T + Square

FGSM and PGD are implemented here directly (no third-party dependency). AutoAttack
uses the official `autoattack` package (https://github.com/fra31/auto-attack) via
`run_standard_evaluation`, imported on demand so FGSM/PGD run even when it is not
installed (pass --no-autoattack). Install AutoAttack with:
    pip install git+https://github.com/fra31/auto-attack

All attacks operate in raw [0, 1] pixel space. Because the training data is
per-channel normalised, the model is wrapped in `NormalizedModel`, which applies
`(x - mean) / std` *inside* the forward pass — so the attacks see valid [0, 1]
images and the eps-budget is the standard pixel-space budget. Each PGD step
projects the perturbation to the eps-ball (`clamp(delta, -eps, +eps)`) and the
image to valid pixels (`clamp(images + delta, 0, 1)`).

Default threat model: L_inf, eps = 8/255, PGD step alpha = 2/255 (the standard
CIFAR benchmark).

Examples
--------
    # Evaluate a pruned (or trained) checkpoint with all attacks
    python attacks.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --n-samples 1000

    # Skip AutoAttack while iterating (it is slow), keep FGSM + PGD
    python attacks.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --no-autoattack

    # Evaluate a bare model on cifar100
    python attacks.py --model vgg19 --dataset cifar100 --weights some_model.pth
"""

import os
import sys
import json
import argparse
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from dataloader import (CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD,
                        TINYIMAGENET_MEAN, TINYIMAGENET_STD, TinyImageNetValDataset)
from utils import set_seed, parse_args_with_config

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}
NORM_STATS = {
    'cifar10': (CIFAR10_MEAN, CIFAR10_STD),
    'cifar100': (CIFAR100_MEAN, CIFAR100_STD),
    'tinyimagenet': (TINYIMAGENET_MEAN, TINYIMAGENET_STD),
}

EPS = 8.0 / 255.0      # L_inf budget for FGSM/PGD (standard CIFAR benchmark)
ALPHA = 2.0 / 255.0    # PGD step size
AA_EPS = 4.0 / 255.0   # L_inf budget for AutoAttack (separate; override with --aa-eps)


# ---------------------------------------------------------------------------
# Normalisation wrapper: attacks feed [0,1] images; normalisation is internal.
# ---------------------------------------------------------------------------
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


# ---------------------------------------------------------------------------
# Clean [0,1] test loaders (NO Normalize — the wrapper does that).
# ---------------------------------------------------------------------------
def get_clean_test_loader(dataset, batch_size=128, n_samples=0, num_workers=2,
                          data_root='./data', tinyimagenet_dir='./tiny-imagenet-200'):
    """Return (loader, mean, std, num_classes) with images in [0, 1]."""
    name = dataset.lower()
    if name == 'cifar10':
        tfm = transforms.ToTensor()
        ds = CIFAR10(root=data_root, train=False, download=True, transform=tfm)
    elif name == 'cifar100':
        tfm = transforms.ToTensor()
        ds = CIFAR100(root=data_root, train=False, download=True, transform=tfm)
    elif name == 'tinyimagenet':
        from torchvision.datasets import ImageFolder
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        train_ds = ImageFolder(os.path.join(tinyimagenet_dir, 'train'))  # class map only
        ds = TinyImageNetValDataset(os.path.join(tinyimagenet_dir, 'val'),
                                    class_to_idx=train_ds.class_to_idx, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    if n_samples and n_samples > 0:
        ds = Subset(ds, list(range(min(n_samples, len(ds)))))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    mean, std = NORM_STATS[name]
    return loader, mean, std, NUM_CLASSES[name]


# ---------------------------------------------------------------------------
# FGSM / PGD — hand-written, L_inf, pixel space [0, 1]
# `model` accepts [0,1] inputs (a NormalizedModel: normalization is internal),
# so the clamp is just: project delta to the eps-ball and the image to [0,1] on
# every step. No per-channel (0-mean)/std .. (1-mean)/std bounds are needed.
# ---------------------------------------------------------------------------
def fgsm_attack(model, images, labels, eps):
    """Single-step FGSM in [0,1] pixel space. Returns adversarial images in [0,1]."""
    images = images.clone().detach()
    images.requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(images), labels)
    grad = torch.autograd.grad(loss, images)[0]
    adv = torch.clamp(images + eps * grad.sign(), 0.0, 1.0)
    return adv.detach()


def pgd_attack(model, images, labels, eps, alpha, steps, random_start=True):
    """L_inf PGD (random start) in [0,1] pixel space. Returns images in [0,1].

    Two clamps every step: delta -> eps-ball (clamp(delta, -eps, +eps)) and
    image -> valid pixels (clamp(images + delta, 0, 1))."""
    images = images.clone().detach()
    if random_start:
        delta = torch.empty_like(images).uniform_(-eps, eps)
        delta.data = torch.clamp(images + delta, 0.0, 1.0) - images
    else:
        delta = torch.zeros_like(images)
    delta.requires_grad_(True)

    for _ in range(steps):
        loss = nn.CrossEntropyLoss()(model(images + delta), labels)
        grad = torch.autograd.grad(loss, delta)[0]
        delta.data = delta + alpha * grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)                   # eps-ball
        delta.data = torch.clamp(images + delta.data, 0.0, 1.0) - images   # valid image
    return (images + delta).detach()


# ---------------------------------------------------------------------------
# AutoAttack — official library (fra31/auto-attack), run in pixel space [0,1].
# Wrapped as a callable(x, y) -> x_adv so it plugs into robust_accuracy's loop;
# each call runs the full standard ensemble (APGD-CE + APGD-T + FAB-T + Square)
# via run_standard_evaluation — the same API used by evaluate_autoattack.py.
#   install:  pip install git+https://github.com/fra31/auto-attack
# ---------------------------------------------------------------------------
class _AutoAttackRunner:
    """Callable wrapper around the official AutoAttack (fra31/auto-attack).

    The model must accept [0,1] inputs (a NormalizedModel); AutoAttack clamps to
    [0,1] internally, so the valid-image constraint and the eps-ball are both in
    pixel space.
    """

    def __init__(self, model, eps=EPS, version='standard', seed=None, verbose=False):
        try:
            from autoattack import AutoAttack
        except ImportError as e:
            raise ImportError(
                "The official AutoAttack package is required. Install it with:\n"
                "    pip install git+https://github.com/fra31/auto-attack\n"
                "or pass --no-autoattack (FGSM/PGD need no extra dependency).") from e
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        self.adversary = AutoAttack(model, norm='Linf', eps=eps, version=version,
                                    seed=seed, verbose=verbose, device=device)

    def __call__(self, x, y):
        # full standard ensemble on this batch, in [0,1] pixel space
        return self.adversary.run_standard_evaluation(x, y, bs=x.shape[0])


# ---------------------------------------------------------------------------
# Build the attack suite. FGSM/PGD use the hand-written functions above (no
# extra deps); AutoAttack uses the official `autoattack` package, imported only
# when requested so FGSM/PGD work even if it is not installed.
# ---------------------------------------------------------------------------
def build_attacks(model, num_classes, eps=EPS, alpha=ALPHA,
                  pgd_steps=(10, 20, 100), include_autoattack=True,
                  aa_version='standard', seed=0, aa_eps=AA_EPS):
    """Return an ordered dict {name: callable(x, y) -> x_adv}. `model` must accept
    [0,1] inputs (i.e. a NormalizedModel). FGSM/PGD use `eps` (default 8/255);
    AutoAttack uses `aa_eps` (default 4/255) — separate so the two budgets can be
    set independently from the CLI (`--eps` and `--aa-eps`)."""
    from functools import partial

    attacks = {}
    attacks['FGSM'] = partial(fgsm_attack, model, eps=eps)
    for steps in pgd_steps:
        attacks[f'PGD-{steps}'] = partial(pgd_attack, model, eps=eps,
                                          alpha=alpha, steps=steps)
    if include_autoattack:
        # official AutoAttack ensemble (APGD-CE + APGD-T + FAB-T + Square)
        attacks['AutoAttack'] = _AutoAttackRunner(
            model, eps=aa_eps, version=aa_version, seed=seed)
    return attacks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def robust_accuracy(model, loader, attack, device):
    """Accuracy on adversarial examples produced by `attack`."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)                  # torchattacks returns [0,1] adv images
        with torch.no_grad():
            correct += model(x_adv).argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_robustness(model, loader, device, num_classes, eps=EPS, alpha=ALPHA,
                        pgd_steps=(10, 20, 100), include_autoattack=True,
                        aa_version='standard', verbose=True, seed=0,
                        aa_eps=AA_EPS):
    """Run clean + all attacks. Returns a results dict."""
    model = model.to(device).eval()
    attacks = build_attacks(model, num_classes, eps, alpha, pgd_steps,
                            include_autoattack, aa_version, seed=seed,
                            aa_eps=aa_eps)

    clean = clean_accuracy(model, loader, device)
    if verbose:
        print(f"  clean       : {clean:6.2f}%")

    results = {'clean_accuracy': round(clean, 4), 'attacks': {}}
    for name, atk in attacks.items():
        acc = robust_accuracy(model, loader, atk, device)
        asr = round(100.0 * (clean - acc) / clean, 4) if clean > 0 else 0.0
        results['attacks'][name] = {'robust_accuracy': round(acc, 4),
                                    'attack_success_rate': asr}
        if verbose:
            print(f"  {name:<11} : {acc:6.2f}%   (ASR {asr:6.2f}%)")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_model(args):
    """Build the model and load weights. Supports checkpoints written by
    main.py / train.py (with metadata) or a bare state_dict via --weights."""
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
    p = argparse.ArgumentParser(description="Adversarial robustness eval: FGSM, PGD, AutoAttack.")
    p.add_argument('--checkpoint', default=None, help='checkpoint from main.py / train.py')
    p.add_argument('--model', default='resnet20', help='model when not using --checkpoint')
    p.add_argument('--weights', default=None, help='bare state_dict to load into --model')
    p.add_argument('--dataset', default='cifar10', choices=list(NUM_CLASSES))
    p.add_argument('--eps', type=float, default=EPS, help='L_inf budget for FGSM/PGD (default 8/255)')
    p.add_argument('--alpha', type=float, default=ALPHA, help='PGD step size (default 2/255)')
    p.add_argument('--aa-eps', type=float, default=AA_EPS,
                   help='L_inf budget for AutoAttack (default 4/255)')
    p.add_argument('--pgd-steps', nargs='+', type=int, default=[10, 20, 100])
    p.add_argument('--no-autoattack', action='store_true', help='skip AutoAttack (it is slow)')
    p.add_argument('--aa-version', default='standard', choices=['standard', 'plus', 'rand'])
    p.add_argument('--n-samples', type=int, default=1000,
                   help='number of test images (0 = full test set; AutoAttack is slow)')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--tinyimagenet-dir', default='./tiny-imagenet-200')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--results-file', default=None, help='optional JSON path to append results')
    p.add_argument('--config', default=None, help='YAML/JSON config file (CLI args override it)')
    args = parse_args_with_config(p)

    set_seed(args.seed)
    device = torch.device(args.device)
    model, dataset, model_name, num_classes, meta = _load_model(args)

    loader, mean, std, num_classes = get_clean_test_loader(
        dataset, batch_size=args.batch_size, n_samples=args.n_samples,
        num_workers=args.num_workers, data_root=args.data_root,
        tinyimagenet_dir=args.tinyimagenet_dir)
    wrapped = NormalizedModel(model, mean, std).to(device).eval()

    print("=" * 70)
    print(f"Robustness eval | {model_name} | {dataset} | n={args.n_samples or 'full'}")
    print(f"L_inf eps={args.eps:.5f} (~{args.eps*255:.2f}/255), alpha={args.alpha:.5f}, "
          f"PGD steps={args.pgd_steps}, AutoAttack={'off' if args.no_autoattack else args.aa_version}")
    if meta:
        print(f"checkpoint meta: {meta}")
    print("=" * 70)

    results = evaluate_robustness(
        wrapped, loader, device, num_classes,
        eps=args.eps, alpha=args.alpha, pgd_steps=tuple(args.pgd_steps),
        include_autoattack=not args.no_autoattack, aa_version=args.aa_version,
        seed=args.seed, aa_eps=args.aa_eps)

    if args.results_file:
        payload = {'model': model_name, 'dataset': dataset, 'eps': args.eps,
                   'eps_over_255': round(args.eps * 255, 4), 'alpha': args.alpha,
                   'pgd_steps': args.pgd_steps, 'n_samples': args.n_samples,
                   'checkpoint': args.checkpoint, 'checkpoint_meta': meta,
                   'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                   **results}
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
        print(f"\nResults appended to {args.results_file}")


if __name__ == '__main__':
    main()
