"""
train.py — standard training pipeline.

  * Optimizer : SGD with momentum  (lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov)
  * Schedule  : step decay (SNIP paper) — MultiStepLR, lr x0.1 at 50% & 75% of
                epochs by default ('--lr-schedule cosine' restores cosine annealing)
  * Loss      : cross-entropy
  * Epochs    : 200 (configurable via --epochs)
  * AMP       : enabled automatically on CUDA

Works for dense models and for pruned models produced by `main.py`. When a
pruned checkpoint (or a mask list) is supplied, the pruned weights are held at
zero throughout training by masking their gradients each step (and re-applying
the mask after the optimizer step), so only the surviving sub-network is trained.

Examples
--------
    # Train a dense model from scratch
    python train.py --model resnet20 --dataset cifar10 --epochs 200

    # Train the sparse sub-network of a pruned checkpoint (mask kept fixed)
    python train.py --checkpoint output/cifar10/resnet20_snip_s0_9.pth --epochs 200
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from dataloader import get_loaders
from attacks import NormalizedModel, NORM_STATS
from utils import (prunable_modules, apply_masks_aligned, prunable_sparsity,
                   set_seed, parse_args_with_config)

# CIFAR/TinyImageNet class counts (used when training a fresh model).
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}


# ---------------------------------------------------------------------------
# AMP helpers (new torch.amp API). Mixed precision is OPTIONAL:
#   amp='auto' -> on iff CUDA, 'on' -> force on, 'off' -> disable.
# ---------------------------------------------------------------------------
def amp_enabled(device, amp='auto'):
    # Tolerate bools (e.g. YAML coercing 'on'/'off' to True/False).
    if amp in ('on', True):
        return True
    if amp in ('off', False):
        return False
    return device.type == 'cuda'      # 'auto'


def _autocast(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


# ---------------------------------------------------------------------------
# Optimizer / scheduler (SGD + momentum, cosine schedule)
# ---------------------------------------------------------------------------
def make_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                     weight_decay=weight_decay, nesterov=nesterov)


def make_scheduler(optimizer, epochs, schedule='multistep', milestones=None, gamma=0.1):
    """Build the LR scheduler.

    SNIP-paper default ('multistep'): MultiStepLR, lr x gamma at `milestones`
    (default = 50% and 75% of total epochs). 'cosine' = CosineAnnealingLR(T_max=epochs);
    'constant' = no decay.
    """
    schedule = (schedule or 'multistep').lower()
    if schedule == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if schedule == 'constant':
        return MultiStepLR(optimizer, milestones=[], gamma=1.0)
    # 'multistep' / 'step' (SNIP): step decay at the milestone epochs
    if not milestones:
        milestones = [int(0.5 * epochs), int(0.75 * epochs)]
    return MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)


# ---------------------------------------------------------------------------
# Mask handling — freeze pruned weights at zero
# ---------------------------------------------------------------------------
def mask_gradients(model, masks):
    """Zero the gradients of pruned weights so SGD never revives them."""
    for m, mk in zip(prunable_modules(model), masks):
        if m.weight.grad is not None:
            m.weight.grad.mul_(mk.to(m.weight.grad.device))


# ---------------------------------------------------------------------------
# One epoch / evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, masks=None,
                    amp_on=True, adv_cfg=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if adv_cfg is not None:                 # PGD adversarial training ([0,1] space)
            from adv_train import pgd_perturb
            inputs = pgd_perturb(model, inputs, targets,
                                 eps=adv_cfg['eps'], alpha=adv_cfg['alpha'],
                                 steps=adv_cfg['steps'])

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp_on):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if masks is not None:
            mask_gradients(model, masks)        # keep pruned weights frozen
        scaler.step(optimizer)
        scaler.update()
        if masks is not None:
            apply_masks_aligned(model, masks)   # re-assert exact zeros

        total_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_on=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with _autocast(device, amp_on):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
def train(model, train_loader, test_loader, device, epochs=200, lr=0.1,
          momentum=0.9, weight_decay=5e-4, nesterov=True,
          lr_schedule='multistep', lr_milestones=None, lr_gamma=0.1, masks=None,
          save_path=None, verbose=True, logger=None, amp='auto', extra_meta=None,
          norm=None, adv_cfg=None):
    """Standard (or adversarial) SGD + cosine training. Returns (best_acc, history).

    `norm` ((mean, std) or None): if given, the model is wrapped in
    `NormalizedModel` for the forward pass so it consumes [0, 1] images and
    normalizes internally — matching the FGSM/PGD/AutoAttack convention. The
    *unwrapped* weights are what gets saved, so checkpoints still load via
    `build_model(...).load_state_dict(...)`.

    `amp` ('auto'|'on'|'off') controls optional mixed precision. `adv_cfg`, if
    given ({'eps','alpha','steps'}), enables PGD adversarial training in [0, 1]
    space (the model normalizes internally). `extra_meta` (dict) is embedded into
    saved checkpoints (e.g. model/dataset/seed) so they are self-describing for
    downstream attack/metric evaluation. If `logger` is given, accuracy/loss/lr
    curves are logged each epoch.

    Each history entry also records 'time' (seconds for that epoch)."""
    extra_meta = extra_meta or {}
    core = model.to(device)
    # Match the attack convention: feed [0,1] images and normalize INSIDE the
    # model. We train the wrapped `net` but save the unwrapped `core` weights.
    net = NormalizedModel(core, norm[0], norm[1]).to(device) if norm is not None else core
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(net, lr, momentum, weight_decay, nesterov)
    scheduler = make_scheduler(optimizer, epochs, schedule=lr_schedule,
                               milestones=lr_milestones, gamma=lr_gamma)
    amp_on = amp_enabled(device, amp)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)
    if verbose:
        _ms = lr_milestones or [int(0.5 * epochs), int(0.75 * epochs)]
        _sched = (f"multistep (x{lr_gamma} at {_ms})"
                  if (lr_schedule or 'multistep').lower() in ('multistep', 'step')
                  else (lr_schedule or 'multistep'))
        print(f"LR: init {lr}, schedule = {_sched}.")
        print(f"Mixed precision (AMP): {'on' if amp_on else 'off'} (mode={amp})."
              + ("  Adversarial training: PGD-%d." % adv_cfg['steps'] if adv_cfg else ""))

    if masks is not None:
        masks = [mk.to(device) for mk in masks]
        apply_masks_aligned(net, masks)   # ensure we start exactly sparse
        if verbose:
            print(f"Training sparse sub-network (sparsity {prunable_sparsity(net)*100:.2f}%).")

    best_acc, history = 0.0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(net, train_loader, optimizer,
                                          criterion, device, scaler, masks, amp_on,
                                          adv_cfg=adv_cfg)
        scheduler.step()
        epoch_time = time.time() - t0
        te_loss, te_acc = evaluate(net, test_loader, criterion, device, amp_on)
        best_acc = max(best_acc, te_acc)
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
                        'test_loss': te_loss, 'test_acc': te_acc,
                        'lr': scheduler.get_last_lr()[0], 'time': epoch_time})

        if logger is not None:
            logger.log_accuracy(epoch, train_acc=tr_acc, test_acc=te_acc,
                                lr=scheduler.get_last_lr()[0],
                                train_loss=tr_loss, test_loss=te_loss)

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | lr {scheduler.get_last_lr()[0]:.4f} | "
                  f"train {tr_acc:5.2f}% ({tr_loss:.3f}) | "
                  f"test {te_acc:5.2f}% ({te_loss:.3f}) | best {best_acc:5.2f}% | "
                  f"{time.time()-t0:.1f}s")

        if save_path and te_acc >= best_acc:
            torch.save({'state_dict': core.state_dict(),   # save UNWRAPPED weights
                        'masks': [mk.detach().cpu() for mk in masks] if masks else None,
                        'epoch': epoch, 'test_acc': te_acc, **extra_meta}, save_path)

    if verbose:
        print(f"Done. Best test accuracy: {best_acc:.2f}%")
    return best_acc, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Standard SGD + cosine training (200 epochs).")
    # what to train
    p.add_argument('--checkpoint', default=None,
                   help='pruned .pth from main.py; trains its sparse sub-network with the mask fixed')
    p.add_argument('--model', default='resnet20', help='model when not using --checkpoint')
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'tinyimagenet'])
    # optimisation
    p.add_argument('--epochs', type=int, default=200, help='number of epochs (default: 200)')
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--no-nesterov', action='store_true', help='disable Nesterov momentum')
    # LR schedule (SNIP paper: step decay)
    p.add_argument('--lr-schedule', default='multistep',
                   choices=['multistep', 'cosine', 'constant'],
                   help='LR schedule (default: multistep step-decay, per SNIP paper)')
    p.add_argument('--lr-milestones', nargs='+', type=int, default=None,
                   help='epochs to decay LR at (default: 50%% and 75%% of --epochs)')
    p.add_argument('--lr-gamma', type=float, default=0.1,
                   help='LR multiplicative decay at each milestone (default: 0.1)')
    # data / misc
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--tinyimagenet-dir', default='./tiny-imagenet-200')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', default='auto', choices=['auto', 'on', 'off'],
                   help='mixed precision: auto (on iff CUDA), on, or off')
    # adversarial training (optional; used by Experiment 5)
    p.add_argument('--adv-train', action='store_true', help='PGD adversarial training')
    p.add_argument('--adv-eps', type=float, default=8 / 255)
    p.add_argument('--adv-alpha', type=float, default=2 / 255)
    p.add_argument('--adv-steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--deterministic', action='store_true',
                   help='force deterministic cuDNN/algorithms (slower, reproducible)')
    p.add_argument('--output-dir', default='./trained')
    # logging
    p.add_argument('--logger', default='tensorboard',
                   choices=['tensorboard', 'wandb', 'none'],
                   help='logging backend for accuracy curves (default: tensorboard)')
    p.add_argument('--log-dir', default='./runs')
    p.add_argument('--config', default=None, help='YAML/JSON config file (CLI args override it)')
    args = parse_args_with_config(p)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    masks = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        dataset = ckpt.get('dataset', args.dataset)
        model_name = ckpt.get('model', args.model)
        num_classes = ckpt.get('num_classes', NUM_CLASSES[dataset])
        model = build_model(model_name, num_classes)
        model.load_state_dict(ckpt['state_dict'])
        masks = ckpt.get('masks', None)
        tag = f"{model_name}_{ckpt.get('pruner', 'pruned')}_{dataset}"
        print(f"Loaded pruned checkpoint: {args.checkpoint} "
              f"({model_name}, {dataset}, target sparsity "
              f"{ckpt.get('target_sparsity', '?')})")
    else:
        dataset = args.dataset
        model_name = args.model
        num_classes = NUM_CLASSES[dataset]
        model = build_model(model_name, num_classes)
        tag = f"{model_name}_dense_{dataset}"
        print(f"Training fresh dense model: {model_name} on {dataset}")

    train_loader, test_loader, _ = get_loaders(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, tinyimagenet_dir=args.tinyimagenet_dir)

    logger = None
    if args.logger not in (None, 'none'):
        from logger import Logger
        logger = Logger(backend=args.logger, log_dir=args.log_dir, run_name=tag,
                        config=vars(args))

    # Embed provenance so the trained checkpoint is self-describing for
    # downstream attack/metric evaluation.
    extra_meta = {'model': model_name, 'dataset': dataset, 'num_classes': num_classes,
                  'pruner': (ckpt.get('pruner') if args.checkpoint else 'dense'),
                  'seed': args.seed}

    adv_cfg = None
    if args.adv_train:
        adv_cfg = {'eps': args.adv_eps, 'alpha': args.adv_alpha,
                   'steps': args.adv_steps}

    save_path = os.path.join(args.output_dir, f"{tag}_best.pth")
    train(model, train_loader, test_loader, device,
          epochs=args.epochs, lr=args.lr, momentum=args.momentum,
          weight_decay=args.weight_decay, nesterov=not args.no_nesterov,
          lr_schedule=args.lr_schedule, lr_milestones=args.lr_milestones,
          lr_gamma=args.lr_gamma,
          masks=masks, save_path=save_path, logger=logger, amp=args.amp,
          extra_meta=extra_meta, norm=NORM_STATS[dataset], adv_cfg=adv_cfg)
    if logger is not None:
        logger.close()
    print(f"Best checkpoint saved to: {save_path}")


if __name__ == '__main__':
    main()
