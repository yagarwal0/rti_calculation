"""
DPaI (Data-Path-aware Initialization) pruner – standalone, self-contained.

Original implementation from: DPaI Original/  (distilled NPB algorithm of
DPaI/npb.py, packaged as DPaI_cifar100_to_cifar10/dpai_pruner.py).

Implements the NPB (Node-Path Balancing) pruning-at-initialization algorithm
from the DPaI codebase, distilled into a single module that exposes:

    masks = dpai_prune(model, sparsity, device, num_steps=500)

The masks list parallels every Conv2d / Linear layer in the model
(in module-iteration order).

DPaI is data-free: it forwards an all-ones tensor, treats the network as a
flow graph, and optimises a per-weight score so that the number of *effective
paths* (and, via alpha/beta, effective nodes and kernels) through the sparse
sub-network is maximised at the target ERK-distributed sparsity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# TopK straight-through estimator
# ---------------------------------------------------------------------------
class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        flat_out = out.flatten()
        flat_out[idx[:k]] = 0.0
        flat_out[idx[k:]] = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


# ---------------------------------------------------------------------------
# ERK (Erdos-Renyi-Kernel) layerwise sparsity distribution
# ---------------------------------------------------------------------------
def ERK_sparsify(model, sparsity=0.9):
    density = 1 - sparsity
    erk_power_scale = 1

    total_params = sum(m.score.numel() for m in model.NPB_modules)
    is_epsilon_valid = False
    dense_layers = set()

    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        for m in model.NPB_modules:
            m.raw_probability = 0
            n_param = np.prod(m.score.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density
            if m in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                m.raw_probability = (np.sum(m.score.shape) / np.prod(m.score.shape)) ** erk_power_scale
                divisor += m.raw_probability * n_param

        epsilon = rhs / divisor
        max_prob = max(m.raw_probability for m in model.NPB_modules)
        if max_prob * epsilon > 1:
            is_epsilon_valid = False
            for m in model.NPB_modules:
                if m.raw_probability == max_prob:
                    dense_layers.add(m)
        else:
            is_epsilon_valid = True

    total_nonzero = 0.0
    for m in model.NPB_modules:
        n_param = np.prod(m.score.shape)
        if m in dense_layers:
            m.sparsity = 0
        else:
            probability_one = epsilon * m.raw_probability
            m.sparsity = 1 - probability_one
        m.num_zeros = int(m.sparsity * m.score.numel())
        total_nonzero += (1 - m.sparsity) * m.score.numel()
    print(f"  ERK overall sparsity {1 - total_nonzero / total_params:.6f}")


# ---------------------------------------------------------------------------
# Mask / weight helpers
# ---------------------------------------------------------------------------
def get_mask_by_score(self):
    return TopK.apply(self.score.abs(), self.num_zeros)


def get_masked_weight(self):
    if self.learn_mask:
        self.register_buffer('mask', self.get_mask())
        return self.mask * self.weight
    else:
        return self.mask * self.weight


def get_weight(self):
    return self.weight


def linear_forward(self, x, weight, bias):
    return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# NPB forward pass (counts effective paths)
# ---------------------------------------------------------------------------
def NPB_forward(self, x):
    if self.npb:
        if self.learn_mask:
            self.register_buffer('mask', self.get_mask())
        self.max_paths = x.max()
        x = self.base_func(x / self.max_paths, self.mask, None)
        return x
    else:
        return self.base_func(x, self.get_weight(), self.bias)


def NPB_dummy_forward(self, x):
    if self.npb:
        return x
    else:
        return self.original_forward(x)


# ---------------------------------------------------------------------------
# NPB registration – patches every Conv2d / Linear
# ---------------------------------------------------------------------------
def NPB_register(model, sparsity, device):
    model.apply(lambda m: setattr(m, "npb", False))
    model.apply(lambda m: setattr(m, "learn_mask", False))
    NPB_modules = []

    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            NPB_modules.append(m)
            setattr(m, 'original_forward', m.forward)

            m.score = nn.Parameter(torch.empty_like(m.weight), requires_grad=True).to(device)
            nn.init.normal_(m.score, 0, 1)
            setattr(m, 'get_weight', get_masked_weight.__get__(m, m.__class__))
            setattr(m, 'get_mask', get_mask_by_score.__get__(m, m.__class__))

            m.sparsity = sparsity
            m.num_zeros = int(m.sparsity * m.score.numel())
            m.register_buffer('mask', m.get_mask())

            if isinstance(m, nn.Linear):
                m.dim_in = (0,)
                m.dim_out = (1,)
                m.view_in = (1, -1)
                m.view_out = (-1, 1)
                setattr(m, 'base_func', linear_forward.__get__(m, m.__class__))
            else:
                m.dim_in = (0, 2, 3)
                m.dim_out = (1, 2, 3)
                m.view_in = (1, -1, 1, 1)
                m.view_out = (-1, 1, 1, 1)
                setattr(m, 'base_func', m._conv_forward)

            setattr(m, 'forward', NPB_forward.__get__(m, m.__class__))

        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.LogSoftmax,
                            nn.ReLU, nn.Dropout)):
            setattr(m, 'original_forward', m.forward)
            setattr(m, 'forward', NPB_dummy_forward.__get__(m, m.__class__))

    model.NPB_modules = NPB_modules
    model.weights = [m.weight for m in NPB_modules]
    model.scores = [m.score for m in NPB_modules]
    ERK_sparsify(model, sparsity)
    model.sparsity = sparsity


# ---------------------------------------------------------------------------
# NPB objective (one iteration)
# ---------------------------------------------------------------------------
def NPB_objective(ones, model, score_optimizer, lr_score_scheduler,
                  alpha=0.0, beta=0.0, update=True, adjust_lr=False):
    model.zero_grad()

    model.apply(lambda m: setattr(m, "npb", True))
    model.apply(lambda m: setattr(m, "learn_mask", True))
    model.zero_grad()

    eff_paths = model(ones)
    cum_max_paths = 0
    for m in model.NPB_modules:
        if hasattr(m, "max_paths"):
            cum_max_paths += m.max_paths.log()
            m.max_paths = 0
    eff_paths = eff_paths.sum().log() + cum_max_paths
    eff_paths.backward()

    eff_params = 0
    eff_nodes = 0
    eff_kernels = 0
    all_layer_eff_nodes = []

    for i, m in enumerate(model.NPB_modules):
        path_grad = torch.zeros_like(m.score) if m.score.grad is None else m.score.grad

        layer_eff_params = path_grad.abs() * m.mask
        layer_eff_nodes = layer_eff_params.sum(m.dim_out).view(m.view_out)
        layer_eff_nodes_hard = (layer_eff_nodes > 0).float()
        all_layer_eff_nodes.append(layer_eff_nodes_hard.sum().item())
        eff_nodes += layer_eff_nodes_hard.sum()

        if len(m.weight.shape) == 4:
            layer_eff_kernels = layer_eff_params.sum((2, 3)).unsqueeze(2).unsqueeze(3)
            layer_eff_kernels_hard = (layer_eff_kernels > 0).float()
            eff_kernels += layer_eff_kernels_hard.sum()
        else:
            layer_eff_params_hard = (layer_eff_params > 0).float()
            eff_kernels += layer_eff_params_hard.sum()

        layer_eff_params_hard = (layer_eff_params > 0).float()
        eff_params += layer_eff_params_hard.sum()

        node_grad = path_grad * (1 - layer_eff_nodes_hard)
        if len(m.weight.shape) == 4:
            kernel_grad = path_grad * (1 - layer_eff_kernels_hard)
        else:
            kernel_grad = path_grad * (1 - layer_eff_params_hard)

        if update:
            m.score.grad = -(
                (1 - alpha) * path_grad
                + alpha * ((1 - beta) * node_grad + beta * kernel_grad)
            )

    if update:
        score_optimizer.step()

    return eff_paths, eff_nodes, eff_kernels, eff_params, all_layer_eff_nodes


# ---------------------------------------------------------------------------
# Public API: run DPaI pruning and return masks
# ---------------------------------------------------------------------------
def dpai_prune(model, sparsity, device, num_steps=500, alpha=0.0, beta=0.0,
               lr_score=0.1, input_shape=(1, 3, 32, 32)):
    """
    Prune *model* at *sparsity* using DPaI / NPB.

    Parameters
    ----------
    input_shape : tuple
        Shape (N, C, H, W) of the all-ones probe input. Defaults to a single
        3x32x32 image (CIFAR / TinyImageNet@32); pass the real shape for others.

    Returns
    -------
    masks : list[Tensor]
        One binary mask per Conv2d / Linear layer (module-iteration order).
    """
    import time

    model.to(device)
    model.eval()

    ones = torch.ones(input_shape, device=device)

    # Register NPB hooks
    NPB_register(model, sparsity, device)

    scores = [m.score for m in model.NPB_modules]
    score_optimizer = torch.optim.Adam(scores, lr=lr_score)
    lr_score_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        score_optimizer,
        milestones=[num_steps // 2, num_steps * 3 // 4],
        gamma=0.5,
    )

    # Initial measurement
    eff_paths, eff_nodes, eff_kernels, eff_params, _ = NPB_objective(
        ones, model, score_optimizer, lr_score_scheduler,
        alpha=alpha, beta=beta, update=False
    )
    print(f"  [DPaI] init — eff_paths={eff_paths.item():.2f}, "
          f"eff_nodes={int(eff_nodes)}, eff_kernels={int(eff_kernels)}, "
          f"eff_params={int(eff_params)}")

    for m in model.NPB_modules:
        m.mask = m.mask.detach().clone()

    # Optimisation loop
    t0 = time.time()
    for step in range(1, num_steps + 1):
        eff_paths, eff_nodes, eff_kernels, eff_params, _ = NPB_objective(
            ones, model, score_optimizer, lr_score_scheduler,
            alpha=alpha, beta=beta, update=True
        )
        if step % 50 == 0:
            print(f"  [DPaI] step {step}/{num_steps} — "
                  f"eff_paths={eff_paths.item():.2f}, eff_nodes={int(eff_nodes)}")

        if eff_nodes < 1:
            print("  [DPaI] WARNING: eff_nodes < 1, stopping early")
            break

    # Final evaluation
    eff_paths, eff_nodes, eff_kernels, eff_params, all_layer_eff_nodes = NPB_objective(
        ones, model, score_optimizer, lr_score_scheduler,
        alpha=alpha, beta=beta, update=False
    )
    print(f"  [DPaI] final — eff_paths={eff_paths.item():.2f}, "
          f"eff_nodes={int(eff_nodes)}, eff_kernels={int(eff_kernels)}, "
          f"eff_params={int(eff_params)}")
    print(f"  [DPaI] pruning took {time.time() - t0:.1f}s")

    # Extract masks
    masks = []
    for m in model.NPB_modules:
        masks.append(m.get_mask().detach().clone().cpu())

    total_mask_params = sum(mk.numel() for mk in masks)
    total_mask_nonzero = sum(mk.sum().item() for mk in masks)
    realized = 1.0 - total_mask_nonzero / float(total_mask_params)
    print(f"  [DPaI] realized mask sparsity: {realized * 100:.4f}%")

    # Clean up NPB state
    model.apply(lambda m: setattr(m, "npb", False))
    model.apply(lambda m: setattr(m, "learn_mask", False))

    return masks
