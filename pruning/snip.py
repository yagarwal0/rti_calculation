"""
SNIP: Single-shot Network Pruning based on Connection Sensitivity (Lee et al., 2019).

Original implementation from: snip_original/snip/snip.py

Usage:
    keep_masks = SNIP(net, keep_ratio, train_dataloader, device)
    # keep_masks is a list of binary masks (one per Conv2d / Linear layer,
    # in module-iteration order). 1 = keep the weight, 0 = prune it.

SNIP is a single-shot, prune-at-initialisation method: it scores every weight
by the sensitivity of the loss to a multiplicative mask on that weight
(|g * theta| evaluated on a single mini-batch), then keeps the top `keep_ratio`
fraction of connections globally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, keep_ratio, train_dataloader, device, loss_fn=F.cross_entropy):
    """Compute SNIP keep-masks at initialisation.

    Returns a list of binary masks (one per Conv2d/Linear, in module order);
    1 = keep, 0 = prune. Connection sensitivity |g * w| is scored on a single
    batch using the network's *actual* initial weights — the same weights the
    mask is applied to and trained — so the mask is not computed for one
    initialisation and applied to another.
    """
    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Fresh copy so scoring doesn't disturb the network we return a mask for.
    net = copy.deepcopy(net)

    # Monkey-patch Conv2d/Linear to multiply the weight by a learnable mask and
    # freeze the weights, so the gradient flows to the mask (SNIP sensitivity).
    # The weights are kept as-is (no re-initialisation): we score exactly the
    # init that will be pruned.
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # One forward/backward to get the connection-sensitivity grads on the masks.
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise (guard all-zero grads).
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    if norm_factor <= 0:
        raise RuntimeError(
            "SNIP: total connection sensitivity is zero (dead gradients) — "
            "check the data batch, loss function, or initialisation.")
    all_scores = all_scores / norm_factor

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    num_params_to_keep = max(1, min(num_params_to_keep, len(all_scores)))
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = [((g / norm_factor) >= acceptable_score).float() for g in grads_abs]

    kept = int(torch.cat([m.flatten() for m in keep_masks]).sum().item())
    total = sum(m.numel() for m in keep_masks)
    print(f"  [snip] kept {kept}/{total} weights "
          f"({100.0 * kept / total:.2f}% dense; target keep {keep_ratio * 100:.2f}%)")

    return keep_masks
