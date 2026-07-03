"""
adv_train.py — PGD adversarial-example generation for adversarial training.

This matches the **eval** attack convention exactly: inputs are raw [0, 1] pixel
images and the model normalizes internally (e.g. `attacks.NormalizedModel`). PGD
therefore runs directly in [0, 1] space — the eps-ball projection and the valid-
pixel clamp are both in pixel space — so training and FGSM/PGD/AutoAttack
evaluation share one identical threat model.

    max_{||delta||_inf <= eps} loss(f(x + delta), y)

`pgd_perturb` returns adversarial [0, 1] inputs ready to feed to the model during
training (standard L_inf PGD adversarial training, Madry et al.).
"""

import torch
import torch.nn as nn


def pgd_perturb(model, x, y, eps=8 / 255, alpha=2 / 255, steps=10,
                random_start=True):
    """Generate L_inf PGD adversarial examples in [0, 1] pixel space.

    Args:
        model: classifier accepting [0, 1] inputs (it normalizes internally).
        x: clean [0, 1] batch.
        y: labels.
        eps, alpha: L_inf budget / step size in [0, 1] pixel space.
        steps: PGD iterations.
        random_start: start from a random point in the eps-ball.
    Returns adversarial [0, 1] inputs (detached).
    """
    x0 = x.detach()
    if random_start:
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp_(0.0, 1.0).detach()
    else:
        x_adv = x0.clone()

    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = criterion(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project to the eps-ball around x0, then clamp to valid pixels
        x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps)
        x_adv = x_adv.clamp(0.0, 1.0).detach()
    if was_training:
        model.train()
    return x_adv
