"""
flops.py — forward-pass FLOPs / MACs counter (sparsity-aware).

Counts multiply-accumulate operations for Conv2d / Linear layers via forward
hooks. With `sparse=True` only non-zero weights are counted, giving the
*effective* compute of a pruned model; `sparse=False` gives the dense baseline.

    from utils import count_flops
    dense  = count_flops(model, (1, 3, 32, 32), sparse=False)
    sparse = count_flops(pruned_model, (1, 3, 32, 32), sparse=True)
    # -> {'macs': ..., 'flops': 2 * macs}
"""

import torch
import torch.nn as nn


@torch.no_grad()
def count_flops(model, input_shape=(1, 3, 32, 32), device='cpu', sparse=True):
    """Return {'macs', 'flops'} for one forward pass on a single input.

    Conv2d MACs  = (#weights counted) * H_out * W_out
    Linear MACs  = (#weights counted)
    'flops' = 2 * macs (one multiply + one add per MAC).
    """
    model = model.to(device).eval()
    total = {'macs': 0}
    handles = []

    def conv_hook(m, inp, out):
        out_spatial = out.shape[-1] * out.shape[-2]
        w = int((m.weight != 0).sum().item()) if sparse else m.weight.numel()
        total['macs'] += w * out_spatial

    def lin_hook(m, inp, out):
        w = int((m.weight != 0).sum().item()) if sparse else m.weight.numel()
        total['macs'] += w

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))

    model(torch.zeros(input_shape, device=device))
    for h in handles:
        h.remove()

    macs = total['macs']
    return {'macs': macs, 'flops': 2 * macs}
