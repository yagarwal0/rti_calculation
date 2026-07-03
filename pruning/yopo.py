"""
YOPO: NMF-reconstruction-error pruning (the "oo_score" method).

Original implementation from: yopo_original/pruning/
  - masked_blocks.py  -> MaskedLinear / MaskedConv2d (below)
  - nmf_utils.py       -> apply_nmf() (below)
  - yopo_utils.py      -> scoring + masking + std-factor search (below)

This single self-contained module reproduces the YOPO pruning pipeline:

  1. convert_to_masked(model, last_layer)         # swap in maskable layers
  2. score_cache = compute_score(model)           # NMF reconstruction-error score
  3. sf = find_optimal_std_factor_global(model, score_cache, sparsity_goal, 'mad')
  4. apply_nmf_masks_global(model, score_cache, 'mad', std_factor=sf)

Each prunable weight matrix W is factorised with rank-`n_components` NMF
(W ~= W_lr H). The per-weight "score" is the absolute reconstruction error
|W - W_lr H|: weights that the low-rank model fails to reconstruct are deemed
important and kept; well-reconstructed (redundant) weights are pruned by
thresholding the score. The threshold is set globally from the score
distribution (mean+k*std, or median+k*MAD) and `std_factor` is searched to hit
a target sparsity. The final classification layer, batchnorm, shortcut and
biases are never pruned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import pickle

# NMF backend
from sklearn.decomposition import NMF
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =============================================================================
# Masked layers  (from pruning/masked_blocks.py)
# =============================================================================
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def set_mask(self, mask):
        self.mask = mask.clone()

    def apply_mask(self):
        self.weight.data *= self.mask

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def set_mask(self, mask):
        self.mask = mask.clone()

    def apply_mask(self):
        self.weight.data *= self.mask

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# =============================================================================
# NMF backend  (from pruning/nmf_utils.py)
# =============================================================================
def apply_nmf(W2D, n_components=7, iter=200):
    """
    Applies Non-negative Matrix Factorization to the given 2D weight matrix.

    Args:
        W2D (np.ndarray): The 2D weight matrix (non-negative).
        n_components (int): The number of components for NMF.
        iter (int): The maximum number of iterations for the solver.

    Returns:
        tuple: A tuple containing the factorized matrices W and H.
    """
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=iter)

    W = model.fit_transform(W2D)
    H = model.components_
    return W, H


# =============================================================================
# Scoring + masking  (from pruning/yopo_utils.py)
# =============================================================================
def compute_score(model, MODEL_NAME=None, n_components=7, nfm_max_iter=200):
    """
    Computes scores for convolutional and linear layers of a model,
    excluding the last classification layer, batch norm, shortcut, and biases.

    Returns:
        dict: computed scores and related statistics for each layer.
    """
    score_cache = {}
    last_layer_name = None

    # Find the name of the last linear layer (classification layer)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer_name = name + '.weight'  # Append .weight to get the parameter name

    for name, param in model.named_parameters():
        # Skip if not a weight parameter, or if it's a bias, batch norm, shortcut, or the last layer's weight
        if 'weight' in name and param.dim() > 1 and 'bn' not in name and 'shortcut' not in name and name != last_layer_name:
            print(f"Computing score for {name}...")
            W_shape = param.shape  # Store original shape
            # Flatten weights to 2D for NMF. Handle Conv2d and Linear layers.
            if param.dim() == 4:  # Conv2d
                W2D = param.data.cpu().numpy().reshape(W_shape[0], -1)
            elif param.dim() == 2:  # Linear
                W2D = param.data.cpu().numpy()

            W2D = np.abs(W2D)  # NMF works on non-negative data, use absolute values

            # Apply NMF
            try:
                W, H = apply_nmf(W2D, n_components=n_components, iter=nfm_max_iter)
                W_reconstructed = np.dot(W, H)
                # Delta Error Score
                score = abs(W2D - W_reconstructed)

                # Compute statistics
                median_score = np.median(score)

                score_cache[name] = {
                    'score': score,
                    'median': median_score,
                    'mad': np.median(np.abs(score - median_score)),
                    'mean': np.mean(score),
                    'std': np.std(score)
                }
            except Exception as e:
                print(f"Error applying NMF to {name}: {e}")
                score_cache[name] = {'error': str(e)}
    if MODEL_NAME is not None:
        filename = f"./oo_score/{MODEL_NAME}_score.pkl"
        with open(filename, "wb") as f:
            pickle.dump(score_cache, f)
        print(f"Score cache saved as {filename}")

    return score_cache


def transfer_score(model, transfer_score_cache, n_components=7, nfm_max_iter=200):
    """
    Computes scores for layers not already present in `transfer_score_cache`,
    excluding the last classification layer, batch norm, shortcut, and biases.
    """
    # Start with the scores from the transfer_score_cache
    score_cache = transfer_score_cache.copy()
    last_layer_name = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer_name = name + '.weight'

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1 and 'bn' not in name and 'shortcut' not in name and name != last_layer_name and name not in transfer_score_cache:
            print(f"Computing transfer score for {name}...")
            W_shape = param.shape
            if param.dim() == 4:  # Conv2d
                W2D = param.data.cpu().numpy().reshape(W_shape[0], -1)
            elif param.dim() == 2:  # Linear
                W2D = param.data.cpu().numpy()

            W2D = np.abs(W2D)

            try:
                W, H = apply_nmf(W2D, n_components=n_components, iter=nfm_max_iter)
                W_reconstructed = np.dot(W, H)
                score = abs(W2D - W_reconstructed)
                median_score = np.median(score)
                score_cache[name] = {
                    'score': score,
                    'median': median_score,
                    'mad': np.median(np.abs(score - median_score)),
                    'mean': np.mean(score),
                    'std': np.std(score)
                }
            except Exception as e:
                print(f"Error applying NMF to {name}: {e}")
                score_cache[name] = {'error': str(e)}

    return score_cache


def get_oo_score(MODEL_NAME):
    """Load a previously saved score cache."""
    filename = f"./oo_score/{MODEL_NAME}_score.pkl"
    with open(filename, "rb") as f:
        score_cache = pickle.load(f)
    print(f"Score cache loaded from {filename}")
    return score_cache


def mask_gradients(model):
    """Applies the mask to the gradients of masked layers (use during training)."""
    for m in model.modules():
        if isinstance(m, (MaskedLinear, MaskedConv2d)):
            if m.weight.grad is not None:
                m.weight.grad *= m.mask


def apply_nmf_masks(model, score_cache, threshold_type, std_factor=1.0):
    """
    Applies per-layer pruning masks based on NMF scores. Skips the last
    classification layer, batch norm, shortcut, and biases.
    """
    last_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer_name = name + '.weight'

    for name, param in model.named_parameters():
        if name in score_cache and 'error' not in score_cache[name]:
            score_data = score_cache[name]
            score = score_data['score']

            if threshold_type == 'std':
                mean_score = score_data['mean']
                std_score = score_data['std']
                threshold = mean_score + std_factor * std_score
            elif threshold_type == 'mad':
                median_score = score_data['median']
                mad_score = score_data['mad']
                threshold = np.clip(median_score + std_factor * mad_score,
                                    np.min(score), np.max(score))
            else:
                print(f"Warning: Unknown threshold type '{threshold_type}'. Skipping masking for layer {name}.")
                continue

            original_shape = param.shape
            mask_np = (score > threshold).astype(np.float32).reshape(original_shape)

            module_name = name.rsplit('.', 1)[0]
            param_module = dict(model.named_modules()).get(module_name)

            if hasattr(param_module, 'set_mask'):
                param_module.set_mask(torch.tensor(mask_np, device=param.device))
                param_module.apply_mask()
            else:
                print(f"Warning: Module for parameter '{name}' does not have a 'set_mask' method. Skipping masking.")


def find_optimal_std_factor(model, score_cache, sparsity_goal, threshold_type, std_factor_range=(0.1, 3.0), num_steps=100):
    """
    Finds the optimal (per-layer) std_factor to achieve a sparsity goal.
    """
    best_std_factor = None
    min_sparsity_diff = float('inf')

    std_factors_to_test = np.linspace(std_factor_range[0], std_factor_range[1], num_steps)

    print(f"Searching for optimal std_factor to achieve {sparsity_goal:.2f}% sparsity with threshold type '{threshold_type}'...")

    total_learnable_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            total_learnable_params += param.numel()
    if total_learnable_params == 0:
        print("No learnable weight parameters found in the model.")
        return None

    for std_factor in std_factors_to_test:
        model_copy = deepcopy(model)
        apply_nmf_masks(model_copy, score_cache, threshold_type, std_factor)

        zero_params = 0
        for name, param in model_copy.named_parameters():
            if 'weight' in name and param.requires_grad:
                zero_params += torch.sum(param == 0).item()

        current_sparsity = (zero_params / total_learnable_params) * 100 if total_learnable_params > 0 else 0
        sparsity_diff = abs(current_sparsity - sparsity_goal)

        if sparsity_diff < min_sparsity_diff:
            min_sparsity_diff = sparsity_diff
            best_std_factor = std_factor

    print(f"Optimal std_factor found: {best_std_factor:.4f} (Achieved sparsity closest to {sparsity_goal:.2f}%)")
    return best_std_factor


def convert_to_masked(model, last_layer):
    """
    Recursively converts standard nn.Linear and nn.Conv2d layers to Masked
    versions, leaving the final classification layer (`last_layer`) untouched.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            masked = MaskedConv2d(module.in_channels, module.out_channels, module.kernel_size,
                                  stride=module.stride, padding=module.padding, dilation=module.dilation,
                                  groups=module.groups, bias=(module.bias is not None))
            masked.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                masked.bias.data.copy_(module.bias.data)
            setattr(model, name, masked)

        elif isinstance(module, nn.Linear):
            # Skip converting the final classification layer
            if module == last_layer:
                continue
            masked = MaskedLinear(module.in_features, module.out_features, bias=(module.bias is not None))
            masked.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                masked.bias.data.copy_(module.bias.data)
            setattr(model, name, masked)

        else:
            convert_to_masked(module, last_layer)  # recurse
    return model


def apply_nmf_masks_global(model, score_cache, threshold_type, std_factor=1.0):
    """
    Applies pruning masks based on a *global* NMF score threshold.
    Skips the last classification layer, batch norm, shortcut, and biases.
    """
    all_scores = []
    for name, score_data in score_cache.items():
        if 'score' in score_data and 'error' not in score_data:
            all_scores.append(score_data['score'].flatten())

    if not all_scores:
        print("No valid score data found in score_cache. Cannot compute global threshold.")
        return

    global_scores = np.concatenate(all_scores)

    if threshold_type == 'std':
        global_mean_score = np.mean(global_scores)
        global_std_score = np.std(global_scores)
        global_threshold = global_mean_score + std_factor * global_std_score
    elif threshold_type == 'mad':
        global_median_score = np.median(global_scores)
        global_mad_score = np.median(np.abs(global_scores - global_median_score))
        global_threshold = np.clip(global_median_score + std_factor * global_mad_score,
                                   np.min(global_scores), np.max(global_scores))
    else:
        print(f"Warning: Unknown threshold type '{threshold_type}'. Skipping global masking.")
        return

    print(f"Global threshold ({threshold_type}, std_factor={std_factor:.2f}): {global_threshold:.6f}")

    last_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer_name = name + '.weight'

    for name, param in model.named_parameters():
        if name in score_cache and 'error' not in score_cache[name] and name != last_layer_name:
            score_data = score_cache[name]
            score = score_data['score']

            original_shape = param.shape
            mask_np = (score > global_threshold).astype(np.float32).reshape(original_shape)

            module_name = name.rsplit('.', 1)[0]
            param_module = dict(model.named_modules()).get(module_name)

            if hasattr(param_module, 'set_mask'):
                param_module.set_mask(torch.tensor(mask_np, device=param.device))
                param_module.apply_mask()
            else:
                print(f"Warning: Module for parameter '{name}' does not have a 'set_mask' method. Skipping masking.")


def find_optimal_std_factor_global(model, score_cache, sparsity_goal, threshold_type="mad", std_factor_range=(0.1, 3.0), num_steps=100):
    """
    Finds the optimal std_factor to achieve a sparsity goal using *global*
    NMF-based pruning.
    """
    best_std_factor = None
    min_sparsity_diff = float('inf')

    std_factors_to_test = np.linspace(std_factor_range[0], std_factor_range[1], num_steps)

    print(f"Searching for optimal std_factor to achieve {sparsity_goal:.2f}% sparsity with threshold type '{threshold_type}'...")

    total_learnable_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            total_learnable_params += param.numel()
    if total_learnable_params == 0:
        print("No learnable weight parameters found in the model.")
        return None

    for std_factor in std_factors_to_test:
        model_copy = deepcopy(model)
        apply_nmf_masks_global(model_copy, score_cache, threshold_type, std_factor)

        zero_params = 0
        for name, param in model_copy.named_parameters():
            if 'weight' in name and param.requires_grad:
                zero_params += torch.sum(param == 0).item()

        current_sparsity = (zero_params / total_learnable_params) * 100 if total_learnable_params > 0 else 0
        sparsity_diff = abs(current_sparsity - sparsity_goal)

        print(f"  Testing std_factor={std_factor:.4f}: Achieved sparsity={current_sparsity:.2f}% (Diff={sparsity_diff:.2f}%)")

        if sparsity_diff < min_sparsity_diff:
            min_sparsity_diff = sparsity_diff
            best_std_factor = std_factor

    print(f"Optimal std_factor found: {best_std_factor:.4f} (Achieved sparsity closest to {sparsity_goal:.2f}%)")
    best_std_factor = round(best_std_factor, 2)
    return best_std_factor


def apply_nmf_masks_global_constrained(model, score_cache, threshold_type, std_factor=1.0, m_min=20, c_min=0):
    """
    Applies global-threshold pruning masks with constraints on minimum per-row
    and per-column mask elements to prevent layer collapse.
    """
    if not all(name in score_cache and 'score' in score_cache[name] and 'error' not in score_cache[name] for name in score_cache):
        print("Score cache is incomplete or contains errors. Cannot apply constrained global masking.")
        return

    all_scores = []
    for name, score_data in score_cache.items():
        if 'score' in score_data and 'error' not in score_data:
            all_scores.append(score_data['score'].flatten())

    if not all_scores:
        print("No valid score data found in score_cache. Cannot compute global threshold.")
        return

    global_scores = np.concatenate(all_scores)

    if threshold_type == 'std':
        global_mean_score = np.mean(global_scores)
        global_std_score = np.std(global_scores)
        global_threshold = global_mean_score + std_factor * global_std_score
    elif threshold_type == 'mad':
        global_median_score = np.median(global_scores)
        global_mad_score = np.median(np.abs(global_scores - global_median_score))
        global_threshold = np.clip(global_median_score + std_factor * global_mad_score,
                                   np.min(global_scores), np.max(global_scores))
    else:
        print(f"Warning: Unknown threshold type '{threshold_type}'. Skipping global masking.")
        return

    print(f"Global threshold ({threshold_type}, std_factor={std_factor:.2f}): {global_threshold:.6f}")

    last_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer_name = name + '.weight'

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad and name in score_cache and 'error' not in score_cache[name] and name != last_layer_name:
            score_data = score_cache[name]
            score = score_data['score']

            original_shape = param.shape
            mask_np = (score > global_threshold).astype(np.float32)

            # --- Enforce constraints ---
            # Enforce minimum per-row elements (output channels/features)
            if mask_np.ndim > 1:
                row_sums = np.sum(mask_np, axis=tuple(range(1, mask_np.ndim)))
                rows_to_fix = np.where(row_sums < m_min)[0]

                if len(rows_to_fix) > 0:
                    for row_idx in rows_to_fix:
                        row_score_flat = score[row_idx].flatten()
                        sorted_indices_flat = np.argsort(row_score_flat)[::-1]
                        current_row_ones = np.sum(mask_np[row_idx])
                        needed_ones = m_min - current_row_ones

                        if needed_ones > 0:
                            indices_to_set_one_flat = sorted_indices_flat[:int(needed_ones)]
                            original_indices = np.unravel_index(indices_to_set_one_flat, score[row_idx].shape)
                            mask_np[row_idx][original_indices] = 1.0

            # Enforce minimum per-column elements (input channels for Conv2d)
            if mask_np.ndim == 4 and c_min > 0:
                col_sums = np.sum(mask_np, axis=(0, 2, 3))
                cols_to_fix = np.where(col_sums < c_min)[0]

                if len(cols_to_fix) > 0:
                    for col_idx in cols_to_fix:
                        col_scores = score[:, col_idx, :, :]
                        col_score_flat = col_scores.flatten()
                        sorted_indices_flat = np.argsort(col_score_flat)[::-1]
                        current_col_ones = np.sum(mask_np[:, col_idx, :, :])
                        needed_ones = c_min - current_col_ones

                        if needed_ones > 0:
                            original_indices_in_col = np.unravel_index(np.arange(col_scores.size).flatten(), col_scores.shape)
                            sorted_original_indices_in_col = tuple(idx[sorted_indices_flat] for idx in original_indices_in_col)
                            indices_to_set_one_original = tuple(idx[:int(needed_ones)] for idx in sorted_original_indices_in_col)
                            global_mask_indices = (indices_to_set_one_original[0],
                                                   np.full_like(indices_to_set_one_original[0], col_idx),
                                                   indices_to_set_one_original[1],
                                                   indices_to_set_one_original[2])
                            mask_np[global_mask_indices] = 1.0

            module_name = name.rsplit('.', 1)[0]
            param_module = dict(model.named_modules()).get(module_name)
            mask_np = mask_np.reshape(original_shape)

            if hasattr(param_module, 'set_mask'):
                mask_tensor = torch.tensor(mask_np, device=param.device)
                param_module.set_mask(mask_tensor)
                param_module.apply_mask()
            else:
                print(f"Warning: Module for parameter '{name}' does not have a 'set_mask' method. Skipping masking.")
