"""
pruning — pruning-at-initialization algorithm implementations.

Modules (import individually; some need extra packages):
    snip      -> SNIP(net, keep_ratio, dataloader, device)          [torch]
    grasp     -> GraSP(net, ratio, dataloader, device, ...)         [torch]
    synflow   -> synflow_prune_100(model, dataloader, device, ...)  [torch, numpy, tqdm]
    dpai      -> dpai_prune(model, sparsity, device, num_steps)     [torch, numpy]
    yopo      -> compute_score / find_optimal_std_factor_global /
                 apply_nmf_masks_global / convert_to_masked          [torch, numpy, scikit-learn]

Imports are intentionally NOT done here so that a missing optional dependency
(e.g. scikit-learn for YOPO) does not break the other algorithms. Import the
submodule you need directly: `from pruning import snip`.
"""
