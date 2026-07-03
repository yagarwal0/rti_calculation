"""
logger.py — experiment logging via TensorBoard (default) or Weights & Biases.

Logs:
    * accuracy curves     train/test accuracy + loss + lr, per epoch
    * robustness curves    robust accuracy per attack vs a step axis (e.g. sparsity)
    * RTI vs sparsity      matplotlib figure of RTI / nRTI against sparsity

A single `Logger` abstracts both backends so calling code is backend-agnostic.

    log = Logger(backend='tensorboard', log_dir='./runs', run_name='resnet20_snip')
    for epoch in ...:
        log.log_accuracy(epoch, train_acc=.., test_acc=.., lr=..)
    log.log_rti_vs_sparsity([0.5,0.9,0.95], rti=[.., .., ..], nrti=[.., .., ..])
    log.close()

TensorBoard:  tensorboard --logdir runs
WandB:        pip install wandb   (backend='wandb')
"""

import os
import json
import datetime


class Logger:
    def __init__(self, backend='tensorboard', log_dir='./runs', run_name=None,
                 project='pruning-robustness', config=None):
        self.backend = (backend or 'none').lower()
        self.run_name = run_name or datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
        self.writer = None
        self.wandb = None
        self.run = None

        if self.backend == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.dir = os.path.join(log_dir, self.run_name)
            self.writer = SummaryWriter(log_dir=self.dir)
            if config:
                self.writer.add_text('config', f"```\n{json.dumps(config, indent=2, default=str)}\n```")
        elif self.backend == 'wandb':
            try:
                import wandb
            except ImportError as e:
                raise ImportError("backend='wandb' requires `pip install wandb`.") from e
            self.wandb = wandb
            os.makedirs(log_dir, exist_ok=True)
            self.run = wandb.init(project=project, name=self.run_name,
                                  config=config, dir=log_dir)
        elif self.backend == 'none':
            pass
        else:
            raise ValueError(f"Unknown logging backend '{backend}'. "
                             "Use 'tensorboard', 'wandb', or 'none'.")

    # ---------------- scalar logging ----------------
    def log_scalar(self, tag, value, step):
        if value is None:
            return
        if self.backend == 'tensorboard':
            self.writer.add_scalar(tag, value, step)
        elif self.backend == 'wandb':
            self.run.log({tag: value}, step=int(step))

    def log_scalars(self, main_tag, tag_value_dict, step):
        """Group several series under one chart (e.g. all attacks together)."""
        clean = {k: v for k, v in tag_value_dict.items() if v is not None}
        if not clean:
            return
        if self.backend == 'tensorboard':
            self.writer.add_scalars(main_tag, clean, step)
        elif self.backend == 'wandb':
            self.run.log({f"{main_tag}/{k}": v for k, v in clean.items()}, step=int(step))

    # ---------------- convenience curves ----------------
    def log_accuracy(self, epoch, train_acc=None, test_acc=None, lr=None,
                     train_loss=None, test_loss=None):
        """Accuracy curves for a training run (one call per epoch)."""
        self.log_scalars('accuracy', {'train': train_acc, 'test': test_acc}, epoch)
        self.log_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)
        self.log_scalar('lr', lr, epoch)

    def log_robustness(self, attacks, step, clean=None):
        """Robustness curves: robust accuracy per attack at a given step.

        `attacks` is {name: robust_accuracy}. Use a meaningful `step`
        (e.g. int(sparsity*100)) to get robust-accuracy-vs-sparsity curves.
        """
        series = dict(attacks)
        if clean is not None:
            series = {'clean': clean, **series}
        self.log_scalars('robustness', series, step)

    # ---------------- figures ----------------
    def log_figure(self, tag, fig, step=0):
        if self.backend == 'tensorboard':
            self.writer.add_figure(tag, fig, step)
        elif self.backend == 'wandb':
            self.run.log({tag: self.wandb.Image(fig)}, step=int(step))
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    def log_rti_vs_sparsity(self, sparsities, rti, nrti=None, tag='RTI_vs_sparsity',
                            method=None, step=0):
        """Plot RTI (and nRTI) against sparsity and log the figure."""
        fig = plot_rti_vs_sparsity(sparsities, rti, nrti, method=method)
        self.log_figure(tag, fig, step)
        # also log raw scalars so the values are queryable
        for s, r in zip(sparsities, rti):
            self.log_scalar('RTI', r, int(round(float(s) * 100)))
        if nrti is not None:
            for s, n in zip(sparsities, nrti):
                self.log_scalar('nRTI', n, int(round(float(s) * 100)))

    def log_robustness_vs_sparsity(self, sparsities, series, tag='robust_acc_vs_sparsity'):
        """Plot robust accuracy curves (one line per attack) against sparsity."""
        fig = plot_robustness_vs_sparsity(sparsities, series)
        self.log_figure(tag, fig)

    def close(self):
        if self.backend == 'tensorboard' and self.writer is not None:
            self.writer.flush()
            self.writer.close()
        elif self.backend == 'wandb' and self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------------------
# Standalone matplotlib figure builders (usable without a Logger)
# ---------------------------------------------------------------------------
def plot_rti_vs_sparsity(sparsities, rti, nrti=None, method=None):
    """RTI (left axis) and optional nRTI (right axis) vs sparsity (%)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = [float(s) * 100 if float(s) <= 1.0 else float(s) for s in sparsities]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    rti = [rti[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, rti, 'o-', color='tab:blue', label='RTI')
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('RTI', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    if nrti is not None:
        nrti = [nrti[i] for i in order]
        ax2 = ax1.twinx()
        ax2.plot(xs, nrti, 's--', color='tab:red', label='nRTI')
        ax2.set_ylabel('nRTI', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.set_title(f"RTI vs Sparsity{f' ({method})' if method else ''}")
    fig.tight_layout()
    return fig


def plot_robustness_vs_sparsity(sparsities, series):
    """Robust accuracy curves vs sparsity. `series` = {attack_name: [acc,...]}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = [float(s) * 100 if float(s) <= 1.0 else float(s) for s in sparsities]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in series.items():
        ys = [ys[i] for i in order]
        ax.plot(xs, ys, 'o-', label=name)
    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Robust accuracy (%)')
    ax.set_title('Robust accuracy vs Sparsity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
