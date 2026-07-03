"""
download_dataset.py — fetch every dataset into the location the code expects.

Datasets and where they land (matching the defaults used everywhere else):

    cifar10       -> <data-root>/cifar-10-batches-py/      (auto, via torchvision)
    cifar100      -> <data-root>/cifar-100-python/         (auto, via torchvision)
    tinyimagenet  -> <tinyimagenet-dir>/                   (downloaded + unzipped)

`<data-root>` defaults to ./data and `<tinyimagenet-dir>` to ./tiny-imagenet-200,
i.e. the same defaults as `dataloader.get_loaders`, `main.py`, the experiment
scripts, etc. — so after running this, every other command just works.

Examples
--------
    python download_dataset.py                       # all three datasets
    python download_dataset.py --datasets cifar10 cifar100
    python download_dataset.py --datasets tinyimagenet --tinyimagenet-dir ./tiny-imagenet-200
    python download_dataset.py --data-root ./data --force   # re-download even if present

CIFAR-10/100 are small and download automatically through torchvision.
TinyImageNet is a ~240 MB zip from cs231n; it is downloaded with a progress
bar and unzipped in place. The script is idempotent: datasets already present
are skipped unless `--force` is given.
"""

import os
import sys
import shutil
import zipfile
import argparse
import tempfile
import urllib.request

TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
# The zip always extracts to a top-level folder with this name.
TINYIMAGENET_ZIP_ROOT = "tiny-imagenet-200"


# ---------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100 (torchvision auto-download)
# ---------------------------------------------------------------------------
def _cifar_present(data_root, folder):
    return os.path.isdir(os.path.join(data_root, folder))


def download_cifar10(data_root, force=False):
    """Download CIFAR-10 train+test into <data-root>/cifar-10-batches-py/."""
    if _cifar_present(data_root, "cifar-10-batches-py") and not force:
        print(f"[cifar10] already present in {data_root} — skipping "
              "(use --force to re-download).")
        return
    from torchvision.datasets import CIFAR10
    os.makedirs(data_root, exist_ok=True)
    print(f"[cifar10] downloading into {data_root} ...")
    CIFAR10(root=data_root, train=True, download=True)
    CIFAR10(root=data_root, train=False, download=True)
    print("[cifar10] done.")


def download_cifar100(data_root, force=False):
    """Download CIFAR-100 train+test into <data-root>/cifar-100-python/."""
    if _cifar_present(data_root, "cifar-100-python") and not force:
        print(f"[cifar100] already present in {data_root} — skipping "
              "(use --force to re-download).")
        return
    from torchvision.datasets import CIFAR100
    os.makedirs(data_root, exist_ok=True)
    print(f"[cifar100] downloading into {data_root} ...")
    CIFAR100(root=data_root, train=True, download=True)
    CIFAR100(root=data_root, train=False, download=True)
    print("[cifar100] done.")


# ---------------------------------------------------------------------------
# TinyImageNet (manual download + unzip)
# ---------------------------------------------------------------------------
def _tinyimagenet_present(tiny_dir):
    return (os.path.isdir(os.path.join(tiny_dir, "train"))
            and os.path.isdir(os.path.join(tiny_dir, "val")))


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        sys.stdout.write(f"\r  downloading... {pct:5.1f}%  ({mb:6.1f}/{total_mb:.1f} MB)")
    else:
        sys.stdout.write(f"\r  downloading... {downloaded / 1e6:6.1f} MB")
    sys.stdout.flush()


def download_tinyimagenet(tiny_dir, force=False):
    """Download and unzip TinyImageNet so that <tiny_dir>/{train,val} exist."""
    if _tinyimagenet_present(tiny_dir) and not force:
        print(f"[tinyimagenet] already present at {tiny_dir} — skipping "
              "(use --force to re-download).")
        return

    tiny_dir = os.path.abspath(tiny_dir)
    parent = os.path.dirname(tiny_dir) or "."
    os.makedirs(parent, exist_ok=True)

    # 1) download the zip to a temp file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
    try:
        print(f"[tinyimagenet] fetching {TINYIMAGENET_URL}")
        urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path, _progress_hook)
        sys.stdout.write("\n")

        # 2) extract into the parent dir -> creates <parent>/tiny-imagenet-200
        print(f"[tinyimagenet] extracting into {parent} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(parent)

        extracted = os.path.join(parent, TINYIMAGENET_ZIP_ROOT)
        # 3) move/rename to the requested directory if it differs
        if os.path.abspath(extracted) != tiny_dir:
            if os.path.exists(tiny_dir):
                if force:
                    shutil.rmtree(tiny_dir)
                else:
                    raise FileExistsError(
                        f"{tiny_dir} already exists; use --force to overwrite.")
            shutil.move(extracted, tiny_dir)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    if not _tinyimagenet_present(tiny_dir):
        raise RuntimeError(
            f"[tinyimagenet] extraction finished but {tiny_dir}/train or /val "
            "is missing — the archive layout may have changed.")
    print(f"[tinyimagenet] done -> {tiny_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Download all datasets into the repository's expected locations.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+',
                   default=['cifar10', 'cifar100', 'tinyimagenet'],
                   choices=['cifar10', 'cifar100', 'tinyimagenet'],
                   help='which datasets to download (default: all)')
    p.add_argument('--data-root', default='./data',
                   help='CIFAR download/cache dir (default: ./data)')
    p.add_argument('--tinyimagenet-dir', default='./tiny-imagenet-200',
                   help='target dir for TinyImageNet (default: ./tiny-imagenet-200)')
    p.add_argument('--force', action='store_true',
                   help='re-download even if the dataset is already present')
    args = p.parse_args()

    print("=" * 70)
    print("Dataset downloader")
    print(f"  datasets         : {', '.join(args.datasets)}")
    print(f"  data-root        : {os.path.abspath(args.data_root)}")
    if 'tinyimagenet' in args.datasets:
        print(f"  tinyimagenet-dir : {os.path.abspath(args.tinyimagenet_dir)}")
    print("=" * 70)

    if 'cifar10' in args.datasets:
        download_cifar10(args.data_root, args.force)
    if 'cifar100' in args.datasets:
        download_cifar100(args.data_root, args.force)
    if 'tinyimagenet' in args.datasets:
        download_tinyimagenet(args.tinyimagenet_dir, args.force)

    print("\nAll requested datasets are ready.")


if __name__ == '__main__':
    main()
