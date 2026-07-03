"""
config.py — experiment configuration files.

Loads a YAML or JSON config and folds it into an argparse parser as defaults, so
every script can be driven by a versioned config file while still allowing
command-line overrides.

Precedence (low -> high):  argparse defaults  <  config file  <  CLI arguments.

Config keys use the argparse `dest` names (underscores), e.g. `batch_size`,
`pgd_steps`. Keys a given script doesn't recognise are ignored (with a note), so
one shared config can drive main.py / train.py / attacks.py / metrics.py.

Usage in a script:
    p = argparse.ArgumentParser(...)
    p.add_argument('--config', default=None)
    ...
    args = parse_args_with_config(p)
"""

import os
import json


def load_config(path):
    """Load a config dict from a .yaml/.yml/.json file. {} if path is falsy."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        if path.lower().endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args_with_config(parser, config_dest='config'):
    """Parse args, applying a config file (named by --<config_dest>) as defaults.

    CLI arguments still override config values because the config is installed
    via `set_defaults` (defaults rank below explicit CLI args).
    """
    pre, _ = parser.parse_known_args()
    cfg_path = getattr(pre, config_dest, None)
    if cfg_path:
        cfg = load_config(cfg_path)
        valid = {a.dest for a in parser._actions}
        unknown = [k for k in cfg if k not in valid]
        if unknown:
            print(f"[config] {cfg_path}: ignoring keys not used by this script: {unknown}")
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in valid})
    return parser.parse_args()
