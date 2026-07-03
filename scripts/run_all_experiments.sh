#!/usr/bin/env bash
# scripts/run_all_experiments.sh
# Run the full experimental protocol (experiments 1..5) end-to-end.
#
# Usage:
#   bash scripts/run_all_experiments.sh                          # uses configs/cifar10.yaml
#   bash scripts/run_all_experiments.sh configs/cifar100.yaml
#   bash scripts/run_all_experiments.sh configs/smoke.yaml       # fast CPU sanity check
#
# Environment overrides (optional):
#   PYTHON=python3 EXPS="1 2 4" bash scripts/run_all_experiments.sh
#   EXTRA="--no-autoattack --epochs 50" bash scripts/run_all_experiments.sh
#
# Each experiment writes a JSON under $RESULTS_DIR (default ./experiments_out).
# Failures in one experiment do NOT stop the others; see the SUMMARY at the end.

set -uo pipefail

# Run from the prune_models/ folder (parent of scripts/).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

CONFIG="${1:-configs/cifar10.yaml}"
PYTHON="${PYTHON:-python}"
EXPS="${EXPS:-1 2 3 4 5}"
RESULTS_DIR="${RESULTS_DIR:-./experiments_out}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
EXTRA="${EXTRA:-}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "=================================================================="
echo " run_all_experiments.sh"
echo " config       : $CONFIG"
echo " experiments  : $EXPS"
echo " results dir  : $RESULTS_DIR"
echo " log dir      : $LOG_DIR"
echo " python       : $PYTHON"
echo " extra args   : $EXTRA"
echo "=================================================================="

declare -a STATUS
for E in $EXPS; do
    case "$E" in
        1) SCRIPT="experiments/exp1_robustness.py" ;;
        2) SCRIPT="experiments/exp2_transfer.py" ;;
        3) SCRIPT="experiments/exp3_symmetric.py" ;;
        4) SCRIPT="experiments/exp4_sparsity_sweep.py" ;;
        5) SCRIPT="experiments/exp5_adv_efficiency.py" ;;
        *) echo "[skip] unknown experiment '$E'"; continue ;;
    esac

    LOG="${LOG_DIR}/exp${E}.log"
    echo
    echo "------------------------------------------------------------------"
    echo "[exp${E}] $SCRIPT  -->  $LOG"
    echo "------------------------------------------------------------------"
    if "$PYTHON" "$SCRIPT" --config "$CONFIG" --results-dir "$RESULTS_DIR" $EXTRA 2>&1 | tee "$LOG"; then
        STATUS+=("exp${E}: OK")
    else
        STATUS+=("exp${E}: FAILED (see $LOG)")
    fi
done

echo
echo "=================================================================="
echo " SUMMARY"
echo "=================================================================="
for s in "${STATUS[@]}"; do echo "  $s"; done

echo
echo "Result JSONs : $RESULTS_DIR/exp*.json"
echo "Plots        : python scripts/plot_results.py --results-dir $RESULTS_DIR --out-dir plots"
