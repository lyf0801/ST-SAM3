#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" stage1_train_SAM3_Adapter.py --dataset ORS_4199 --labeled_percent 5%
"${PYTHON_BIN}" stage2_select_reliable_subset.py --dataset ORS_4199 --labeled_percent 5%
"${PYTHON_BIN}" stage3_retrain_SAM3_Adapter.py --dataset ORS_4199 --labeled_percent 5%
"${PYTHON_BIN}" stage4_select_reliable_subset.py --dataset ORS_4199 --labeled_percent 5%
"${PYTHON_BIN}" stage5_re_retrain_SAM3_Adapter.py --dataset ORS_4199 --labeled_percent 5%
"${PYTHON_BIN}" stage6_select_reliable_subnet.py --dataset ORS_4199 --labeled_percent 5%
"${PYTHON_BIN}" stage7_re_re_retrain_SAM3_Adapter.py --dataset ORS_4199 --labeled_percent 5%
