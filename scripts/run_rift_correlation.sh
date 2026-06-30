#!/usr/bin/env bash
python correlate_rift.py --config configs/rift_correlation.yaml --rows_json ${1:-outputs/rift_eval/rows.json}
