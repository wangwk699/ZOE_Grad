#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=(opt-1.3b opt-6.7b llama2-7b llama2-13b llama3-8b qwen3-8b)
TASKS=(SST2 RTE CB BoolQ WSC WIC MultiRC SQuAD)
METHODS=(STE HTGE Uniform Normal)

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for method in "${METHODS[@]}"; do
      "$ROOT/scripts/run.sh" "$model" "$task" "$method" "$@"
    done
  done
done
