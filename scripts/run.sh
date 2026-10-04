#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: scripts/run.sh MODEL TASK METHOD [extra train_main.py options...]" >&2
  exit 2
fi

MODEL_ALIAS=$1
TASK=$2
METHOD=$3
shift 3

case "$MODEL_ALIAS" in
  opt-1.3b|opt-6.7b) MODEL_ID="facebook/$MODEL_ALIAS"; CHECKPOINT="$MODEL_ALIAS-w4a16.pth"; LET=1 ;;
  llama2-7b) MODEL_ID="meta-llama/Llama-2-7b-hf"; CHECKPOINT="Llama-2-7b-w4a16.pth"; LET=0 ;;
  llama2-13b) MODEL_ID="meta-llama/Llama-2-13b-hf"; CHECKPOINT="Llama-2-13b-w4a16.pth"; LET=0 ;;
  llama3-8b) MODEL_ID="meta-llama/Meta-Llama-3-8B"; CHECKPOINT="Meta-Llama-3-8B-w4a16.pth"; LET=0 ;;
  qwen3-8b) MODEL_ID="Qwen/Qwen3-8B"; CHECKPOINT="Qwen3-8B-w4a16.pth"; LET=0 ;;
  *) echo "Unsupported model: $MODEL_ALIAS" >&2; exit 2 ;;
esac

case "$TASK" in
  SST2|RTE|CB|BoolQ|WSC|WIC|MultiRC) CLASSIFICATION=true ;;
  SQuAD) CLASSIFICATION=false ;;
  *) echo "Unsupported task: $TASK" >&2; exit 2 ;;
esac

case "$METHOD" in
  STE|HTGE|Uniform) DELTA=0.285 ;;
  Normal) DELTA=0.15 ;;
  *) echo "Unsupported estimator: $METHOD" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
RESUME=${RESUME_PATH:-"./pre_quantized_models/$CHECKPOINT"}
if [[ ! -f "$RESUME" ]]; then
  echo "Missing checkpoint: $RESUME" >&2
  exit 1
fi

if [[ "$TASK" == SQuAD ]]; then
  NUM_EVAL=${NUM_EVAL:-300}
  NUM_DEV=${NUM_DEV:-30}
else
  NUM_EVAL=${NUM_EVAL:-1000}
  NUM_DEV=${NUM_DEV:-10}
fi

if [[ "$MODEL_ALIAS" == opt-6.7b && "$TASK" == SQuAD ]]; then
  CALIB_DATASET=${CALIB_DATASET:-squad}
else
  CALIB_DATASET=${CALIB_DATASET:-wikitext2}
fi
export HF_HOME=${HF_HOME:-"$PWD/cache/hf"}

OUTPUT_DIR=${OUTPUT_DIR:-"./logs/$MODEL_ALIAS/$TASK/$METHOD"}
ARGS=(
  --model "$MODEL_ID"
  --task_name "$TASK"
  --trainer "$METHOD"
  --quant_method omni
  --wbits "${WBITS:-4}"
  --abits "${ABITS:-16}"
  --lwc
  --resume "$RESUME"
  --train
  --train_as_classification "$CLASSIFICATION"
  --epochs 0
  --calib_dataset "$CALIB_DATASET"
  --cache_dir ./cache
  --q_output_dir "$OUTPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --max_steps "${STEPS:-5000}"
  --learning_rate "${LR:-1e-6}"
  --lr_scheduler_type "${LR_SCHEDULER:-constant_with_warmup}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --delta "${DELTA_OVERRIDE:-$DELTA}"
  --t "${T:-16}"
  --max_length "${MAX_LENGTH:-2048}"
  --train_batch_size "${BATCH_SIZE:-1}"
  --logging_steps "${LOGGING_STEPS:-100}"
  --save_strategy no
  --eval_strategy no
  --num_train "${NUM_TRAIN:-1000}"
  --num_eval "$NUM_EVAL"
  --num_dev "$NUM_DEV"
)
if [[ "$LET" == 1 ]]; then
  ARGS+=(--let)
fi

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'python train_main.py'
  printf ' %q' "${ARGS[@]}" "$@"
  printf '\n'
else
  python train_main.py "${ARGS[@]}" "$@"
fi
