#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 {check|smoke|train|infer|eval|all} [--seed N] [--inference-engine eager|compiled] [--batch-size N] [options]" >&2
  exit 2
fi
shift

SEED="${SEED:-0}"
INFER_MODE="both"
EVAL_TASK="both"
DATA_ROOT="${DATA_ROOT:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-/home/jiahao/task/csgo_benchmark_v2_eval_general}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
PUFFIN_PYTHON="${PUFFIN_PYTHON:-python}"
OUTPUT_ROOT_EXPLICIT=0
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/Puffin}"
CHECKPOINT=""
CHECKPOINT_EXPLICIT=0
WORK_DIR=""
HAS_MAX_SAMPLES=0
HAS_STEPS=0
EXTRA_ARGS=()
INFER_ARGS=()
HAS_MAPS=0

while (($#)); do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --mode)
      INFER_MODE="$2"
      shift 2
      ;;
    --task)
      EVAL_TASK="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --shared-eval-dir)
      SHARED_EVAL_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      OUTPUT_ROOT_EXPLICIT=1
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      CHECKPOINT_EXPLICIT=1
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --steps|--cfg-scale|--max-samples)
      if [[ "$1" == "--max-samples" ]]; then
        HAS_MAX_SAMPLES=1
      fi
      if [[ "$1" == "--steps" ]]; then
        HAS_STEPS=1
      fi
      INFER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --inference-engine|--batch-size|--decoder-chunk-size)
      INFER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --maps)
      HAS_MAPS=1
      shift
      map_args=()
      while (($#)) && [[ "$1" != --* ]]; do
        map_args+=("$1")
        shift
      done
      if ((${#map_args[@]} == 0)); then
        echo "--maps requires at least one map name" >&2
        exit 2
      fi
      INFER_ARGS+=(--maps "${map_args[@]}")
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$MODE" == "smoke" && "$OUTPUT_ROOT_EXPLICIT" == "0" ]]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/csgo_benchmark_v2_smoke/Puffin"
fi
if [[ "$HAS_MAPS" == "1" && "$MODE" != "infer" ]]; then
  echo "--maps is only valid with infer; all/smoke/eval require complete benchmark coverage" >&2
  exit 2
fi
if [[ "$HAS_MAX_SAMPLES" == "1" && "$MODE" != "infer" && "$MODE" != "smoke" ]]; then
  echo "--max-samples is diagnostic-only and is valid only with infer or smoke" >&2
  exit 2
fi
if [[ "$INFER_MODE" != "both" && "$MODE" != "infer" ]]; then
  echo "--mode is only configurable for infer; all and smoke require both generation tracks" >&2
  exit 2
fi
if [[ "$CHECKPOINT_EXPLICIT" == "1" && "$MODE" != "infer" ]]; then
  echo "--checkpoint is only valid with infer; all must use the checkpoint it just trained" >&2
  exit 2
fi

export DATA_ROOT SHARED_EVAL_DIR OUTPUT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SEED_ROOT="$OUTPUT_ROOT/seed_$SEED"

run_train() {
  "$PUFFIN_PYTHON" "$PROJECT_ROOT/train_seen10.py" --seed "$SEED" \
    --data-root "$DATA_ROOT" --shared-eval-dir "$SHARED_EVAL_DIR" "${EXTRA_ARGS[@]}"
}

run_infer() {
  local selected_checkpoint="$CHECKPOINT"
  if [[ -z "$selected_checkpoint" ]]; then
    if [[ "$MODE" == "all" && -n "$WORK_DIR" ]]; then
      if [[ "$WORK_DIR" = /* ]]; then
        selected_checkpoint="$WORK_DIR/best.pth"
      else
        selected_checkpoint="$PROJECT_ROOT/$WORK_DIR/best.pth"
      fi
    else
      selected_checkpoint="$SEED_ROOT/checkpoints/best.pth"
    fi
  fi
  "$PUFFIN_PYTHON" "$PROJECT_ROOT/infer_seen10.py" \
    --mode "$INFER_MODE" --seed "$SEED" --checkpoint "$selected_checkpoint" \
    --data-root "$DATA_ROOT" --shared-eval-dir "$SHARED_EVAL_DIR" \
    --output-root "$OUTPUT_ROOT" "${INFER_ARGS[@]}"
}

run_one_eval() {
  local task="$1"
  "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" "$task" \
    --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
    --pred-root "$SEED_ROOT/$task/gen_imgs" \
    --data-root "$DATA_ROOT" \
    --output "$SEED_ROOT/evaluation_shared/$task"
}

run_eval() {
  case "$EVAL_TASK" in
    both)
      run_one_eval discrete
      run_one_eval continuous
      ;;
    discrete|continuous)
      run_one_eval "$EVAL_TASK"
      ;;
    *)
      echo "--task must be discrete, continuous, or both" >&2
      exit 2
      ;;
  esac
}

run_smoke_eval() {
  "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke discrete \
    --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
    --pred-root "$SEED_ROOT/discrete/gen_imgs" \
    --data-root "$DATA_ROOT" --limit 1
  "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke continuous \
    --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
    --pred-root "$SEED_ROOT/continuous/gen_imgs" \
    --data-root "$DATA_ROOT" --max-clips 1 --frame-only
}

case "$MODE" in
  check)
    python_cmd=("$PUFFIN_PYTHON")
    "${python_cmd[@]}" -m compileall -q \
      "$PROJECT_ROOT/csgo_seen10" \
      "$PROJECT_ROOT/train_seen10.py" \
      "$PROJECT_ROOT/infer_seen10.py"
    "${python_cmd[@]}" -m unittest discover -s "$PROJECT_ROOT/tests" -p 'test_csgo_seen10.py'
    "${python_cmd[@]}" "$PROJECT_ROOT/train_seen10.py" --dataset-only --seed "$SEED" \
      --data-root "$DATA_ROOT" --shared-eval-dir "$SHARED_EVAL_DIR" "${EXTRA_ARGS[@]}"
    ;;
  smoke)
    python_cmd=("$PUFFIN_PYTHON")
    "${python_cmd[@]}" -m compileall -q \
      "$PROJECT_ROOT/csgo_seen10" \
      "$PROJECT_ROOT/train_seen10.py" \
      "$PROJECT_ROOT/infer_seen10.py"
    "${python_cmd[@]}" -m unittest discover -s "$PROJECT_ROOT/tests" -p 'test_csgo_seen10.py'
    "${python_cmd[@]}" "$PROJECT_ROOT/train_seen10.py" --dataset-only --seed "$SEED" \
      --data-root "$DATA_ROOT" --shared-eval-dir "$SHARED_EVAL_DIR" "${EXTRA_ARGS[@]}"
    "${python_cmd[@]}" "$PROJECT_ROOT/train_seen10.py" --smoke --seed "$SEED" \
      --data-root "$DATA_ROOT" --shared-eval-dir "$SHARED_EVAL_DIR" "${EXTRA_ARGS[@]}"
    if [[ -n "$WORK_DIR" ]]; then
      if [[ "$WORK_DIR" = /* ]]; then
        CHECKPOINT="$WORK_DIR/iter_1.pth"
      else
        CHECKPOINT="$PROJECT_ROOT/$WORK_DIR/iter_1.pth"
      fi
    else
      CHECKPOINT="$SEED_ROOT/smoke/checkpoints/iter_1.pth"
    fi
    if [[ "$HAS_MAX_SAMPLES" == "0" ]]; then
      INFER_ARGS+=(--max-samples 1)
    fi
    if [[ "$HAS_STEPS" == "0" ]]; then
      INFER_ARGS+=(--steps 1)
    fi
    run_infer
    run_smoke_eval
    ;;
  train)
    run_train
    ;;
  infer)
    run_infer
    ;;
  eval)
    run_eval
    ;;
  all)
    run_train
    run_infer
    run_eval
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
