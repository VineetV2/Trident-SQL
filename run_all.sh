#!/bin/bash
# One command, end to end: BIRD dev with the final pipeline (the base stages with their own column finding + our
# Plan-and-Fragment generation, query tree and Cross-Writer Agreement; dev 70%), on one Linux machine with
# 2x 80 GB NVIDIA GPUs. No Slurm needed.
#
#   cp pipeline.env.example pipeline.env     # set WORK (a folder with ~200 GB free) and GPUS
#   bash run_all.sh                          # = setup, then run (about 30 h of GPU time for all 1,534 questions)
#
# Commands:
#   setup                 install uv, make the pipeline's Python environment (one, from requirements.lock),
#                         download gpt-oss-120b and Qwen3-Embedding-0.6B, BIRD dev and the filtered BIRD train file,
#                         build our writer's prep. Needs internet. Every step is skipped when its result is already
#                         there, so setup can be rerun safely.
#   run [--smoke N]       start vLLM, run pipeline.py (every stage, then the score), stop vLLM. If it stops,
#                         the same command resumes.
#                         --smoke N: only the first N dev questions, shorter outputs (a quick test; NOT the real
#                         setting); results in <RUNS_DIR>_debug. The few-shot index is still built over all train
#                         questions (about 4 h) unless FEWSHOT_INDEX points at a finished one.
#   run --preflight       check config, settings and inputs; no GPU, no model call.
#   check                 show what is in place; changes nothing.
#   all (default)         setup + run.
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE=$ROOT_DIR
CONF=${PIPELINE_ENV:-$ROOT_DIR/pipeline.env}
[ -f "$CONF" ] || { echo "!! $CONF missing: run  cp pipeline.env.example pipeline.env  and set WORK"; exit 2; }
set -a; # shellcheck disable=SC1090
source "$CONF"; set +a
: "${WORK:?set WORK in pipeline.env}"
GPUS=${GPUS:-0,1}; RUN_NAME=${RUN_NAME:-dev-run}
PIPE_PY_ENV=${PIPE_PY_ENV:-$WORK/envs/pipeline}
HF_HOME=${HF_HOME:-$WORK/hf}
BIRD_ROOT=${BIRD_ROOT:-$WORK/data/bird}
DEV_DB_ROOT=${DEV_DB_ROOT:-$BIRD_ROOT/dev/dev_databases}
PREP_DIR=${PREP_DIR:-$WORK/data/prep/dev}
VSTORE=${VSTORE:-$WORK/vector_store}
FEWSHOT_INDEX=${FEWSHOT_INDEX:-$WORK/few_shot_index}
RUNS_DIR=${RUNS_DIR:-$WORK/runs}
LOGS=$WORK/logs
export HF_HOME UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/cache/uv} TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$WORK/cache/tiktoken}
export PATH=$HOME/.local/bin:$PATH
MODEL=openai/gpt-oss-120b; MODEL_REV=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
EMB=Qwen/Qwen3-Embedding-0.6B; EMB_REV=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
EMB_DIR=$HF_HOME/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/$EMB_REV
BIRD_DEV_URL=https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
BIRD_TRAIN_FILTERED_URL=https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/main/data/train-00000-of-00001.jsonl

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "!! $*" >&2; exit 2; }
fetch() {  # url, output file (skipped when present)
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
snapshot_ok() {  # repo id, revision: the snapshot exists and no download is half done
  local d=$HF_HOME/hub/models--${1//\//--}
  [ -f "$d/snapshots/$2/config.json" ] && [ -z "$(find "$d/blobs" -name '*.incomplete' 2>/dev/null | head -1)" ]
}

# ------------------------------------------------------------------ checks
check_gpus() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs NVIDIA GPUs"
  local n=0 g mem name
  for g in ${GPUS//,/ }; do
    mem=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') \
      || die "GPU $g not found (GPUS=$GPUS)"
    name=$(nvidia-smi -i "$g" --query-gpu=name --format=csv,noheader)
    [ "${mem:-0}" -ge 79000 ] || die "GPU $g ($name) has ${mem} MiB; gpt-oss-120b here needs 80 GB GPUs (A100/H100 80GB)"
    n=$((n + 1))
  done
  [ "$n" -ge 2 ] || die "needs 2 GPUs in GPUS (got $GPUS)"
  log "GPUs: $GPUS OK"
}
cmd_check() {  # prints what is in place; returns 1 if something run needs is missing
  local ok=0
  item() { if eval "$2"; then echo "  [x] $1"; else echo "  [ ] $1"; ok=1; fi; }
  echo "pipeline.env: $CONF   WORK=$WORK"
  item "uv" "command -v uv >/dev/null"
  item "pipeline env $PIPE_PY_ENV" "[ -x '$PIPE_PY_ENV/bin/python' ]"
  item "model $MODEL@${MODEL_REV:0:7}" "snapshot_ok $MODEL $MODEL_REV"
  item "model $EMB@${EMB_REV:0:7}" "snapshot_ok $EMB $EMB_REV"
  item "BIRD dev $BIRD_ROOT/dev/dev.json" "[ -f '$BIRD_ROOT/dev/dev.json' ]"
  item "BIRD dev databases $DEV_DB_ROOT" "[ \$(ls -d '$DEV_DB_ROOT'/*/ 2>/dev/null | wc -l) -ge 11 ]"
  item "BIRD train file $BIRD_ROOT/train/train.json" "[ -f '$BIRD_ROOT/train/train.json' ]"
  item "our writer's prep $PREP_DIR" "[ \$(ls '$PREP_DIR'/*/columns.json 2>/dev/null | wc -l) -ge 11 ]"
  echo "  value index: $VSTORE ($(ls "$VSTORE"/*/success_flag 2>/dev/null | wc -l) databases built; the run builds the rest)"
  echo "  few-shot index: $FEWSHOT_INDEX ($([ -f "$FEWSHOT_INDEX/manifest.json" ] && echo built || echo 'built by the run, about 4 h'))"
  return $ok
}

# ------------------------------------------------------------------ setup
setup_uv() {
  if command -v uv >/dev/null; then log "uv: present ($(uv --version))"; return; fi
  log "uv: installing (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null || die "uv install failed"
}
ENV_CHECK="import vllm, torch, sqlglot, yaml, huggingface_hub, sentence_transformers, loguru; assert vllm.__version__ == '0.23.0'"
setup_pipeline_env() {
  if [ -x "$PIPE_PY_ENV/bin/python" ] && "$PIPE_PY_ENV/bin/python" -c "$ENV_CHECK" 2>/dev/null; then
    log "pipeline env: present ($PIPE_PY_ENV)"; return
  fi
  log "pipeline env: creating $PIPE_PY_ENV (Python 3.12, exact versions from requirements.lock)"
  uv venv --python 3.12 "$PIPE_PY_ENV"
  uv pip install --python "$PIPE_PY_ENV/bin/python" -r "$CODE/requirements.lock"
  "$PIPE_PY_ENV/bin/python" -c "$ENV_CHECK; import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
}
setup_models() {
  local hf=$PIPE_PY_ENV/bin/hf; [ -x "$hf" ] || hf=$PIPE_PY_ENV/bin/huggingface-cli
  if snapshot_ok "$MODEL" "$MODEL_REV"; then log "model $MODEL: present"; else
    log "model $MODEL@$MODEL_REV: downloading (about 65 GB; the original/ and metal/ copies are skipped)"
    HF_HUB_OFFLINE=0 "$hf" download "$MODEL" --revision "$MODEL_REV" --exclude "original/*" --exclude "metal/*"
  fi
  if snapshot_ok "$EMB" "$EMB_REV"; then log "model $EMB: present"; else
    log "model $EMB@$EMB_REV: downloading"
    HF_HUB_OFFLINE=0 "$hf" download "$EMB" --revision "$EMB_REV"
  fi
  snapshot_ok "$MODEL" "$MODEL_REV" && snapshot_ok "$EMB" "$EMB_REV" || die "model download incomplete"
}
setup_bird_dev() {
  if [ -f "$BIRD_ROOT/dev/dev.json" ] && [ "$(ls -d "$DEV_DB_ROOT"/*/ 2>/dev/null | wc -l)" -ge 11 ]; then
    log "BIRD dev: present"; return
  fi
  log "BIRD dev: downloading $BIRD_DEV_URL"
  fetch "$BIRD_DEV_URL" "$WORK/downloads/dev.zip"
  rm -rf "$WORK/downloads/dev_unzip" && mkdir -p "$WORK/downloads/dev_unzip" "$BIRD_ROOT"
  unzip -q -o "$WORK/downloads/dev.zip" -d "$WORK/downloads/dev_unzip"
  [ -d "$BIRD_ROOT/dev" ] || mv "$WORK/downloads/dev_unzip/dev_20240627" "$BIRD_ROOT/dev"
  [ -d "$BIRD_ROOT/dev/dev_databases" ] || (cd "$BIRD_ROOT/dev" && unzip -q -o dev_databases.zip)
  rm -rf "$WORK/downloads/dev_unzip" "$BIRD_ROOT/dev/__MACOSX"
  "$PIPE_PY_ENV/bin/python" -c "import json,sys; n=len(json.load(open(sys.argv[1]))); assert n==1534, n; print('dev.json:', n, 'questions')" \
    "$BIRD_ROOT/dev/dev.json"
}
setup_bird_train() {
  if [ -f "$BIRD_ROOT/train/train.json" ]; then log "BIRD train file: present"; return; fi
  log "BIRD train file: downloading birdsql/bird23-train-filtered (the few-shot pool)"
  fetch "$BIRD_TRAIN_FILTERED_URL" "$BIRD_ROOT/train/train_filtered.jsonl"
  # JSON lines to one JSON list; a missing question_id becomes the line index (from 0), a missing difficulty ""
  "$PIPE_PY_ENV/bin/python" - "$BIRD_ROOT/train/train_filtered.jsonl" "$BIRD_ROOT/train/train.json" <<'PY'
import json, sys
records = []
for question_id, line in enumerate(open(sys.argv[1], encoding="utf-8")):
    line = line.strip()
    if not line:
        continue
    record = json.loads(line)
    record.setdefault("question_id", question_id)
    record.setdefault("difficulty", "")
    records.append(record)
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(records, ensure_ascii=False, indent=2))
print("train.json:", len(records), "questions")
PY
}
setup_prep() {
  if [ "$(ls "$PREP_DIR"/*/columns.json 2>/dev/null | wc -l)" -ge 11 ]; then log "our writer's prep: present"; return; fi
  log "our writer's prep: building $PREP_DIR"
  (cd "$CODE" && "$PIPE_PY_ENV/bin/python" -m stage0_dataset.prep.build --db-root "$DEV_DB_ROOT" --out-root "$PREP_DIR")
}
setup_tiktoken() {
  "$PIPE_PY_ENV/bin/python" -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('o200k_base', 'cl100k_base')]" \
    && log "tiktoken encodings: cached in $TIKTOKEN_CACHE_DIR"
}
cmd_setup() {
  mkdir -p "$WORK" "$LOGS"
  log "free space under $WORK: $(df -h "$WORK" | awk 'NR==2{print $4}') (needs about 200 GB)"
  setup_uv; setup_pipeline_env; setup_models; setup_bird_dev; setup_bird_train; setup_prep; setup_tiktoken
  log "setup done"; cmd_check || true
}

# ------------------------------------------------------------------ run
cmd_run() {
  local smoke="" preflight=""
  while [ $# -gt 0 ]; do
    case $1 in
      --smoke) smoke=${2:?--smoke needs a number}; shift 2;;
      --preflight) preflight=1; shift;;
      *) die "unknown option $1 (run [--smoke N] [--preflight])";;
    esac
  done
  cmd_check >/dev/null || { cmd_check || true; die "something is missing: run  bash run_all.sh setup"; }
  mkdir -p "$LOGS" "$RUNS_DIR"
  local run_dir=$RUNS_DIR/$RUN_NAME
  if [ -n "$smoke" ]; then
    [[ "$smoke" =~ ^[1-9][0-9]*$ ]] || die "--smoke needs a whole number"
    run_dir=${RUNS_DIR}_debug/$RUN_NAME
    log "SMOKE run: first $smoke questions, shorter outputs (not the real setting); results in $run_dir"
  fi
  # the pipeline's own arguments; the vLLM address is added by serve_and_run_local.sh (TP_BASE_URL)
  local args=(--run-dir "$run_dir" --data-root "$BIRD_ROOT" --db-root "$DEV_DB_ROOT" --prep "$PREP_DIR"
              --vector-store "$VSTORE" --fewshot-index "$FEWSHOT_INDEX" --embed-model "$EMB_DIR")
  [ -n "$smoke" ] && args+=(--max-samples "$smoke")
  if [ -n "$preflight" ]; then
    (cd "$CODE" && "$PIPE_PY_ENV/bin/python" pipeline.py "${args[@]}" --base-url http://unused --preflight)
    return
  fi
  check_gpus
  export B2_SHARED_EMBED_GPU=1 PIPE_PY_ENV PIPE_LOGS=$LOGS GPUS VLLM_READY_MIN=${VLLM_READY_MIN:-90}
  local runlog=$LOGS/run_${RUN_NAME}_$(date +%Y%m%dT%H%M%S).log
  log "run $RUN_NAME: log $runlog (vLLM start takes 5-75 min)"
  bash "$CODE/serving/serve_and_run_local.sh" bash -c 'exec python pipeline.py "$@" --base-url "$TP_BASE_URL"' _ "${args[@]}" \
    2>&1 | tee -a "$runlog"
}

case ${1:-all} in
  setup) cmd_setup;;
  run) shift; cmd_run "$@";;
  check) cmd_check;;
  all) cmd_setup; cmd_run;;
  -h|--help|help) sed -n '2,24p' "$0";;
  *) die "usage: bash run_all.sh [setup|run [--smoke N] [--preflight]|check|all]";;
esac
