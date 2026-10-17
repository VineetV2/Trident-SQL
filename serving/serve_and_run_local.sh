#!/bin/bash
# For a machine WITHOUT Slurm: start a vLLM server for gpt-oss-120b (settings
# from [model] and [server] in configs/pipeline_config.toml) on 2 GPUs, wait until it answers, run the client command
# given as arguments, then stop the server. Same server arguments and environment as the cluster jobs
# (--disable-custom-all-reduce, --seed 0, expandable segments, no FlashInfer sampler, harmony date pinned to
# 2026-09-23).
# Usage (normally called by run_all.sh at the repo root):
#   PIPE_PY_ENV=<python env with vLLM 0.23.0> HF_HOME=<model cache> PIPE_LOGS=<dir> \
#     bash serving/serve_and_run_local.sh <command...>
# Optional: GPUS (default 0,1), VLLM_READY_MIN (default 90), VLLM_GPU_UTIL, B2_SHARED_EMBED_GPU=1 (the embedding model
# shares GPU 0 with vLLM, whose memory share drops to 0.80, as in the cluster runs), PORT, TIKTOKEN_CACHE_DIR.
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
: "${PIPE_PY_ENV:?set PIPE_PY_ENV (the Python env with vLLM 0.23.0)}" "${HF_HOME:?set HF_HOME (model cache)}"
PIPE_LOGS=${PIPE_LOGS:-$CODE/../logs}; mkdir -p "$PIPE_LOGS"
RUN_ID=$(date +%Y%m%dT%H%M%S)
PORT=${PORT:-$((8000 + RANDOM % 900))}
cd "$CODE"
# activate the env: a venv has bin/activate; a conda env is put on PATH directly
if [ -f "$PIPE_PY_ENV/bin/activate" ]; then source "$PIPE_PY_ENV/bin/activate"; else export PATH=$PIPE_PY_ENV/bin:$PATH CONDA_PREFIX=$PIPE_PY_ENV; fi

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0             # FlashInfer's sampler JIT-compiles and needs nvcc
export VLLM_SYSTEM_START_DATE=2026-09-23         # pin harmony "current date" so prompts don't change by run date
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$CODE/../work_cache/tiktoken}
export TP_BASE_URL="http://127.0.0.1:${PORT}"
unset NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_CUMEM_ENABLE

export MODEL_CFG=${MODEL_CFG:-configs/pipeline_config.toml}
read -r MODEL REV TP MAXLEN SERVED <<<"$(python - <<'PY'
import os, tomllib; c=tomllib.load(open(os.environ["MODEL_CFG"], "rb")); m, s = c["model"], c["server"]
print(m["name"], m["hf_revision"], s["tensor_parallel_size"], s["max_model_len"], s.get("served_model_name") or m["name"])
PY
)"
export CUDA_VISIBLE_DEVICES=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}
echo "run=$RUN_ID host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
echo "model=$MODEL rev=$REV tp=$TP max_model_len=$MAXLEN served_as=$SERVED cfg=$MODEL_CFG port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

IFS=, read -ra GPU_LIST <<<"$CUDA_VISIBLE_DEVICES"
[ "${#GPU_LIST[@]}" -ge "$TP" ] || { echo "!! need >= $TP GPUs, got ${#GPU_LIST[@]} (GPUS=$CUDA_VISIBLE_DEVICES)"; exit 2; }
VLLM_GPUS=$(IFS=,; echo "${GPU_LIST[*]:0:$TP}")
if [ "${#GPU_LIST[@]}" -gt "$TP" ]; then
  export B2_EMBED_GPU=${GPU_LIST[$TP]}              # a spare GPU: the embedding model runs there alone
elif [ -n "${B2_SHARED_EMBED_GPU:-}" ]; then
  export B2_EMBED_GPU=${GPU_LIST[0]}                # shared with vLLM: vLLM capped lower below
  VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.80}
fi
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
echo "vllm gpus=$VLLM_GPUS gpu_memory_utilization=$VLLM_GPU_UTIL embedding gpu=${B2_EMBED_GPU:-none}"

# refuse GPUs that already hold memory or need a reset (leftovers make vLLM fail at start)
if nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | grep -q "requires reset"; then
  echo "!! a GPU reports 'requires reset': not starting"; exit 3
fi
for g in ${VLLM_GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${used:-0}" -le "${GPU_CLEAN_MAX_MB:-2000}" ] || { echo "!! GPU $g already holds ${used} MiB: not starting vLLM"; exit 3; }
done

# vLLM runs in its own process group (API server, engine, tensor-parallel workers); cleanup stops the whole group
VLLM_PGID="" CLIENT_PID=""
group_alive() { [ -n "$VLLM_PGID" ] && pgrep -g "$VLLM_PGID" >/dev/null 2>&1; }
stop_vllm() {
  [ -n "$VLLM_PGID" ] || return 0
  local sig
  for sig in INT TERM KILL; do
    group_alive || break
    kill -s "$sig" -- "-$VLLM_PGID" 2>/dev/null || true
    for _ in $(seq 1 30); do group_alive || break; sleep 1; done
  done
  if group_alive; then echo "WARNING: vLLM group $VLLM_PGID still alive after SIGKILL"; else echo "vllm stopped at $(date -Is)"; fi
}
# the client (pipeline.py) stops its running stage on TERM (up to 10 s, then SIGKILL): wait for it before vLLM goes away
CLIENT_STOP_WAIT_S=30
stop_client() {
  [ -n "$CLIENT_PID" ] && kill -0 "$CLIENT_PID" 2>/dev/null || return 0
  kill -TERM "$CLIENT_PID" 2>/dev/null || true
  local _
  for _ in $(seq 1 "$CLIENT_STOP_WAIT_S"); do kill -0 "$CLIENT_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$CLIENT_PID" 2>/dev/null; then
    echo "WARNING: client $CLIENT_PID still running ${CLIENT_STOP_WAIT_S} s after TERM: SIGKILL"
    kill -KILL "$CLIENT_PID" 2>/dev/null || true
  fi
  wait "$CLIENT_PID" 2>/dev/null || true
}
cleanup() {
  local rc=$?
  set +e
  trap - EXIT
  # a second Ctrl-C, or a closed output pipe (run_all.sh's tee dies on Ctrl-C too), must not cut the cleanup short
  trap '' INT TERM PIPE
  stop_client
  stop_vllm
  exit "$rc"
}
trap cleanup EXIT
trap 'trap "" PIPE; echo "signal received at $(date -Is); stopping" || true; exit 143' INT TERM

LOG="$PIPE_LOGS/vllm-${RUN_ID}.log"
CUDA_VISIBLE_DEVICES=$VLLM_GPUS setsid python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --revision "$REV" --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" --host 127.0.0.1 --port "$PORT" --seed 0 \
  --disable-custom-all-reduce \
  ${VLLM_EXTRA_ARGS:-} > "$LOG" 2>&1 &
VLLM_PID=$!
VLLM_PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
echo "vllm pid=$VLLM_PID log=$LOG"

VLLM_READY_MIN=${VLLM_READY_MIN:-90}
for i in $(seq 1 $((VLLM_READY_MIN * 6))); do
  if curl -sf "$TP_BASE_URL/v1/models" >/dev/null; then echo "vllm ready after $((i*10)) s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "!! vllm died; tail of $LOG:"; tail -50 "$LOG"; exit 1; fi
  if [ $((i % 6)) -eq 0 ] && nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>&1 | grep -q "requires reset"; then
    echo "!! a GPU reports 'requires reset' while vLLM starts: stopping"; exit 3
  fi
  sleep 10
done
curl -sf "$TP_BASE_URL/v1/models" >/dev/null || { echo "!! vllm not ready after $VLLM_READY_MIN min"; tail -50 "$LOG"; exit 1; }

echo "running: $*"
# the client sees only the embedding GPU (if any), so its "cuda" never lands on a vLLM-only GPU
if [ -n "${B2_EMBED_GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$B2_EMBED_GPU; fi
"$@" &
CLIENT_PID=$!
wait "$CLIENT_PID"
echo "end=$(date -Is)"
