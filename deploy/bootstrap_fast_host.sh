#!/usr/bin/env bash
# Run from an already-authorized Linux GPU host. This never rents a host or
# switches accounts. Initial pulls/downloads use the network; inference cannot.
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077

command -v python3 >/dev/null
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
command -v timeout >/dev/null
[[ $(uname -s) == Linux ]] || { echo 'A Linux NVIDIA host is required.' >&2; exit 1; }
[ -f deploy/.env ] || cp deploy/.env.example deploy/.env

# Read .env as data, never source/eval shell code. Explicit process environment
# wins, followed by existing .env values, then the documented defaults.
setting() {
  python3 - "$1" "$2" <<'PY'
import os, pathlib, sys
name, default = sys.argv[1:]
values = {}
for line in pathlib.Path('deploy/.env').read_text().splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('\"\'')
value = os.environ.get(name, values.get(name, default))
if '\n' in value or '\r' in value:
    raise SystemExit('Invalid multiline deployment setting')
print(value)
PY
}
upsert() {
  python3 - "$1" "$2" <<'PY'
import pathlib, sys
name, value = sys.argv[1:]
if '\n' in value or '\r' in value:
    raise SystemExit('Invalid multiline deployment setting')
path = pathlib.Path('deploy/.env')
lines = [line for line in path.read_text().splitlines() if not line.startswith(name + '=')]
# Single quoting preserves literal '$', '#' and spaces in Docker Compose.
if "'" in value:
    raise SystemExit('Single quotes are unsupported in deployment setting values')
lines.append(name + "='" + value + "'")
path.write_text('\n'.join(lines) + '\n')
path.chmod(0o600)
PY
}

MODEL_DIR=$(setting FORTE_FAST_MODEL_DIR /data/models/Qwen3-VL-30B-A3B-Instruct)
MODEL_REPO=$(setting FORTE_FAST_MODEL_REPO Qwen/Qwen3-VL-30B-A3B-Instruct)
MODEL_REVISION=$(setting FORTE_FAST_MODEL_REVISION '')
SOURCE_IMAGE=$(setting FORTE_FAST_VLLM_SOURCE_IMAGE vllm/vllm-openai:v0.31.0-cu129)
MODEL_IMAGE=$(setting FORTE_MODEL_IMAGE forte-model:vllm-0.31.0)
GPU_ID=$(setting FORTE_INFERENCE_GPU_ID 0)
SECRETS_DIR=$(setting FORTE_SECRETS_DIR ./secrets)
READY_TIMEOUT=$(setting FORTE_FAST_READY_TIMEOUT_SECONDS 900)
if [[ ! $MODEL_REVISION =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo 'Set FORTE_FAST_MODEL_REVISION to the reviewed 40-character Hugging Face commit SHA.' >&2
  exit 1
fi
MODEL_REVISION=${MODEL_REVISION,,}
if [[ ! $READY_TIMEOUT =~ ^[1-9][0-9]*$ ]] || (( READY_TIMEOUT > 3600 )); then
  echo 'FORTE_FAST_READY_TIMEOUT_SECONDS must be an integer in 1..3600.' >&2
  exit 1
fi
if [[ $MODEL_DIR != /* || $MODEL_DIR == / || $MODEL_DIR == *','* ]]; then
  echo 'FORTE_FAST_MODEL_DIR must be an absolute directory without commas.' >&2
  exit 1
fi

echo '1/6 Check host and pull the configured model runtime'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker info --format '{{.ServerVersion}}' >/dev/null
docker compose version --short
docker pull "$SOURCE_IMAGE"
docker tag "$SOURCE_IMAGE" "$MODEL_IMAGE"

echo '2/6 Prepare the pinned model snapshot'
if [ -f "$MODEL_DIR/config.json" ]; then
  # A snapshot of another revision is never silently reused or overwritten.
  python3 tools/fast_host_preflight.py --snapshot-only --model-dir "$MODEL_DIR" \
    --model-repo "$MODEL_REPO" --model-revision "$MODEL_REVISION"
else
  # Resume interrupted downloads in a revision-specific staging directory;
  # promote only a complete, validated snapshot into the serving mount.
  DOWNLOAD_DIR="${MODEL_DIR}.download-${MODEL_REVISION}"
  mkdir -p "$DOWNLOAD_DIR"
  docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$DOWNLOAD_DIR,dst=/out" \
    --env HF_HOME=/tmp/huggingface --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env "FORTE_DOWNLOAD_REPO=$MODEL_REPO" --env "FORTE_DOWNLOAD_REVISION=$MODEL_REVISION" \
    --entrypoint python3 -i "$MODEL_IMAGE" - <<'PY'
import json, os, pathlib
from huggingface_hub import HfApi, snapshot_download
repo = os.environ['FORTE_DOWNLOAD_REPO']
revision = os.environ['FORTE_DOWNLOAD_REVISION']
if HfApi().model_info(repo, revision=revision).sha.lower() != revision:
    raise SystemExit('Resolved model revision differs from requested immutable pin')
snapshot_download(repo_id=repo, revision=revision, local_dir='/out')
receipt = {'schema_version': 'forte.fast-model-snapshot.v1', 'repo_id': repo, 'revision': revision}
pathlib.Path('/out/.forte-model-receipt.json').write_text(json.dumps(receipt) + '\n')
PY
  python3 tools/fast_host_preflight.py --snapshot-only --model-dir "$DOWNLOAD_DIR" \
    --model-repo "$MODEL_REPO" --model-revision "$MODEL_REVISION"
  # rmdir refuses a nonempty pre-existing destination. Never replace weights
  # or unrelated user files when a download is promoted.
  if [ -e "$MODEL_DIR" ]; then rmdir "$MODEL_DIR"; fi
  mv -T -- "$DOWNLOAD_DIR" "$MODEL_DIR"
fi

echo '3/6 Check actual CUDA/bf16 and container model-file access'
python3 tools/fast_host_preflight.py --model-dir "$MODEL_DIR" --model-repo "$MODEL_REPO" \
  --model-revision "$MODEL_REVISION" --image "$MODEL_IMAGE" --gpu "$GPU_ID"

echo '4/6 Build application images and preserve deployment secrets'
docker build -f deploy/Dockerfile -t forte-review:base .
docker build -f deploy/Dockerfile.fastpatch --build-arg BASE_IMAGE=forte-review:base -t forte-review:fast .
if [[ $SECRETS_DIR == /* ]]; then SECRETS_PATH=$SECRETS_DIR; else SECRETS_PATH="deploy/$SECRETS_DIR"; fi
python3 deploy/prepare_secrets.py --directory "$SECRETS_PATH"
upsert FORTE_APP_IMAGE forte-review:fast
upsert FORTE_MODEL_IMAGE "$MODEL_IMAGE"
upsert FORTE_FAST_MODEL_DIR "$MODEL_DIR"
upsert FORTE_FAST_MODEL_REPO "$MODEL_REPO"
upsert FORTE_FAST_MODEL_REVISION "$MODEL_REVISION"
upsert FORTE_INFERENCE_GPU_ID "$GPU_ID"
upsert FORTE_SECRETS_DIR "$SECRETS_DIR"
for name in FORTE_BIND_HOST FORTE_CORS_ALLOWED_ORIGINS FORTE_PUBLIC_API_BASE_URL FORTE_TRUSTED_SCENE_CONTEXT; do
  if [[ -v $name ]]; then upsert "$name" "${!name}"; fi
done

echo '5/6 Start one worker and wait for all readiness checks'
cd deploy
compose=(docker compose --env-file .env -f compose.yaml -f compose.fast.yaml)
"${compose[@]}" config --quiet
ready=false
deadline=$((SECONDS + READY_TIMEOUT))
if ! timeout "$READY_TIMEOUT" "${compose[@]}" up -d --scale worker=1; then
  echo 'Container startup failed or timed out; deployment is not ready.' >&2
  "${compose[@]}" ps
  exit 1
fi
while (( SECONDS < deadline )); do
  if "${compose[@]}" exec -T api python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz',timeout=5)" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
if [[ $ready != true ]]; then
  echo "Readiness failed within ${READY_TIMEOUT}s; deployment is not ready. Existing containers are retained for inspection." >&2
  "${compose[@]}" ps
  exit 1
fi

echo '6/6 Ready; run full quality and public-API parallel-load evaluation before cutover'
"${compose[@]}" ps
echo 'API credentials remain in deploy/secrets/api_keys. Bind address is preserved from deploy/.env (default: loopback).'
