#!/usr/bin/env bash
# run_openrouter_free.sh — test1v2 exp grid over OpenRouter's free models, ONE MODEL AT A TIME.
#
# Sequential by design: OpenRouter's free tier is rate-limited per account, so overlapping runs
# would just produce 429s. Each model gets the full 101-point recall-proof grid, then the next
# starts. A model that fails outright does not stop the sequence.
#
# Usage: OPENROUTER_API_KEY=sk-or-v1-... ./run_openrouter_free.sh [func]
set -u
FUNC=${1:-exp}
HERE=$(cd "$(dirname "$0")" && pwd)
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

MODELS=(
  "nvidia/nemotron-nano-9b-v2:free"
  "nvidia/nemotron-3-nano-30b-a3b:free"
  "nvidia/nemotron-3-super-120b-a12b:free"
  "nvidia/nemotron-3-ultra-550b-a55b:free"
  "google/gemma-4-31b-it:free"
  "openai/gpt-oss-20b:free"
)

for m in "${MODELS[@]}"; do
  # tag: strip provider prefix and the :free suffix, keep it filesystem-safe
  tag="${FUNC}_$(echo "$m" | sed 's|.*/||; s|:free||; s|[^A-Za-z0-9._-]|-|g')_orfree"
  echo "=============================================================="
  echo "[$(date +%H:%M:%S)] $m  ->  $tag"
  echo "=============================================================="
  "$HERE/test1v2/run_test1v2.py" --func "$FUNC" \
    --base-url https://openrouter.ai/api/v1 --model "$m" \
    --api-key-env OPENROUTER_API_KEY \
    --concurrency 4 --timeout 900 --retries 2 --tag "$tag" \
    || echo "[$(date +%H:%M:%S)] $m FAILED (rc=$?) - continuing"
done
echo "[$(date +%H:%M:%S)] all done"
