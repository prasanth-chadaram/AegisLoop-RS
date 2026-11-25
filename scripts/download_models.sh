#!/usr/bin/env bash
# Download real model weights OUTSIDE the git repository.
# Nothing written by this script is ever committed (see .gitignore).
set -euo pipefail

DEFAULT_DIR="/mnt/data/aegisloop-models"
if [[ -n "${AEGIS_MODEL_DIR:-}" ]]; then
  MODEL_DIR="$AEGIS_MODEL_DIR"
elif [[ -d "/mnt/data" && -w "/mnt/data" ]]; then
  MODEL_DIR="$DEFAULT_DIR"
else
  MODEL_DIR="$HOME/.cache/aegisloop-models"
fi

DEPTH_URL="https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx"
VLM_URL="https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf"
PROJ_URL="https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf"

DEPTH_FILE="$MODEL_DIR/depth-anything-v2-small.onnx"
VLM_FILE="$MODEL_DIR/SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf"
PROJ_FILE="$MODEL_DIR/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf"

mkdir -p "$MODEL_DIR"

fetch() {
  local url="$1" dest="$2" min_bytes="$3"
  if [[ -f "$dest" ]]; then
    local size
    size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if (( size >= min_bytes )); then
      echo "skip  $dest ($size bytes)"
      return 0
    fi
    echo "remove incomplete $dest ($size bytes)"
    rm -f "$dest"
  fi
  echo "fetch $url"
  echo "  -> $dest"
  curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  sha256sum "$dest"
}

case "${1:-all}" in
  depth) fetch "$DEPTH_URL" "$DEPTH_FILE" 25000000 ;;
  vlm)
    fetch "$VLM_URL" "$VLM_FILE" 120000000
    fetch "$PROJ_URL" "$PROJ_FILE" 180000000
    ;;
  all)
    fetch "$DEPTH_URL" "$DEPTH_FILE" 25000000
    fetch "$VLM_URL" "$VLM_FILE" 120000000
    fetch "$PROJ_URL" "$PROJ_FILE" 180000000
    ;;
  *)
    echo "usage: $0 [all|depth|vlm]" >&2
    exit 2
    ;;
esac

echo "model dir: $MODEL_DIR"
echo "real-model bench: cargo run --release --features onnx -- bench --real-model --model-dir '$MODEL_DIR'"
