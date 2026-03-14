#!/usr/bin/env bash
set -euo pipefail

echo "=== Takkeli Mission Init ==="

# Check tooling
echo "Checking tooling..."
uv --version
ruff --version
ty --version
python3 --version

# Install workspace dependencies
echo "Installing workspace dependencies..."
uv sync

# Verify workspace structure
echo "Verifying workspace members..."
for dir in 01_data_filtering 02_pretraining 03_alignment 04_inference_eval; do
  if [ -d "$dir" ]; then
    echo "  Found: $dir"
  else
    echo "  Missing: $dir"
  fi
done

echo "=== Init Complete ==="
