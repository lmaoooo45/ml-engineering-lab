#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ml-engineering-lab
python -m venv .venv
export UV_CACHE_DIR=/workspace/.cache/uv
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip check --python .venv/bin/python
.venv/bin/python -m pytest -q
