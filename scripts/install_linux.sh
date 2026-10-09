#!/usr/bin/env bash
set -euo pipefail
# Run under Ubuntu/WSL2 with Python 3.10, from the repository root.
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.2 wheel==0.45.1
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install mmcv==2.1.0 --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
python -m pip install -r requirements-inference.txt
python -m pip install --no-deps -e .
python -m pip install pytest==8.3.5
python -m pip check
python -m pytest -q
python -m prelabel doctor
