#!/usr/bin/env bash
set -Eeuo pipefail

echo "Running installation steps in $(pwd)"
export PATH="$HOME/.local/bin:$PATH"

SCR="/netscratch/${USER:-$UID}"
mkdir -p "$SCR/.cache/torch" "$SCR/tmp" "$SCR/logs"
export XDG_CACHE_HOME="$SCR/.cache"
export TORCH_HOME="$SCR/.cache/torch"
export TMPDIR="$SCR/tmp"
export PYTHONUNBUFFERED=1

chmod +x install.sh
./install.sh 

if poetry run python - <<'PY' >/dev/null 2>&1
import torch, cv2
assert torch.__version__.startswith("2.3.")
assert hasattr(cv2.dnn, "DictValue")
PY
then
  echo "Venv already OK – skipping heavy installs."
else
  poetry run python -m pip install --upgrade pip
  poetry run python -m pip install -e .
  poetry run python -m pip install tqdm==4.66.1 optuna==4.2.1 psutil==5.9.0 stable-baselines3

  poetry run python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
  poetry run python -m pip install "opencv-python-headless>=4.8,<4.10"

  poetry run python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
fi

poetry run python - <<'PY'
import torch, cv2
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("cv2:", cv2.__version__, "dnn.DictValue:", hasattr(cv2.dnn, "DictValue"))
PY

stdbuf -oL -eL poetry run python -u pgtg/train.py
