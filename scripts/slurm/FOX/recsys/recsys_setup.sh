#!/bin/bash
# ONE-TIME setup on a FOX login node (compute nodes inherit ~/.local).
# --no-deps everywhere so pip can NEVER pull a user-site numpy/torch that
# would shadow the NLPL module stack.
set -o errexit
source "$(dirname "$0")/modules.sh"

# The module stack already ships tensorboard 2.18.0 built against its own
# protobuf 5.28 runtime. Any user-site tensorboard SHADOWS it and breaks
# torch.utils.tensorboard with a protobuf Gencode/Runtime VersionError
# (observed with user-site TB 2.21 / gencode 6.31 on 2026-07-26), so:
# install NO tensorboard, and remove any user-site copy that earlier setup
# runs left behind. (pip uninstall can't do this — it resolves the module
# copy first and dies on permissions.)
rm -rf "$HOME"/.local/lib/python3.12/site-packages/tensorboard \
       "$HOME"/.local/lib/python3.12/site-packages/tensorboard-*.dist-info \
       "$HOME"/.local/lib/python3.12/site-packages/tensorboard_data_server* \
       "$HOME"/.local/bin/tensorboard

pip install --user --no-deps \
  recbole==1.2.1 colorlog colorama thop tabulate texttable

# tensorboard: NOT installed at all. FOX has no 2024a-toolchain tensorboard
# module (only 3.9/3.10 builds — loading them would mix toolchains) and no
# further user-site installs are wanted. recbole only touches tensorboard
# via torch.utils.tensorboard.SummaryWriter, which src/recsys/recbole_grid.py
# stubs out when the package is absent (the grid never writes tb logs).
python - <<'PY'
import pandas, sklearn, numpy, torch, recbole
assert recbole.__version__ == "1.2.1", recbole.__version__
from src.recsys.recbole_grid import build_model
for name in ["SASRec", "GRU4Rec", "BERT4Rec"]:
    build_model(name, 1000, torch.device("cpu"))
print("stack + recbole 1.2.1 + grid imports OK")
PY
echo "SETUP OK"
