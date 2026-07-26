#!/bin/bash
# ONE-TIME setup on a FOX login node (compute nodes inherit ~/.local).
# --no-deps everywhere so pip can NEVER pull a user-site numpy/torch that
# would shadow the NLPL module stack.
set -o errexit
source "$(dirname "$0")/modules.sh"
pip install --user --no-deps \
  recbole==1.2.1 colorlog colorama thop tabulate texttable \
  tensorboard absl-py grpcio markdown protobuf werkzeug tensorboard-data-server
python - <<'PY'
import pandas, sklearn, numpy, torch, recbole
assert recbole.__version__ == "1.2.1", recbole.__version__
from src.recsys.recbole_grid import build_model
for name in ["SASRec", "GRU4Rec", "BERT4Rec"]:
    build_model(name, 1000, torch.device("cpu"))
print("stack + recbole 1.2.1 + grid imports OK")
PY
echo "SETUP OK"
