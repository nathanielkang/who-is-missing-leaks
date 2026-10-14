#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install --user -q statsmodels scikit-learn pandas numpy scipy
python3 -c "import numpy,pandas,sklearn,statsmodels,scipy; print('DEPS_OK', numpy.__version__)"
