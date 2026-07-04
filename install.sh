#!/usr/bin/env bash
# Hermes Statistical Tools — dependency installer
# Installs Python packages needed by the four statistical tools.
set -euo pipefail

echo "=== Hermes Statistical Tools Installer ==="
echo ""

# Detect Python
PYTHON="${PYTHON:-python3}"
echo "Python: $($PYTHON --version)"

# Detect pip
if command -v uv &>/dev/null; then
    PIP="uv pip install"
    echo "Installer: uv"
else
    PIP="$PYTHON -m pip install"
    echo "Installer: pip"
fi

echo ""
echo "Installing dependencies..."

# PSPP tool: scipy, numpy
# StatsModels: statsmodels, pandas
# SEM: semopy
# IRT: girth
$PIP "numpy>=1.26,<2" "scipy>=1.12,<2" 2>&1 | tail -1
$PIP "pandas>=2.0,<3" 2>&1 | tail -1
$PIP "statsmodels>=0.14,<1" 2>&1 | tail -1
$PIP "semopy>=2.3,<3" 2>&1 | tail -1
$PIP "girth>=0.8,<1" 2>&1 | tail -1

echo ""
echo "=== Verifying ==="
$PYTHON -c "
import numpy; print(f'numpy {numpy.__version__}')
import scipy; print(f'scipy {scipy.__version__}')
import pandas; print(f'pandas {pandas.__version__}')
import statsmodels; print(f'statsmodels {statsmodels.__version__}')
import semopy; print(f'semopy {semopy.__version__}')
import importlib.metadata; print(f'girth {importlib.metadata.version(\"girth\")}')
" 2>&1

echo ""
echo "=== Setup Complete ==="
echo "Copy tools/*.py to your Hermes checkout:"
echo "  cp tools/*.py ~/.hermes/hermes-agent/tools/"
echo "Then add to toolsets.py medical toolset:"
echo "  'medical': {'tools': ['pspp','statsmodels','sem','irt'], ...}"
