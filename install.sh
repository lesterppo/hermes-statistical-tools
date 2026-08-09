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
# Medical_Ext: lifelines (survival) + statsmodels (power); meta is pure numpy/scipy
# All versions live in requirements.txt (single source of truth).
if [ -f requirements.txt ]; then
    $PIP -r requirements.txt 2>&1 | tail -3
else
    $PIP "numpy>=1.26,<2" "scipy>=1.12,<2" 2>&1 | tail -1
    $PIP "pandas>=2.0,<3" 2>&1 | tail -1
    $PIP "statsmodels>=0.14,<1" 2>&1 | tail -1
    $PIP "semopy>=2.3,<3" 2>&1 | tail -1
    $PIP "girth>=0.8,<1" 2>&1 | tail -1
    $PIP "lifelines>=0.29,<1" 2>&1 | tail -1
fi

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
echo ""
echo "NOTE — deployment options:"
echo ""
echo "  A) Plugin (RECOMMENDED — survives \`hermes update\`):"
echo "     The plugin at ~/.hermes/plugins/hermes_local_tools/ is the"
echo "     canonical runtime home. The git tree (tools/, toolsets.py,"
echo "     tools_config.py) is reset by every \`hermes update\`, wiping"
echo "     anything copied there. Sync edited tool files to the plugin:"
echo "       mkdir -p ~/.hermes/plugins/hermes_local_tools"
echo "       cp tools/pspp_tool.py tools/statsmodels_tool.py tools/sem_tool.py \\"
echo "          tools/irt_tool.py tools/medical_ext_tool.py \\"
echo "          ~/.hermes/plugins/hermes_local_tools/"
echo ""
echo "  B) Legacy git-tree install (breaks on next update):"
echo "     cp tools/*.py ~/.hermes/hermes-agent/tools/"
echo "     Then add to toolsets.py medical toolset:"
echo "       'medical': {'tools': ['pspp','statsmodels','sem','irt','medical_ext'], ...}"
echo ""
echo "SYNC NOTE: when you edit tool files in this repo, always copy them"
echo "to BOTH ~/.hermes/plugins/hermes_local_tools/ AND this repo before"
echo "pushing — the plugin copy is what Hermes actually runs."
