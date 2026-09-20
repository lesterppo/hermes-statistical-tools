#!/usr/bin/env bash
# install.sh — Hermes Statistical Tools installer (plugin-based)
#
# Installs Python dependencies AND the tools as a Hermes PLUGIN under
# $HERMES_HOME/plugins/ so that `hermes update` cannot wipe them. The old
# behaviour (copy into hermes-agent/tools/) is still available with
# --legacy, but it is documented as lossy: updates reset that directory
# and any toolsets.py entry with it.
#
# Usage:
#   ./install.sh              # deps + install/refresh the plugin (DEFAULT)
#   ./install.sh --legacy     # deps + copy into hermes-agent/tools/ (update-lossy)
#   ./install.sh --uninstall  # remove the plugin
#   ./install.sh --check      # report backend availability only

set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_NAME="hermes_statistical_tools"
PLUGIN_DIR="$HERMES_HOME/plugins/$PLUGIN_NAME"
legacy=0
uninstall=0
check_only=0

for arg in "$@"; do
    case "$arg" in
        --legacy) legacy=1 ;;
        --uninstall) uninstall=1 ;;
        --check) check_only=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

echo "Hermes Statistical Tools — Installer"
echo "===================================="

# ---------------------------------------------------------------------------
# 0. Uninstall
# ---------------------------------------------------------------------------
if [ "$uninstall" -eq 1 ]; then
    if [ -d "$PLUGIN_DIR" ]; then
        rm -rf "$PLUGIN_DIR"
        echo "✓ removed $PLUGIN_DIR"
        echo "  (restart Hermes / the gateway to unload the toolset)"
    else
        echo "nothing to remove at $PLUGIN_DIR"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Backend detection
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
echo
echo "Python: $($PYTHON --version)"
echo
echo "Backends detected:"
HAS_NUMPY=0; HAS_SCIPY=0; HAS_PANDAS=0; HAS_SM=0; HAS_SEMOPY=0; HAS_GIRTH=0; HAS_LIFELINES=0
$PYTHON -c "import numpy" >/dev/null 2>&1 && HAS_NUMPY=1
$PYTHON -c "import scipy" >/dev/null 2>&1 && HAS_SCIPY=1
$PYTHON -c "import pandas" >/dev/null 2>&1 && HAS_PANDAS=1
$PYTHON -c "import statsmodels" >/dev/null 2>&1 && HAS_SM=1
$PYTHON -c "import semopy" >/dev/null 2>&1 && HAS_SEMOPY=1
$PYTHON -c "import girth" >/dev/null 2>&1 && HAS_GIRTH=1
$PYTHON -c "import lifelines" >/dev/null 2>&1 && HAS_LIFELINES=1

if [ "$HAS_NUMPY" -eq 1 ] && [ "$HAS_SCIPY" -eq 1 ]; then
    echo "  ✓ numpy+scipy  — pspp (pure Python, 45 types)"
else
    echo "  ✗ numpy/scipy  — install: pip install numpy scipy"
fi
if [ "$HAS_PANDAS" -eq 1 ]; then
    echo "  ✓ pandas       — statsmodels, sem, medical_ext"
else
    echo "  ✗ pandas       — install: pip install pandas"
fi
if [ "$HAS_SM" -eq 1 ]; then
    echo "  ✓ statsmodels  — statsmodels tool + medical_ext power"
else
    echo "  ✗ statsmodels  — install: pip install statsmodels"
fi
if [ "$HAS_SEMOPY" -eq 1 ]; then
    echo "  ✓ semopy       — sem"
else
    echo "  ✗ semopy       — install: pip install semopy"
fi
if [ "$HAS_GIRTH" -eq 1 ]; then
    echo "  ✓ girth        — irt"
else
    echo "  ✗ girth        — install: pip install girth"
fi
if [ "$HAS_LIFELINES" -eq 1 ]; then
    echo "  ✓ lifelines    — medical_ext survival"
else
    echo "  ✗ lifelines    — install: pip install lifelines"
fi
echo "  ✓ stdlib path  — pspp meta/power sub-analyses (no backend needed)"

if [ "$check_only" -eq 1 ]; then
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Python dependencies (versions pinned in requirements.txt)
# ---------------------------------------------------------------------------
# PSPP tool: scipy, numpy
# StatsModels: statsmodels, pandas
# SEM: semopy
# IRT: girth
# Medical_Ext: lifelines (survival) + statsmodels (power); meta is pure numpy/scipy
# All versions live in requirements.txt (single source of truth).
echo
echo "Installing dependencies..."
# uv refuses outside a venv unless --system; fall back to pip on any failure
# (deps may already be present — a failed install must not kill the plugin copy).
if [ -n "${VIRTUAL_ENV:-}" ] && command -v uv &>/dev/null; then
    PIP="uv pip install"
    echo "Installer: uv (venv)"
elif command -v uv &>/dev/null; then
    PIP="uv pip install --system"
    echo "Installer: uv (--system)"
else
    PIP="$PYTHON -m pip install"
    echo "Installer: pip"
fi

if [ -f requirements.txt ]; then
    # shellcheck disable=SC2086
    { $PIP -r requirements.txt 2>&1 | tail -3; } || {
        echo "  ⚠ uv install failed, trying pip"; $PYTHON -m pip install -r requirements.txt 2>&1 | tail -3;
    } || echo "  ⚠ dependency install failed — continuing (backends may already exist)"
else
    # shellcheck disable=SC2086
    { $PIP "numpy>=1.24,<3" "scipy>=1.10,<2" "pandas>=2.0,<3" 2>&1 | tail -1; } || true
    # shellcheck disable=SC2086
    { $PIP "statsmodels>=0.14,<1" 2>&1 | tail -1; } || true
    # shellcheck disable=SC2086
    { $PIP "semopy>=2.3,<3" 2>&1 | tail -1; } || true
    # shellcheck disable=SC2086
    { $PIP "girth>=0.8,<1" 2>&1 | tail -1; } || true
    # shellcheck disable=SC2086
    { $PIP "lifelines>=0.29,<1" 2>&1 | tail -1; } || true
fi

echo
echo "=== Verifying ==="
$PYTHON -c "
import numpy; print(f'numpy {numpy.__version__}')
import scipy; print(f'scipy {scipy.__version__}')
import pandas; print(f'pandas {pandas.__version__}')
import statsmodels; print(f'statsmodels {statsmodels.__version__}')
import semopy; print(f'semopy {semopy.__version__}')
import importlib.metadata; print(f'girth {importlib.metadata.version(\"girth\")}')
import lifelines; print(f'lifelines {lifelines.__version__}')
" 2>&1

# ---------------------------------------------------------------------------
# 3. Locate the Hermes install
# ---------------------------------------------------------------------------
if [ ! -d "$HERMES_HOME" ]; then
    echo
    echo "ERROR: Hermes home not found at $HERMES_HOME" >&2
    echo "       Install Hermes Agent first: https://github.com/NousResearch/hermes-agent" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Legacy copy path (kept for compatibility, documented as lossy)
# ---------------------------------------------------------------------------
if [ "$legacy" -eq 1 ]; then
    HERMES_TOOLS="$HERMES_HOME/hermes-agent/tools"
    TOOLSETS_FILE="$HERMES_HOME/hermes-agent/toolsets.py"
    if [ ! -d "$HERMES_TOOLS" ]; then
        echo "ERROR: $HERMES_TOOLS not found" >&2
        exit 1
    fi
    echo
    echo "WARNING: --legacy copies into the Hermes git tree. The next"
    echo "         'hermes update' deletes these files and the toolsets.py"
    echo "         entry, leaving 'Unknown toolsets: medical'. Prefer the"
    echo "         plugin install (no flag)."
    for tool in pspp_tool.py statsmodels_tool.py sem_tool.py irt_tool.py medical_ext_tool.py; do
        cp "tools/$tool" "$HERMES_TOOLS/$tool"
        echo "  ✓ $tool → $HERMES_TOOLS/$tool"
    done
    if ! grep -q '"medical"' "$TOOLSETS_FILE" 2>/dev/null; then
        echo
        echo "Add the toolset manually to $TOOLSETS_FILE:"
        echo '  "medical": {"description": "Medical research — PSPP, statsmodels, SEM, IRT, survival/meta/power",'
        echo '              "tools": ["pspp","statsmodels","sem","irt","medical_ext"],'
        echo '              "includes": []},'
    fi
    echo
    echo "Then: hermes tools enable medical && start a new session."
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. Plugin install (recommended — survives `hermes update`)
# ---------------------------------------------------------------------------
echo
echo "Installing plugin → $PLUGIN_DIR"

# Collision check: hermes_medical_tools registers the overlapping `medical`
# toolset (including its own `pspp`). Duplicate registrations overwrite each
# other (last wins), so flag any plugin already referencing these names.
collisions=""
for other in "$HERMES_HOME"/plugins/*/; do
    [ -d "$other" ] || continue
    base="$(basename "$other")"
    [ "$base" = "$PLUGIN_NAME" ] && continue
    if grep -rqs "pspp_tool\|statsmodels_tool\|sem_tool\|irt_tool\|medical_ext_tool\|hermes_medical_tools" "$other" 2>/dev/null; then
        collisions="$collisions $base"
    fi
done
if [ -n "$collisions" ]; then
    echo
    echo "  ⚠ These plugins already reference the same tool names:$collisions"
    echo "    (hermes_medical_tools registers the overlapping 'medical'"
    echo "    toolset incl. its own 'pspp' — last registration wins.)"
    echo "    Duplicate registrations overwrite each other. Disable one set:"
    for c in $collisions; do
        echo "      hermes plugins disable $(basename "$c")"
    done
fi

mkdir -p "$PLUGIN_DIR/tools"
cp plugin/plugin.yaml "$PLUGIN_DIR/plugin.yaml"
cp plugin/__init__.py "$PLUGIN_DIR/__init__.py"
cp plugin/tools/__init__.py "$PLUGIN_DIR/tools/__init__.py"
for tool in pspp_tool.py statsmodels_tool.py sem_tool.py irt_tool.py medical_ext_tool.py; do
    cp "tools/$tool" "$PLUGIN_DIR/tools/$tool"
    echo "  ✓ tools/$tool"
done

echo
echo "Enabled toolset: medical (pspp, statsmodels, sem, irt, medical_ext)"
echo
echo "Next steps:"
echo "  1. hermes plugins enable $PLUGIN_NAME      # if not auto-enabled"
echo "  2. restart Hermes (or the gateway) so the plugin loads"
echo "  3. the 'medical' toolset is auto-enabled for the platform"
echo
echo "Verify with:  hermes tools list | grep -E 'pspp|statsmodels|sem|irt|medical_ext'"
echo
echo "SYNC NOTE: when you edit tool files in this repo, re-run ./install.sh"
echo "(or copy tools/*.py to $PLUGIN_DIR/tools/) before pushing — the plugin"
echo "copy is what Hermes actually runs."
