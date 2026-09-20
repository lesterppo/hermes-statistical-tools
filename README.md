# Hermes Statistical Tools

AI-agent-native statistical analysis tools for the [Hermes Agent](https://github.com/NousResearch/hermes-agent) framework.

Five tools covering 60+ statistical test types — all with compact JSON output, lazy imports, and thread-safe design.

> **For an autonomous AI agent setting this up**, read [`QUICKSTART.md`](QUICKSTART.md) first — it has the exact install + invocation contract. The full integration guide is [`AGENTS.md`](AGENTS.md).

## Tools

| Tool | Lines | Types | Package | Description |
|------|-------|-------|---------|-------------|
| PSPP | 2,376 | 45 | scipy/numpy | Pure Python PSPP 2.0.0 command set |
| StatsModels | 832 | 6 | statsmodels | Mixed models, GEE, RM-ANOVA, MICE |
| SEM | 362 | 2 | semopy | Structural Equation Modeling |
| IRT | 550 | 7 | girth | Item Response Theory |
| Medical Extended | ~690 | 7 | lifelines + statsmodels | Survival (KM/Cox/log-rank), meta-analysis, power/sample-size |

## Quick Install

```bash
git clone https://github.com/lesterppo/hermes-statistical-tools.git
cd hermes-statistical-tools
bash install.sh
```

## Deploy into Hermes (Plugin — recommended)

`./install.sh` (no flags) installs the tools as the `hermes_statistical_tools`
plugin under `~/.hermes/plugins/`, **outside** the Hermes git tree, so
`hermes update` cannot wipe them:

```bash
bash install.sh              # deps + plugin install/refresh (DEFAULT)
bash install.sh --check      # backend availability only
bash install.sh --legacy     # copy into hermes-agent/tools/ (update-lossy)
bash install.sh --uninstall  # remove the plugin
```

Then enable + verify:

```bash
hermes plugins enable hermes_statistical_tools   # if not auto-enabled
# restart Hermes (or the gateway) so the plugin loads
hermes tools list | grep -E 'pspp|statsmodels|sem|irt|medical_ext'
```

The plugin registers the `medical` toolset
(`pspp`, `statsmodels`, `sem`, `irt`, `medical_ext`) via two-step
registration — each tool module self-registers on import, then
`register(ctx)` marks them plugin-owned. No core Hermes files are modified.
See `plugin/__init__.py` for the pattern.

> **Overlap warning:** the sibling [`hermes-medical-tools`](https://github.com/lesterppo/hermes-medical-tools)
> plugin registers the same `medical` toolset (including its own `pspp`).
> Enabling both registers those names twice — last registration wins. The
> installer flags the collision; disable one set
> (`hermes plugins disable <name>`) if results look off.

### Legacy git-tree install (breaks on next update)

```bash
cp tools/*.py <hermes-agent>/tools/
```

Then register the `medical` toolset in `<hermes-agent>/toolsets.py`:

```python
"medical": {
    "description": "Medical research — PSPP, statsmodels, SEM, IRT, survival/meta/power",
    "tools": ["pspp", "statsmodels", "sem", "irt", "medical_ext"],
    "includes": []
},
```

## Verify

```bash
cd tools
python3 -c "
import importlib, tools.registry as reg
for m in ['pspp_tool','statsmodels_tool','sem_tool','irt_tool','medical_ext_tool']:
    importlib.import_module(m)
print('medical_ext OK:', reg.registry.get_entry('medical_ext').toolset)
"
```

## Usage

Every tool has a top-level `<name>_run(action=..., **kwargs)` returning a JSON
string. Errors return `{"e": "message"}`.

```python
from tools.medical_ext_tool import medical_ext_run

# Survival — Kaplan-Meier by group
result = medical_ext_run(action="km", d=csv, t="T", e="E", g="grp")
# Survival — Cox proportional hazards
result = medical_ext_run(action="cox", d=csv, t="T", e="E", p=["age","trt"])
# Meta-analysis — fixed-effect + DerSimonian-Laird random-effects
result = medical_ext_run(action="forest", d="yi,vi\n0.5,0.1\n0.2,0.15\n0.8,0.12\n")
# Power — solve sample size for Cohen's d
result = medical_ext_run(action="ttest", d_es=0.5, n=0, pw=0.8, a=0.05, alt="two-sided")
```

Other tools follow the same pattern:

```python
from tools.pspp_tool import pspp_run
result = pspp_run(t="ttest", d="g,v\nA,10\nB,20\n", a="g", v="v", g=["A","B"])

from tools.statsmodels_tool import statsmodels_run
result = statsmodels_run(action="mlm", d=csv, o="score", p=["trt","t"], g="s")

from tools.sem_tool import sem_run
result = sem_run(action="sem_fit", d=csv, desc="f1 =~ x1 + x2 + x3\nf2 ~ f1")

from tools.irt_tool import irt_run
result = irt_run(action="irt_rasch", d=csv_binary, method="jml")
```

See [`QUICKSTART.md`](QUICKSTART.md) for the complete action catalog and the
power/solve convention, and [`AGENTS.md`](AGENTS.md) for design patterns.

## Design

All tools follow consistent patterns:
- **CSV input** via `d` parameter (survival/meta); structured params for power
- **Compact JSON output** with short keys (`c`, `s`, `z`, `p`, `d`, `a`, `ll`, `hr`)
- **Action-based routing** (one schema per tool, many operations)
- **Lazy imports** (thread-safe, heavy packages load on first use)
- **Privacy-safe** (zero hardcoded paths, zero secrets)

## Tests

```bash
python3 -m pytest tests/ -q
```

Pure-stdlib + pytest, synthetic data, no network.

## License

MIT — see individual tool files for author credits.
