# Hermes Statistical Tools

AI-agent-native statistical analysis tools for the [Hermes Agent](https://github.com/NousResearch/hermes-agent) framework.

Four tools covering 60 statistical test types — all with compact JSON output, lazy imports, and thread-safe design.

## Tools

| Tool | Lines | Types | Package | Description |
|------|-------|-------|---------|-------------|
| PSPP | 2,376 | 45 | scipy/numpy | Pure Python PSPP 2.0.0 command set |
| StatsModels | 832 | 6 | statsmodels | Mixed models, GEE, RM-ANOVA, MICE |
| SEM | 362 | 2 | semopy | Structural Equation Modeling |
| IRT | 550 | 7 | girth | Item Response Theory |

## Quick Install

```bash
git clone https://github.com/lesterppo/hermes-statistical-tools.git
cd hermes-statistical-tools
bash install.sh
```

## Usage

Copy tools into your Hermes checkout and register in `toolsets.py`:

```python
# In toolsets.py:
"medical": {
    "description": "Medical research — PSPP, statsmodels, SEM, IRT, ...",
    "tools": ["pspp", "statsmodels", "sem", "irt"],
    "includes": []
},
```

### PSPP Tool

```python
from tools.pspp_tool import pspp_run
# 45 test types: desc, ttest, anova, reg, logistic, factor, surv, meta, ...
result = pspp_run(t="ttest", d="g,v\nA,10\nB,20\n", a="g", v="v", g=["A","B"])
```

### StatsModels Tool

```python
from tools.statsmodels_tool import statsmodels_run
# Mixed linear model
result = statsmodels_run(action="mlm", d="s,score,trt,t\n1,85,A,1\n...", o="score", p=["trt","t"], g="s")
```

### SEM Tool

```python
from tools.sem_tool import sem_run
desc = "f1 =~ x1 + x2 + x3\nf2 =~ y1 + y2 + y3\nf2 ~ f1"
result = sem_run(action="sem_fit", d=csv_data, desc=desc)
```

### IRT Tool

```python
from tools.irt_tool import irt_run
# Rasch model
result = irt_run(action="irt_rasch", d=csv_binary, method="jml")
# Score respondents
result = irt_run(action="irt_score", d=responses, diff="-1.5,0.3,1.2")
```

## Design

All tools follow consistent patterns:
- **CSV input** via `d` parameter
- **Compact JSON output** with short keys (`c`, `s`, `z`, `p`, `d`, `a`, `ll`)
- **Action-based routing** (one schema per tool, many operations)
- **Lazy imports** (thread-safe, heavy packages load on first use)
- **Privacy-safe** (zero hardcoded paths, zero secrets)

See [AGENTS.md](AGENTS.md) for the full AI agent integration guide.

## License

MIT — see individual tool files for author credits.
