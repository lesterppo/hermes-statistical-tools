# QUICKSTART — for AI Agents

This repo provides **AI-agent-native statistical tools** that plug into the
[Hermes Agent](https://github.com/NousResearch/hermes-agent) framework. Each
tool exposes one schema with an `action` parameter and returns compact JSON
with short keys. This file tells an autonomous agent exactly how to install,
verify, and invoke them.

## 1. Prerequisites

- A Python 3.10+ virtualenv (the hermes-agent `.venv` is the usual target).
- The hermes-agent repo at `~/hermes-agent` (or any checkout where `tools/`
  and `toolsets.py` live). If you only want to *test* the tools standalone,
  you still need `tools/registry.py` importable (see §4).

## 2. Install dependencies

```bash
PIP="python3 -m pip"   # or: <venv>/bin/python -m pip
$PIP install -r requirements.txt
# Medical Extended needs lifelines for survival analysis:
$PIP install "lifelines>=0.29,<1"
```

`install.sh` does this for you (idempotent):

```bash
bash install.sh
```

## 3. Deploy into a Hermes checkout

```bash
# from this repo root
cp tools/*.py <hermes-agent>/tools/

# Register the toolset. In <hermes-agent>/toolsets.py add (or merge into)
# the TOOLSETS dict:
"medical": {
    "description": "Medical research — PSPP, statsmodels, SEM, IRT, survival/meta/power",
    "tools": ["pspp", "statsmodels", "sem", "irt", "medical_ext"],
    "includes": []
},
```

After copying, tools are auto-discovered (any `tools/*.py` with a top-level
`registry.register(...)` is imported automatically). They become available to
the agent only when their toolset is enabled for the active platform.

## 4. Verify (standalone, no Hermes needed)

```bash
cd <this-repo>/tools
python3 -c "
import importlib, tools.registry as reg
for m in ['pspp_tool','statsmodels_tool','sem_tool','irt_tool','medical_ext_tool']:
    importlib.import_module(m)
e = reg.registry.get_entry('medical_ext')
print('medical_ext registered:', e is not None, '| toolset:', e.toolset)
print('check_fn passes:', e.check_fn())
"
```

## 5. Invocation contract

Every tool has a top-level `<name>_run(action=..., **kwargs)` function that
returns a **JSON string**. Errors return `{"e": "message"}` — never a raw
traceback.

### Medical Extended (`tools/medical_ext_tool.py`) — the headless tool

Three families, native action names:

**Survival** (lifelines)
```python
from tools.medical_ext_tool import medical_ext_run

# Kaplan-Meier, whole cohort
medical_ext_run(action="km", d=csv, t="T", e="E")
# Kaplan-Meier stratified + log-rank
medical_ext_run(action="km", d=csv, t="T", e="E", g="grp")
# Cox proportional hazards
medical_ext_run(action="cox", d=csv, t="T", e="E", p=["age","trt"])
# Pairwise log-rank
medical_ext_run(action="logrank", d=csv, t="T", e="E", g="grp")
```
CSV for survival: columns `t` (duration), `e` (1=event,0=censor),
optional `g` (group), and any `p` predictors (numeric).

**Meta-analysis** (statsmodels 0.15+ native `combine_effects`, else pure-numpy DL)
```python
medical_ext_run(action="forest",
    d="yi,vi,label\n0.5,0.1,StudyA\n0.2,0.15,StudyB\n0.8,0.12,StudyC\n")
```
CSV: `yi` (effect size), `vi` (variance of yi) or `se` (std error), optional
`label`. Returns fixed-effect + DerSimonian-Laird random-effects pooled
estimates, heterogeneity Q/I²/tau², and per-study weights. tau² may be
slightly negative when Q<df (valid DL result).

**Power / sample-size** (statsmodels) — pass any 3 of the 4 quantities; set
the unknown one to `0` and it is solved.
```python
# Solve sample size (nobs) for Cohen's d=0.5, power=0.8, alpha=0.05
medical_ext_run(action="ttest", d_es=0.5, n=0, pw=0.8, a=0.05,
                ratio=1.0, alt="two-sided")
# One-way ANOVA power
medical_ext_run(action="anova", n=0, f=0.3, pw=0.8, a=0.05, k=3)
# Two-proportion power (uses p1/p2 instead of effect size)
medical_ext_run(action="prop", n=0, p1=0.4, p2=0.55, pw=0.8, a=0.05,
                ratio=1.0, alt="two-sided")
```
Power keys: `n`=nobs per group, `d_es`=Cohen's d (ttest), `f`=Cohen's f
(anova), `p1`/`p2`=proportions (prop), `pw`=power, `a`=alpha,
`ratio`=n2/n1, `alt`=two-sided|larger|smaller, `k`=groups. Solve the one
set to `0`.

## 6. Output format

Compact JSON, short keys. Examples (survival/meta/power):
- `hr` hazard ratio, `c` coef, `s` std err, `z` z-value, `p` p-value
- `l`/`u` CI lower/upper, `n` nobs, `fe`/`re` fixed/random-effects
- `Q`/`I2`/`tau2` heterogeneity; `forest` per-study rows with `w_fe`/`w_re`

## 7. Privacy

Zero hardcoded paths, zero secrets, zero personal identifiers. All file
operations (if any) must use `get_hermes_home()`. Safe to clone and run as-is.

## 8. Tests

```bash
python3 -m pytest tests/ -q
```
Pure-stdlib + pytest, synthetic data, no network. 17 tests cover the Medical
Extended tool (survival, meta both code paths, power all directions).
