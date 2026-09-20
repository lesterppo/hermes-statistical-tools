---
name: statsmodels-tool
description: Build, test, and extend the agent-native statsmodels statistical tool.
version: 1.1.0
author: Peter (lesterppo)
license: MIT
tags: [statsmodels, statistics, medical, tools, agent-native]
---

# StatsModels Tool — Agent-Native Advanced Statistical Analysis

6 action types wrapping statsmodels 0.14.6 for analyses the PSPP tool (pure scipy/numpy)
cannot do: Mixed Models, GEE, Ordinal GEE, Nominal GEE, Repeated Measures ANOVA,
and MICE with pooled analysis (Rubin's Rules).

## When to Use

- Running mixed-effects models, GEE, or repeated measures ANOVA in Hermes
- Multiple imputation with pooled inference via MICE
- The PSPP tool cannot do these analyses (they require statsmodels)

## Quick Reference

```
statsmodels(action="mlm",      d="CSV", o="y", p=["x1","x2"], g="subject")
statsmodels(action="gee",      d="CSV", o="y", p=["x1"],      g="id", fam="binomial", cov="exchangeable")
statsmodels(action="gee_ord",  d="CSV", o="y", p=["x1"],      g="id", cov="exchangeable")
statsmodels(action="gee_nom",  d="CSV", o="y", p=["x1"],      g="id")
statsmodels(action="anova_rm", d="CSV", o="y", g="subject",   w=["time"])
statsmodels(action="mice",     d="CSV", o="y", p=["x1","x2"], n=10, mod="ols")
```

## Action Catalog

| Action | StatsModels API | Description |
|--------|----------------|-------------|
| `mlm` | `sm.MixedLM` | Linear mixed models (random intercepts) |
| `gee` | `sm.GEE` | Generalized Estimating Equations |
| `gee_ord` | `sm.OrdinalGEE` | Ordinal GEE (ordered categorical) |
| `gee_nom` | `sm.NominalGEE` | Nominal GEE (experimental — falls back gracefully) |
| `anova_rm` | `AnovaRM` | Repeated measures ANOVA (within-subject only) |
| `mice` | `MICE` | Multiple imputation + pooled analysis |

### GEE Families

`gaussian`, `binomial`, `poisson`, `gamma`

### GEE Covariance Structures

`independence`, `exchangeable`, `ar1`, `unstructured`

### MICE Models

`ols` (linear regression), `logit` (logistic), `poisson`

## Output Format

Compact JSON with short keys:
- `f`: fixed effects → `c`=coef, `s`=std_err, `z`=z/t_val, `p`=p_val, `l`=lower_ci, `u`=upper_ci
- `r`: random effects summary (mlm only) → `mean`, `sd`, `n`
- `rv`: random effects variance components
- `ll`: log-likelihood, `aic`, `bic`
- `cnvg`: convergence (bool)
- `th`: thresholds (gee_ord only)
- `fmi`: fraction of missing information (mice only)
- `n`: number of observations, `imp`: number of imputations

## Key Features

- **Categorical variable handling**: String predictors are auto-converted to
  dummy variables (drop_first) for proper statistical treatment
- **Lazy import**: statsmodels (~30 MB) is only imported on first actual call
- **Thread-safe**: Double-checked locking pattern for lazy import
- **Error messages**: Actionable for the LLM (e.g., "Try gee_ord instead" for
  NominalGEE NotImplementedError)

## Pitfalls

- **NominalGEE is experimental** in statsmodels 0.14.6 — may raise
  NotImplementedError. The tool catches it with an actionable message;
  fall back to `gee_ord` or binary-encoded `gee`.
- **AnovaRM does NOT support between-subject factors** — there is no `b`
  param on `statsmodels_run`. For mixed between-within designs, use `mlm`.
- **AnovaRM within-only**: requires at least one within-subject factor `w`.
- **Intercept-only models**: pass `p=[]` for null models — tool
  auto-creates constant-only design matrix.
- **Small datasets** (< 10 observations) cause SVD convergence failures in
  MICE. Use more observations or reduce imputation count.
- **Categorical predictors** are dummy-coded with `drop_first=True`. The
  omitted category becomes the reference. Coefficient interpretation
  changes accordingly.
- **MixedLM convergence** can fail with very small datasets or near-zero
  variance components. The `cnvg` field indicates convergence status.
- **GEE with binomial family** on small datasets may produce all-null
  parameter estimates due to non-convergence.
- **Ordinal GEE with string outcomes**: strings are encoded alphabetically
  (A->0, B->1, C->2). If order matters, pre-encode the outcome column as
  integers in the CSV before calling this tool.

## Verification

```bash
cd ~/.hermes/hermes-agent && python3 -c "
import json
from tools.statsmodels_tool import statsmodels_run, _check_statsmodels
assert _check_statsmodels()

# MLM
r = statsmodels_run(action='mlm', d='s,score,trt,t\n1,85,A,1\n1,88,A,2\n2,78,A,1\n2,80,A,2\n3,70,B,1\n3,72,B,2\n', o='score', p=['trt','t'], g='s')
d = json.loads(r)
assert 'e' not in d
assert 'trt_B' in d['f']
print('OK')
"
```

Or from the repo: `python3 -m pytest tests/ -q` (42 tests, must stay green).

## File Location

`tools/statsmodels_tool.py` — 6 actions, registry-registered as
`"statsmodels"` in `"medical"` toolset. Requires statsmodels >= 0.14 and
pandas.
