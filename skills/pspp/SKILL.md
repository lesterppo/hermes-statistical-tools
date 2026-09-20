---
name: pspp-tool
description: Build, test, and extend the agent-native PSPP statistical tool.
version: 1.1.0
author: Peter (lesterppo)
tags: [pspp, statistics, medical, tools, agent-native]
---

# PSPP Tool — Agent-Native Statistical Analysis

45 test types, pure Python (scipy/numpy), zero PSPP binary dependency.
Agent specifies test type + variable names — no SPSS syntax.

## When to Use

- Running any statistical test in Hermes
- Extending the tool with new test types
- Debugging numerical issues (convergence, zero variance, singular matrices)
- Cross-validating results against PSPP 2.0.0 or scipy

## Quick Reference

```
pspp(t="ttest", d="g,v\n1,10\n...", a="g", v="v", g=[1,2])
pspp(t="reg",   d="x,y\n1,2\n...", o="y", p=["x"])
pspp(t="anova", d="g,v\n...", a="g", v="v", posthoc="tukey")
pspp(t="surv",  d="time,event,group\n...", o="event", v="time", a="group")
pspp(t="meta",  d="study,effect,se\n...", v="study", effect_col="effect", se_col="se")
pspp(t="eval",  d="test,ref\n...", v="test", pos_ref=1)
pspp(t="power", d="x\n1\n...", power_test="ttest", es_val=0.5, calc="n")
pspp(t="evalue", d="x\n1\n...", or_val=2.5)
```

## Test Type Catalog

**Descriptive (5):** desc, freq, examine, means, crosstab
**Compare means (4):** ttest, pttest, ttest1, anova
**Non-parametric (7):** mw, kw, wilcoxon, friedman, sign, ks, runs
**Correlation (3):** corr, spearman, partial
**Regression (5):** reg, logistic, poisson, negbin, quantreg
**Advanced (10):** factor, reliability, roc, eval, surv, meta, blandaltman, kappa, evalue, power
**Causal/ML (5):** dca, mediation, pscore, gam, lasso
**Other (6):** zip, adf, beta, hedgesg, omega, rank

## Schema Params (post-fix)

Beyond the core `t/d/a/v/g/o/w/p/c/mu/paired/posthoc/rotate/n/pos`:

- `pos_ref` — positive reference class for `eval` (confusion matrix vs reference)
- `effect_col`, `se_col` — effect-size / SE column names for `meta`
- `power_test` (`ttest|prop|anova`), `alpha`, `es_val`, `ratio` (n2/n1),
  `k` (groups), `p0`/`p1` (proportions), `calc` (`n` = sample size, `power` = power)
- `or_val`, `rr_val` — direct OR/RR value for `evalue` (no 2x2 table needed)
- `tvar` — time variable name for `surv` when it differs from the value column
- `tau`, `lam`, `df`/`df_val`, `lags` — quantreg / lasso / ADF params

The registry handler mirrors **every** `pspp_run` param — a missing one
surfaces as an `unexpected keyword` error.

## Signed Effect Sizes + Direction

Cohen's `d`, Hedges' `g`, and rank-biserial are **signed**: positive means
`g[1]` (second group label) scores higher than `g[0]`. Outputs carry a
`direction` field, e.g. `"B minus A (positive = B higher)"` — never take
`abs()` before interpreting.

## surv / meta Output Notes

- `surv`: Kaplan-Meier keys are **float** time values (`str(k)`, never
  `int(k)`) — fractional event times survive. Group labels kept verbatim
  (no int truncation); 2-group log-rank reports `g1`/`g2` labels.
- `meta`: study labels preserved from the `v` column (up to 100 studies);
  forest output pairs each `label` with its effect/CI.
- `roc`: one point per distinct score, counted **after** ties at that score.
- `evalue`: OR/RR ≤ 1 (protective) is computed on the reciprocal with a
  `note` flag — the E-value is always ≥ 1.

## Pitfalls

- **Tukey HSD SE must use `/2` factor:** `se = sqrt(mse/2 * (1/n1 + 1/n2))`
- **Games-Howell uses Studentized Range, not t-distribution** — Welch SE
  `sqrt(v1/n1 + v2/n2)` + Welch-Satterthwaite df; already family-wise
  corrected, like Tukey (no extra BH step)
- **Partial correlation df = n − 2 − k** (two targets + k controls)
- **BH FDR needs `minimum.accumulate` backwards for monotonicity**
- **McNemar needs `max(0, abs(b-c)-1)` guard for balanced pairs**
- **OLS must use `lstsq()`, never `inv()`** — prevents crashes on collinearity
- **Zero variance crashes Welch DF** — guard with early return
- **Zero-cell OR needs Haldane-Anscombe (+0.5) correction**
- **`_rnd` is None-safe for NaN/inf** (incl. numpy scalars) — never passes
  NaN into `scipy.t.sf()`; guard raw t-values with `math.isnan(tv)`
- **BSpline.design_matrix fails on small data** — use truncated power basis instead
- **Factor analysis defaults matter** — verify n-components default after edits
- **`_ANALYSES` dict opening must survive patch operations** — verify after bulk inserts

## Verification

```bash
cd ~/.hermes/hermes-agent && source .venv/bin/activate
python -c "import tools.pspp_tool; print('OK')"
python -c "from tools.pspp_tool import pspp_run; import json; r=pspp_run(t='ttest',d='g,v\n1,10\n2,20',a='g',v='v',g=[1,2]); print(json.loads(r)['k']['t'])"
```

Or from the repo: `python3 -m pytest tests/ -q` (42 tests, must stay green).

## File Location

`tools/pspp_tool.py` — 45 types, pure Python, registry-registered as `"pspp"` in `"medical"` toolset.
