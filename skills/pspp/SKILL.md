---
name: pspp-tool
description: Build, test, and extend the agent-native PSPP statistical tool.
version: 1.0.0
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
pspp(t="meta",  d="study,effect,se\n...", v="study")
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

## Pitfalls

- **Tukey HSD SE must use `/2` factor:** `se = sqrt(mse/2 * (1/n1 + 1/n2))`
- **Games-Howell uses Studentized Range, not t-distribution**
- **BH FDR needs `minimum.accumulate` backwards for monotonicity**
- **McNemar needs `max(0, abs(b-c)-1)` guard for balanced pairs**
- **OLS must use `lstsq()`, never `inv()`** — prevents crashes on collinearity
- **Zero variance crashes Welch DF** — guard with early return
- **Zero-cell OR needs Haldane-Anscombe (+0.5) correction**
- **NaN t-values crash `scipy.t.sf()`** — guard with `math.isnan(tv)`
- **BSpline.design_matrix fails on small data** — use truncated power basis instead
- **`p_adjust` function has `alpha` param conflict** — be careful with naming
- **Registry handler must mirror ALL function params** — missing params cause `unexpected keyword` errors
- **`_ANALYSES` dict opening must survive patch operations** — verify after bulk inserts

## Verification

```bash
cd ~/.hermes/hermes-agent && source .venv/bin/activate
python -c "import tools.pspp_tool; print('OK')"
python -c "from tools.pspp_tool import pspp_run; import json; r=pspp_run(t='ttest',d='g,v\n1,10\n2,20',a='g',v='v',g=[1,2]); print(json.loads(r)['k']['t'])"
```

## File Location

`tools/pspp_tool.py` — 2376 lines, 45 types, pure Python, registry-registered as `"pspp"` in `"medical"` toolset.
