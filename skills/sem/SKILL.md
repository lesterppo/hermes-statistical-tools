---
name: sem-tool
description: Build, test, and extend the agent-native SEM Hermes tool.
version: 1.1.0
author: Peter (lesterppo)
license: MIT
tags: [sem, semopy, statistics, medical, agent-native]
---

# SEM Tool — Agent-Native Structural Equation Modeling

Wraps semopy 2.3.11. Together with PSPP (45 types), statsmodels (6 actions),
and IRT (7 actions), it completes the Hermes statistical suite:
descriptive → mixed models → SEM → IRT.

## When to Use

- Fitting latent-factor / path models in Hermes
- Inspecting standardized vs raw parameter estimates
- Debugging semopy API mismatches or session-cache issues

## Quick Reference

```
sem(action="sem_fit",     d="CSV", desc="f1 =~ x1 + x2 + x3\nf2 ~ f1")
sem(action="sem_inspect",          desc="f1 =~ x1 + x2 + x3\nf2 ~ f1", mode="std_est")
```

Model syntax (lavaan-like):
```
f1 =~ x1 + x2 + x3    # latent factor defined by indicators
f2 ~ f1                # structural regression
x1 ~~ x1               # variance
```

Output: parameter estimates (`lval/op/rval/c/s/z/p`) + fit indices
(CFI, RMSEA, GFI, AIC, BIC, chi2).

## Session Cache (post-fix)

- Cache key is **`desc` + sha256(data)** (`_cache_key`), LRU-capped at 32
  models, thread-safe. Refitting the same syntax on *different* data no
  longer returns a stale model.
- `sem_inspect` accepts optional `d`: with `d` it looks up the exact
  desc+data fit; **without `d` it falls back to the most recent fit for
  that `desc`** (back-compat). Either way, inspect only works after a
  `sem_fit` in the same session.

## Pitfalls

- `inspect(std_est=True)` not `mode='std_est'` — actual semopy API
- Fixed parameters (first loading) have `'-'` for SE/z-value — handled gracefully
- calc_stats returns DataFrame; fit indices extracted via `.iloc[0]`
- `sem_run` signature is `(action, d, desc, mode)` — the registry handler
  must mirror all four

## Verification

```bash
python3 -c "
from tools.sem_tool import sem_run, _check_sem
assert _check_sem()
print('OK')
"
```

Or from the repo: `python3 -m pytest tests/ -q` (42 tests, must stay green).

## Dependencies

- semopy>=2.3,<3

## File Location

- `tools/sem_tool.py` — registry: `sem` in `medical` toolset
