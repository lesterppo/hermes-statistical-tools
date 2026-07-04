---
name: sem-irt-tools
description: Build, test, and extend the SEM and IRT Hermes tools.
version: 1.0.0
author: Peter (lesterppo)
license: MIT
tags: [sem, irt, semopy, girth, statistics, medical, agent-native]
---

# SEM + IRT Tools — Agent-Native SEM and IRT

Two tools wrapping semopy 2.3.11 (SEM) and girth 0.8.0 (IRT). Together with
PSPP (45 types) and statsmodels (6 types), these complete the Hermes statistical
analysis suite: descriptive → mixed models → SEM → IRT.

## SEM Tool (sem_tool.py, 362 lines)

Actions: `sem_fit`, `sem_inspect`

Model syntax (lavaan-like):
```
f1 =~ x1 + x2 + x3    # latent factor defined by indicators
f2 ~ f1                # structural regression
x1 ~~ x1               # variance
```

Output: parameter estimates (lval/op/rval/c/s/z/p) + fit indices (CFI, RMSEA, GFI, AIC, BIC, chi2).

### Pitfalls
- `inspect(std_est=True)` not `mode='std_est'` — actual semopy API
- Cached per-session: `sem_inspect` only works after `sem_fit` in same session
- Fixed parameters (first loading) have `'-'` for SE/z-value — handled gracefully
- calc_stats returns DataFrame; fit indices extracted via `.iloc[0]`

## IRT Tool (irt_tool.py, 550 lines)

Actions: `irt_rasch`, `irt_2pl`, `irt_3pl`, `irt_grm`, `irt_pcm`, `irt_score`, `irt_ctt`

### girth Naming Quirk
girth MML output swaps field names:
- `"Difficulty"` → person ability (theta)
- `"Ability"` → item difficulty (beta)
- Our `_build_model_out` remaps correctly

### Orientation
- Model fitting: data as (n_people, n_items) — girth accepts both orientations
- Scoring: data MUST be transposed to (n_items, n_people)
- Binary data: auto-converted 1/2 → 0/1

### 3PL Limitations
girth 0.8.0 threepl_mml has a scipy optimizer bug that crashes on many
datasets. Tool catches this gracefully with an actionable error message
suggesting irt_2pl or more data.

## Verification
```bash
python3 -c "
from tools.sem_tool import sem_run, _check_sem
from tools.irt_tool import irt_run, _check_irt
assert _check_sem() and _check_irt()
print('OK')
"
```

## Dependencies
- semopy>=2.3,<3
- girth>=0.8,<1

## File Locations
- `tools/sem_tool.py` — registry: `sem` in `medical` toolset
- `tools/irt_tool.py` — registry: `irt` in `medical` toolset
