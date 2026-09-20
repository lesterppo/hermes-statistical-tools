---
name: irt-tool
description: Build, test, and extend the agent-native IRT Hermes tool.
version: 1.1.0
author: Peter (lesterppo)
license: MIT
tags: [irt, girth, statistics, medical, agent-native]
---

# IRT Tool — Agent-Native Item Response Theory

Wraps girth 0.8.0: Rasch/1PL, 2PL, 3PL, GRM/PCM for polytomous data, ability
scoring, and CTT statistics. Together with PSPP, statsmodels, and SEM, it
completes the Hermes statistical suite.

## When to Use

- Fitting IRT models or scoring respondents in Hermes
- Debugging girth orientation / naming / optimizer issues
- Extending scoring (new estimators, SE methods)

## Quick Reference

```
irt(action="irt_rasch", d="CSV", method="jml")                 # or method="mml"
irt(action="irt_2pl",   d="CSV")
irt(action="irt_3pl",   d="CSV")
irt(action="irt_grm",   d="CSV")                               # polytomous 0,1,2,...
irt(action="irt_pcm",   d="CSV")
irt(action="irt_score", d="CSV", diff="-1.5,0.3,1.2", disc="1.0,1.2,0.8", model="2pl")
irt(action="irt_score", d="CSV", diff="...", disc="...", guess="0.2,0.15,0.25", model="3pl")
irt(action="irt_ctt",   d="CSV")
```

Caller data layout is **always rows = people, cols = items**. Binary items:
0/1 (1/2 auto-converted). Polytomous: 0,1,2,… ordered categories.

## Orientation (post-fix)

**Every** girth estimator and ability function expects items × people.
The tool centralizes this in `_to_girth()` (transpose people×items →
items×people) — no call site transposes by hand. Passing people×items
directly to girth silently swaps items for respondents (item estimates come
back person-length), which is why the old code produced garbage lengths.

## girth Result Naming (post-fix, verified on girth 0.8.0)

- `"Difficulty"` → **item** difficulties (n_items; 2D thresholds for GRM/PCM)
- `"Discrimination"` → **item** discriminations (n_items; scalar broadcast
  for Rasch/1PL fixed-discrimination output)
- `"Guessing"` → item guessing, 3PL only (n_items)
- `"Ability"` → **person** abilities (n_people)

The earlier "swapped names" theory was wrong — it was the orientation bug.
`_build_model_out` maps these canonically.

## Length Validation (post-fix)

Every vector is validated against the data shape before reporting:

- `diff`/`disc`/`g` must be item-length, `a` (ability) person-length
- Mismatches are **omitted** with `warn_d` / `warn_disc` / `warn_a` flags —
  a person-length vector is never reported as an item parameter or vice versa
- `irt_score` rejects `diff`/`disc`/`guess` whose length ≠ n_items with an error
- GRM/PCM do listwise deletion of missing rows (`.astype(int)` on NaN raises,
  so NaN rows are dropped first — same as the binary path)

## 3PL: Shim + Native Scorer (post-fix)

- girth 0.8.0's `threepl_mml` crashes on scipy ≥ 1.14 (size-1 array through
  `fminbound`). `_apply_scipy_compat_shim()` wraps the module-level
  `fminbound` to squeeze scalar outputs — 3PL fitting works instead of
  erroring. If the shim ever fails, the graceful error path (suggest
  `irt_2pl` / more data) still applies.
- girth 0.8.0's `ability_eap/mle/map` take **no guessing parameter**, so
  3PL scoring (`model="3pl"` + nonzero `guess`) uses the native
  `_score_3pl_mle`: per-respondent bounded MLE under the 3PL ICC with SE
  from observed information. Output `method` reads `"mle-3pl"` with an `se`
  array; perfect/zero scores clip to ±6 (MLE is ±infinity there).
- Scoring `method` accepts `mle`/`eap`/`map`; estimation-only values
  (`jml`/`mml`) fall back to `mle` instead of erroring.

## Pitfalls

- Binary data auto-converts 1/2 → 0/1; anything else non-0/1 is rejected
- `irt_run` signature is `(action, d, method, diff, disc, guess, model)` —
  the registry handler must mirror all seven
- GRM/PCM need genuine polytomous (multi-category) input
- Small samples make MML estimates unstable — prefer `irt_rasch`/`irt_2pl`

## Verification

```bash
python3 -c "
from tools.irt_tool import irt_run, _check_irt
assert _check_irt()
print('OK')
"
```

Or from the repo: `python3 -m pytest tests/ -q` (42 tests, must stay green).

## Dependencies

- girth>=0.8,<1

## File Location

- `tools/irt_tool.py` — registry: `irt` in `medical` toolset
