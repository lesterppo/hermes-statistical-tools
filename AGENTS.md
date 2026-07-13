# Hermes Statistical Tools — AI Agent Guide

Five AI-agent-native statistical tools for the Hermes Agent framework.
All tools follow the same patterns: CSV data input, compact JSON output
with short keys, action-based routing, lazy imports, thread-safe.

## Quick Start for AI Agents

```bash
# Install
bash install.sh

# Copy tools into your Hermes checkout
cp tools/*.py ~/.hermes/hermes-agent/tools/

# Register in your toolsets.py medical toolset:
# "medical": {"tools": ["pspp", "statsmodels", "sem", "irt", "medical_ext"], ...}

# Verify
python3 -c "
from tools.pspp_tool import _check_pspp
from tools.statsmodels_tool import _check_statsmodels
from tools.sem_tool import _check_sem
from tools.irt_tool import _check_irt
from tools.medical_ext_tool import _check_backends
print('All OK')
"
```

## Tool Catalog

### 1. PSPP Tool (`pspp_tool.py`, 2376 lines, 45 types)
- **Package**: scipy, numpy (no PSPP binary)
- **Actions**: t parameter specifies test type
- **Schema**: `PSPP_SCHEMA` (1780 chars)
- **Registration**: `registry.register(name="pspp", toolset="medical", ...)`
- **Skill**: `skills/pspp/SKILL.md`

### 2. StatsModels Tool (`statsmodels_tool.py`, 832 lines, 6 types)
- **Package**: statsmodels >= 0.14
- **Actions**: mlm, gee, gee_ord, gee_nom, anova_rm, mice
- **Schema**: `STATSMODELS_SCHEMA` (1858 chars)
- **Registration**: `registry.register(name="statsmodels", toolset="medical", ...)`
- **Skill**: `skills/statsmodels/SKILL.md`

### 3. SEM Tool (`sem_tool.py`, 362 lines, 2 types)
- **Package**: semopy >= 2.3
- **Actions**: sem_fit, sem_inspect
- **Schema**: `SEM_SCHEMA`
- **Registration**: `registry.register(name="sem", toolset="medical", ...)`
- **Skill**: `skills/sem/SKILL.md`

### 4. IRT Tool (`irt_tool.py`, 550 lines, 7 types)
- **Package**: girth >= 0.8
- **Actions**: irt_rasch, irt_2pl, irt_3pl, irt_grm, irt_pcm, irt_score, irt_ctt
- **Schema**: `IRT_SCHEMA`
- **Registration**: `registry.register(name="irt", toolset="medical", ...)`
- **Skill**: `skills/irt/SKILL.md`

### 5. Medical Extended Tool (`medical_ext_tool.py`, ~690 lines, 7 types)
- **Package**: lifelines >= 0.29 (survival), statsmodels >= 0.14 (power),
  pure numpy/scipy (meta — statsmodels 0.14.6 lacks effect_sizes)
- **Actions**: km, cox, logrank (survival); forest (meta-analysis);
  ttest, anova, prop (power/sample-size)
- **Input**: survival/meta use CSV via `d` (columns t,e[,g,p] / yi,vi[,label]);
  power uses structured params (d_es/f/p1/p2/n/pw/a/ratio/alt/k)
- **Schema**: `MEDICAL_EXT_SCHEMA`
- **Registration**: `registry.register(name="medical_ext", toolset="medical", ...)`
- **Pass 0 for the quantity to solve** in power actions (nobs, effect_size,
  or power); the tool solves the missing one via statsmodels power formulas.

## Design Patterns (for extending or building new tools)

### Pattern 1: Action-Based Routing
One tool, one schema, many operations via `action` parameter.

### Pattern 2: Lazy Import
Thread-safe double-checked locking. Heavy packages (~30MB) only load on first use.

### Pattern 3: Compact Output
Short JSON keys: `c`=coef, `s`=std_err, `z`=z_val, `p`=p_val, `d`=difficulty,
`a`=ability, `ll`=log_likelihood, `aic`/`bic`, `hr`=hazard_ratio, `n`=nobs.

### Pattern 4: CSV Input
Survival/meta data arrives as CSV string via `d` parameter. Power uses
structured numeric params. Tools parse internally.

### Pattern 5: Error Messages
All errors return `{"e": "message"}` — never raw stack traces. Messages must be actionable for the LLM.

### Pattern 6: Registry Registration
```python
from tools.registry import registry
registry.register(
    name="tool_name",
    toolset="medical",
    schema=SCHEMA,
    handler=lambda args, **kw: dispatch_function(...),
    check_fn=lambda: True,  # or import check
    emoji="📊",
)
```

## Privacy Guarantee
- Zero hardcoded user paths (uses `get_hermes_home()`)
- Zero API keys or secrets
- Zero personal identifiers
- All file operations scoped to HERMES_HOME
