#!/usr/bin/env python3
"""
StatsModels — AI-agent-native tool wrapping statsmodels 0.14.6 for advanced
statistical analyses that the PSPP tool (pure scipy/numpy) cannot do.

Actions (6):
  mlm      — Linear Mixed Models (MixedLM)
  gee      — Generalised Estimating Equations
  gee_ord  — Ordinal GEE (ordered categorical outcomes)
  gee_nom  — Nominal GEE (unordered categorical outcomes)
  anova_rm — Repeated Measures ANOVA (AnovaRM)
  mice     — Multiple Imputation by Chained Equations

All data comes as CSV string via the `d` parameter (same pattern as PSPP tool).
Output is compact JSON with short keys: c=coef, s=std_err, z=z_val, p=p_val,
l=lower_ci, u=upper_ci, v=variance, ll=log_likelihood, aic, bic.
"""

from __future__ import annotations

import csv
import io
import json
import math
import threading
from typing import Any, Dict, List, Optional

import numpy as np

# Thread-safe lazy loading — statsmodels + pandas are heavy (~30 MB)
_sm = None
_sm_anova = None
_sm_mice = None
_pd = None
_sm_lock = threading.Lock()
_sm_import_error: Optional[str] = None


def _check_statsmodels() -> bool:
    """Gate: statsmodels + pandas must be importable."""
    global _sm_import_error
    try:
        import statsmodels.api  # noqa: F401
        import pandas  # noqa: F401
        return True
    except ImportError as e:
        _sm_import_error = str(e)
        return False


def _ensure_statsmodels():
    """Lazy import statsmodels modules thread-safely, once."""
    global _sm, _sm_anova, _sm_mice, _pd, _sm_import_error
    if _sm is not None:
        return
    if _sm_import_error is not None:
        raise ImportError(_sm_import_error)
    with _sm_lock:
        if _sm is not None:
            return
        if _sm_import_error is not None:
            raise ImportError(_sm_import_error)
        try:
            import statsmodels.api as _sm_mod
            from statsmodels.stats.anova import AnovaRM as _AnovaRM
            import statsmodels.imputation.mice as _mice_mod
            import pandas as _pd_mod

            _sm = _sm_mod
            _sm_anova = _AnovaRM
            _sm_mice = _mice_mod
            _pd = _pd_mod
        except ImportError as e:
            _sm_import_error = str(e)
            raise


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _ok(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _err(msg: str) -> str:
    return _ok({"e": msg})


def _parse_csv(csv_data: str):
    """Parse CSV string into pandas DataFrame."""
    _ensure_statsmodels()
    return _pd.read_csv(io.StringIO(csv_data.strip()))


def _to_numeric_series(series):
    """Convert a pandas Series to float, encoding categoricals as integer codes."""
    import pandas as pd
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    try:
        return series.astype(float)
    except (ValueError, TypeError):
        # Categorical — encode as integer codes, then float
        codes = series.astype("category").cat.codes
        return codes.astype(float)


def _expand_categoricals(df, exog_cols):
    """Replace categorical exog columns with dummy variables (drop_first).

    Returns (new_df, new_exog_cols).
    Continuous/numeric columns pass through unchanged.
    """
    import pandas as pd

    new_exog = []
    df_new = df.copy()

    for c in exog_cols:
        if c not in df_new.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_new[c]):
            new_exog.append(c)
        else:
            # Try numeric conversion first (strings like "1", "2")
            try:
                df_new[c] = df_new[c].astype(float)
                new_exog.append(c)
            except (ValueError, TypeError):
                # True categorical — create dummies
                dummies = pd.get_dummies(df_new[c], prefix=c, drop_first=True)
                for dcol in dummies.columns:
                    df_new[dcol] = dummies[dcol].astype(float)
                    new_exog.append(dcol)

    return df_new, new_exog


def _clean_arrays(df, endog_col, exog_cols, grp_col=None):
    """Extract endog/exog/groups arrays, drop rows with missing values.

    Categorical exog columns are expanded into dummy variables (drop_first).
    Group columns are encoded as integer codes if non-numeric.
    """
    exog_cols = list(exog_cols) if exog_cols else []

    # Expand categoricals BEFORE numeric conversion
    df, exog_cols = _expand_categoricals(df, exog_cols)

    cols = [endog_col]
    if exog_cols:
        cols.extend(exog_cols)
    if grp_col:
        cols.append(grp_col)
    existing = [c for c in cols if c in df.columns]

    df_clean = df[existing].copy()

    # Convert endog and group columns to numeric
    df_clean[endog_col] = _to_numeric_series(df_clean[endog_col])
    if grp_col and grp_col in df_clean.columns:
        df_clean[grp_col] = _to_numeric_series(df_clean[grp_col])

    df_clean = df_clean.dropna()
    if len(df_clean) == 0:
        raise ValueError("no complete cases after dropping missing values")

    y = df_clean[endog_col].astype(float)
    if exog_cols:
        X = df_clean[exog_cols].astype(float)
    else:
        # Intercept-only model — create constant column
        import pandas as _pd_local
        X = _pd_local.DataFrame({"const": 1.0}, index=df_clean.index)
    groups = df_clean[grp_col] if grp_col else None

    # Add intercept constant (all models need it without formula API).
    # Skip if we already created a constant-only DataFrame for intercept-only.
    if X is not None and len(X.columns) > 0 and not (
        len(X.columns) == 1 and X.columns[0] == "const" and (X["const"] == 1.0).all()
    ):
        X = _sm.add_constant(X, has_constant="add")

    return y, X, groups, len(df_clean)


def _fmt_float(v: float) -> Optional[float]:
    """Round and guard against inf/nan for JSON serialisation.

    Very small p-values (below 1e-8) are returned as floats for compactness
    — the consuming LLM can recognise scientific notation in JSON.
    """
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        f = float(v)
        # For very small p-values, preserve precision
        if 0 < abs(f) < 1e-8:
            return f  # let JSON serialise as scientific notation
        return round(f, 6)
    return v


def _sget(obj: Any, attr: str) -> Any:
    """Safe getattr: return None when the attribute is missing OR when
    accessing it raises (e.g. NominalGEE's llf/aic raise NotImplementedError
    instead of being absent — hasattr() would propagate those)."""
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _fmt_result(result, is_mixed: bool = False) -> dict:
    """Extract compact summary from a statsmodels result object.

    Works for: MixedLM, GEE, OrdinalGEE, NominalGEE results.

    Note: several fit-stat attributes (llf, aic, bic) are properties that
    RAISE NotImplementedError on NominalGEE instead of being absent, and
    hasattr() does not swallow non-AttributeError exceptions — so every
    attribute access here goes through the _sget() safe getter.
    """
    out: Dict[str, Any] = {}

    # Model fit stats
    _ll = _sget(result, "llf")
    if _ll is not None:
        out["ll"] = _fmt_float(_ll)
    _aic = _sget(result, "aic")
    if _aic is not None:
        out["aic"] = _fmt_float(_aic)
    _bic = _sget(result, "bic")
    if _bic is not None:
        out["bic"] = _fmt_float(_bic)
    _scale = _sget(result, "scale")
    if _scale is not None:
        out["v"] = _fmt_float(_scale)

    # Fixed effects
    params = result.params
    try:
        bse = result.bse
    except Exception:
        try:
            bse = result.bse_fe
        except Exception:
            bse = None

    try:
        tvals = result.tvalues
    except Exception:
        tvals = None

    try:
        pvals = result.pvalues
    except Exception:
        pvals = None

    try:
        ci = result.conf_int()
    except Exception:
        ci = None

    out["f"] = {}
    for name in params.index:
        # Skip variance component params in mixed models
        if is_mixed and (name.startswith("Group") or name.endswith(" Var")):
            continue
        row: Dict[str, Optional[float]] = {}
        row["c"] = _fmt_float(params.get(name))
        if bse is not None and name in bse:
            row["s"] = _fmt_float(bse.get(name))
        if tvals is not None and name in tvals:
            row["z"] = _fmt_float(tvals.get(name))
        if pvals is not None and name in pvals:
            row["p"] = _fmt_float(pvals.get(name))
        if ci is not None and name in ci.index:
            row["l"] = _fmt_float(ci.loc[name].iloc[0])
            row["u"] = _fmt_float(ci.loc[name].iloc[1])
        out["f"][name] = row

    # Random effects (mixed models only)
    if is_mixed and hasattr(result, "random_effects"):
        re_dict = result.random_effects
        if re_dict:
            out["r"] = {}
            # Collect group-level REs
            if isinstance(re_dict, dict):
                # Summarise: mean and range of random intercepts
                vals = []
                for g in re_dict:
                    v = re_dict[g]
                    if hasattr(v, "iloc"):
                        vals.append(float(v.iloc[0]))
                    elif isinstance(v, (int, float)):
                        vals.append(float(v))
                    elif isinstance(v, np.ndarray):
                        vals.append(float(v[0]))
                if vals:
                    out["r"]["mean"] = _fmt_float(np.mean(vals))
                    out["r"]["sd"] = _fmt_float(np.std(vals, ddof=1))
                    out["r"]["n"] = len(vals)

            # Variance components from random effects covariance
            if hasattr(result, "cov_re") and result.cov_re is not None:
                rec = result.cov_re
                out["rv"] = {}
                if isinstance(rec, np.ndarray):
                    if rec.ndim == 2:
                        for i in range(rec.shape[0]):
                            out["rv"][str(i)] = _fmt_float(float(rec[i, i]))
                    elif rec.ndim == 1:
                        out["rv"]["0"] = _fmt_float(float(rec[0]))

    # Convergence info
    _cnvg = _sget(result, "converged")
    if _cnvg is not None:
        out["cnvg"] = bool(_cnvg)

    return out


def _fmt_anova(result) -> dict:
    """Extract AnovaRM results into compact format."""
    out = {"anova": {}}
    table = result.anova_table
    for factor in table.index:
        entry: Dict[str, Any] = {}
        row = table.loc[factor]
        for col in table.columns:
            # Match specific column patterns — order matters:
            # 'Pr > F' contains 'F' so must check 'Pr' first
            if "Pr" in col or col == "p" or col == "p-unc":
                entry["p"] = _fmt_float(row[col])
            elif col == "F Value" or col == "F":
                entry["f"] = _fmt_float(row[col])
            elif "Num DF" in col:
                entry["ndf"] = int(row[col])
            elif "Den DF" in col:
                entry["ddf"] = int(row[col])
        out["anova"][str(factor)] = entry
    return out


# ═══════════════════════════════════════════════════════════════════
# Family / Covariance Structure Mappers
# ═══════════════════════════════════════════════════════════════════

_FAMILY_MAP = {
    "gaussian": lambda: _sm.families.Gaussian(),
    "binomial": lambda: _sm.families.Binomial(),
    "poisson": lambda: _sm.families.Poisson(),
    "gamma": lambda: _sm.families.Gamma(),
}

_COV_MAP = {
    "independence": lambda: _sm.cov_struct.Independence(),
    "exchangeable": lambda: _sm.cov_struct.Exchangeable(),
    "ar1": lambda: _sm.cov_struct.Autoregressive(),
    "unstructured": lambda: _sm.cov_struct.Unstructured(),
}


# ═══════════════════════════════════════════════════════════════════
# Action Handlers
# ═══════════════════════════════════════════════════════════════════


def _action_mlm(args: dict) -> str:
    """Linear Mixed Model (sm.MixedLM)."""
    endog = args.get("o", "")
    exog = args.get("p", [])
    grp = args.get("g", "")

    if not endog:
        return _err("mlm requires 'o' (endog/dependent variable)")
    if not grp:
        return _err("mlm requires 'g' (grouping variable)")

    df = _parse_csv(args["d"])
    if endog not in df.columns:
        return _err(f"endog column '{endog}' not in data")
    if grp not in df.columns:
        return _err(f"group column '{grp}' not in data")
    for c in exog:
        if c not in df.columns:
            return _err(f"exog column '{c}' not in data")

    y, X, groups, n = _clean_arrays(df, endog, exog, grp)

    model = _sm.MixedLM(y, X, groups=groups)
    result = model.fit()

    out = _fmt_result(result, is_mixed=True)
    out["n"] = n
    return _ok(out)


def _action_gee(args: dict) -> str:
    """Generalised Estimating Equations (sm.GEE)."""
    endog = args.get("o", "")
    exog = args.get("p", [])
    grp = args.get("g", "")
    family = args.get("fam", "gaussian")
    cov_struct = args.get("cov", "exchangeable")

    if not endog:
        return _err("gee requires 'o' (endog/dependent variable)")
    if not grp:
        return _err("gee requires 'g' (grouping variable)")

    if family not in _FAMILY_MAP:
        return _err(f"unknown family '{family}'. valid: {list(_FAMILY_MAP)}")
    if cov_struct not in _COV_MAP:
        return _err(
            f"unknown cov structure '{cov_struct}'. valid: {list(_COV_MAP)}"
        )

    df = _parse_csv(args["d"])
    if endog not in df.columns:
        return _err(f"endog column '{endog}' not in data")
    if grp not in df.columns:
        return _err(f"group column '{grp}' not in data")
    for c in exog:
        if c not in df.columns:
            return _err(f"exog column '{c}' not in data")

    y, X, groups, n = _clean_arrays(df, endog, exog, grp)

    model = _sm.GEE(
        y, X, groups=groups, family=_FAMILY_MAP[family](),
        cov_struct=_COV_MAP[cov_struct]()
    )
    result = model.fit()

    out = _fmt_result(result)
    out["n"] = n
    return _ok(out)


def _action_gee_ord(args: dict) -> str:
    """Ordinal GEE (sm.OrdinalGEE) for ordered categorical outcomes."""
    endog = args.get("o", "")
    exog = args.get("p", [])
    grp = args.get("g", "")
    cov_struct = args.get("cov", "exchangeable")

    if not endog:
        return _err("gee_ord requires 'o' (endog/dependent variable)")
    if not grp:
        return _err("gee_ord requires 'g' (grouping variable)")

    if cov_struct not in _COV_MAP:
        return _err(
            f"unknown cov structure '{cov_struct}'. valid: {list(_COV_MAP)}"
        )

    df = _parse_csv(args["d"])
    if endog not in df.columns:
        return _err(f"endog column '{endog}' not in data")
    if grp not in df.columns:
        return _err(f"group column '{grp}' not in data")
    for c in exog:
        if c not in df.columns:
            return _err(f"exog column '{c}' not in data")

    y, X, groups, n = _clean_arrays(df, endog, exog, grp)

    model = _sm.OrdinalGEE(
        y.astype(int), X, groups=groups, cov_struct=_COV_MAP[cov_struct]()
    )
    result = model.fit()

    out = _fmt_result(result)

    # Add threshold info
    if hasattr(result, "transform_threshold_params"):
        try:
            thresh = result.transform_threshold_params()
            out["th"] = [_fmt_float(float(t)) for t in thresh]
        except Exception:
            pass

    out["n"] = n
    return _ok(out)


def _action_gee_nom(args: dict) -> str:
    """Nominal GEE (sm.NominalGEE) for unordered categorical outcomes.

    Note: NominalGEE is experimental in statsmodels and may raise
    NotImplementedError on some data configurations.
    """
    endog = args.get("o", "")
    exog = args.get("p", [])
    grp = args.get("g", "")
    cov_struct = args.get("cov", "exchangeable")

    if not endog:
        return _err("gee_nom requires 'o' (endog/dependent variable)")
    if not grp:
        return _err("gee_nom requires 'g' (grouping variable)")

    if cov_struct not in _COV_MAP:
        return _err(
            f"unknown cov structure '{cov_struct}'. valid: {list(_COV_MAP)}"
        )

    df = _parse_csv(args["d"])
    if endog not in df.columns:
        return _err(f"endog column '{endog}' not in data")
    if grp not in df.columns:
        return _err(f"group column '{grp}' not in data")
    for c in exog:
        if c not in df.columns:
            return _err(f"exog column '{c}' not in data")

    y, X, groups, n = _clean_arrays(df, endog, exog, grp)

    try:
        model = _sm.NominalGEE(
            y.astype(int), X, groups=groups, cov_struct=_COV_MAP[cov_struct]()
        )
        result = model.fit()
    except NotImplementedError:
        return _err(
            "NominalGEE is not fully implemented for this data configuration "
            "in statsmodels 0.14.6. Try using gee_ord (Ordinal GEE) for "
            "ordered outcomes, or gee with binary/multinomial encoding."
        )

    out = _fmt_result(result)
    out["n"] = n
    return _ok(out)


def _action_anova_rm(args: dict) -> str:
    """Repeated Measures ANOVA (AnovaRM) — within-subject factors only.

    statsmodels 0.14.6 AnovaRM does not support between-subject factors.
    For mixed between-within designs, use the 'mlm' action instead.
    """
    endog = args.get("o", "")  # depvar
    grp = args.get("g", "")    # subject
    within = args.get("w", [])

    if not endog:
        return _err("anova_rm requires 'o' (dependent variable)")
    if not grp:
        return _err("anova_rm requires 'g' (subject identifier)")
    if not within:
        return _err(
            "anova_rm requires at least one within-subject factor ('w'). "
            "For between-subject or mixed designs, use the 'mlm' action."
        )

    df = _parse_csv(args["d"])
    cols = [endog, grp]
    if within:
        cols.extend(within)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        return _err(f"columns not in data: {missing}")

    df_clean = df[cols].dropna()
    if len(df_clean) == 0:
        return _err("no complete cases after dropping missing values")

    try:
        model = _sm_anova(
            df_clean, depvar=endog, subject=grp,
            within=within if within else None,
        )
        result = model.fit()
        out = _fmt_anova(result)
        out["n"] = len(df_clean)
        return _ok(out)
    except NotImplementedError as e:
        return _err(f"AnovaRM error: {e}")
    except Exception as e:
        return _err(f"AnovaRM fit failed: {e}")


def _action_mice(args: dict) -> str:
    """Multiple Imputation by Chained Equations (MICE) with pooled analysis.

    Imputes missing values and fits a model (OLS/Logit/Poisson) across
    all imputed datasets, returning pooled coefficients (Rubin's Rules).
    """
    imp_count = args.get("n", 5)
    model_type = args.get("mod", "ols")
    endog = args.get("o", "")
    exog = args.get("p", [])

    if not endog:
        return _err("mice requires 'o' (outcome/dependent variable)")
    if not exog:
        return _err("mice requires 'p' (predictor variables)")

    # Map model type to statsmodels class
    model_map = {
        "ols": _sm.OLS,
        "logit": _sm.Logit,
        "poisson": _sm.Poisson,
    }
    if model_type not in model_map:
        return _err(
            f"unknown model type '{model_type}'. valid: {list(model_map)}"
        )

    df = _parse_csv(args["d"])

    # Build formula string from column names
    # Use "1" for intercept-only models (empty exog)
    rhs = " + ".join(exog) if exog else "1"
    formula = f"{endog} ~ {rhs}"

    try:
        mice_data = _sm_mice.MICEData(df)
        mice = _sm_mice.MICE(formula, model_map[model_type], mice_data)
        result = mice.fit(n_imputations=imp_count)
    except np.linalg.LinAlgError as e:
        return _err(
            f"MICE linear algebra error (likely from small dataset): {e}. "
            "Try with more observations or fewer variables with missing data."
        )
    except Exception as e:
        return _err(f"MICE analysis failed: {type(e).__name__}: {e}")

    # Build compact output from MICEResults
    out: Dict[str, Any] = {
        "imp": imp_count,
        "n": len(df),
    }

    # Pooled model fit stats
    if hasattr(result, "llf"):
        out["ll"] = _fmt_float(float(result.llf))

    # Fixed effects (pooled via Rubin's Rules)
    out["f"] = {}
    params = result.params
    bse = result.bse
    tvals = result.tvalues
    pvals = result.pvalues
    exog_names = result.exog_names

    for i, name in enumerate(exog_names):
        row: Dict[str, Optional[float]] = {}
        # MICEResults.params/bse/tvalues/pvalues are numpy arrays, not Series
        if params is not None:
            row["c"] = _fmt_float(float(params[i]))
        if bse is not None:
            row["s"] = _fmt_float(float(bse[i]))
        if tvals is not None:
            row["z"] = _fmt_float(float(tvals[i]))
        if pvals is not None:
            row["p"] = _fmt_float(float(pvals[i]))
        out["f"][name] = row

    # Fraction of missing information (numpy array, same order as exog_names)
    if hasattr(result, "frac_miss_info"):
        fmi = result.frac_miss_info
        if hasattr(fmi, "__len__") and exog_names and len(fmi) == len(exog_names):
            out["fmi"] = {
                str(name): _fmt_float(float(fmi[i]))
                for i, name in enumerate(exog_names)
            }
        else:
            out["fmi"] = [_fmt_float(float(v)) for v in fmi]

    return _ok(out)


# ═══════════════════════════════════════════════════════════════════
# Main Dispatcher
# ═══════════════════════════════════════════════════════════════════

def statsmodels_run(
    action: str = "",
    d: str = "",
    o: str = "",
    p: Optional[List[str]] = None,
    g: str = "",
    fam: str = "gaussian",
    cov: str = "exchangeable",
    w: Optional[List[str]] = None,
    n: int = 5,
    mod: str = "ols",
) -> str:
    """Dispatch to the appropriate action handler.

    Args:
        action: mlm, gee, gee_ord, gee_nom, anova_rm, mice
        d: CSV data string with header row
        o: outcome/dependent variable column name
        p: predictor/independent variable column names
        g: grouping/subject variable column name
        fam: GEE family (gaussian, binomial, poisson, gamma)
        cov: GEE covariance structure (independence, exchangeable, ar1, unstructured)
        w: within-subject factors (anova_rm)
        b: between-subject factors (anova_rm)
        n: number of imputations (mice, default 5)
    """
    if p is None:
        p = []
    if w is None:
        w = []

    args = {
        "action": action,
        "d": d,
        "o": o,
        "p": p if isinstance(p, list) else [p],
        "g": g,
        "fam": fam,
        "cov": cov,
        "w": w if isinstance(w, list) else [w],
        "n": int(n) if n else 5,
        "mod": mod if mod else "ols",
    }

    try:
        _ensure_statsmodels()
    except ImportError as e:
        return _err(f"statsmodels not available: {e}")

    try:
        if action == "mlm":
            return _action_mlm(args)
        elif action == "gee":
            return _action_gee(args)
        elif action == "gee_ord":
            return _action_gee_ord(args)
        elif action == "gee_nom":
            return _action_gee_nom(args)
        elif action == "anova_rm":
            return _action_anova_rm(args)
        elif action == "mice":
            return _action_mice(args)
        else:
            valid = ["mlm", "gee", "gee_ord", "gee_nom", "anova_rm", "mice"]
            return _err(f"unknown action '{action}'. valid: {valid}")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════

STATSMODELS_SCHEMA = {
    "name": "statsmodels",
    "description": (
        "Advanced stats via statsmodels. "
        "Actions: mlm (MixedLM), gee (GEE), gee_ord (Ordinal GEE), "
        "gee_nom (Nominal GEE, experimental), anova_rm (Within-subject RM ANOVA), "
        "mice (Multiple Imputation with pooled analysis). "
        "Data as CSV string in 'd'. "
        "GEE families: gaussian, binomial, poisson, gamma. "
        "Cov structures: independence, exchangeable, ar1, unstructured. "
        "MICE models: ols, logit, poisson. "
        "For gee_ord, string outcomes are encoded alphabetically. "
        "If order matters, pre-encode the outcome column as integers "
        "in the CSV before calling this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["mlm", "gee", "gee_ord", "gee_nom", "anova_rm", "mice"],
                "description": "Analysis to run.",
            },
            "d": {
                "type": "string",
                "description": "CSV data with header row.",
            },
            "o": {
                "type": "string",
                "description": "Outcome/dependent variable column name.",
            },
            "p": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Predictor/independent variable column names.",
            },
            "g": {
                "type": "string",
                "description": "Grouping/subject identifier column name.",
            },
            "fam": {
                "type": "string",
                "enum": ["gaussian", "binomial", "poisson", "gamma"],
                "description": "GEE distribution family (default: gaussian).",
            },
            "cov": {
                "type": "string",
                "enum": ["independence", "exchangeable", "ar1", "unstructured"],
                "description": "GEE covariance structure (default: exchangeable).",
            },
            "w": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Within-subject factors (anova_rm). Required for anova_rm. For between-subject or mixed designs use mlm instead.",
            },
            "n": {
                "type": "integer",
                "description": "Number of imputations for MICE (default 5).",
            },
            "mod": {
                "type": "string",
                "enum": ["ols", "logit", "poisson"],
                "description": "MICE target model: OLS, logit, or poisson (default: ols).",
            },
        },
        "required": ["action", "d"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

from tools.registry import registry

registry.register(
    name="statsmodels",
    toolset="medical",
    schema=STATSMODELS_SCHEMA,
    handler=lambda args, **kw: statsmodels_run(
        action=args.get("action", ""),
        d=args.get("d", ""),
        o=args.get("o", ""),
        p=args.get("p"),
        g=args.get("g", ""),
        fam=args.get("fam", "gaussian"),
        cov=args.get("cov", "exchangeable"),
        w=args.get("w"),
        n=args.get("n", 5),
        mod=args.get("mod", "ols"),
    ),
    check_fn=_check_statsmodels,
    emoji="📊",
)
