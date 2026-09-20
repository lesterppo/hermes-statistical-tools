#!/usr/bin/env python3
"""
Medical Extended — AI-agent-native headless tool wrapping three analysis
families the rest of the Hermes medical suite does not cover:

  survival — time-to-event analysis (Kaplan-Meier, Cox PH, log-rank)
             backend: lifelines
  meta     — meta-analysis (pooled effect, forest summary, heterogeneity)
             backend: pure numpy/scipy (statsmodels 0.14.6 lacks effect_sizes)
  power    — sample-size / power / effect-size planning
             backend: statsmodels.stats.power + proportion_effectsize

Design mirrors the sibling medical tools (pspp/statsmodels/sem/irt):
  * agent passes a native `action` name (e.g. cox, km, logrank, forest, ttest)
  * data arrives as a CSV string in `d` (survival) or as structured params
  * output is compact JSON with short keys: c=coef, s=se, z=z/stat, p=p_val,
    l=lower_ci, u=upper_ci, hr=hazard_ratio, n=nobs, etc.
  * gated via check_fn so it only appears when the backends import.

All runners are pure computation — no browser, no UI (headless).
"""

from __future__ import annotations

import io
import json
import math
import re
import threading
from typing import Any, Dict, List, Optional

import numpy as np

# Thread-safe lazy loading — lifelines and statsmodels are heavy.
_lf = None          # lifelines module
_lf_stat = None     # lifelines.statistics
_pd = None
_sm = None
_sm_power = None
_lock = threading.Lock()
_lazy_err: Optional[str] = None


def _check_backends() -> bool:
    """Gate: lifelines must be importable (survival) — meta/power degrade gracefully."""
    global _lazy_err
    try:
        import lifelines  # noqa: F401
        import pandas  # noqa: F401
        return True
    except ImportError as e:
        _lazy_err = str(e)
        return False


def _ensure_lifelines():
    global _lf, _lf_stat, _pd, _sm, _sm_power, _lazy_err
    if _lf is not None:
        return
    if _lazy_err is not None:
        raise ImportError(_lazy_err)
    with _lock:
        if _lf is not None:
            return
        if _lazy_err is not None:
            raise ImportError(_lazy_err)
        try:
            import lifelines as _lf_mod
            from lifelines import statistics as _lf_stat_mod
            import pandas as _pd_mod
            try:
                import statsmodels.api as _sm_mod
                from statsmodels.stats import power as _sm_power_mod
            except ImportError:
                _sm_mod = None
                _sm_power_mod = None
            _lf = _lf_mod
            _lf_stat = _lf_stat_mod
            _pd = _pd_mod
            _sm = _sm_mod
            _sm_power = _sm_power_mod
        except ImportError as e:
            _lazy_err = str(e)
            raise


# ═══════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════

def _ok(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _err(msg: str) -> str:
    return _ok({"e": msg})


def _f(v: Any) -> Optional[float]:
    """Round and guard inf/nan for JSON."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    if 0 < abs(fv) < 1e-8:
        return fv  # preserve tiny p-values as scientific notation
    return round(fv, 6)


def _parse_csv(csv_data: str):
    _ensure_lifelines()
    return _pd.read_csv(io.StringIO(csv_data.strip()))


def _to_num(series):
    """Coerce a pandas Series to float (encode categoricals as codes if needed)."""
    import pandas as pd
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    try:
        return series.astype(float)
    except (ValueError, TypeError):
        return series.astype("category").cat.codes.astype(float)


# ═══════════════════════════════════════════════════════════════════
# SURVIVAL family
# ═══════════════════════════════════════════════════════════════════

def _surv_km(args: dict) -> str:
    """Kaplan-Meier survival estimate for one group (or whole cohort).

    d: CSV with duration col (T) and event col (E). Optional group col (g).
    If g given, returns per-group KM + log-rank across groups.
    """
    Tcol = args.get("t", "")
    Ecol = args.get("e", "")
    gcol = args.get("g", "")
    if not Tcol or not Ecol:
        return _err("km requires 't' (duration) and 'e' (event indicator)")
    df = _parse_csv(args["d"])
    if Tcol not in df.columns or Ecol not in df.columns:
        return _err(f"km requires columns '{Tcol}' and '{Ecol}' in data")
    T = _to_num(df[Tcol]).values
    E = _to_num(df[Ecol]).values
    out: Dict[str, Any] = {}

    if gcol and gcol in df.columns:
        # Group labels preserved verbatim (no int truncation — float
        # durations and string groups both survive).
        import pandas as _pd_local
        codes, uniques = _pd_local.factorize(df[gcol])
        G = codes.astype(float)
        groups = sorted(set(int(x) for x in codes))
        glabel = {int(c): str(uniques[c]) for c in range(len(uniques))}
        out["g"] = {}
        for gi in groups:
            mask = codes == gi
            km = _lf.KaplanMeierFitter()
            km.fit(T[mask], event_observed=E[mask])
            ts = km.survival_function_["KM_estimate"]
            ci = km.confidence_interval_
            surv = {str(k): _f(v) for k, v in ts.items()}
            lo = {str(k): _f(v) for k, v in ci.iloc[:, 0].items()}
            hi = {str(k): _f(v) for k, v in ci.iloc[:, 1].items()}
            et = km.event_table
            at_risk_last = int(et["at_risk"].iloc[-1]) if "at_risk" in et else None
            out["g"][glabel[gi]] = {
                "n": int(mask.sum()),
                "median": _f(km.median_survival_time_),
                "at_risk": at_risk_last,
                "surv": surv,
                "lo": lo,
                "hi": hi,
            }
        # Log-rank across groups
        if len(groups) == 2:
            a, b = groups
            ma, mb = (codes == a), (codes == b)
            res = _lf_stat.logrank_test(
                T[ma], T[mb], event_observed_A=E[ma], event_observed_B=E[mb]
            )
            out["logrank"] = {"z": _f(res.test_statistic), "p": _f(res.p_value),
                              "g1": glabel[a], "g2": glabel[b]}
        else:
            try:
                res = _lf_stat.multivariate_logrank_test(T, codes, E)
                out["logrank"] = {"z": _f(res.test_statistic), "p": _f(res.p_value),
                                  "k": len(groups)}
            except Exception:
                pass
    else:
        km = _lf.KaplanMeierFitter()
        km.fit(T, event_observed=E)
        ts = km.survival_function_["KM_estimate"]
        ci = km.confidence_interval_
        out = {
            "n": int(len(T)),
            "median": _f(km.median_survival_time_),
            "surv": {str(k): _f(v) for k, v in ts.items()},
            "lo": {str(k): _f(v) for k, v in ci.iloc[:, 0].items()},
            "hi": {str(k): _f(v) for k, v in ci.iloc[:, 1].items()},
        }
        et = km.event_table
        if "at_risk" in et:
            out["at_risk"] = int(et["at_risk"].iloc[-1])
    return _ok(out)


def _surv_cox(args: dict) -> str:
    """Cox proportional hazards regression.

    d: CSV with duration (t), event (e), and predictors (p, list).
    Returns HR, coef, SE, z, p, CI per covariate + concordance index.
    """
    Tcol = args.get("t", "")
    Ecol = args.get("e", "")
    pcols = args.get("p", []) or []
    if not Tcol or not Ecol:
        return _err("cox requires 't' (duration) and 'e' (event indicator)")
    if not pcols:
        return _err("cox requires at least one predictor in 'p'")
    df = _parse_csv(args["d"])
    if Tcol not in df.columns or Ecol not in df.columns:
        return _err(f"cox requires columns '{Tcol}' and '{Ecol}' in data")
    for c in pcols:
        if c not in df.columns:
            return _err(f"cox predictor column '{c}' not in data")
    data = df[[Tcol, Ecol] + list(pcols)].copy()
    for c in pcols:
        data[c] = _to_num(data[c])
    data[Tcol] = _to_num(data[Tcol])
    data[Ecol] = _to_num(data[Ecol])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return _err("cox needs >= 3 complete cases")
    if data[Ecol].sum() == 0:
        return _err("cox requires at least one event (E=1)")

    cph = _lf.CoxPHFitter()
    cph.fit(data, duration_col=Tcol, event_col=Ecol)
    s = cph.summary
    f: Dict[str, Any] = {}
    for name in s.index:
        row = s.loc[name]
        f[str(name)] = {
            "c": _f(row["coef"]),
            "hr": _f(row["exp(coef)"]),
            "s": _f(row["se(coef)"]),
            "z": _f(row["z"]),
            "p": _f(row["p"]),
            "l": _f(row["exp(coef) lower 95%"]),
            "u": _f(row["exp(coef) upper 95%"]),
        }
    out = {
        "n": int(len(data)),
        "events": int(data[Ecol].sum()),
        "cindex": _f(cph.concordance_index_),
        "ll": _f(cph.log_likelihood_),
        "f": f,
    }
    return _ok(out)


def _surv_logrank(args: dict) -> str:
    """Pairwise log-rank test between two groups (g column with 2 levels).

    d: CSV with duration (t), event (e), group (g).
    """
    Tcol = args.get("t", "")
    Ecol = args.get("e", "")
    gcol = args.get("g", "")
    if not Tcol or not Ecol or not gcol:
        return _err("logrank requires 't', 'e', and 'g' columns")
    df = _parse_csv(args["d"])
    for c in (Tcol, Ecol, gcol):
        if c not in df.columns:
            return _err(f"logrank requires column '{c}' in data")
    T = _to_num(df[Tcol]).values
    E = _to_num(df[Ecol]).values
    import pandas as _pd_lr
    codes, uniques = _pd_lr.factorize(df[gcol])
    if len(uniques) != 2:
        return _err(f"logrank needs exactly 2 groups, found {len(uniques)}: {list(uniques)}")
    labels = [str(u) for u in uniques]
    ma, mb = (codes == 0), (codes == 1)
    res = _lf_stat.logrank_test(
        T[ma], T[mb], event_observed_A=E[ma], event_observed_B=E[mb]
    )
    return _ok({"z": _f(res.test_statistic), "p": _f(res.p_value),
                "g1": labels[0], "g2": labels[1],
                "n1": int(ma.sum()), "n2": int(mb.sum())})


# ═══════════════════════════════════════════════════════════════════
# META family  (pure numpy/scipy — statsmodels 0.14.6 lacks effect_sizes)
# ═══════════════════════════════════════════════════════════════════

def _meta_forest(args: dict) -> str:
    """Inverse-variance meta-analysis with fixed-effect + DerSimonian-Laird random-effects.

    d: CSV (or inline) with one row per study. Columns: yi (effect), vi (var),
    optional label. Returns pooled FE/RE estimates, heterogeneity (Q, I^2, tau^2),
    forest summary (weights), and z-test of RE estimate.
    """
    d = args.get("d", "")
    if not d.strip():
        return _err("forest requires 'd' with columns: yi, vi (optional label)")
    try:
        df = _parse_csv(d)
    except Exception as e:
        return _err(f"forest could not parse data: {e}")
    yi_c, vi_c, lab_c = None, None, None
    for cand in ("yi", "effect", "estimate", "logrr", "logor"):
        if cand in df.columns:
            yi_c = cand
            break
    for cand in ("vi", "var", "se2"):
        if cand in df.columns:
            vi_c = cand
            break
    for cand in ("label", "study", "name", "id", "author"):
        if cand in df.columns:
            lab_c = cand
            break
    if "se" in df.columns and vi_c is None:
        df["_vi_from_se"] = df["se"].astype(float) ** 2
        vi_c = "_vi_from_se"
    if yi_c is None or vi_c is None:
        return _err("forest requires effect column (yi) and variance column (vi or se)")
    for c in (df[yi_c], df[vi_c]):
        try:
            c.astype(float)
        except (ValueError, TypeError):
            return _err("forest effect/variance columns must be numeric")
    yi = df[yi_c].astype(float).values
    vi = df[vi_c].astype(float).values
    labels = df[lab_c].astype(str).values if lab_c else [f"st{i+1}" for i in range(len(yi))]
    if len(yi) < 2:
        return _err("forest needs >= 2 studies")
    if np.any(vi <= 0):
        return _err("forest variance (vi) must be > 0 for all studies")

    k = len(yi)

    # Prefer statsmodels 0.15+ native meta-analysis (validated vs R metafor);
    # fall back to a pure-numpy DerSimonian-Laird implementation on 0.14.x.
    try:
        from statsmodels.stats.meta_analysis import combine_effects
        res = combine_effects(yi, vi, method_re="dl", row_names=list(labels))
        fe = float(res.mean_effect_fe)
        var_fe = float(res.var_eff_w_fe)
        re = float(res.mean_effect_re)
        var_re = float(res.var_eff_w_re)
        tau2 = float(res.tau2)  # may be slightly negative (Q < df) — keep raw DL value
        Q = float(res.q)
        dfree = k - 1
        from scipy import stats as sps
        Qp = float(sps.chi2.sf(Q, dfree))
        I2 = max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0
        z = re / math.sqrt(var_re) if var_re > 0 else 0.0
        p_re = float(2 * (1 - sps.norm.cdf(abs(z))))
        method_re = "der_simonian_laird"
        method_fe = "inverse_variance_fixed"
        # per-study weights (use FE weights for the forest display)
        w = 1.0 / vi
        wre = 1.0 / (vi + tau2) if tau2 > 0 else w
    except Exception:
        # Pure-numpy fallback (statsmodels < 0.15)
        w = 1.0 / vi
        fe = float(np.sum(w * yi) / np.sum(w))
        var_fe = 1.0 / np.sum(w)
        Q = float(np.sum(w * (yi - fe) ** 2))
        dfree = k - 1
        from scipy import stats as sps
        Qp = float(sps.chi2.sf(Q, dfree))
        denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = (Q - dfree) / denom if denom > 0 else 0.0  # raw DL; may be negative
        wre = 1.0 / (vi + tau2)
        re = float(np.sum(wre * yi) / np.sum(wre))
        var_re = 1.0 / np.sum(wre)
        I2 = max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0
        z = re / math.sqrt(var_re)
        p_re = float(2 * (1 - sps.norm.cdf(abs(z))))
        method_re = "der_simonian_laird"
        method_fe = "inverse_variance_fixed"

    # Forest rows with weights
    forest = []
    for i in range(k):
        wi = w[i]
        wgt_fe = float(wi / np.sum(w))
        wgt_re = float(wre[i] / np.sum(wre)) if tau2 > 0 else wgt_fe
        forest.append({
            "label": str(labels[i]),
            "yi": _f(yi[i]),
            "vi": _f(vi[i]),
            "se": _f(math.sqrt(vi[i])),
            "w_fe": _f(wgt_fe),
            "w_re": _f(wgt_re),
        })

    return _ok({
        "k": k,
        "fe": {"est": _f(fe), "se": _f(math.sqrt(var_fe)),
               "l": _f(fe - 1.96 * math.sqrt(var_fe)),
               "u": _f(fe + 1.96 * math.sqrt(var_fe)),
               "z": _f(fe / math.sqrt(var_fe)),
               "p": _f(2 * (1 - sps.norm.cdf(abs(fe / math.sqrt(var_fe))))),
               "method": method_fe},
        "re": {"est": _f(re), "se": _f(math.sqrt(var_re)),
               "l": _f(re - 1.96 * math.sqrt(var_re)),
               "u": _f(re + 1.96 * math.sqrt(var_re)),
               "tau2": _f(tau2), "z": _f(z), "p": _f(p_re),
               "method": method_re},
        "hetero": {"Q": _f(Q), "Qdf": dfree, "Qp": _f(Qp),
                   "I2": _f(I2), "tau2": _f(tau2)},
        "forest": forest,
    })


# ═══════════════════════════════════════════════════════════════════
# POWER family  (statsmodels.stats.power)
# ═══════════════════════════════════════════════════════════════════

def _power_ttest(args: dict) -> str:
    """Two-sample t-test power/sample-size.

    Solve the one missing quantity: nobs1 (n per group), effect_size (Cohen's d),
    power, or alpha. Provide the three you know; 'ratio' for unequal groups.
    """
    if _sm_power is None:
        return _err("statsmodels.stats.power unavailable")
    from statsmodels.stats.power import TTestIndPower
    p = TTestIndPower()
    nobs1 = _noneable(args.get("n"))
    es = _noneable(args.get("d_es"))
    power = _noneable(args.get("pw"))
    alpha = _noneable(args.get("a"), default=0.05)
    ratio = _noneable(args.get("ratio"), default=1.0)
    alt = args.get("alt", "two-sided")
    if alt not in ("two-sided", "larger", "smaller"):
        return _err("alt must be two-sided|larger|smaller")
    try:
        val = p.solve_power(effect_size=es, nobs1=nobs1, alpha=alpha,
                            power=power, ratio=ratio, alternative=alt)
    except Exception as e:
        return _err(f"ttest solve failed: {type(e).__name__}: {e}")
    return _ok({"nobs1": _f(val) if nobs1 is None else _f(nobs1),
                "effect_size": _f(val) if es is None else _f(es),
                "power": _f(val) if power is None else _f(power),
                "alpha": _f(alpha), "ratio": _f(ratio), "alt": alt})


def _power_anova(args: dict) -> str:
    """One-way ANOVA (F-test) power/sample-size.

    effect_size = Cohen's f. k = number of groups. Solve missing of n/pw/es/alpha.
    """
    if _sm_power is None:
        return _err("statsmodels.stats.power unavailable")
    from statsmodels.stats.power import FTestAnovaPower
    p = FTestAnovaPower()
    nobs = _noneable(args.get("n"))
    es = _noneable(args.get("f"))
    power = _noneable(args.get("pw"))
    alpha = _noneable(args.get("a"), default=0.05)
    k = int(_noneable(args.get("k"), default=3))
    if k < 2:
        return _err("anova k (groups) must be >= 2")
    try:
        val = p.solve_power(effect_size=es, nobs=nobs, alpha=alpha,
                            power=power, k_groups=k)
    except Exception as e:
        return _err(f"anova solve failed: {type(e).__name__}: {e}")
    return _ok({"nobs": _f(val) if nobs is None else _f(nobs),
                "effect_size": _f(val) if es is None else _f(es),
                "power": _f(val) if power is None else _f(power),
                "alpha": _f(alpha), "k_groups": k})


def _power_prop(args: dict) -> str:
    """Two-proportion z-test power/sample-size.

    Provide two of {p1, p2, n, pw}; alpha default 0.05. Effect size derived
    from proportions via proportion_effectsize.
    """
    if _sm_power is None:
        return _err("statsmodels.stats.power unavailable")
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    p1 = _noneable(args.get("p1"))
    p2 = _noneable(args.get("p2"))
    nobs = _noneable(args.get("n"))
    power = _noneable(args.get("pw"))
    alpha = _noneable(args.get("a"), default=0.05)
    ratio = _noneable(args.get("ratio"), default=1.0)
    if p1 is None or p2 is None:
        return _err("prop requires both p1 and p2 (proportions)")
    if not (0 < p1 < 1) or not (0 < p2 < 1):
        return _err("prop proportions must be in (0,1)")
    try:
        es = float(proportion_effectsize(p2, p1))
    except Exception as e:
        return _err(f"prop effect size failed: {e}")
    p = NormalIndPower()
    try:
        val = p.solve_power(effect_size=es, nobs1=nobs, alpha=alpha,
                            power=power, ratio=ratio, alternative="two-sided")
    except Exception as e:
        return _err(f"prop solve failed: {type(e).__name__}: {e}")
    return _ok({"effect_size": _f(es),
                "nobs1": _f(val) if nobs is None else _f(nobs),
                "power": _f(power) if power is None else _f(power),
                "alpha": _f(alpha), "p1": _f(p1), "p2": _f(p2),
                "ratio": _f(ratio)})


def _noneable(v, default=None):
    """Resolve a possibly-missing numeric arg; None/0 means 'solve for this'.

    0 is treated as a sentinel for 'unspecified' because a real nobs/power/
    alpha/effect_size is never 0 for these study-design computations.
    """
    if v is None:
        return default
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(fv) or math.isinf(fv) or fv == 0.0:
        return default
    return fv


# ═══════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════

def medical_ext_run(
    action: str = "",
    d: str = "",
    t: str = "",
    e: str = "",
    g: str = "",
    p: Optional[List[str]] = None,
    # meta/power params passed as JSON string or via individual kwargs
    n=None, pw=None, a=None, ratio=None, alt=None, f=None, k=None,
    p1=None, p2=None, d_es=None, **meta_power_extra
) -> str:
    """Dispatch to the appropriate analysis sub-handler.

    survival: km, cox, logrank
    meta:     forest
    power:    ttest, anova, prop
    """
    if p is None:
        p = []
    # Collect meta/power params (could also arrive json-encoded in d for those)
    mp = {
        "n": n, "pw": pw, "a": a, "ratio": ratio, "alt": alt,
        "f": f, "k": k, "p1": p1, "p2": p2, "d_es": d_es,
    }
    # If caller passed a JSON meta/power payload in 'n' or 'd', parse it.
    for key in ("n", "pw", "a", "ratio", "alt", "f", "k", "p1", "p2", "d_es"):
        if isinstance(mp.get(key), str) and mp[key].strip().startswith("{"):
            try:
                payload = json.loads(mp[key])
                for kk, vv in payload.items():
                    mp[kk] = vv
            except Exception:
                pass

    args = {
        "d": d, "t": t, "e": e, "g": g,
        "p": p if isinstance(p, list) else [p],
        **mp,
    }
    # meta/power: allow a JSON blob as 'd' when no CSV header expected
    if action in ("forest",) and d and not _looks_like_csv(d):
        try:
            payload = json.loads(d)
            for kk, vv in payload.items():
                args[kk] = vv
            args["d"] = _df_to_csv(payload)
        except Exception:
            pass

    try:
        _ensure_lifelines()
    except ImportError as ex:
        return _err(f"backend unavailable: {ex}")

    try:
        if action == "km":
            return _surv_km(args)
        elif action == "cox":
            return _surv_cox(args)
        elif action == "logrank":
            return _surv_logrank(args)
        elif action == "forest":
            return _meta_forest(args)
        elif action == "ttest":
            return _power_ttest(args)
        elif action == "anova":
            return _power_anova(args)
        elif action == "prop":
            return _power_prop(args)
        else:
            valid = ["km", "cox", "logrank", "forest", "ttest", "anova", "prop"]
            return _err(f"unknown action '{action}'. valid: {valid}")
    except Exception as ex:
        return _err(f"{type(ex).__name__}: {ex}")


def _looks_like_csv(s: str) -> bool:
    head = s.strip().splitlines()[0] if s.strip() else ""
    return "," in head and not head.strip().startswith("{")


def _df_to_csv(payload: dict) -> str:
    import csv as _csv
    buf = io.StringIO()
    cols = list(payload.keys())
    w = _csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    nrows = max((len(v) if isinstance(v, (list, tuple)) else 1) for v in payload.values())
    for i in range(nrows):
        row = {}
        for c in cols:
            v = payload[c]
            row[c] = v[i] if isinstance(v, (list, tuple)) and i < len(v) else v
        w.writerow(row)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════

MEDICAL_EXT_SCHEMA = {
    "name": "medical_ext",
    "description": (
        "Headless medical statistics: survival (km/cox/logrank), meta-analysis "
        "(forest), and power/sample-size (ttest/anova/prop). "
        "Survival: data as CSV in 'd' with duration 't' and event 'e' columns; "
        "predictors 'p' (list), optional group 'g'. "
        "Meta (forest): CSV in 'd' with effect 'yi' and variance 'vi' (or 'se'); "
        "optional 'label'. Returns fixed-effect + DerSimonian-Laird random-effects "
        "pooled estimates and heterogeneity (Q, I2, tau2). Uses statsmodels 0.15+ "
        "native combine_effects when available, else a pure-numpy DL fallback. "
        "Power: pass any three of nobs(n)/effect_size(d or f)/power(pw)/alpha(a); "
        "prop takes p1,p2 instead of effect size. "
        "Output is compact JSON with short keys (hr, c, s, z, p, l, u, n)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["km", "cox", "logrank", "forest", "ttest", "anova", "prop"],
                "description": "Analysis family + method.",
            },
            "d": {
                "type": "string",
                "description": "CSV data with header row (survival: t,e[,g,p]; meta: yi,vi[,label]).",
            },
            "t": {"type": "string", "description": "Duration/time column (survival)."},
            "e": {"type": "string", "description": "Event indicator column (survival, 1=event)."},
            "g": {"type": "string", "description": "Group column (survival, for stratified KM / log-rank)."},
            "p": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Predictor columns (cox).",
            },
            "n": {"type": "number", "description": "Sample size (power: nobs1 per group; anova: total nobs)."},
            "pw": {"type": "number", "description": "Target power (power)."},
            "a": {"type": "number", "description": "Alpha/significance level (default 0.05)."},
            "d_es": {"type": "number", "description": "Effect size Cohen's d (ttest). Separate from data param 'd'."},
            "f": {"type": "number", "description": "Effect size Cohen's f (anova)."},
            "ratio": {"type": "number", "description": "Group size ratio n2/n1 (power, default 1.0)."},
            "alt": {"type": "string", "enum": ["two-sided", "larger", "smaller"], "description": "Alternative (ttest)."},
            "k": {"type": "integer", "description": "Number of groups (anova, default 3)."},
            "p1": {"type": "number", "description": "Proportion group 1 (prop)."},
            "p2": {"type": "number", "description": "Proportion group 2 (prop)."},
        },
        "required": ["action"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

from tools.registry import registry  # noqa: E402

registry.register(
    name="medical_ext",
    toolset="medical",
    schema=MEDICAL_EXT_SCHEMA,
    handler=lambda args, **kw: medical_ext_run(
        action=args.get("action", ""),
        d=args.get("d", ""),
        t=args.get("t", ""),
        e=args.get("e", ""),
        g=args.get("g", ""),
        p=args.get("p"),
        n=args.get("n"),
        pw=args.get("pw"),
        a=args.get("a"),
        ratio=args.get("ratio"),
        alt=args.get("alt"),
        f=args.get("f"),
        k=args.get("k"),
        p1=args.get("p1"),
        p2=args.get("p2"),
        d_es=args.get("d_es"),
    ),
    check_fn=_check_backends,
    emoji="🧬",
)
