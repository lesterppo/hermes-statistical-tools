#!/usr/bin/env python3
"""
IRT — AI-agent-native tool wrapping girth 0.8.0 for Item Response Theory.

Actions (7):
  irt_rasch  — Rasch/1PL model (JML or MML)
  irt_2pl    — 2-parameter logistic model (MML)
  irt_3pl    — 3-parameter logistic model (MML)
  irt_grm    — Graded Response Model for polytomous items (MML)
  irt_pcm    — Partial Credit Model (MML)
  irt_score  — Score respondents given item parameters
  irt_ctt    — Classical Test Theory statistics

Data comes as CSV string. For binary items, values should be 0/1 or 1/2.
For polytomous items, values should be 0,1,2,... (ordered categories).
"""

from __future__ import annotations

import io
import json
import math
import threading
from typing import Any, Dict, List, Optional

import numpy as np

# Thread-safe lazy loading
_girth = None
_girth_lock = threading.Lock()
_girth_import_error: Optional[str] = None


def _check_irt() -> bool:
    global _girth_import_error
    try:
        import girth  # noqa: F401
        return True
    except ImportError as e:
        _girth_import_error = str(e)
        return False


def _ensure_irt():
    global _girth, _girth_import_error
    if _girth is not None:
        return
    if _girth_import_error is not None:
        raise ImportError(_girth_import_error)
    with _girth_lock:
        if _girth is not None:
            return
        if _girth_import_error is not None:
            raise ImportError(_girth_import_error)
        try:
            import girth as _girth_mod
            _girth = _girth_mod
            _apply_scipy_compat_shim()
        except ImportError as e:
            _girth_import_error = str(e)
            raise


def _apply_scipy_compat_shim():
    """Restore scalar-return semantics for girth's internal fminbound calls.

    girth 0.8.0's threepl_mml path funnels through _mml_abstract, whose
    objective returns a size-1 array. scipy <1.14 tolerated that inside
    fminbound; scipy ≥1.14 returns an array, which then fails on assignment
    ("setting an array element with a sequence") — making threepl_mml fail
    on ALL data. Wrapping the module-level fminbound reference to squeeze
    the objective output restores the old behaviour. Defensive: any failure
    here just leaves girth unpatched (3PL keeps its graceful error path).
    """
    global _girth
    try:
        import sys
        from scipy import optimize as _opt
        _orig = _opt.fminbound

        def _scalar_fminbound(func, *a, **k):
            def _wrapped(x):
                try:
                    return float(np.asarray(func(x)).squeeze())
                except (TypeError, ValueError):
                    return func(x)
            r = _orig(_wrapped, *a, **k)
            try:
                return float(np.asarray(r).squeeze())
            except (TypeError, ValueError):
                return r

        for _mod in (
            "girth.unidimensional.dichotomous.rasch_mml",
            "girth.unidimensional.dichotomous.threepl_mml",
        ):
            _m = sys.modules.get(_mod)
            if _m is not None and getattr(_m, "fminbound", None) is _orig:
                setattr(_m, "fminbound", _scalar_fminbound)
    except Exception:
        pass


def _to_girth(data: np.ndarray) -> np.ndarray:
    """Transpose (n_people, n_items) → girth's (n_items, n_people) layout.

    EVERY girth estimator and ability function expects items × people;
    passing people × items silently swaps items for respondents (item
    estimates come back person-length). Centralise the transpose here so
    no call site can forget it.
    """
    return np.ascontiguousarray(data.T)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _ok(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _err(msg: str) -> str:
    return _ok({"e": msg})


def _fmt(v: float) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        f = float(v)
        if 0 < abs(f) < 1e-8:
            return f
        return round(f, 6)
    return v


def _parse_matrix(csv_data: str):
    """Parse CSV string into numpy 2D float array. NaN-safe for missing data."""
    import pandas as pd
    df = pd.read_csv(io.StringIO(csv_data.strip()))
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return arr


def _validate_binary(data: np.ndarray) -> np.ndarray:
    """Ensure binary data is 0/1. If 1/2, convert to 0/1."""
    # Mask NaN values for comparison
    finite = data[~np.isnan(data)]
    if len(finite) == 0:
        return data
    unique = np.unique(finite)
    if len(unique) <= 2:
        if min(unique) == 1 and max(unique) == 2:
            data = np.where(~np.isnan(data), data - 1, data)
    return data


def _prepare_for_girth(data: np.ndarray) -> np.ndarray:
    """Drop rows/columns with any NaN values (listwise deletion for IRT).
    Convert to int for girth compatibility."""
    # Drop rows with NaN
    valid_rows = ~np.isnan(data).any(axis=1)
    clean = data[valid_rows]
    if len(clean) == 0:
        raise ValueError("no complete cases after dropping missing values")
    return clean.astype(int)


# ═══════════════════════════════════════════════════════════════════
# Output helpers — girth's canonical result-dict naming (verified against
# girth 0.8.0 with correctly-oriented items × people input):
#   "Difficulty"     → item difficulty estimates (n_items; 2D thresholds
#                       for GRM/PCM: n_items × n_thresholds)
#   "Discrimination" → item discrimination estimates (n_items)
#   "Guessing"       → item guessing estimates, 3PL only (n_items)
#   "Ability"        → person ability estimates (n_people)
# Lengths are validated against the data shape; anything person-length is
# never reported as an item parameter (and vice versa).
# ═══════════════════════════════════════════════════════════════════


def _build_model_out(result: dict, data: np.ndarray, polytomous: bool = False) -> dict:
    """Build compact output from girth result dict."""
    n_items = data.shape[1]
    n_people = data.shape[0]
    out: Dict[str, Any] = {
        "n_items": n_items,
        "n_people": n_people,
    }

    # Item difficulty — 1D for dichotomous, 2D (thresholds) for polytomous.
    item_diff = result.get("Difficulty")
    if item_diff is None:
        for alt in ("Beta", "item_difficulty", "difficulties"):
            if result.get(alt) is not None:
                item_diff = result.get(alt)
                break
    if item_diff is not None:
        try:
            arr = np.asarray(item_diff, dtype=float)
            if polytomous and arr.ndim == 2 and arr.shape[0] == n_items:
                out["d"] = [[_fmt(float(v)) for v in row] for row in arr]
            else:
                vals = arr.flatten()
                if len(vals) == n_items:
                    out["d"] = [_fmt(float(v)) for v in vals]
                else:
                    out["warn_d"] = (
                        f"difficulty length {len(vals)} != n_items {n_items}; omitted"
                    )
        except (TypeError, ValueError):
            pass
    else:
        out["warn_d"] = "no item-difficulty vector in estimator output (rasch-jml)"

    item_disc = result.get("Discrimination")
    if item_disc is not None:
        try:
            vals = np.asarray(item_disc, dtype=float).flatten()
            if len(vals) == n_items:
                out["disc"] = [_fmt(float(v)) for v in vals]
            elif len(vals) == 1:
                # Scalar constraint (e.g. onepl_mml fixed discrimination)
                out["disc"] = [_fmt(float(vals[0]))] * n_items
            else:
                out["warn_disc"] = (
                    f"discrimination length {len(vals)} != n_items {n_items}; omitted"
                )
        except (TypeError, ValueError):
            pass

    # Guessing (3PL only)
    guess = result.get("Guessing")
    if guess is not None:
        try:
            gvals = np.asarray(guess, dtype=float).flatten()
            if len(gvals) == n_items:
                out["g"] = [_fmt(float(v)) for v in gvals]
        except (TypeError, ValueError):
            pass

    # Person ability — must be person-length.
    person_theta = result.get("Ability")
    if person_theta is not None:
        try:
            vals = np.asarray(person_theta, dtype=float).flatten()
            if len(vals) == n_people:
                out["a"] = [_fmt(float(v)) for v in vals]
            else:
                out["warn_a"] = (
                    f"ability length {len(vals)} != n_people {n_people}; omitted"
                )
        except (TypeError, ValueError):
            pass

    # Fit stats
    for key in ("AIC", "BIC"):
        val = result.get(key)
        if val is not None and isinstance(val, dict):
            out[key.lower()] = {k: _fmt(float(v)) for k, v in val.items()}

    return out


# ═══════════════════════════════════════════════════════════════════
# Action Handlers
# ═══════════════════════════════════════════════════════════════════


def _action_rasch(args: dict) -> str:
    """Rasch / 1PL model."""
    method = args.get("method", "jml")
    d = args.get("d", "")

    if not d:
        return _err("irt_rasch requires 'd' (CSV data)")

    _ensure_irt()
    data = _validate_binary(_parse_matrix(d))
    data = _prepare_for_girth(data)

    try:
        if method == "mml":
            result = _girth.onepl_mml(_to_girth(data))
        else:
            result = _girth.rasch_jml(_to_girth(data))
    except Exception as e:
        return _err(f"Rasch {method.upper()} failed: {type(e).__name__}: {e}")

    out = _build_model_out(result, data)
    out["method"] = method
    return _ok(out)


def _action_2pl(args: dict) -> str:
    """2PL model (MML)."""
    d = args.get("d", "")

    if not d:
        return _err("irt_2pl requires 'd' (CSV data)")

    _ensure_irt()
    data = _validate_binary(_parse_matrix(d))
    data = _prepare_for_girth(data)

    try:
        result = _girth.twopl_mml(_to_girth(data))
    except Exception as e:
        return _err(f"2PL MML failed: {type(e).__name__}: {e}")

    out = _build_model_out(result, data)
    return _ok(out)


def _action_3pl(args: dict) -> str:
    """3PL model (MML). Note: experimental — may fail on small datasets."""
    d = args.get("d", "")

    if not d:
        return _err("irt_3pl requires 'd' (CSV data)")

    _ensure_irt()
    data = _validate_binary(_parse_matrix(d))
    data = _prepare_for_girth(data)

    try:
        result = _girth.threepl_mml(_to_girth(data))
    except Exception as e:
        return _err(
            f"3PL MML failed: {type(e).__name__}: {e}. "
            "3PL estimation is numerically sensitive — try with more items "
            "(>= 10) and more respondents (>= 200), or use irt_2pl instead."
        )

    out = _build_model_out(result, data)
    return _ok(out)


def _action_grm(args: dict) -> str:
    """Graded Response Model for polytomous items (MML)."""
    d = args.get("d", "")

    if not d:
        return _err("irt_grm requires 'd' (CSV data)")

    _ensure_irt()
    data = _parse_matrix(d)
    # Listwise deletion identical to the binary path — .astype(int) on NaN
    # raises instead of dropping.
    if np.isnan(data).any():
        valid = ~np.isnan(data).any(axis=1)
        data = data[valid]
        if len(data) == 0:
            return _err("no complete cases after dropping missing values")
    data = data.astype(int)

    try:
        result = _girth.grm_mml(_to_girth(data))
    except Exception as e:
        return _err(f"GRM MML failed: {type(e).__name__}: {e}")

    out = _build_model_out(result, data, polytomous=True)
    return _ok(out)


def _action_pcm(args: dict) -> str:
    """Partial Credit Model (MML)."""
    d = args.get("d", "")

    if not d:
        return _err("irt_pcm requires 'd' (CSV data)")

    _ensure_irt()
    data = _parse_matrix(d)
    if np.isnan(data).any():
        valid = ~np.isnan(data).any(axis=1)
        data = data[valid]
        if len(data) == 0:
            return _err("no complete cases after dropping missing values")
    data = data.astype(int)

    try:
        result = _girth.pcm_mml(_to_girth(data))
    except Exception as e:
        return _err(f"PCM MML failed: {type(e).__name__}: {e}")

    out = _build_model_out(result, data, polytomous=True)
    return _ok(out)


def _action_score(args: dict) -> str:
    """Score respondent ability given item parameters."""
    d = args.get("d", "")
    diff_str = args.get("diff", "")
    disc_str = args.get("disc", "")
    guess_str = args.get("guess", "")
    method = args.get("method", "mle")
    model = args.get("model", "2pl")
    if method not in ("mle", "eap", "map"):
        # "jml"/"mml" are estimation methods (rasch), not scoring methods
        method = "mle"

    if not d:
        return _err("irt_score requires 'd' (CSV response data)")
    if not diff_str:
        return _err("irt_score requires 'diff' (item difficulties, comma-separated)")

    _ensure_irt()
    data = _validate_binary(_parse_matrix(d))
    data = _prepare_for_girth(data)

    # Parse item parameters from comma-separated strings
    try:
        diff = np.array([float(x.strip()) for x in diff_str.split(",")])
    except ValueError:
        return _err("diff must be comma-separated numbers, e.g. '-1.5,0.3,1.2'")

    disc = np.ones_like(diff)
    if disc_str:
        try:
            disc = np.array([float(x.strip()) for x in disc_str.split(",")])
        except ValueError:
            return _err("disc must be comma-separated numbers")

    guess = np.zeros_like(diff)
    if guess_str:
        try:
            guess = np.array([float(x.strip()) for x in guess_str.split(",")])
        except ValueError:
            return _err("guess must be comma-separated numbers")

    if len(diff) != data.shape[1]:
        return _err(
            f"diff has {len(diff)} values but data has {data.shape[1]} items"
        )
    if len(disc) != data.shape[1]:
        return _err(
            f"disc has {len(disc)} values but data has {data.shape[1]} items"
        )
    if len(guess) != data.shape[1]:
        return _err(
            f"guess has {len(guess)} values but data has {data.shape[1]} items"
        )

    try:
        # girth ability functions expect data as (n_items, n_people).
        data_T = _to_girth(data.astype(float))
        use_3pl = (model == "3pl" and float(np.max(guess)) > 0)
        if use_3pl:
            # girth 0.8.0's ability_eap/mle/map take NO guessing parameter
            # (4th positional is an options dict) — so 3PL scoring is done
            # here directly: per-respondent bounded MLE under the 3PL ICC
            # P = g + (1-g) * logit(a*(theta-b)), with SE from observed info.
            ability, se = _score_3pl_mle(data, diff, disc, guess)
            method = "mle-3pl"
        elif method == "eap":
            ability = _girth.ability_eap(data_T, diff, disc)
            se = None
        elif method == "map":
            ability = _girth.ability_map(data_T, diff, disc)
            se = None
        else:
            ability = _girth.ability_mle(data_T, diff, disc)
            se = None
    except Exception as e:
        return _err(f"Ability scoring failed: {type(e).__name__}: {e}")

    out = {
        "method": method,
        "model": model,
        "n": len(ability),
        "a": [_fmt(float(v)) for v in ability],
    }
    if se is not None:
        out["se"] = [_fmt(float(v)) for v in se]
    return _ok(out)


def _score_3pl_mle(data: np.ndarray, diff: np.ndarray, disc: np.ndarray,
                   guess: np.ndarray):
    """Per-respondent MLE of theta under the 3PL model.

    Maximises sum_j [y*log P + (1-y)*log(1-P)] over theta in [-6, 6] with
    P_j = g_j + (1-g_j)/(1+exp(-a_j (theta-b_j))). Perfect/zero scores are
    clipped to the boundary (their MLE is +/-infinity). Returns
    (abilities, standard_errors); SE from observed information, None-safe.
    """
    from scipy.optimize import minimize_scalar
    n_people = data.shape[0]
    a = np.asarray(disc, dtype=float)
    b = np.asarray(diff, dtype=float)
    g = np.clip(np.asarray(guess, dtype=float), 0.0, 0.5)
    theta_hat = np.zeros(n_people)
    se_hat = np.full(n_people, np.nan)
    for i in range(n_people):
        y = data[i].astype(float)

        def _nll(th):
            z = np.clip(a * (th - b), -30, 30)
            p = g + (1 - g) / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        if np.all(y == 1):
            theta_hat[i] = 6.0
            continue
        if np.all(y == 0):
            theta_hat[i] = -6.0
            continue
        try:
            res = minimize_scalar(_nll, bounds=(-6, 6), method="bounded",
                                  options={"xatol": 1e-4})
            th = float(res.x)
        except (ValueError, TypeError):
            th = 0.0
        theta_hat[i] = th
        # Observed information at the MLE for SE
        h = 1e-4
        try:
            info = (_nll(th + h) - 2 * _nll(th) + _nll(th - h)) / (h * h)
            if info > 0:
                se_hat[i] = float(1 / math.sqrt(info))
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return theta_hat, se_hat


def _action_ctt(args: dict) -> str:
    """Classical Test Theory statistics."""
    d = args.get("d", "")

    if not d:
        return _err("irt_ctt requires 'd' (CSV data)")

    _ensure_irt()
    data = _validate_binary(_parse_matrix(d))
    data = _prepare_for_girth(data)
    n_items = data.shape[1]
    n_people = data.shape[0]

    # Compute CTT manually (girth's ctt_statistics module has no public functions)
    total_scores = data.sum(axis=1)

    out: Dict[str, Any] = {
        "n_items": n_items,
        "n_people": n_people,
        "items": [],
    }

    item_diffs = []
    item_discs = []

    for j in range(n_items):
        item_scores = data[:, j]
        p_value = float(np.mean(item_scores))
        # Point-biserial correlation (item discrimination)
        item_total = np.column_stack([item_scores, total_scores])
        # Remove the item from total for corrected correlation
        corrected_total = total_scores - item_scores
        if np.std(corrected_total) > 0 and np.std(item_scores) > 0:
            r_pb = float(np.corrcoef(item_scores, corrected_total)[0, 1])
        else:
            r_pb = 0.0

        entry: Dict[str, Optional[float]] = {
            "p": _fmt(p_value),
            "r": _fmt(r_pb),
        }
        out["items"].append(entry)
        item_diffs.append(p_value)
        item_discs.append(r_pb)

    # Cronbach's alpha
    item_vars = data.var(axis=0, ddof=1)
    total_var = total_scores.var(ddof=1)
    if total_var > 0 and n_items > 1:
        alpha = (n_items / (n_items - 1)) * (1 - sum(item_vars) / total_var)
        out["alpha"] = _fmt(float(alpha))

    # Mean and SD of total scores
    out["mean"] = _fmt(float(np.mean(total_scores)))
    out["sd"] = _fmt(float(np.std(total_scores, ddof=1)))

    return _ok(out)


# ═══════════════════════════════════════════════════════════════════
# Main Dispatcher
# ═══════════════════════════════════════════════════════════════════


def irt_run(
    action: str = "",
    d: str = "",
    method: str = "jml",
    diff: str = "",
    disc: str = "",
    guess: str = "",
    model: str = "2pl",
) -> str:
    """Dispatch to IRT action handler.

    Args:
        action: irt_rasch, irt_2pl, irt_3pl, irt_grm, irt_pcm, irt_score, irt_ctt
        d: CSV data string (item responses: rows=people, cols=items)
        method: Estimation method — 'jml' or 'mml' (rasch only), 'mle'/'eap'/'map' (score)
        diff: Comma-separated item difficulties (irt_score)
        disc: Comma-separated item discriminations (irt_score)
        guess: Comma-separated guessing parameters (irt_score)
        model: Scoring model — '2pl' or '3pl' (irt_score)
    """
    args = {
        "action": action,
        "d": d,
        "method": method,
        "diff": diff,
        "disc": disc,
        "guess": guess,
        "model": model,
    }

    try:
        _ensure_irt()
    except ImportError as e:
        return _err(f"girth not available: {e}")

    try:
        if action == "irt_rasch":
            return _action_rasch(args)
        elif action == "irt_2pl":
            return _action_2pl(args)
        elif action == "irt_3pl":
            return _action_3pl(args)
        elif action == "irt_grm":
            return _action_grm(args)
        elif action == "irt_pcm":
            return _action_pcm(args)
        elif action == "irt_score":
            return _action_score(args)
        elif action == "irt_ctt":
            return _action_ctt(args)
        else:
            valid = [
                "irt_rasch", "irt_2pl", "irt_3pl", "irt_grm",
                "irt_pcm", "irt_score", "irt_ctt",
            ]
            return _err(f"unknown action '{action}'. valid: {valid}")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════

IRT_SCHEMA = {
    "name": "irt",
    "description": (
        "IRT via girth. Fit Rasch/1PL/2PL/3PL models or GRM/PCM for polytomous. "
        "Score respondents with known item params. CTT statistics. "
        "Data as CSV in 'd' (rows=people, cols=items). Binary items: 0/1. "
        "Polytomous items: 0,1,2,... ordered categories. "
        "Rasch method: jml (default) or mml. "
        "Score method: mle (default), eap, or map."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "irt_rasch", "irt_2pl", "irt_3pl",
                    "irt_grm", "irt_pcm", "irt_score", "irt_ctt",
                ],
                "description": "Analysis: Rasch/1PL, 2PL, 3PL, GRM, PCM, ability scoring, or CTT.",
            },
            "d": {
                "type": "string",
                "description": "CSV data with header row. Rows=respondents, columns=items. Binary: 0/1. Polytomous: 0,1,2,...",
            },
            "method": {
                "type": "string",
                "enum": ["jml", "mml", "mle", "eap", "map"],
                "description": "Estimation method. For irt_rasch: 'jml' or 'mml'. For irt_score: 'mle', 'eap', or 'map'.",
            },
            "diff": {
                "type": "string",
                "description": "For irt_score: comma-separated item difficulties, e.g. '-1.5,0.3,1.2'",
            },
            "disc": {
                "type": "string",
                "description": "For irt_score: comma-separated item discriminations, e.g. '1.0,1.2,0.8'",
            },
            "guess": {
                "type": "string",
                "description": "For irt_score: comma-separated guessing parameters (3PL only), e.g. '0.2,0.15,0.25'",
            },
            "model": {
                "type": "string",
                "enum": ["2pl", "3pl"],
                "description": "For irt_score: scoring model type (default: 2pl).",
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
    name="irt",
    toolset="medical",
    schema=IRT_SCHEMA,
    handler=lambda args, **kw: irt_run(
        action=args.get("action", ""),
        d=args.get("d", ""),
        method=args.get("method", "jml"),
        diff=args.get("diff", ""),
        disc=args.get("disc", ""),
        guess=args.get("guess", ""),
        model=args.get("model", "2pl"),
    ),
    check_fn=_check_irt,
    emoji="📏",
)
