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
        except ImportError as e:
            _girth_import_error = str(e)
            raise


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
# Output helpers — girth MML has swapped naming in its results dict:
#   "Difficulty"   → person ability estimates (theta)
#   "Ability"      → item difficulty estimates (beta)
#   "Discrimination" → item discrimination (alpha)
# We remap to canonical IRT names for agent consumption.
# ═══════════════════════════════════════════════════════════════════


def _build_model_out(result: dict, data: np.ndarray, polytomous: bool = False) -> dict:
    """Build compact output from girth result dict."""
    n_items = data.shape[1]
    n_people = data.shape[0]
    out: Dict[str, Any] = {
        "n_items": n_items,
        "n_people": n_people,
    }

    # Item difficulty — 1D for dichotomous, 2D for polytomous.
    # Rasch/JML returns person/thetas under different keys and may carry
    # NO item-difficulty vector at all — report what exists, warn on gaps.
    item_diff = result.get("Ability")
    if item_diff is None:
        # JML variants stash difficulties under alternate keys
        for alt in ("Beta", "Difficulty", "item_difficulty", "difficulties"):
            if result.get(alt) is not None:
                item_diff = result.get(alt)
                break
    if item_diff is not None:
        try:
            arr = np.asarray(item_diff, dtype=float)
            if polytomous and arr.ndim == 2:
                out["d"] = [[_fmt(float(v)) for v in row] for row in arr]
            else:
                vals = arr.flatten()
                if len(vals) == n_items:
                    out["d"] = [_fmt(float(v)) for v in vals]
                elif len(vals) == n_people:
                    # Mislabeled person-ability vector — keep under a, not d
                    out["a"] = [_fmt(float(v)) for v in vals]
                    out["warn_d"] = "item difficulties absent; person abilities in a"
                else:
                    out["warn_d"] = f"difficulty length {len(vals)} matches neither n_items {n_items} nor n_people {n_people}"
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
            elif len(vals) == n_people:
                out["warn_disc"] = (
                    f"discrimination length {len(vals)} = n_people, not n_items {n_items}; "
                    "estimator returned person-length vector — omitted"
                )
            else:
                out["warn_disc"] = f"discrimination length {len(vals)} != n_items {n_items}; omitted"
        except (TypeError, ValueError):
            pass

    # Guessing (3PL only)
    guess = result.get("Guessing")
    if guess is not None:
        out["g"] = [_fmt(float(v)) for v in guess.flatten()]

    # Person ability — must be person-length; item-length vectors here are
    # mislabeled difficulties, not abilities.
    person_theta = result.get("Difficulty")
    if person_theta is not None:
        try:
            vals = np.asarray(person_theta, dtype=float).flatten()
            if len(vals) == n_people:
                out["a"] = [_fmt(float(v)) for v in vals]
            elif len(vals) == n_items and "d" not in out:
                out["d"] = [_fmt(float(v)) for v in vals]
            elif "a" not in out:
                out["a"] = [_fmt(float(v)) for v in vals[:n_people]] if len(vals) > n_people else [_fmt(float(v)) for v in vals]
                if len(vals) != n_people:
                    out["warn_a"] = f"ability length {len(vals)} != n_people {n_people}"
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
            result = _girth.onepl_mml(data)
        else:
            result = _girth.rasch_jml(data)
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
        result = _girth.twopl_mml(data)
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
        result = _girth.threepl_mml(data)
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
        result = _girth.grm_mml(data)
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
        result = _girth.pcm_mml(data)
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
        # girth ability functions expect data as (n_items, n_people)
        data_T = data.astype(float).T
        if method == "eap":
            try:
                ability = _girth.ability_eap(data_T, diff, disc)
            except TypeError:
                ability = _girth.ability_eap(data_T, diff, disc, guess)
        elif method == "map":
            try:
                ability = _girth.ability_map(data_T, diff, disc)
            except TypeError:
                ability = _girth.ability_map(data_T, diff, disc, guess)
        else:
            # MLE has no guessing parameter in girth — for 3PL scoring
            # (model=3pl + guess) fall back to EAP which honours it.
            if model == "3pl" and float(np.max(guess)) > 0:
                try:
                    ability = _girth.ability_eap(data_T, diff, disc, guess)
                    method = "eap"
                except TypeError:
                    ability = _girth.ability_mle(data_T, diff, disc)
            else:
                ability = _girth.ability_mle(data_T, diff, disc)
    except Exception as e:
        return _err(f"Ability scoring failed: {type(e).__name__}: {e}")

    return _ok({
        "method": method,
        "model": model,
        "n": len(ability),
        "a": [_fmt(float(v)) for v in ability],
    })


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
