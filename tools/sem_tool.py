#!/usr/bin/env python3
"""
SEM — AI-agent-native tool wrapping semopy 2.3.11 for Structural Equation Modeling.

Actions (2):
  sem_fit     — Fit SEM model with lavaan-style syntax, return estimates + fit indices
  sem_inspect — Inspect fitted model parameters (raw or std_est mode)

Model description uses lavaan-like syntax:
  f1 =~ x1 + x2 + x3      # latent factor defined by indicators (=~)
  f2 ~ f1                  # structural regression
  x1 ~~ x1                 # variance (optional — auto-added)

Output is compact JSON with short keys.
"""

from __future__ import annotations

import io
import json
import math
import threading
from typing import Any, Dict, Optional

import numpy as np

# Thread-safe lazy loading
_semopy = None
_sem_lock = threading.Lock()
_sem_import_error: Optional[str] = None


def _check_sem() -> bool:
    global _sem_import_error
    try:
        import semopy  # noqa: F401
        return True
    except ImportError as e:
        _sem_import_error = str(e)
        return False


def _ensure_sem():
    global _semopy, _sem_import_error
    if _semopy is not None:
        return
    if _sem_import_error is not None:
        raise ImportError(_sem_import_error)
    with _sem_lock:
        if _semopy is not None:
            return
        if _sem_import_error is not None:
            raise ImportError(_sem_import_error)
        try:
            import semopy as _sem_mod
            _semopy = _sem_mod
        except ImportError as e:
            _sem_import_error = str(e)
            raise


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _ok(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _err(msg: str) -> str:
    return _ok({"e": msg})


def _fmt(v: float) -> Optional[float]:
    """Round float, guard NaN/inf."""
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


# ═══════════════════════════════════════════════════════════════════
# Model Cache (per-session, in-memory)
# ═══════════════════════════════════════════════════════════════════

_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 32


def _cache_key(desc: str, d: str) -> str:
    import hashlib
    h = hashlib.sha256(d.strip().encode("utf-8")).hexdigest()[:16]
    return desc.strip() + "\x00" + h


def _cache_put(key: str, model: Any) -> None:
    with _cache_lock:
        if key in _model_cache:
            _model_cache.pop(key)
        while len(_model_cache) >= _CACHE_MAX:
            _model_cache.pop(next(iter(_model_cache)))
        _model_cache[key] = model


# ═══════════════════════════════════════════════════════════════════
# Action Handlers
# ═══════════════════════════════════════════════════════════════════


def _action_fit(args: dict) -> str:
    """Fit SEM model and return parameter estimates + fit indices."""
    desc = args.get("desc", "")
    d = args.get("d", "")

    if not desc:
        return _err("sem_fit requires 'desc' (model description in lavaan-like syntax)")
    if not d:
        return _err("sem_fit requires 'd' (CSV data)")

    _ensure_sem()
    import pandas as pd

    data = pd.read_csv(io.StringIO(d.strip()))

    try:
        model = _semopy.Model(desc)
        model.fit(data)
    except Exception as e:
        return _err(f"SEM fit failed: {type(e).__name__}: {e}")

    # Cache the model for later inspection
    key = _cache_key(desc, d)
    _cache_put(key, model)

    # Parameter estimates
    insp = model.inspect()
    out: Dict[str, Any] = {"params": []}
    for _, row in insp.iterrows():
        entry: Dict[str, Optional[Any]] = {
            "lv": str(row["lval"]),
            "op": str(row["op"]),
            "rv": str(row["rval"]),
        }
        est = row.get("Estimate")
        if est is not None and not isinstance(est, str) and not (isinstance(est, float) and math.isnan(est)):
            try:
                entry["c"] = _fmt(float(est))
            except (ValueError, TypeError):
                pass
        se = row.get("Std. Err")
        if se is not None and not isinstance(se, str) and not (isinstance(se, float) and math.isnan(se)):
            try:
                entry["s"] = _fmt(float(se))
            except (ValueError, TypeError):
                pass
        zv = row.get("z-value")
        if zv is not None and not isinstance(zv, str) and not (isinstance(zv, float) and math.isnan(zv)):
            try:
                entry["z"] = _fmt(float(zv))
            except (ValueError, TypeError):
                pass
        pv = row.get("p-value")
        if pv is not None and not isinstance(pv, str) and not (isinstance(pv, float) and math.isnan(pv)):
            try:
                entry["p"] = _fmt(float(pv))
            except (ValueError, TypeError):
                pass
        out["params"].append(entry)

    # Fit statistics
    try:
        stats = _semopy.calc_stats(model)
        fit: Dict[str, Optional[float]] = {}
        for attr_name, key in [
            ("chi2", "chi2"),
            ("DoF", "df"),
            ("GFI", "gfi"),
            ("AGFI", "agfi"),
            ("CFI", "cfi"),
            ("NFI", "nfi"),
            ("TLI", "tli"),
            ("RMSEA", "rmsea"),
            ("AIC", "aic"),
            ("BIC", "bic"),
            ("LogLik", "ll"),
        ]:
            val = getattr(stats, attr_name, None)
            if val is not None:
                try:
                    # calc_stats returns DataFrame; attributes are Series
                    if hasattr(val, "iloc"):
                        fit[key] = _fmt(float(val.iloc[0]))
                    else:
                        fit[key] = _fmt(float(val))
                except (ValueError, TypeError, IndexError):
                    pass
        if fit:
            out["fit"] = fit
    except Exception:
        pass

    out["n"] = len(data)
    return _ok(out)


def _action_inspect(args: dict) -> str:
    """Inspect a cached fitted model."""
    desc = args.get("desc", "")
    d = args.get("d", "")
    mode = args.get("mode", "raw")

    if not desc:
        return _err("sem_inspect requires 'desc' (same model description as used in fit)")

    _ensure_sem()

    if d:
        key = _cache_key(desc, d)
        model = _model_cache.get(key)
    else:
        # Back-compat: no data hash — match the most recent fit for this desc
        model = None
        with _cache_lock:
            for k in reversed(list(_model_cache.keys())):
                if k.split("\x00", 1)[0] == desc.strip():
                    model = _model_cache[k]
                    break
    if model is None:
        return _err(
            f"No fitted model cached for this description. Run sem_fit first "
            f"with the same 'desc' in this session."
        )

    try:
        # semopy.inspect(std_est=True) for standardised, default for raw
        if mode == "std_est":
            insp = model.inspect(std_est=True)
        else:
            insp = model.inspect()
    except Exception as e:
        return _err(f"SEM inspect failed: {type(e).__name__}: {e}")

    if insp is None:
        return _err(f"inspect(mode='{mode}') returned None — mode may not be available")

    out: Dict[str, Any] = {"mode": mode, "params": []}
    for _, row in insp.iterrows():
        entry: Dict[str, Optional[Any]] = {
            "lv": str(row["lval"]),
            "op": str(row["op"]),
            "rv": str(row["rval"]),
        }
        est = row.get("Estimate")
        if est is not None and not isinstance(est, str) and not (isinstance(est, float) and math.isnan(est)):
            try:
                entry["c"] = _fmt(float(est))
            except (ValueError, TypeError):
                pass
        se = row.get("Std. Err")
        if se is not None and not isinstance(se, str) and not (isinstance(se, float) and math.isnan(se)):
            try:
                entry["s"] = _fmt(float(se))
            except (ValueError, TypeError):
                pass
        zv = row.get("z-value")
        if zv is not None and not isinstance(zv, str) and not (isinstance(zv, float) and math.isnan(zv)):
            try:
                entry["z"] = _fmt(float(zv))
            except (ValueError, TypeError):
                pass
        pv = row.get("p-value")
        if pv is not None and not isinstance(pv, str) and not (isinstance(pv, float) and math.isnan(pv)):
            try:
                entry["p"] = _fmt(float(pv))
            except (ValueError, TypeError):
                pass
        out["params"].append(entry)

    return _ok(out)


# ═══════════════════════════════════════════════════════════════════
# Main Dispatcher
# ═══════════════════════════════════════════════════════════════════


def sem_run(
    action: str = "",
    d: str = "",
    desc: str = "",
    mode: str = "raw",
) -> str:
    """Dispatch to SEM action handler.

    Args:
        action: sem_fit, sem_inspect
        d: CSV data string (for fit)
        desc: Model description in lavaan-like syntax
        mode: Inspection mode — 'raw' or 'std_est' (standardised)
    """
    args = {"action": action, "d": d, "desc": desc, "mode": mode}

    try:
        _ensure_sem()
    except ImportError as e:
        return _err(f"semopy not available: {e}")

    try:
        if action == "sem_fit":
            return _action_fit(args)
        elif action == "sem_inspect":
            return _action_inspect(args)
        else:
            return _err(f"unknown action '{action}'. valid: ['sem_fit', 'sem_inspect']")
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════

SEM_SCHEMA = {
    "name": "sem",
    "description": (
        "SEM via semopy. Fit lavaan-style models and inspect results. "
        "Syntax: f1 =~ x1+x2 (latent by indicators), f2 ~ f1 (regression), "
        "x1 ~~ x1 (variance). Actions: sem_fit (fit + fit indices), "
        "sem_inspect (raw or std_est params). "
        "Data as CSV string in 'd'. Model syntax in 'desc'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sem_fit", "sem_inspect"],
                "description": "sem_fit: fit model + return estimates + fit indices. sem_inspect: inspect cached model.",
            },
            "d": {
                "type": "string",
                "description": "CSV data with header row (for sem_fit).",
            },
            "desc": {
                "type": "string",
                "description": "Model description in lavaan-like syntax, e.g. 'f1 =~ x1+x2+x3\\nf2 =~ y1+y2+y3\\nf2 ~ f1'",
            },
            "mode": {
                "type": "string",
                "enum": ["raw", "std_est"],
                "description": "Inspection mode for sem_inspect: raw (unstandardised) or std_est (standardised).",
            },
        },
        "required": ["action", "desc"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

from tools.registry import registry

registry.register(
    name="sem",
    toolset="medical",
    schema=SEM_SCHEMA,
    handler=lambda args, **kw: sem_run(
        action=args.get("action", ""),
        d=args.get("d", ""),
        desc=args.get("desc", ""),
        mode=args.get("mode", "raw"),
    ),
    check_fn=_check_sem,
    emoji="🧩",
)
