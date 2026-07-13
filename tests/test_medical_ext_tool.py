#!/usr/bin/env python3
"""Tests for the medical_ext headless tool (survival / meta / power).

Pure unit tests: synthetic data, real backends, no network.
Mirrors the Hermes skill-authoring standard (stdlib + pytest only).
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the tool module directly (it self-registers with the hermes registry,
# but we exercise the handler in-process to avoid importing the whole agent).
_HERE = Path(__file__).resolve().parent
_TOOLS = _HERE.parent / "tools"
sys.path.insert(0, str(_TOOLS))
mod = importlib.import_module("medical_ext_tool")
H = lambda args: json.loads(mod.medical_ext_run(**args))


@pytest.fixture(scope="module")
def surv_csv():
    np.random.seed(1)
    n = 200
    T = np.round(np.random.exponential(scale=15, size=n), 2)
    E = (np.random.rand(n) < 0.75).astype(int)
    grp = np.random.randint(0, 2, size=n)
    age = np.round(np.random.normal(60, 10, n), 1)
    csv = "T,E,grp,age\n" + "\n".join(
        f"{T[i]},{E[i]},{grp[i]},{age[i]}" for i in range(n)
    )
    csv_risk = "T,E,age\n" + "\n".join(f"{T[i]},{E[i]},{age[i]}" for i in range(n))
    return csv, csv_risk


def test_registry_registration():
    import tools.registry as reg
    importlib.import_module("medical_ext_tool")
    assert reg.registry.get_entry("medical_ext") is not None
    assert reg.registry.get_entry("medical_ext").toolset == "medical"


def test_km_stratified(surv_csv):
    csv, _ = surv_csv
    out = H({"action": "km", "d": csv, "t": "T", "e": "E", "g": "grp"})
    assert "g" in out and out["g"], "stratified KM must return per-group curves"
    assert "logrank" in out and "p" in out["logrank"]


def test_km_whole_cohort(surv_csv):
    _, csv_risk = surv_csv
    out = H({"action": "km", "d": csv_risk, "t": "T", "e": "E"})
    assert "median" in out and "surv" in out
    assert "at_risk" in out  # must be JSON-serialisable int, not numpy int64


def test_cox(surv_csv):
    _, csv_risk = surv_csv
    out = H({"action": "cox", "d": csv_risk, "t": "T", "e": "E", "p": ["age"]})
    assert "hr" in out["f"]["age"]
    assert "p" in out["f"]["age"]
    assert "cindex" in out


def test_logrank(surv_csv):
    csv, _ = surv_csv
    out = H({"action": "logrank", "d": csv, "t": "T", "e": "E", "g": "grp"})
    assert "z" in out and "p" in out


def test_meta_forest():
    csv = ("yi,vi,label\n0.5,0.1,StudyA\n0.2,0.15,StudyB\n"
           "0.8,0.12,StudyC\n0.35,0.08,StudyD\n")
    out = H({"action": "forest", "d": csv})
    assert out["k"] == 4
    assert len(out["forest"]) == 4
    # weights (fixed-effect) must sum to ~1
    wsum = sum(r["w_fe"] for r in out["forest"])
    assert abs(wsum - 1.0) < 1e-6
    assert "Q" in out["hetero"] and "I2" in out["hetero"]
    assert "fe" in out and "re" in out
    # tau2 is the raw DerSimonian-Laird estimate (may be slightly negative when Q<df)
    assert isinstance(out["hetero"]["tau2"], float)


def test_meta_forest_numpy_fallback():
    """Force the pure-numpy DL fallback (simulate statsmodels < 0.15)."""
    import medical_ext_tool as m
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.endswith("meta_analysis"):
            raise ImportError("simulated <0.15")
        return real_import(name, *a, **k)

    csv = ("yi,vi,label\n0.5,0.1,StudyA\n0.2,0.15,StudyB\n"
           "0.8,0.12,StudyC\n0.35,0.08,StudyD\n")
    builtins.__import__ = fake_import
    try:
        out = H({"action": "forest", "d": csv})
    finally:
        builtins.__import__ = real_import
    assert out["k"] == 4
    assert isinstance(out["hetero"]["tau2"], float)
    assert abs(out["fe"]["est"] - 0.463333) < 1e-4


def test_meta_forest_paths_agree():
    """statsmodels native path and numpy fallback must give identical FE/RE."""
    import medical_ext_tool as m
    csv = ("yi,vi,label\n1.2,0.3,A\n0.9,0.25,B\n1.5,0.4,C\n0.7,0.2,D\n1.0,0.35,E\n")
    out_native = H({"action": "forest", "d": csv})
    # Fallback via hiding meta_analysis import
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.endswith("meta_analysis"):
            raise ImportError("simulated <0.15")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        out_fb = H({"action": "forest", "d": csv})
    finally:
        builtins.__import__ = real_import
    assert abs(out_native["fe"]["est"] - out_fb["fe"]["est"]) < 1e-6
    assert abs(out_native["re"]["est"] - out_fb["re"]["est"]) < 1e-6
    assert abs(out_native["hetero"]["Q"] - out_fb["hetero"]["Q"]) < 1e-6



def test_power_ttest_solve_n():
    out = H({"action": "ttest", "d_es": 0.5, "n": 0, "pw": 0.8,
             "a": 0.05, "ratio": 1.0, "alt": "two-sided"})
    assert 63 < out["nobs1"] < 65


def test_power_ttest_reverse_solve_d():
    out = H({"action": "ttest", "n": 64, "d_es": 0, "pw": 0.8,
             "a": 0.05, "alt": "two-sided"})
    assert abs(out["effect_size"] - 0.5) < 0.02


def test_power_anova_solve_n():
    out = H({"action": "anova", "n": 0, "f": 0.3, "pw": 0.8, "a": 0.05, "k": 3})
    assert out["nobs"] > 0


def test_power_anova_round_trip():
    fwd = H({"action": "anova", "n": 0, "f": 0.3, "pw": 0.8, "a": 0.05, "k": 3})
    back = H({"action": "anova", "n": fwd["nobs"], "f": 0, "pw": 0.8, "a": 0.05, "k": 3})
    assert abs(back["effect_size"] - 0.3) < 0.02


def test_power_prop_solve_n():
    out = H({"action": "prop", "p1": 0.4, "p2": 0.5, "n": 0, "pw": 0.8, "a": 0.05})
    assert out["nobs1"] > 0
    assert "effect_size" in out


def test_error_unknown_action():
    out = H({"action": "bogus"})
    assert "e" in out


def test_error_cox_no_predictors(surv_csv):
    _, csv_risk = surv_csv
    out = H({"action": "cox", "d": csv_risk, "t": "T", "e": "E", "p": []})
    assert "e" in out


def test_error_km_missing_column(surv_csv):
    csv, _ = surv_csv
    out = H({"action": "km", "d": csv, "t": "X", "e": "E"})
    assert "e" in out


def test_error_prop_bad_proportions():
    out = H({"action": "prop", "p1": 1.5, "p2": 0.5, "n": 0, "pw": 0.8})
    assert "e" in out
