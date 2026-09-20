"""Property checks locking the review fixes (pspp/irt/sem/medical_ext).

Pure-synthetic, no network. Run: python3 -m pytest tests/ -q
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import pspp_tool as p
import medical_ext_tool as m


def _parse(s):
    return json.loads(s)


def test_evalue_protective_no_crash():
    r = _parse(p.pspp_run(t="evalue", d="x,y\n1,1\n0,0\n", or_val=0.9))["k"]
    assert "E_value" in r, r
    r = _parse(p.pspp_run(t="evalue", d="x,y\n1,1\n0,0\n", rr_val=0.5))["k"]
    assert "E_value" in r, r
    r = _parse(p.pspp_run(t="evalue", d="x,y\n1,1\n0,0\n", or_val=2.5))["k"]
    assert r["E_value"] > 2.5, r


def test_factor_default_nonempty():
    d = "a,b,c\n1,2,3\n2,3,4\n3,4,5\n4,5,6\n5,6,7\n6,7,8\n"
    r = _parse(p.pspp_run(t="factor", d=d, v=["a", "b", "c"]))
    assert r["k"]["components"], r


def test_power_n_honoured():
    r1 = _parse(p.pspp_run(t="power", d="x\n1\n", calc="power",
                           power_test="ttest", n=50))
    r2 = _parse(p.pspp_run(t="power", d="x\n1\n", calc="power",
                           power_test="ttest", n=30))
    assert r1["k"]["n1"] == 50, r1
    assert r1["k"]["power"] > r2["k"]["power"], (r1, r2)
    # ANOVA power implemented (was 'not yet implemented')
    ra = _parse(p.pspp_run(t="power", d="x\n1\n", calc="power",
                           power_test="anova", n=30))
    assert "power" in ra["k"], ra


def test_kappa_string_categories():
    d = "r1,r2\na,a\na,b\nb,a\nb,b\na,a\n"
    r = _parse(p.pspp_run(t="kappa", d=d, a="r1", v="r2"))
    assert "kappa" in r["k"], r


def test_ttest_zero_variance_warn_not_fatal():
    d = "g,v\nA,5\nA,5\nB,5\nB,5\n"
    r = _parse(p.pspp_run(t="ttest", d=d, a="g", v="v", g=["A", "B"]))
    assert r["k"]["warn"], r
    assert r["k"]["m1"] == 5.0, r


def test_roc_threshold_consistent():
    d = "y,s\n1,0.9\n1,0.8\n1,0.7\n0,0.3\n0,0.2\n0,0.1\n"
    r = _parse(p.pspp_run(t="roc", d=d, a="y", v="s"))["k"]
    assert r["AUC"] == 1.0, r
    # threshold must separate: all positives above, all negatives below/equal
    assert 0.3 < r["threshold"] <= 0.7, r


def test_surv_logrank_separated():
    d = ("t,e,g\n1,1,A\n2,1,A\n3,0,A\n10,1,B\n11,1,B\n12,0,B\n20,1,A\n21,0,B\n")
    r = _parse(p.pspp_run(t="surv", d=d, o="e", v="t", a="g"))["k"]
    assert "logrank" in r, r
    assert r["logrank"]["p"] is not None, r


def test_medical_ext_km_float_times():
    csv = "T,E,g\n15.23,1,A\n15.91,1,A\n20.5,0,B\n22.1,1,B\n"
    r = _parse(m.medical_ext_run(action="km", d=csv, t="T", e="E", g="g"))
    pts = r["g"]["A"]["surv"]
    assert "15.23" in pts and "15.91" in pts, pts


def test_medical_ext_forest_label():
    csv = "yi,vi,label\n0.5,0.1,StudyA\n0.2,0.15,StudyB\n"
    r = _parse(m.medical_ext_run(action="forest", d=csv))
    assert r["forest"][0]["label"] == "StudyA", r["forest"]
