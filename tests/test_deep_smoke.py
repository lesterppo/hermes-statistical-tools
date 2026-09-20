"""Deep live-smoke regression tests (added 2026-09-20).

Locks in fixes found by the full 45-action + 4-tool live sweep:
- IRT data orientation (girth wants items x people), canonical output mapping,
  scipy-compat shim for threepl_mml, manual 3PL ability scoring.
- statsmodels _fmt_result safe attribute access (NominalGEE llf/aic raise
  NotImplementedError; hasattr() does not swallow it).
- medical_ext power-ttest alt default.
- pspp: signed effect sizes, partial df, Games-Howell q + no double
  Bonferroni, logistic separation warning, beta/omega/pscore notes,
  inf-safe _rnd (strict JSON).
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import numpy as np
import pytest
from scipy import stats as S

from pspp_tool import pspp_run  # noqa: E402
from statsmodels_tool import statsmodels_run  # noqa: E402
from irt_tool import irt_run  # noqa: E402
from medical_ext_tool import medical_ext_run  # noqa: E402
from sem_tool import sem_run  # noqa: E402


def J(out):
    return json.loads(out, parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError(f"non-strict JSON constant: {x}")))


@pytest.fixture()
def rng_data():
    rng = np.random.RandomState(0)
    n = 60
    g = np.array(["A"] * 30 + ["B"] * 30)
    x1 = rng.normal(10, 2, n)
    x2 = rng.normal(5, 1.5, n)
    y = 2 * x1 + rng.normal(0, 2, n)
    return rng, n, g, x1, x2, y


def _csv(cols):
    m = len(next(iter(cols.values())))
    return "\n".join([",".join(cols.keys())] +
                     [",".join(str(cols[k][i]) for k in cols) for i in range(m)])


# ── pspp: all 45 actions return strict-JSON payloads ──────────────
def test_pspp_all_45_actions_pass(rng_data):
    rng, n, g, x1, x2, y = rng_data
    x3 = rng.normal(0, 1, n)
    biny = (x1 > 10).astype(int)
    cnty = rng.poisson(3, n)
    D = _csv({"grp": g, "x1": np.round(x1, 3), "x2": np.round(x2, 3),
              "y": np.round(y, 3), "bin": biny, "cnt": cnty})
    calls = {
        "desc": dict(d=D, v=["x1", "x2"]), "freq": dict(d=D, a="grp"),
        "examine": dict(d=D, v="x1", a="grp"), "means": dict(d=D, v="x1", a="grp"),
        "crosstab": dict(d=_csv({"a": list(np.random.choice(["m", "f"], n)),
                                         "b": list(np.random.choice(["y", "n"], n))}),
                         a="a", v="b"),
        "ttest": dict(d=D, a="grp", v="x1", g=["A", "B"]),
        "pttest": dict(d=_csv({"pre": np.round(x1, 3), "post": np.round(x1 + 1, 3)}),
                       a="pre", v="post"),
        "ttest1": dict(d=D, v="x1", mu=10.0), "anova": dict(d=D, a="grp", v="x1"),
        "mw": dict(d=D, a="grp", v="x1", g=["A", "B"]), "kw": dict(d=D, a="grp", v="x1"),
        "wilcoxon": dict(d=_csv({"pre": np.round(x1, 3), "post": np.round(x1 + 1, 3)}),
                         a="pre", v="post"),
        "friedman": dict(d=_csv({"c1": np.round(x1, 3), "c2": np.round(x2, 3),
                                         "c3": np.round(x3, 3)}), v=["c1", "c2", "c3"]),
        "sign": dict(d=_csv({"pre": np.round(x1, 3), "post": np.round(x1 + 1, 3)}),
                     a="pre", v="post"),
        "ks": dict(d=D, v="x1"), "runs": dict(d=D, v="x1"),
        "corr": dict(d=D, v=["x1", "x2", "y"]), "spearman": dict(d=D, v=["x1", "x2"]),
        "partial": dict(d=D, v=["x1", "y"], c=["x2"]),
        "reg": dict(d=D, o="y", p=["x1", "x2"]),
        "logistic": dict(d=_csv({"bin": biny, "x1": np.round(x1, 3)}), o="bin", p=["x1"]),
        "poisson": dict(d=_csv({"cnt": cnty, "x1": np.round(x1, 3)}), o="cnt", p=["x1"]),
        "factor": dict(d=_csv({"i1": np.round(x1, 3), "i2": np.round(x2, 3),
                                       "i3": np.round(x3, 3), "i4": np.round(x1, 3),
                                       "i5": np.round(x2, 3), "i6": np.round(x3, 3)}),
                       v=["i1", "i2", "i3", "i4", "i5", "i6"], n=2),
        "reliability": dict(d=_csv({"i1": np.round(x1, 3), "i2": np.round(x1, 3),
                                            "i3": np.round(x1, 3)}), v=["i1", "i2", "i3"]),
        "roc": dict(d=_csv({"bin": biny, "x1": np.round(x1, 3)}), a="bin", v="x1", pos=1),
        "eval": dict(d=_csv({"truth": [0, 1, 0, 1, 1, 0] * 10,
                                      "pred": [0, 1, 1, 1, 0, 0] * 10}),
                     a="truth", v="pred", pos_ref=1),
        "surv": dict(d=_csv({"tt": np.round(rng.exponential(10, n), 3),
                                      "ev": rng.binomial(1, .7, n), "grp": g}),
                     o="ev", v="tt", a="grp"),
        "meta": dict(d=_csv({"study": ["s1", "s2", "s3"],
                                      "eff": ["0.5", "0.3", "0.7"], "se": ["0.1", "0.15", "0.12"]}),
                     v="study", effect_col="eff", se_col="se"),
        "blandaltman": dict(d=_csv({"m1": np.round(x1, 3), "m2": np.round(x1, 3)}),
                            a="m1", v="m2"),
        "kappa": dict(d=_csv({"r1": [1, 1, 0, 1] * 15, "r2": [1, 0, 0, 1] * 15}),
                      a="r1", v="r2"),
        "evalue": dict(d=D, or_val=2.5),
        "power": dict(d=D, power_test="ttest", calc="n", es_val=0.5, alpha=0.05),
        "rank": dict(d=D, v="x1"),
        "dca": dict(d=_csv({"bin": biny,
                                     "prob": np.round(np.clip((x1 - 6) / 8, .01, .99), 3)}),
                    a="bin", v="prob"),
        "quantreg": dict(d=D, o="y", p=["x1"], tau=0.5),
        "negbin": dict(d=_csv({"cnt": cnty, "x1": np.round(x1, 3)}), o="cnt", p=["x1"]),
        "mediation": dict(d=_csv({"X": np.round(x1, 3), "M": np.round(x2, 3),
                                          "Y": np.round(y, 3)}), o="Y", a="X", v="M"),
        "pscore": dict(d=_csv({"treat": rng.binomial(1, .4, n), "x1": np.round(x1, 3),
                                       "x2": np.round(x2, 3)}), a="treat", p=["x1", "x2"]),
        "gam": dict(d=D, o="y", p=["x1"]),
        "zip": dict(d=_csv({"cnt": cnty, "x1": np.round(x1, 3)}), o="cnt", p=["x1"]),
        "lasso": dict(d=D, o="y", p=["x1", "x2"], lam=0.1),
        "adf": dict(d=_csv({"ts": np.round(np.cumsum(rng.normal(0, 1, n)), 3)}), v="ts"),
        "beta": dict(d=_csv({"prop": np.round(np.clip(rng.beta(2, 5, n), .01, .99), 4),
                                      "x1": np.round(x1, 3)}), o="prop", p=["x1"]),
        "hedgesg": dict(d=D, a="grp", v="x1", g=["A", "B"]),
        "omega": dict(d=_csv({"i1": np.round(x1, 3), "i2": np.round(x1, 3),
                                      "i3": np.round(x1, 3)}), v=["i1", "i2", "i3"]),
    }
    assert len(calls) == 45
    failed = {k: J(pspp_run(t=k, **kw)).get("e") for k, kw in calls.items()
              if "e" in J(pspp_run(t=k, **kw))}
    assert not failed, failed


def test_ttest_matches_scipy(rng_data):
    _, _, g, x1, _, _ = rng_data
    D = _csv({"grp": g, "x1": np.round(x1, 3)})
    j = J(pspp_run(t="ttest", d=D, a="grp", v="x1", g=["A", "B"]))
    ref = S.ttest_ind(x1[:30], x1[30:])
    assert abs(j["k"]["p"] - round(float(ref.pvalue), 4)) < 1e-3


def test_hedgesg_signed_with_direction(rng_data):
    _, _, g, x1, _, _ = rng_data
    D = _csv({"grp": g, "x1": np.round(x1, 3)})
    j = J(pspp_run(t="hedgesg", d=D, a="grp", v="x1", g=["A", "B"]))["k"]
    assert j["g"] < 0  # B mean < A mean in this seed
    assert "direction" in j and j["direction"].startswith("B minus A")


def test_partial_df(rng_data):
    rng, n, _, x1, x2, y = rng_data
    D = _csv({"x1": np.round(x1, 3), "x2": np.round(x2, 3), "y": np.round(y, 3)})
    j = J(pspp_run(t="partial", d=D, v=["x1", "y"], c=["x2"]))
    assert j["k"]["df"] == n - 3  # n - 2 targets - 1 control


def test_games_howell_matches_reference():
    rng = np.random.RandomState(3)
    A, Bc, C = rng.normal(0, 1, 25), rng.normal(1, 2, 30), rng.normal(0.5, 1.5, 20)
    D = "g,v\n" + "\n".join([f"A,{v}" for v in A] + [f"B,{v}" for v in Bc] +
                            [f"C,{v}" for v in C])
    j = J(pspp_run(t="anova", d=D, a="g", v="v", posthoc="gh"))["k"]
    ab = next(p for p in j["posthoc"] if p["i"] == "A" and p["j"] == "B")
    m1, m2 = A.mean(), Bc.mean()
    v1, v2 = A.var(ddof=1), Bc.var(ddof=1)
    Se = v1 / len(A) + v2 / len(Bc)
    t = abs(m1 - m2) / np.sqrt(Se)
    df = Se ** 2 / ((v1 / len(A)) ** 2 / (len(A) - 1) + (v2 / len(Bc)) ** 2 / (len(Bc) - 1))
    ref = float(S.studentized_range.sf(t * np.sqrt(2), 3, df))
    assert abs(ab["p"] - round(ref, 4)) < 1e-3


def test_logistic_separation_warning():
    rng = np.random.RandomState(0)
    xs = np.concatenate([rng.normal(-2, 1, 30), rng.normal(2, 1, 30)])
    D = "x,bin\n" + "\n".join(f"{v:.3f},{int(v > 0)}" for v in xs)
    j = J(pspp_run(t="logistic", d=D, o="bin", p=["x"]))["k"]
    assert "warn_separation" in j


def test_beta_omega_pscore_notes(rng_data):
    rng, n, g, x1, _, y = rng_data
    D = _csv({"prop": np.round(np.clip(rng.beta(2, 5, n), .01, .99), 4),
              "x1": np.round(x1, 3)})
    assert "note" in J(pspp_run(t="beta", d=D, o="prop", p=["x1"]))["k"]
    I = _csv({"i1": np.round(x1, 3), "i2": np.round(x1, 3), "i3": np.round(x1, 3)})
    assert "note" in J(pspp_run(t="omega", d=I, v=["i1", "i2", "i3"]))["k"]
    P = _csv({"treat": rng.binomial(1, .4, n), "x1": np.round(x1, 3),
              "x2": np.round(rng.normal(0, 1, n), 3)})
    assert "note" in J(pspp_run(t="pscore", d=P, a="treat", p=["x1", "x2"]))["k"]


# ── statsmodels ───────────────────────────────────────────────────
def test_gee_nom_passes():
    rng = np.random.RandomState(1)
    n = 80
    subj = np.repeat(np.arange(20), 4)
    x = rng.normal(0, 1, n)
    D = "g,x,nom\n" + "\n".join(
        f"{subj[i]},{x[i]:.3f},{int(np.random.choice([0, 1, 2]))}" for i in range(n))
    j = J(statsmodels_run(action="gee_nom", d=D, o="nom", p=["x"], g="g"))
    assert "e" not in j and "f" in j


def test_statsmodels_core_actions():
    rng = np.random.RandomState(1)
    n = 80
    subj = np.repeat(np.arange(20), 4)
    time = np.tile([0, 1, 2, 3], 20)
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 1, n)
    LONG = _csv({"subj": subj, "time": time, "x": np.round(x, 3), "y": np.round(y, 3)})
    assert "e" not in J(statsmodels_run(action="mlm", d=LONG, o="y", p=["x"], g="subj"))
    assert "e" not in J(statsmodels_run(action="gee", d=LONG, o="y", p=["x"], g="subj"))
    assert "e" not in J(statsmodels_run(action="anova_rm", d=LONG, o="y",
                                        g="subj", w=["time"]))


# ── IRT ───────────────────────────────────────────────────────────
def _irt_binary(seed=1, people=200, items=10):
    rng = np.random.RandomState(seed)
    th = rng.normal(0, 1, people)
    b = np.linspace(-2, 2, items)
    P = 1 / (1 + np.exp(-(th[:, None] - b[None, :])))
    B = (rng.uniform(size=(people, items)) < P).astype(int)
    C = ",".join(f"i{j + 1}" for j in range(items)) + "\n" + "\n".join(
        ",".join(str(v) for v in row) for row in B)
    return B, C


def test_irt_2pl_recovers_difficulty_order():
    B, C = _irt_binary()
    j = J(irt_run(action="irt_2pl", d=C))
    assert "e" not in j
    assert len(j["d"]) == 10 and len(j["disc"]) == 10  # item-length, not person-length
    true_b = list(np.linspace(-2, 2, 10))
    rho = S.spearmanr(j["d"], true_b).statistic
    assert rho > 0.95  # estimates recover true difficulty ordering


def test_irt_3pl_works():
    _, C = _irt_binary()
    j = J(irt_run(action="irt_3pl", d=C))
    assert "e" not in j, j.get("e")
    assert len(j["d"]) == 10 and "g" in j


def test_irt_score_3pl_monotone():
    B, C = _irt_binary(people=100, items=8)
    j = J(irt_run(action="irt_score", d=C, diff=",".join(["0"] * 8),
                  disc=",".join(["1"] * 8), guess=",".join(["0.2"] * 8), model="3pl"))
    assert "e" not in j, j.get("e")
    rho = S.spearmanr(j["a"], B.sum(axis=1)).statistic
    assert rho > 0.9


def test_irt_missing_data():
    B, C = _irt_binary(people=100, items=8)
    rng = np.random.RandomState(2)
    Bm = B.astype(float)
    Bm[rng.uniform(size=B.shape) < 0.1] = np.nan
    BMC = "i1,i2,i3,i4,i5,i6,i7,i8\n" + "\n".join(
        ",".join("" if np.isnan(v) else str(int(v)) for v in row) for row in Bm)
    j = J(irt_run(action="irt_2pl", d=BMC))
    assert "e" not in j and j["n_people"] < 100


# ── medical_ext ───────────────────────────────────────────────────
def test_medical_ext_power_ttest_default_alt():
    j = J(medical_ext_run(action="ttest", n=0, pw=0.8, a=0.05, d_es=0.5))
    assert "e" not in j and j["alt"] == "two-sided"


def test_medical_ext_forest_km():
    j = J(medical_ext_run(action="forest", d=json.dumps(
        {"study": ["a", "b"], "yi": [0.5, 0.3], "se": [0.1, 0.15],
         "label": ["A", "B"]})))
    assert "e" not in j and j["k"] == 2
    rng = np.random.RandomState(1)
    n = 60
    SC = _csv({"T": np.round(rng.exponential(10, n), 4),
               "E": rng.binomial(1, .7, n),
               "grp": np.where(rng.binomial(1, .5, n) == 1, "T", "C")})
    assert "e" not in J(medical_ext_run(action="km", d=SC, t="T", e="E", g="grp"))


# ── sem ───────────────────────────────────────────────────────────
def test_sem_fit_inspect_same_desc_different_data():
    rng = np.random.RandomState(1)
    n = 80
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 1, n)

    def _mk(sh):
        return _csv({"x1": np.round(x + sh, 3), "x2": np.round(x + sh, 3),
                     "x3": np.round(x + sh, 3), "y1": np.round(y + sh, 3),
                     "y2": np.round(y + sh, 3), "y3": np.round(y + sh, 3)})

    desc = "f1 =~ x1+x2+x3\nf2 =~ y1+y2+y3\nf2 ~ f1"
    assert "e" not in J(sem_run(action="sem_fit", d=_mk(0), desc=desc))
    assert "e" not in J(sem_run(action="sem_fit", d=_mk(5), desc=desc))
    j = J(sem_run(action="sem_inspect", d=_mk(5), desc=desc))
    assert "params" in j and len(j["params"]) > 0
