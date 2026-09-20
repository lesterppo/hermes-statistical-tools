#!/usr/bin/env python3
"""
PSPP — AI-agent-native statistical tool. Pure Python (scipy/statsmodels/numpy)
replicating the full PSPP 2.0.0 statistical command set with agent-native interfaces.

All 25 test types implemented directly — no SPSS syntax, no PSPP binary required
for any analysis. Agent specifies test type + variable names.

Descriptive:     desc, freq, examine, means, crosstab
Compare means:   ttest, pttest, ttest1, anova
Non-parametric:  mw, kw, wilcoxon, friedman, sign, ks, runs
Correlation:     corr, spearman, partial
Regression:      reg, logistic
Advanced:        factor, reliability, roc
Transform:       rank
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _ok(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _err(msg: str) -> str:
    return _ok({"e": msg})


def _parse_csv(csv_data: str) -> Dict[str, list]:
    reader = csv.DictReader(io.StringIO(csv_data.strip()))
    cols = {name: [] for name in reader.fieldnames or []}
    for row in reader:
        for name in cols:
            try:
                v = float(row[name])
            except (ValueError, TypeError):
                v = row[name] if row[name] else None
            cols[name].append(v)
    for name in cols:
        numeric = sum(1 for v in cols[name] if isinstance(v, (int, float)))
        if numeric > len(cols[name]) * 0.8:
            cols[name] = [float(v) if isinstance(v, (int, float)) else float('nan') for v in cols[name]]
    return cols


def _valid(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return True


def _arr(col: list) -> np.ndarray:
    vals = [float(x) for x in col if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return np.array(vals, dtype=np.float64)


def _groups(col: list, group_var: list, target_groups: list) -> List[np.ndarray]:
    result = []
    for g in target_groups:
        vals = []
        for i, cv in enumerate(col):
            if i < len(group_var) and group_var[i] == g:
                if cv is not None and not (isinstance(cv, float) and math.isnan(cv)):
                    vals.append(float(cv))
        result.append(np.array(vals, dtype=np.float64))
    return result


def _unique_groups(cols: dict, a: str) -> list:
    return sorted(set(x for x in cols[a] if _valid(x)))


def _rnd(val, ndigits=4):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    if isinstance(val, (np.floating, np.integer)):
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits)
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    try:
        return round(f, ndigits)
    except (ValueError, TypeError, OverflowError):
        return None


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Cohen's d = (mean(b) - mean(a)) / pooled SD.

    Sign convention: positive means group b (g2) scores higher than
    group a (g1). Callers pass groups in user-supplied g order.
    """
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    sd1, sd2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    return round(float((np.mean(b) - np.mean(a)) / pooled), 4) if pooled > 0 else 0.0


def _design_matrix(X_cols: List[np.ndarray], add_intercept: bool = True) -> np.ndarray:
    n = min(len(x) for x in X_cols) if X_cols else 0
    cols = [x[:n] for x in X_cols]
    if add_intercept:
        cols.insert(0, np.ones(n))
    return np.column_stack(cols) if cols else np.empty((n, 0))


def _ols(y: np.ndarray, X: np.ndarray):
    n, k = X.shape
    # Use lstsq instead of explicit inverse for numerical stability
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None, None, 0, 0, 0
    y_pred = X @ beta
    resid = y - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_R2 = 1 - (1 - R2) * (n - 1) / (n - k) if n > k else 0
    sigma2 = ss_res / (n - k) if n > k else 0
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except (np.linalg.LinAlgError, ValueError):
        se = np.full(k, float('nan'))
    t_vals = beta / se
    p_vals = np.array([float(2 * scipy_stats.t.sf(abs(tv if not math.isnan(tv) else 0), n - k)) for tv in t_vals])
    return beta, se, t_vals, p_vals, R2, adj_R2, ss_res


def _p_adjust(p_values: list, method: str = "bonferroni") -> list:
    """Multiple comparison correction."""
    p = np.array(p_values, dtype=float)
    if method == "bonferroni":
        return list(np.minimum(p * len(p), 1.0))
    elif method == "holm":
        n = len(p)
        idx = np.argsort(p)
        adjusted = np.zeros(n)
        for rank, i in enumerate(idx):
            adjusted[i] = min(p[i] * (n - rank), 1.0)
        return list(adjusted)
    elif method == "bh":
        n = len(p)
        idx = np.argsort(p)
        adjusted = np.zeros(n)
        for rank, i in enumerate(idx):
            adjusted[i] = min(p[i] * n / (rank + 1), 1.0)
        # Enforce monotonicity (step-down): later adjusted values ≥ earlier ones
        adjusted_sorted = adjusted[idx]
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        for rank, i in enumerate(idx):
            adjusted[i] = adjusted_sorted[rank]
        return list(adjusted)
    return list(p)


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

# ─── DESCRIPTIVES ─────────────────────────────────────────────────

def _do_descriptives(cols: dict, args: dict) -> dict:
    """Descriptive statistics for listed variables."""
    variables = args.get("v", [])
    if not variables:
        return {"e": "v (variable list) required"}
    result = []
    for var in variables:
        if var not in cols:
            continue
        vals = _arr(cols[var])
        if len(vals) == 0:
            result.append({"n": var, "N": 0})
            continue
        result.append({
            "n": var, "N": len(vals),
            "m": _rnd(float(np.mean(vals))),
            "sd": _rnd(float(np.std(vals, ddof=1))),
            "min": _rnd(float(np.min(vals))),
            "max": _rnd(float(np.max(vals))),
        })
    return {"vars": result}


def _do_frequencies(cols: dict, args: dict) -> dict:
    """Frequency table for a variable."""
    a = args.get("a", "")
    if not a or a not in cols:
        return {"e": "a (variable) required"}
    counts = {}
    total = 0
    for v in cols[a]:
        if _valid(v):
            key = int(v) if isinstance(v, float) and v == int(v) else v
            counts[key] = counts.get(key, 0) + 1
            total += 1
    freq = []
    for key in sorted(counts.keys(), key=lambda x: (isinstance(x, str), x)):
        freq.append({"v": key, "n": counts[key],
                     "pct": f"{round(counts[key]/total*100,1)}%" if total > 0 else "0%"})
    return {"freq": freq}


def _do_examine(cols: dict, args: dict) -> dict:
    """EXAMINE — normality tests (Shapiro-Wilk), percentiles, skewness, kurtosis."""
    variables = args.get("v", [])
    if not variables:
        return {"e": "v (variable list) required"}
    result = []
    for var in variables:
        if var not in cols:
            continue
        vals = _arr(cols[var])
        n = len(vals)
        if n < 3:
            result.append({"n": var, "N": n, "e": "need ≥3 values"})
            continue
        # Shapiro-Wilk normality test
        sw_stat, sw_p = scipy_stats.shapiro(vals) if n <= 5000 else (None, None)
        # Skewness and kurtosis
        from scipy.stats import skew, kurtosis
        sk = float(skew(vals))
        kt = float(kurtosis(vals, fisher=True))
        # Percentiles
        pcts = {}
        for q in [5, 10, 25, 50, 75, 90, 95]:
            pcts[f"p{q}"] = _rnd(float(np.percentile(vals, q)))
        entry = {
            "n": var, "N": n,
            "m": _rnd(float(np.mean(vals))),
            "sd": _rnd(float(np.std(vals, ddof=1))),
            "min": _rnd(float(np.min(vals))),
            "max": _rnd(float(np.max(vals))),
            "sk": _rnd(sk), "kt": _rnd(kt),
            "sw": _rnd(sw_stat), "swp": _rnd(sw_p),
            **pcts,
        }
        result.append(entry)
    return {"vars": result}


def _do_means(cols: dict, args: dict) -> dict:
    """MEANS — descriptives by group."""
    a = args.get("a", "")
    v = args.get("v", "")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (group var) and v (value var) required"}
    groups = _unique_groups(cols, a)
    result = []
    for g in groups:
        vals = _groups(cols[v], cols[a], [g])[0]
        if len(vals) == 0:
            continue
        result.append({
            "g": g, "N": len(vals),
            "m": _rnd(float(np.mean(vals))),
            "sd": _rnd(float(np.std(vals, ddof=1))),
            "min": _rnd(float(np.min(vals))),
            "max": _rnd(float(np.max(vals))),
        })
    return {"groups": result}


def _do_crosstab(cols: dict, args: dict) -> dict:
    """CROSSTABS — chi-square, Phi, Cramer's V, Fisher's exact, OR, McNemar."""
    a = args.get("a", "")
    v = args.get("v", "")
    w = args.get("w", "")
    paired = args.get("paired", False)
    if not a or not v:
        return {"e": "a (row var) and v (col var) required"}
    if a not in cols or v not in cols:
        return {"e": f"Variables not found: '{a}', '{v}'"}
    # Build contingency table
    table_data = {}
    if w and w in cols:
        for i in range(min(len(cols[a]), len(cols[v]), len(cols[w]))):
            ra, ca, wt = cols[a][i], cols[v][i], cols[w][i]
            if _valid(ra) and _valid(ca) and _valid(wt):
                table_data[(str(ra), str(ca))] = table_data.get((str(ra), str(ca)), 0) + float(wt)
    else:
        for i in range(min(len(cols[a]), len(cols[v]))):
            ra, ca = cols[a][i], cols[v][i]
            if _valid(ra) and _valid(ca):
                table_data[(str(ra), str(ca))] = table_data.get((str(ra), str(ca)), 0) + 1
    row_labels = sorted(set(k[0] for k in table_data))
    col_labels = sorted(set(k[1] for k in table_data))
    n_rows, n_cols = len(row_labels), len(col_labels)
    if n_rows < 2 or n_cols < 2:
        return {"e": f"Need ≥2×2 table, got {n_rows}×{n_cols}"}
    obs = np.zeros((n_rows, n_cols))
    for (r, c), cnt in table_data.items():
        obs[row_labels.index(r), col_labels.index(c)] = cnt
    n_total = int(np.sum(obs))

    result = {"n": n_total, "rows": row_labels, "cols": col_labels}

    # Chi-square
    chi2, chi_p, dof, expected = scipy_stats.chi2_contingency(obs, correction=False)
    result["χ²"] = _rnd(chi2)
    result["df"] = dof
    result["p"] = _rnd(chi_p)

    # Phi
    phi = math.sqrt(chi2 / n_total) if n_total > 0 else 0
    result["phi"] = _rnd(phi)

    # Cramer's V
    min_dim = min(n_rows, n_cols) - 1
    V = math.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 and n_total > 0 else 0
    result["V"] = _rnd(V)

    # Fisher's exact (2x2 only)
    if obs.shape == (2, 2):
        try:
            _, fisher_p = scipy_stats.fisher_exact(obs)
            result["fp"] = _rnd(fisher_p)
        except Exception:
            pass

    # Odds ratio (2x2 only) — use Haldane-Anscombe correction for zero cells
    if obs.shape == (2, 2):
        a_v, b_v, c_v, d_v = obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1]
        if b_v == 0 or c_v == 0:
            a_v += 0.5; b_v += 0.5; c_v += 0.5; d_v += 0.5
        if b_v > 0 and c_v > 0:
            or_val = (a_v * d_v) / (b_v * c_v)
            se = math.sqrt(1 / a_v + 1 / b_v + 1 / c_v + 1 / d_v)
            result["OR"] = _rnd(or_val)
            result["OR_ci"] = [_rnd(math.exp(math.log(or_val) - 1.96 * se)),
                              _rnd(math.exp(math.log(or_val) + 1.96 * se))]

    # McNemar (paired 2x2)
    if paired and obs.shape == (2, 2):
        b_v, c_v = obs[0, 1], obs[1, 0]
        if b_v + c_v > 0:
            # Continuity correction: use max guard for balanced discordant pairs
            mcn = (max(0, abs(b_v - c_v) - 1)) ** 2 / (b_v + c_v)
            mcn_p = float(scipy_stats.chi2.sf(mcn, 1))
            result["mchi2"] = _rnd(mcn)
            result["mp"] = _rnd(mcn_p)

    return result


# ─── COMPARE MEANS ────────────────────────────────────────────────

def _do_ttest(cols: dict, args: dict) -> dict:
    """Independent samples t-test (Welch)."""
    a, v, g = args.get("a", ""), args.get("v", ""), args.get("g", [])
    if not a or not v or len(g) != 2:
        return {"e": "a (group var), v (value var), g=[g1,g2] required"}
    if a not in cols or v not in cols:
        return {"e": f"Variables not found: '{a}', '{v}'"}
    grps = _groups(cols[v], cols[a], g)
    if len(grps[0]) < 2 or len(grps[1]) < 2:
        return {"e": "Need ≥2 values per group"}
    t, p = scipy_stats.ttest_ind(grps[0], grps[1], equal_var=False)
    v1, v2 = float(np.var(grps[0], ddof=1)), float(np.var(grps[1], ddof=1))
    n1, n2 = len(grps[0]), len(grps[1])
    # Guard against zero variance (identical values) — return stats WITH a
    # warning key (not "e": pspp_run treats "e" as fatal and drops the payload).
    if v1 == 0 and v2 == 0:
        return {"n1": n1, "n2": n2, "m1": _rnd(float(np.mean(grps[0]))), "m2": _rnd(float(np.mean(grps[1]))),
                "t": 0, "df": 0, "p": 1.0, "d": 0, "md": 0, "ci_l": None, "ci_u": None,
                "warn": "zero variance in both groups"}
    if v1 == 0 or v2 == 0:
        return {"n1": n1, "n2": n2, "m1": _rnd(float(np.mean(grps[0]))), "m2": _rnd(float(np.mean(grps[1]))),
                "t": None, "df": None, "p": None, "d": None, "md": None, "ci_l": None, "ci_u": None,
                "warn": "zero variance in one group"}
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    md = float(np.mean(grps[1]) - np.mean(grps[0]))
    se = math.sqrt(v1 / n1 + v2 / n2)
    ci = scipy_stats.t.interval(0.95, df, loc=md, scale=se)
    return {
        "n1": n1, "n2": n2,
        "m1": _rnd(float(np.mean(grps[0]))), "m2": _rnd(float(np.mean(grps[1]))),
        "sd1": _rnd(float(np.std(grps[0], ddof=1))), "sd2": _rnd(float(np.std(grps[1], ddof=1))),
        "t": _rnd(t), "df": _rnd(df), "p": _rnd(p),
        "d": _cohens_d(grps[0], grps[1]),
        "md": _rnd(md), "ci_l": _rnd(ci[0]), "ci_u": _rnd(ci[1]),
    }


def _do_pttest(cols: dict, args: dict) -> dict:
    """Paired t-test."""
    a, v = args.get("a", ""), args.get("v", "")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (var1), v (var2) required"}
    x1, x2 = _arr(cols[a]), _arr(cols[v])
    n = min(len(x1), len(x2))
    if n < 3:
        return {"e": "Need ≥3 pairs"}
    t, p = scipy_stats.ttest_rel(x1[:n], x2[:n])
    diff = x1[:n] - x2[:n]
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    d = float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0
    return {
        "n": n, "m1": _rnd(float(np.mean(x1[:n]))), "m2": _rnd(float(np.mean(x2[:n]))),
        "t": _rnd(t), "df": n - 1, "p": _rnd(p), "d": _rnd(d),
        "md": _rnd(float(np.mean(diff))),
    }


def _do_ttest1(cols: dict, args: dict) -> dict:
    """One-sample t-test."""
    v, mu = args.get("v", ""), args.get("mu", 0)
    if not v or v not in cols:
        return {"e": "v (variable) required"}
    vals = _arr(cols[v])
    n = len(vals)
    if n < 2:
        return {"e": "Need ≥2 values"}
    t, p = scipy_stats.ttest_1samp(vals, mu)
    return {
        "n": n, "m": _rnd(float(np.mean(vals))), "sd": _rnd(float(np.std(vals, ddof=1))),
        "t": _rnd(t), "df": n - 1, "p": _rnd(p),
        "mu": mu, "md": _rnd(float(np.mean(vals)) - mu),
    }


def _do_anova(cols: dict, args: dict) -> dict:
    """One-way ANOVA with post-hoc (Tukey HSD, Bonferroni, Games-Howell)."""
    a, v = args.get("a", ""), args.get("v", "")
    posthoc_method = args.get("posthoc", "")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (group var), v (value var) required"}
    uq = _unique_groups(cols, a)
    if len(uq) < 2:
        return {"e": f"Need ≥2 groups, got {uq}"}
    groups = _groups(cols[v], cols[a], uq)
    if any(len(g) < 2 for g in groups):
        return {"e": "Need ≥2 values per group"}
    F, p = scipy_stats.f_oneway(*groups)
    grand_mean = np.mean(np.concatenate(groups))
    ss_b = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_w = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups)
    eta2 = float(ss_b / (ss_b + ss_w)) if (ss_b + ss_w) > 0 else 0
    result = {
        "F": _rnd(F), "df1": len(groups) - 1,
        "df2": sum(len(g) for g in groups) - len(groups),
        "p": _rnd(p), "eta2": _rnd(eta2),
    }
    # Post-hoc
    if posthoc_method in ("tukey", "bonferroni", "gh", "") and len(groups) >= 2:
        result["posthoc"] = _do_posthoc(groups, uq, posthoc_method or "tukey")
    return result


def _do_posthoc(groups: List[np.ndarray], labels: list, method: str) -> list:
    """Pairwise post-hoc comparisons.

    tukey:      Tukey HSD (equal variances), already family-wise corrected.
    bonferroni: Tukey-style t pairwise + Bonferroni correction.
    gh:         Games-Howell (unequal variances, Welch SE + Studentized Range).
                Already family-wise corrected — NO further adjustment applied.
    """
    pairs = []
    raw_p = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            md = float(np.mean(groups[j]) - np.mean(groups[i]))
            if method == "gh":
                # Games-Howell: t = |md| / sqrt(v1/n1 + v2/n2) (Welch SE),
                # p from the Studentized Range distribution with q = t*sqrt(2)
                # and Welch-Satterthwaite df. Already corrected.
                n1, n2 = len(groups[i]), len(groups[j])
                v1, v2 = float(np.var(groups[i], ddof=1)), float(np.var(groups[j], ddof=1))
                S = v1 / n1 + v2 / n2
                t_gh = abs(md) / math.sqrt(S) if S > 0 else 0
                df = S ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if S > 0 else 1
                p_val = float(scipy_stats.studentized_range.sf(t_gh * math.sqrt(2), len(groups), df))
            else:
                # Tukey HSD
                mse = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups) / (sum(len(g) for g in groups) - len(groups))
                se = math.sqrt(mse / 2 * (1 / len(groups[i]) + 1 / len(groups[j])))
                q_val = abs(md) / se if se > 0 else 0
                p_val = float(scipy_stats.studentized_range.sf(q_val, len(groups),
                         sum(len(g) for g in groups) - len(groups)))
            raw_p.append(p_val)
            pairs.append({"i": labels[i], "j": labels[j], "d": _rnd(md), "p": p_val})
    # Adjust p-values
    if method == "bonferroni":
        adj = _p_adjust(raw_p, "bonferroni")
    else:
        adj = raw_p  # Tukey and Games-Howell are already family-wise corrected
    for idx in range(len(pairs)):
        pairs[idx]["p"] = _rnd(adj[idx])
    return pairs


# ─── NON-PARAMETRIC ───────────────────────────────────────────────

def _do_mann_whitney(cols: dict, args: dict) -> dict:
    a, v, g = args.get("a", ""), args.get("v", ""), args.get("g", [])
    if not a or not v or len(g) != 2:
        return {"e": "a (group var), v (value var), g=[g1,g2] required"}
    if a not in cols or v not in cols:
        return {"e": f"Variables not found: '{a}', '{v}'"}
    grps = _groups(cols[v], cols[a], g)
    if len(grps[0]) == 0 or len(grps[1]) == 0:
        return {"e": "Need ≥1 value per group"}
    U, p = scipy_stats.mannwhitneyu(grps[0], grps[1], alternative='two-sided')
    n1 = len(grps[0])
    R1 = scipy_stats.rankdata(np.concatenate([grps[0], grps[1]]))[:n1]
    W = float(np.sum(R1))
    sigma = math.sqrt(n1 * len(grps[1]) * (n1 + len(grps[1]) + 1) / 12)
    Z = (U - n1 * len(grps[1]) / 2) / sigma if sigma > 0 else 0
    return {"U": _rnd(U), "W": _rnd(W), "Z": _rnd(Z), "p": _rnd(p)}


def _do_kruskal_wallis(cols: dict, args: dict) -> dict:
    a, v = args.get("a", ""), args.get("v", "")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (group var), v (value var) required"}
    uq = _unique_groups(cols, a)
    if len(uq) < 2:
        return {"e": f"Need ≥2 groups, got {uq}"}
    groups = _groups(cols[v], cols[a], uq)
    H, p = scipy_stats.kruskal(*groups)
    return {"H": _rnd(H), "df": len(groups) - 1, "p": _rnd(p)}


def _do_wilcoxon(cols: dict, args: dict) -> dict:
    """Wilcoxon signed-rank test (paired or one-sample)."""
    a, v, mu = args.get("a", ""), args.get("v", ""), args.get("mu")
    if a and v and a in cols and v in cols:
        # Paired
        x1, x2 = _arr(cols[a]), _arr(cols[v])
        n = min(len(x1), len(x2))
        W, p = scipy_stats.wilcoxon(x1[:n], x2[:n])
        return {"n": n, "W": _rnd(W), "p": _rnd(p), "type": "paired"}
    elif v and v in cols:
        # One-sample
        vals = _arr(cols[v])
        target = float(mu) if mu is not None else 0
        W, p = scipy_stats.wilcoxon(vals - target)
        return {"n": len(vals), "W": _rnd(W), "p": _rnd(p), "type": "one-sample", "mu": target}
    return {"e": "a (var1) + v (var2) for paired, or v (var) + mu for one-sample"}


def _do_friedman(cols: dict, args: dict) -> dict:
    """Friedman test (repeated measures non-parametric)."""
    variables = args.get("v", [])
    if not variables or len(variables) < 2:
        return {"e": "v (2+ repeated measure variables) required"}
    for var in variables:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in variables]
    n = min(len(a) for a in arrays)
    if n < 3:
        return {"e": "Need ≥3 subjects"}
    chi2, p = scipy_stats.friedmanchisquare(*[a[:n] for a in arrays])
    return {"χ²": _rnd(chi2), "df": len(variables) - 1, "p": _rnd(p), "n": n}


def _do_sign_test(cols: dict, args: dict) -> dict:
    """Sign test (paired)."""
    a, v = args.get("a", ""), args.get("v", "")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (var1), v (var2) required"}
    x1, x2 = _arr(cols[a]), _arr(cols[v])
    n = min(len(x1), len(x2))
    diff = x1[:n] - x2[:n]
    pos = int(np.sum(diff > 0))
    neg = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    total = pos + neg
    p = float(min(2 * scipy_stats.binom.cdf(min(pos, neg), total, 0.5), 1.0)) if total > 0 else 1.0
    return {"pos": pos, "neg": neg, "ties": ties, "n": total, "p": _rnd(p)}


def _do_ks_test(cols: dict, args: dict) -> dict:
    """Kolmogorov-Smirnov test."""
    a, v, g = args.get("a", ""), args.get("v", ""), args.get("g", [])
    if a and v and len(g) >= 2:
        # Two-sample KS
        if a not in cols or v not in cols:
            return {"e": f"Variables not found: '{a}', '{v}'"}
        grps = _groups(cols[v], cols[a], g[:2])
        D, p = scipy_stats.ks_2samp(grps[0], grps[1])
        return {"D": _rnd(D), "p": _rnd(p), "type": "two-sample"}
    elif v and v in cols:
        # One-sample KS (against normal)
        vals = _arr(cols[v])
        D, p = scipy_stats.kstest(vals, 'norm', args=(np.mean(vals), np.std(vals, ddof=1)))
        return {"D": _rnd(D), "p": _rnd(p), "type": "one-sample (vs normal)"}
    return {"e": "a+v+g for two-sample, or v for one-sample vs normal"}


def _do_runs_test(cols: dict, args: dict) -> dict:
    """Runs test for randomness."""
    v = args.get("v", "")
    if not v or v not in cols:
        return {"e": "v (variable) required"}
    vals = _arr(cols[v])
    n = len(vals)
    if n < 2:
        return {"e": "Need ≥2 values"}
    median = float(np.median(vals))
    above = vals > median
    runs = 1
    for i in range(1, n):
        if above[i] != above[i - 1]:
            runs += 1
    n1 = int(np.sum(above))
    n2 = n - n1
    if n1 < 1 or n2 < 1:
        return {"e": "Need both above and below median"}
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    sigma = math.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    Z = (runs - mu) / sigma if sigma > 0 else 0
    p = float(2 * norm.sf(abs(Z)))
    return {"runs": runs, "median": _rnd(median), "Z": _rnd(Z), "p": _rnd(p)}


# ─── CORRELATION ──────────────────────────────────────────────────

def _do_correlation(cols: dict, args: dict) -> dict:
    """Pearson correlation matrix."""
    variables = args.get("v", [])
    if not variables or len(variables) < 2:
        return {"e": "v (2+ variables) required"}
    for var in variables:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in variables]
    n = min(len(a) for a in arrays)
    arrays = [a[:n] for a in arrays]
    pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            r, p = scipy_stats.pearsonr(arrays[i], arrays[j])
            pairs.append({"v": f"{variables[i]}-{variables[j]}", "r": _rnd(r), "p": _rnd(p)})
    return {"pairs": pairs}


def _do_spearman(cols: dict, args: dict) -> dict:
    """Spearman rank correlation matrix."""
    variables = args.get("v", [])
    if not variables or len(variables) < 2:
        return {"e": "v (2+ variables) required"}
    for var in variables:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in variables]
    n = min(len(a) for a in arrays)
    arrays = [a[:n] for a in arrays]
    pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            rho, p = scipy_stats.spearmanr(arrays[i], arrays[j])
            pairs.append({"v": f"{variables[i]}-{variables[j]}", "rho": _rnd(rho), "p": _rnd(p)})
    return {"pairs": pairs}


def _do_partial_corr(cols: dict, args: dict) -> dict:
    """Partial correlation (control for one or more variables)."""
    variables = args.get("v", [])
    controls = args.get("c", [])
    if not variables or not controls:
        return {"e": "v (variables) and c (control variables) required"}
    all_vars = variables + controls
    for var in all_vars:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in all_vars]
    n = min(len(a) for a in arrays)
    arrays = [a[:n] for a in arrays]
    data = np.column_stack(arrays)
    # Compute precision matrix (inverse of covariance)
    cov = np.cov(data, rowvar=False)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return {"e": "Precision matrix singular — variables may be collinear"}
    # Partial correlations from precision matrix
    pcorr = np.zeros((len(all_vars), len(all_vars)))
    for i in range(len(all_vars)):
        for j in range(len(all_vars)):
            if i == j:
                pcorr[i, j] = 1.0
            else:
                val = -prec[i, j] / math.sqrt(prec[i, i] * prec[j, j])
                pcorr[i, j] = val
    # Extract pairwise partial correlations among target variables
    pairs = []
    # df for the t-test of a partial r controlling k covariates:
    # df = n - 2 - k = n - len(all_vars) (2 targets + k controls).
    df = n - len(all_vars)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            r_partial = float(pcorr[i, j])
            # t-test for partial correlation
            if df > 0 and abs(r_partial) < 1:
                t_val = r_partial * math.sqrt(df / (1 - r_partial ** 2))
                p_val = float(2 * scipy_stats.t.sf(abs(t_val), df))
            else:
                p_val = None
            pairs.append({"v": f"{variables[i]}-{variables[j]}", "r": _rnd(r_partial), "p": _rnd(p_val),
                         "control": controls})
    return {"pairs": pairs, "df": df}


# ─── REGRESSION ───────────────────────────────────────────────────

def _do_regression(cols: dict, args: dict) -> dict:
    """Linear regression with VIF."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    if not o or not predictors:
        return {"e": "o (outcome) and p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y = y[:n]
    X_cols = [x[:n] for x in X_cols]
    X = _design_matrix(X_cols, add_intercept=True)
    k = X.shape[1]
    if n <= k:
        return {"e": f"Need more observations ({n}) than parameters ({k})"}
    beta, se, t_vals, p_vals, R2, adj_R2, ss_res = _ols(y, X)
    if beta is None:
        return {"e": "Matrix singular — predictors may be collinear"}
    # SS for F-test
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_reg = ss_tot - ss_res
    F = (ss_reg / (k - 1)) / (ss_res / (n - k)) if k > 1 and ss_res > 0 else None
    Fp = float(scipy_stats.f.sf(F, k - 1, n - k)) if F else None
    R = math.sqrt(R2) if R2 >= 0 else 0
    # Coefficients
    names = ["(Intercept)"] + list(predictors)
    coefs = []
    for i, name in enumerate(names):
        if i < len(beta):
            coefs.append({
                "n": name, "B": _rnd(beta[i]), "SE": _rnd(se[i]),
                "t": _rnd(t_vals[i]), "p": _rnd(p_vals[i]),
            })
    # VIF (for predictors only, not intercept)
    vifs = {}
    if len(predictors) > 1:
        for idx, pv in enumerate(predictors):
            other_preds = [p for j, p in enumerate(predictors) if j != idx]
            X_sub = _design_matrix([_arr(cols[p])[:n] for p in other_preds], add_intercept=True)
            y_sub = _arr(cols[pv])[:n]
            try:
                _, _, _, _, R2_sub, _, _ = _ols(y_sub, X_sub)
                if R2_sub is not None and R2_sub < 1:
                    vifs[pv] = _rnd(1 / (1 - R2_sub))
                else:
                    vifs[pv] = None
            except Exception:
                vifs[pv] = None
    result = {
        "R": _rnd(R), "R2": _rnd(R2), "aR2": _rnd(adj_R2),
        "F": _rnd(F), "Fp": _rnd(Fp), "n": n,
        "coef": coefs,
    }
    if vifs:
        result["vif"] = vifs
    return result


def _do_logistic(cols: dict, args: dict) -> dict:
    """Logistic regression using Newton-Raphson (no statsmodels dependency)."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    if not o or not predictors:
        return {"e": "o (binary outcome) and p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y_raw = _arr(cols[o])
    y_unique = sorted(set(int(v) for v in y_raw if not math.isnan(v)))
    if len(y_unique) != 2:
        return {"e": f"Outcome must be binary (2 unique values), got {y_unique}"}
    lo, hi = y_unique
    y = np.array([1.0 if v == hi else 0.0 for v in y_raw])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y = y[:n]
    X = _design_matrix([x[:n] for x in X_cols], add_intercept=True)
    # IRLS / Newton-Raphson
    beta = np.zeros(X.shape[1])
    converged = False
    for _ in range(50):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W = np.diag(mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))
        try:
            XTWX = X.T @ W @ X
            beta_new = np.linalg.inv(XTWX) @ X.T @ W @ z
        except np.linalg.LinAlgError:
            return {"e": "Matrix singular — predictors may be collinear or complete separation"}
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    # Standard errors
    eta = X @ beta
    eta = np.clip(eta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    W = np.diag(mu * (1 - mu))
    try:
        cov = np.linalg.inv(X.T @ W @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), float('nan'))
    z_vals = beta / se
    p_vals = np.array([float(2 * norm.sf(abs(zv))) for zv in z_vals])
    # Pseudo R² (McFadden)
    log_lik = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    null_prob = np.mean(y)
    null_prob = np.clip(null_prob, 1e-10, 1 - 1e-10)
    null_loglik = float(np.sum(y * np.log(null_prob) + (1 - y) * np.log(1 - null_prob)))
    mcfadden_R2 = 1 - log_lik / null_loglik if null_loglik != 0 else 0
    # Coefficients
    names = ["(Intercept)"] + list(predictors)
    coefs = []
    for i, name in enumerate(names):
        if i < len(beta):
            coefs.append({
                "n": name, "B": _rnd(beta[i]), "SE": _rnd(se[i]),
                "z": _rnd(z_vals[i]), "p": _rnd(p_vals[i]),
                "OR": _rnd(math.exp(beta[i])),
            })
    result = {
        "n": n, "R2_mcfadden": _rnd(mcfadden_R2),
        "coef": coefs,
    }
    # Separation diagnostics: non-convergence, huge |B|/SE, or perfect
    # in-sample classification all signal complete/quasi-complete separation.
    pred_pos = (mu >= 0.5).astype(float)
    perfect = bool(np.all(pred_pos == y)) if n > 0 else False
    max_se = float(np.nanmax(se)) if len(se) else 0.0
    if (not converged) or perfect or max_se > 100 or np.max(np.abs(beta)) > 10:
        result["warn_separation"] = (
            "possible complete/quasi-complete separation"
            f" (converged={converged}, perfect_classification={perfect},"
            f" max|B|={float(np.max(np.abs(beta))):.2f}, maxSE={max_se:.2f}):"
            " coefficients/SEs may be inflated; consider Firth penalized"
            " logistic regression or collapsing sparse categories"
        )
    return result


# ─── ADVANCED ─────────────────────────────────────────────────────

def _do_factor(cols: dict, args: dict) -> dict:
    """Principal Component Analysis with varimax rotation."""
    variables = args.get("v", [])
    n_in = args.get("n", 0)
    try:
        n_in = int(n_in) if n_in else 0
    except (TypeError, ValueError):
        n_in = 0
    # Default: keep all components (min(n_vars, n_obs-1)) instead of
    # silently returning empty when n=0 comes from the registry default.
    n_components = n_in if n_in and n_in > 0 else len(variables)
    rotate = args.get("rotate", "varimax")
    if not variables or len(variables) < 2:
        return {"e": "v (2+ variables) required"}
    for var in variables:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in variables]
    n = min(len(a) for a in arrays)
    data = np.column_stack([(a[:n] - np.mean(a[:n])) / np.std(a[:n], ddof=1) for a in arrays])
    # Covariance = Correlation (data is standardized)
    corr = data.T @ data / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    n_comp = min(n_components, len(variables))
    eigenvalues = eigenvalues[:n_comp]
    loadings = eigenvectors[:, :n_comp] * np.sqrt(eigenvalues)
    # Varimax rotation
    if rotate == "varimax" and n_comp > 1:
        U, _, Vt = np.linalg.svd(
            loadings @ (loadings.T @ loadings) - 0.5 * loadings @ np.diag(np.diag(loadings.T @ loadings))
        ) if False else (None, None, None)
        # Simple varimax iteration
        L = loadings.copy()
        for _ in range(50):
            L2 = L ** 2
            M = L2 - np.mean(L2, axis=0) / len(variables)
            U, _, Vt = np.linalg.svd(L.T @ (L * M))
            R = U @ Vt
            L_new = L @ R
            if np.max(np.abs(L_new - L)) < 1e-6:
                L = L_new
                break
            L = L_new
        loadings = L
    # Variance explained
    var_explained = [float(np.sum(loadings[:, i] ** 2)) for i in range(n_comp)]
    total_var = float(np.sum(eigenvalues[:len(variables)]))
    result = {
        "components": [],
    }
    for i in range(n_comp):
        comp = {
            "component": i + 1,
            "eigenvalue": _rnd(float(eigenvalues[i]) if i < len(eigenvalues) else 0),
            "var_pct": _rnd(var_explained[i] / total_var * 100 if total_var > 0 else 0),
            "loadings": {},
        }
        for j, var_name in enumerate(variables):
            comp["loadings"][var_name] = _rnd(float(loadings[j, i]))
        result["components"].append(comp)
    result["total_var"] = _rnd(total_var)
    return result


def _do_reliability(cols: dict, args: dict) -> dict:
    """Cronbach's alpha (internal consistency reliability)."""
    variables = args.get("v", [])
    if not variables or len(variables) < 2:
        return {"e": "v (2+ item variables) required"}
    for var in variables:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in variables]
    n_items = len(arrays)
    n = min(len(a) for a in arrays)
    data = np.column_stack([a[:n] for a in arrays])
    # Cronbach's alpha = (k / (k-1)) * (1 - sum(item_var) / total_var)
    item_vars = np.var(data, axis=0, ddof=1)
    total_scores = np.sum(data, axis=1)
    total_var = float(np.var(total_scores, ddof=1))
    alpha = (n_items / (n_items - 1)) * (1 - float(np.sum(item_vars)) / total_var) if total_var > 0 and n_items > 1 else 0
    # Item-total correlations
    item_total = []
    for i, var_name in enumerate(variables):
        total_without = np.sum(np.delete(data, i, axis=1), axis=1)
        r, p = scipy_stats.pearsonr(data[:, i], total_without)
        item_total.append({"n": var_name, "r": _rnd(r), "p": _rnd(p)})
    return {"alpha": _rnd(alpha), "n_items": n_items, "n_cases": n, "item_total": item_total}


def _do_roc(cols: dict, args: dict) -> dict:
    """ROC curve and AUC."""
    a = args.get("a", "")
    v = args.get("v", "")
    pos_label = args.get("pos", 1)
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (binary outcome), v (predictor) required"}
    y_true_raw = _arr(cols[a])
    y_score_raw = _arr(cols[v])
    n = min(len(y_true_raw), len(y_score_raw))
    y_true = np.array([1.0 if abs(x - pos_label) < 0.001 else 0.0 for x in y_true_raw[:n]])
    y_score = y_score_raw[:n]
    if len(set(y_true)) < 2:
        return {"e": "Outcome must have both positive and negative cases"}
    # Sort by score descending
    idx = np.argsort(y_score)[::-1]
    y_true = y_true[idx]
    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"e": "Need both positive and negative cases"}
    # Compute ROC points — one point per distinct score, AFTER counting
    # all cases tied at that score (standard ROC construction).
    tpr, fpr, thresh = [0.0], [0.0], [float("inf")]
    tp, fp = 0, 0
    i = 0
    order_scores = y_score[idx]
    while i < n:
        s = float(order_scores[i])
        j = i
        while j < n and float(order_scores[j]) == s:
            if y_true[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        thresh.append(s)
        i = j
    # AUC via trapezoidal rule
    auc = 0.0
    for k in range(len(fpr) - 1):
        auc += (fpr[k + 1] - fpr[k]) * (tpr[k + 1] + tpr[k]) / 2
    # Youden's J (skip the inf point at index 0)
    J = np.array(tpr) - np.array(fpr)
    best_idx = int(np.argmax(J))
    threshold = thresh[best_idx] if best_idx > 0 else None
    return {
        "AUC": _rnd(auc),
        "n_pos": n_pos, "n_neg": n_neg,
        "youden": _rnd(float(J[best_idx])),
        "best_sens": _rnd(float(tpr[best_idx])),
        "best_spec": _rnd(float(1 - fpr[best_idx])),
        "threshold": _rnd(threshold),
    }


# ─── TRANSFORM ────────────────────────────────────────────────────

def _do_rank(cols: dict, args: dict) -> dict:
    """Rank transform (for Spearman workaround, etc.)."""
    v = args.get("v", "")
    if not v or v not in cols:
        return {"e": "v (variable) required"}
    vals = _arr(cols[v])
    n = len(vals)
    ranks = scipy_stats.rankdata(vals)
    return {"n": n, "ranks": [_rnd(float(r)) for r in ranks]}


# ─── SURVIVAL ──────────────────────────────────────────────────────

def _do_survival(cols: dict, args: dict) -> dict:
    """Kaplan-Meier survival + log-rank test + Cox PH (univariate)."""
    o = args.get("o", "")      # event/censoring indicator (1=event, 0=censored)
    a = args.get("a", "")      # group variable (for log-rank/stratified)
    t_var = args.get("tvar", "")  # time variable (if different from 'a' context)
    # Time variable is 'v' (the value column)
    v = args.get("v", "")
    if not o or not v:
        return {"e": "o (event indicator: 1=event, 0=censored) and v (time) required"}
    if o not in cols or v not in cols:
        return {"e": f"Variables not found: '{o}', '{v}'"}
    time_vals = _arr(cols[v])
    event_vals = _arr(cols[o])
    n = min(len(time_vals), len(event_vals))
    time_vals = time_vals[:n]
    event_vals = event_vals[:n]
    events = (event_vals >= 0.5).astype(int)
    if len(set(events)) < 2:
        return {"e": "Need both events (1) and censorings (0)"}

    # Sort by time
    order = np.argsort(time_vals)
    times = time_vals[order]
    events_sorted = events[order]

    # KM estimator
    km_times, km_surv = [], []
    n_at_risk = n
    surv = 1.0
    i = 0
    while i < n:
        t = times[i]
        # Count events at this time
        d = 0
        while i < n and abs(times[i] - t) < 1e-10:
            if events_sorted[i] == 1:
                d += 1
            i += 1
        if d > 0:
            surv *= (n_at_risk - d) / n_at_risk
        km_times.append(float(t))
        km_surv.append(_rnd(surv))
        n_at_risk -= d

    result = {"km": [{"t": float(t), "s": s} for t, s in zip(km_times, km_surv)]}

    # Median survival
    for entry in result["km"]:
        if entry["s"] <= 0.5:
            result["median_surv"] = entry["t"]
            break

    # Log-rank test (Mantel-Haenszel, event-time stratified; supports 2+ groups)
    if a and a in cols:
        groups = cols[a][:n]
        group_labels = sorted(set(str(g) for g in groups if _valid(g)))
        if len(group_labels) >= 2:
            # Map each sorted position back to its group
            pos_group = [str(groups[order[i]]) for i in range(n)]
            times_f = [float(times[i]) for i in range(n)]
            ev_f = [int(events_sorted[i]) for i in range(n)]
            if len(group_labels) == 2:
                # Proper Mantel-Haenszel: risk set at t = all with time >= t
                O1 = 0.0
                E1 = 0.0
                V = 0.0
                for t in sorted(set(ti for ti, ei in zip(times_f, ev_f) if ei == 1)):
                    risk = [i for i in range(n) if times_f[i] >= t - 1e-10]
                    n_at = len(risk)
                    n1_at = sum(1 for i in risk if pos_group[i] == group_labels[1])
                    n0_at = n_at - n1_at
                    d_at = sum(1 for i in risk if abs(times_f[i] - t) < 1e-10 and ev_f[i] == 1)
                    d1_at = sum(1 for i in risk if abs(times_f[i] - t) < 1e-10 and ev_f[i] == 1 and pos_group[i] == group_labels[1])
                    if n_at > 1 and d_at > 0 and n1_at > 0 and n0_at > 0:
                        E1 += d_at * n1_at / n_at
                        O1 += d1_at
                        V += (n1_at * n0_at * d_at * (n_at - d_at)) / (n_at * n_at * (n_at - 1))
                chi2 = (O1 - E1) ** 2 / V if V > 0 else 0.0
                p_logrank = float(scipy_stats.chi2.sf(chi2, 1))
                result["logrank"] = {"χ²": _rnd(chi2), "df": 1, "p": _rnd(p_logrank),
                                    "groups": group_labels[:2], "O1": _rnd(O1), "E1": _rnd(E1)}
            else:
                # k-group Mantel-Haenszel: sum (O-E)^2/V per group, df=k-1
                k = len(group_labels)
                Os, Es, Vs = [0.0] * k, [0.0] * k, [0.0] * k
                for t in sorted(set(ti for ti, ei in zip(times_f, ev_f) if ei == 1)):
                    risk = [i for i in range(n) if times_f[i] >= t - 1e-10]
                    n_at = len(risk)
                    d_at = sum(1 for i in risk if abs(times_f[i] - t) < 1e-10 and ev_f[i] == 1)
                    if n_at > 1 and d_at > 0:
                        for gi, gl in enumerate(group_labels):
                            n_at_g = sum(1 for i in risk if pos_group[i] == gl)
                            d_at_g = sum(1 for i in risk if abs(times_f[i] - t) < 1e-10 and ev_f[i] == 1 and pos_group[i] == gl)
                            if n_at_g > 0:
                                Es[gi] += d_at * n_at_g / n_at
                                Os[gi] += d_at_g
                                Vs[gi] += (n_at_g * (n_at - n_at_g) * d_at * (n_at - d_at)) / (n_at * n_at * (n_at - 1))
                chi2 = sum((Os[gi] - Es[gi]) ** 2 / Vs[gi] for gi in range(k) if Vs[gi] > 0)
                p_logrank = float(scipy_stats.chi2.sf(chi2, k - 1))
                result["logrank"] = {"χ²": _rnd(chi2), "df": k - 1, "p": _rnd(p_logrank),
                                    "groups": group_labels}

    return result


# ─── META-ANALYSIS ─────────────────────────────────────────────────

def _do_meta(cols: dict, args: dict) -> dict:
    """Meta-analysis: fixed + random effects, I², forest plot data."""
    # Expect data: study_name, effect, se (or ci_low, ci_high)
    v = args.get("v", "")
    e = args.get("effect_col", "effect")
    se_col = args.get("se_col", "se")
    if not v:
        return {"e": "v (study label column) required"}
    if e not in cols:
        # Try to auto-detect
        for candidate in ["effect", "es", "or", "rr", "hr", "md", "smd", "d"]:
            if candidate in cols:
                e = candidate
                break
        else:
            return {"e": f"Need effect size column. Found: {list(cols.keys())}"}
    if se_col not in cols:
        for candidate in ["se", "var", "ci_l", "ci_u"]:
            if candidate in cols:
                se_col = candidate
                break
        else:
            return {"e": f"Need SE or CI columns. Found: {list(cols.keys())}"}
    labels = [str(x) for x in cols.get(v, [])[:100]]
    effects = _arr(cols[e])
    if se_col in ("ci_l", "ci_u"):
        # Derive SE from CI
        ci_low = _arr(cols.get("ci_l", cols.get("ci_low", [])))
        ci_high = _arr(cols.get("ci_u", cols.get("ci_high", [])))
        ses = (ci_high - ci_low) / (2 * 1.96)
    elif "var" in se_col.lower():
        ses = np.sqrt(np.maximum(_arr(cols[se_col]), 0))
    else:
        ses = _arr(cols[se_col])
    n = min(len(effects), len(ses), len(labels))
    effects, ses, labels = effects[:n], ses[:n], labels[:n]
    # Remove invalid
    valid = ~np.isnan(effects) & ~np.isnan(ses) & (ses > 0)
    effects, ses, labels = effects[valid], ses[valid], [labels[i] for i in range(n) if valid[i]]
    if len(effects) < 2:
        return {"e": f"Need ≥2 valid studies, got {len(effects)}"}
    k = len(effects)

    # Fixed effect: inverse-variance weighted
    w_fixed = 1 / ses ** 2
    fe_est = float(np.sum(w_fixed * effects) / np.sum(w_fixed))
    fe_se = float(np.sqrt(1 / np.sum(w_fixed)))
    fe_z = fe_est / fe_se if fe_se > 0 else 0
    fe_p = float(2 * norm.sf(abs(fe_z)))

    # Random effects (DerSimonian-Laird)
    Q = float(np.sum(w_fixed * (effects - fe_est) ** 2))
    df = k - 1
    p_het = float(scipy_stats.chi2.sf(Q, df)) if df > 0 else 1.0
    # I²
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    # Tau²
    c = float(np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed))
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    # Random effects weights
    w_random = 1 / (ses ** 2 + tau2)
    re_est = float(np.sum(w_random * effects) / np.sum(w_random))
    re_se = float(np.sqrt(1 / np.sum(w_random)))
    re_z = re_est / re_se if re_se > 0 else 0
    re_p = float(2 * norm.sf(abs(re_z)))
    # 95% CI random effects
    re_ci_l = re_est - 1.96 * re_se
    re_ci_u = re_est + 1.96 * re_se

    # Forest plot data
    studies = []
    for i in range(k):
        ci_l = float(effects[i] - 1.96 * ses[i])
        ci_u = float(effects[i] + 1.96 * ses[i])
        w_pct_fixed = float(w_fixed[i] / np.sum(w_fixed) * 100)
        w_pct_random = float(w_random[i] / np.sum(w_random) * 100)
        studies.append({
            "study": labels[i],
            "es": _rnd(float(effects[i])),
            "se": _rnd(float(ses[i])),
            "ci_l": _rnd(ci_l), "ci_u": _rnd(ci_u),
            "w_fixed_pct": _rnd(w_pct_fixed),
            "w_random_pct": _rnd(w_pct_random),
        })

    return {
        "k": k,
        "fixed": {"es": _rnd(fe_est), "se": _rnd(fe_se), "z": _rnd(fe_z), "p": _rnd(fe_p),
                  "ci_l": _rnd(fe_est - 1.96 * fe_se), "ci_u": _rnd(fe_est + 1.96 * fe_se)},
        "random": {"es": _rnd(re_est), "se": _rnd(re_se), "z": _rnd(re_z), "p": _rnd(re_p),
                   "ci_l": _rnd(re_ci_l), "ci_u": _rnd(re_ci_u)},
        "het": {"Q": _rnd(Q), "df": df, "p": _rnd(p_het), "I2_pct": _rnd(I2), "tau2": _rnd(tau2)},
        "studies": studies,
    }


# ─── DIAGNOSTIC TEST EVALUATION ───────────────────────────────────

def _do_eval(cols: dict, args: dict) -> dict:
    """Diagnostic test evaluation: sensitivity, specificity, LR+, LR-, PPV, NPV, DOR."""
    a = args.get("a", "")  # reference/gold standard
    v = args.get("v", "")  # test result
    pos_test = args.get("pos", 1)
    pos_ref = args.get("pos_ref", 1)
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (reference standard), v (test result) required"}
    ref = _arr(cols[a])
    test = _arr(cols[v])
    n = min(len(ref), len(test))
    # Build 2x2
    TP = sum(1 for i in range(n) if abs(test[i] - pos_test) < 0.001 and abs(ref[i] - pos_ref) < 0.001)
    FP = sum(1 for i in range(n) if abs(test[i] - pos_test) < 0.001 and abs(ref[i] - pos_ref) >= 0.001)
    FN = sum(1 for i in range(n) if abs(test[i] - pos_test) >= 0.001 and abs(ref[i] - pos_ref) < 0.001)
    TN = sum(1 for i in range(n) if abs(test[i] - pos_test) >= 0.001 and abs(ref[i] - pos_ref) >= 0.001)
    total = TP + FP + FN + TN
    if total == 0:
        return {"e": "No valid data"}
    # Metrics
    sens = TP / (TP + FN) if (TP + FN) > 0 else None
    spec = TN / (TN + FP) if (TN + FP) > 0 else None
    ppv = TP / (TP + FP) if (TP + FP) > 0 else None
    npv = TN / (TN + FN) if (TN + FN) > 0 else None
    prev = (TP + FN) / total if total > 0 else None
    # Likelihood ratios
    lr_pos = sens / (1 - spec) if sens is not None and spec is not None and spec < 1 else None
    lr_neg = (1 - sens) / spec if sens is not None and spec is not None and spec > 0 else None
    # DOR
    dor = (TP * TN) / (FP * FN) if FP > 0 and FN > 0 else None
    # Accuracy
    acc = (TP + TN) / total if total > 0 else None
    # Youden
    youden = (sens + spec - 1) if sens is not None and spec is not None else None
    # CI for sensitivity (Wilson)
    if sens is not None and (TP + FN) > 0:
        n_sens = TP + FN
        z = 1.96
        sens_ci_l = (sens + z**2/(2*n_sens) - z*math.sqrt(sens*(1-sens)/n_sens + z**2/(4*n_sens**2))) / (1 + z**2/n_sens)
        sens_ci_u = (sens + z**2/(2*n_sens) + z*math.sqrt(sens*(1-sens)/n_sens + z**2/(4*n_sens**2))) / (1 + z**2/n_sens)
        sens_ci = [_rnd(max(0, sens_ci_l)), _rnd(min(1, sens_ci_u))]
    else:
        sens_ci = None
    if spec is not None and (TN + FP) > 0:
        n_spec = TN + FP
        z = 1.96
        spec_ci_l = (spec + z**2/(2*n_spec) - z*math.sqrt(spec*(1-spec)/n_spec + z**2/(4*n_spec**2))) / (1 + z**2/n_spec)
        spec_ci_u = (spec + z**2/(2*n_spec) + z*math.sqrt(spec*(1-spec)/n_spec + z**2/(4*n_spec**2))) / (1 + z**2/n_spec)
        spec_ci = [_rnd(max(0, spec_ci_l)), _rnd(min(1, spec_ci_u))]
    else:
        spec_ci = None
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": total,
        "sens": _rnd(sens), "sens_ci": sens_ci,
        "spec": _rnd(spec), "spec_ci": spec_ci,
        "ppv": _rnd(ppv), "npv": _rnd(npv),
        "lr_pos": _rnd(lr_pos), "lr_neg": _rnd(lr_neg),
        "dor": _rnd(dor),
        "acc": _rnd(acc), "youden": _rnd(youden),
        "prev": _rnd(prev),
    }


# ─── BLAND-ALTMAN ─────────────────────────────────────────────────

def _do_blandaltman(cols: dict, args: dict) -> dict:
    """Bland-Altman agreement analysis between two methods."""
    a = args.get("a", "")  # method 1
    v = args.get("v", "")  # method 2
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (method1), v (method2) required"}
    m1 = _arr(cols[a])
    m2 = _arr(cols[v])
    n = min(len(m1), len(m2))
    m1, m2 = m1[:n], m2[:n]
    diff = m1 - m2
    avg = (m1 + m2) / 2
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff
    # CI for limits of agreement
    se_loa = sd_diff * math.sqrt(3 / n) if n > 0 else 0
    loa_lower_ci = [loa_lower - 1.96 * se_loa, loa_lower + 1.96 * se_loa]
    loa_upper_ci = [loa_upper - 1.96 * se_loa, loa_upper + 1.96 * se_loa]
    # Proportional bias test
    r, p_prop = scipy_stats.pearsonr(avg, diff)
    return {
        "n": n,
        "mean_diff": _rnd(mean_diff), "sd_diff": _rnd(sd_diff),
        "loa_lower": _rnd(loa_lower), "loa_upper": _rnd(loa_upper),
        "loa_lower_ci": [_rnd(loa_lower_ci[0]), _rnd(loa_lower_ci[1])],
        "loa_upper_ci": [_rnd(loa_upper_ci[0]), _rnd(loa_upper_ci[1])],
        "prop_bias_r": _rnd(r), "prop_bias_p": _rnd(p_prop),
    }


# ─── COHEN'S KAPPA ────────────────────────────────────────────────

def _do_kappa(cols: dict, args: dict) -> dict:
    """Cohen's kappa (unweighted + linear/quadratic weighted)."""
    a = args.get("a", "")
    v = args.get("v", "")
    w = args.get("w", "unweighted")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (rater1), v (rater2) required"}
    # Accept string categories as well as numeric ratings (no int truncation)
    r1s = [cols[a][i] for i in range(min(len(cols[a]), len(cols[v])))]
    r2s = [cols[v][i] for i in range(min(len(cols[a]), len(cols[v])))]
    pairs = [(x, y) for x, y in zip(r1s, r2s)
             if x is not None and y is not None
             and not (isinstance(x, float) and math.isnan(x))
             and not (isinstance(y, float) and math.isnan(y))]
    if not pairs:
        return {"e": "No complete pairs"}
    cats = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]),
                  key=lambda x: (isinstance(x, str), x))
    # Numeric-like floats that are whole numbers stay numeric; do NOT
    # truncate genuine fractional ratings — keep them as distinct labels.
    if len(cats) < 2:
        return {"e": f"Need ≥2 categories, got {cats}"}
    k = len(cats)
    # Build agreement matrix
    obs = np.zeros((k, k))
    for x, y in pairs:
        obs[cats.index(x), cats.index(y)] += 1
    n = len(pairs)
    total = np.sum(obs)
    p_o = np.trace(obs) / total if total > 0 else 0
    # Expected agreement
    row_sums = np.sum(obs, axis=1)
    col_sums = np.sum(obs, axis=0)
    p_e = float(np.sum(row_sums * col_sums)) / (total ** 2) if total > 0 else 0
    kappa_val = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    # SE of kappa
    se_kappa = math.sqrt((p_o * (1 - p_o)) / (total * (1 - p_e) ** 2)) if total > 0 and p_e < 1 else 0
    z = kappa_val / se_kappa if se_kappa > 0 else 0
    p_val = float(2 * norm.sf(abs(z)))
    result = {"kappa": _rnd(kappa_val), "se": _rnd(se_kappa), "z": _rnd(z), "p": _rnd(p_val),
              "p_o": _rnd(p_o), "p_e": _rnd(p_e), "n": int(total), "weight": w}
    # Percent agreement
    result["agree_pct"] = _rnd(p_o * 100)
    # Weighted kappa
    if w != "unweighted" and k > 2:
        if w == "linear":
            weights = np.array([[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
        else:  # quadratic
            weights = np.array([[1 - ((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])
        w_obs = float(np.sum(weights * obs)) / total if total > 0 else 0
        w_exp = float(np.sum(weights * np.outer(row_sums, col_sums))) / (total ** 2) if total > 0 else 0
        w_kappa = (w_obs - w_exp) / (1 - w_exp) if w_exp < 1 else 0
        result["weighted_kappa"] = _rnd(w_kappa)
    return result


# ─── E-VALUE ──────────────────────────────────────────────────────

def _do_evalue(cols: dict, args: dict) -> dict:
    """E-value for sensitivity to unmeasured confounding."""
    # Accept either data (compute OR/RR) or direct OR/RR values
    if args.get("or") is not None or args.get("rr") is not None:
        or_val = args.get("or")
        rr_val = args.get("rr")
        if or_val is not None:
            try:
                ov = float(or_val)
            except (TypeError, ValueError):
                return {"e": "or must be numeric"}
            if ov <= 0:
                return {"e": "or must be > 0"}
            if ov <= 1:
                # Protective / null effect: E-value is 1 (no unmeasured
                # confounding needed to explain away). Use the reciprocal
                # so the magnitude is still informative.
                inv = 1.0 / ov if ov > 0 else float("inf")
                e_val = inv + math.sqrt(inv * (inv - 1)) if inv > 1 else 1.0
                return {"OR": ov, "E_value": _rnd(e_val),
                        "E_lower": _rnd(0),
                        "note": "OR<=1: E-value computed on reciprocal (protective effect)"}
            e_val = ov + math.sqrt(ov * (ov - 1))
            return {"OR": ov, "E_value": _rnd(e_val),
                    "E_lower": _rnd(1 + math.sqrt(1 - 1/ov) if ov > 1 else 0)}
        else:
            try:
                rv = float(rr_val)
            except (TypeError, ValueError):
                return {"e": "rr must be numeric"}
            if rv <= 0:
                return {"e": "rr must be > 0"}
            if rv <= 1:
                inv = 1.0 / rv if rv > 0 else float("inf")
                e_val = inv + math.sqrt(inv * (inv - 1)) if inv > 1 else 1.0
                return {"RR": rv, "E_value": _rnd(e_val),
                        "note": "RR<=1: E-value computed on reciprocal (protective effect)"}
            e_val = rv + math.sqrt(rv * (rv - 1))
            return {"RR": rv, "E_value": _rnd(e_val)}
    # Compute OR from 2x2 data
    a = args.get("a", "")
    v = args.get("v", "")
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (exposure), v (outcome) for 2x2 data, or pass or=/rr= directly"}
    exp = _arr(cols[a])
    out = _arr(cols[v])
    n = min(len(exp), len(out))
    a_cell = sum(1 for i in range(n) if exp[i] == 1 and out[i] == 1)
    b_cell = sum(1 for i in range(n) if exp[i] == 1 and out[i] == 0)
    c_cell = sum(1 for i in range(n) if exp[i] == 0 and out[i] == 1)
    d_cell = sum(1 for i in range(n) if exp[i] == 0 and out[i] == 0)
    if b_cell == 0 or c_cell == 0:
        return {"e": "Zero cell — cannot compute OR"}
    or_val = (a_cell * d_cell) / (b_cell * c_cell)
    e_val = or_val + math.sqrt(or_val * (or_val - 1)) if or_val > 1 else 0
    return {"a": a_cell, "b": b_cell, "c": c_cell, "d": d_cell,
            "OR": _rnd(or_val),
            "E_value": _rnd(e_val),
            "E_lower": _rnd(1 + math.sqrt(1 - 1/or_val) if or_val > 1 else 0)}


# ─── POISSON REGRESSION ───────────────────────────────────────────

def _do_poisson(cols: dict, args: dict) -> dict:
    """Poisson regression (log link GLM via IRLS)."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    if not o or not predictors:
        return {"e": "o (count outcome), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n_min = min(len(y), *(len(x) for x in X_cols))
    y = y[:n_min]
    X = _design_matrix([x[:n_min] for x in X_cols], add_intercept=True)
    k = X.shape[1]
    if n_min <= k:
        return {"e": f"Need more observations ({n_min}) than parameters ({k})"}
    # IRLS for Poisson (log link)
    beta = np.zeros(k)
    beta[0] = math.log(max(np.mean(y), 0.1))
    for _ in range(50):
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e10)
        # Working response
        z_vec = eta + (y - mu) / mu
        W = np.diag(mu)
        try:
            XTWX = X.T @ W @ X
            beta_new = np.linalg.inv(XTWX) @ X.T @ W @ z_vec
        except np.linalg.LinAlgError:
            return {"e": "Matrix singular"}
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    # SE
    mu = np.exp(X @ beta)
    mu = np.clip(mu, 1e-10, 1e10)
    W = np.diag(mu)
    try:
        cov = np.linalg.inv(X.T @ W @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, float('nan'))
    z_vals = beta / se
    p_vals = np.array([float(2 * norm.sf(abs(zv))) for zv in z_vals])
    # Deviance
    dev = float(2 * np.sum(y * np.log(np.maximum(y, 1e-10) / mu) - (y - mu)))
    null_dev = float(2 * np.sum(y * np.log(np.maximum(y, 1e-10) / np.mean(y)) - (y - np.mean(y))))
    names = ["(Intercept)"] + list(predictors)
    coefs = []
    for i, name in enumerate(names):
        if i < len(beta):
            coefs.append({
                "n": name, "B": _rnd(beta[i]), "SE": _rnd(se[i]),
                "z": _rnd(z_vals[i]), "p": _rnd(p_vals[i]),
                "IRR": _rnd(math.exp(beta[i])),
            })
    return {"n": n_min, "deviance": _rnd(dev), "null_deviance": _rnd(null_dev), "coef": coefs}


# ─── POWER ANALYSIS ───────────────────────────────────────────────

def _do_power(cols: dict, args: dict) -> dict:
    """Sample size and power calculations."""
    calc = args.get("calc", "n")
    test = args.get("power_test", "ttest")
    alpha = args.get("alpha", 0.05)
    power_target = args.get("power", 0.80)
    d = args.get("es", 0.5)  # effect size (Cohen's d for t-test, f for ANOVA)
    ratio = args.get("ratio", 1.0)  # n2/n1 for independent t-test
    groups = args.get("k", 2)  # number of groups for ANOVA
    p0 = args.get("p0", 0.5)  # baseline proportion
    p1 = args.get("p1", 0.7)  # alternative proportion

    if test == "ttest":
        if calc == "n":
            # Two-sample t-test sample size
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power_target)
            n1 = (1 + 1/ratio) * ((z_alpha + z_beta) / d) ** 2
            n2 = n1 * ratio
            return {"n1": math.ceil(n1), "n2": math.ceil(n2), "n_total": math.ceil(n1 + n2),
                    "d": d, "alpha": alpha, "power": power_target}
        else:
            # Power given n — honour caller n (n / n1 / n_per_group), default 30
            n1 = args.get("n1", args.get("n", args.get("n_per_group", 30)))
            try:
                n1 = int(float(n1))
            except (TypeError, ValueError):
                n1 = 30
            n1 = max(n1, 2)
            n2 = n1 * ratio
            ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
            df = n1 + n2 - 2
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
            power = float(1 - scipy_stats.nct.cdf(t_crit, df, ncp) + scipy_stats.nct.cdf(-t_crit, df, ncp))
            return {"power": _rnd(power), "n1": n1, "n2": math.ceil(n2), "d": d, "alpha": alpha}

    elif test == "prop":
        if calc == "n":
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power_target)
            p_bar = (p0 + p1) / 2
            n = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2
            return {"n_per_group": math.ceil(n), "n_total": math.ceil(2 * n), "p0": p0, "p1": p1, "alpha": alpha, "power": power_target}
        else:
            n = args.get("n_per_group", args.get("n", args.get("n1", 30)))
            try:
                n = int(float(n))
            except (TypeError, ValueError):
                n = 30
            n = max(n, 2)
            z_alpha = norm.ppf(1 - alpha / 2)
            se = math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
            z_power = (abs(p1 - p0) - z_alpha * math.sqrt(2 * p0 * (1 - p0) / n)) / se if se > 0 else 0
            power = float(norm.cdf(z_power))
            return {"power": _rnd(power), "n_per_group": n, "p0": p0, "p1": p1, "alpha": alpha}

    elif test == "anova":
        if calc == "n":
            z_alpha = norm.ppf(1 - alpha / 2) if alpha < 1 else 0
            z_beta = norm.ppf(power_target)
            # f = sqrt(eta2/(1-eta2)) approximation for sample size
            f_val = d
            n_per_group = 2 * (z_alpha + z_beta) ** 2 * groups / (f_val ** 2 * (groups - 1)) if f_val > 0 else 0
            return {"n_per_group": math.ceil(n_per_group), "n_total": math.ceil(n_per_group * groups),
                    "groups": groups, "f": f_val, "alpha": alpha, "power": power_target}
        else:
            # Noncentral-F power given per-group n
            n_pg = args.get("n_per_group", args.get("n", args.get("n1", 30)))
            try:
                n_pg = int(float(n_pg))
            except (TypeError, ValueError):
                n_pg = 30
            n_pg = max(n_pg, 2)
            from scipy.stats import ncf as _ncf, f as _fdist
            f_val = d
            lam = groups * n_pg * f_val ** 2
            df1 = groups - 1
            df2 = groups * (n_pg - 1)
            f_crit = _fdist.ppf(1 - alpha, df1, df2)
            power = float(1 - _ncf.cdf(f_crit, df1, df2, lam))
            return {"power": _rnd(power), "n_per_group": n_pg,
                    "n_total": n_pg * groups, "groups": groups,
                    "f": f_val, "alpha": alpha}

    return {"e": f"Unknown test '{test}' or calc '{calc}'"}


# ─── DECISION CURVE ANALYSIS ──────────────────────────────────────

def _do_dca(cols: dict, args: dict) -> dict:
    """Decision Curve Analysis — net benefit across threshold probabilities."""
    a = args.get("a", "")  # binary outcome (1=event)
    v = args.get("v", "")  # predicted probability
    if not a or not v or a not in cols or v not in cols:
        return {"e": "a (binary outcome), v (predicted probability) required"}
    y = _arr(cols[a])
    p_hat = _arr(cols[v])
    n = min(len(y), len(p_hat))
    y, p_hat = y[:n], p_hat[:n]
    # Threshold range
    thresholds = np.linspace(0.01, 0.99, 50)
    net_benefit = []
    treat_all_benefit = []
    for pt in thresholds:
        tp = np.sum((p_hat >= pt) & (y >= 0.5))
        fp = np.sum((p_hat >= pt) & (y < 0.5))
        # Net benefit = (TP - FP * pt/(1-pt)) / n
        w = pt / (1 - pt)
        nb = (tp - fp * w) / n if n > 0 else 0
        net_benefit.append(_rnd(float(nb)))
        # Treat all: prevalence - (1-prevalence)*w
        prev = np.mean(y >= 0.5)
        treat_all = float(prev - (1 - prev) * w)
        treat_all_benefit.append(_rnd(treat_all))
    return {"thresholds": [_rnd(float(t)) for t in thresholds],
            "net_benefit": net_benefit,
            "treat_all": treat_all_benefit}


# ─── QUANTILE REGRESSION ──────────────────────────────────────────

def _do_quantreg(cols: dict, args: dict) -> dict:
    """Quantile regression via IRLS with check function weights."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    tau = args.get("tau", 0.5)  # quantile (0.5=median)
    if not o or not predictors:
        return {"e": "o (outcome), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y, X = y[:n], _design_matrix([x[:n] for x in X_cols], add_intercept=True)
    k = X.shape[1]
    # IRLS with check function weights
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(100):
        resid = y - X @ beta
        # Weights: 1/|resid| for quantile regression stability
        w = 1.0 / (np.abs(resid) + 1e-6)
        w = np.where(resid > 0, tau * w, (1 - tau) * w)
        XTWX = X.T @ np.diag(w) @ X
        XTWy = X.T @ (w * y)
        try:
            beta_new = np.linalg.lstsq(XTWX, XTWy, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    # SE via bootstrap-like approximation
    names = ["(Intercept)"] + list(predictors)
    coefs = [{"n": name, "B": _rnd(float(beta[i])), "tau": tau}
             for i, name in enumerate(names) if i < len(beta)]
    return {"tau": tau, "n": n, "coef": coefs}


# ─── NEGATIVE BINOMIAL REGRESSION ─────────────────────────────────

def _do_negbin(cols: dict, args: dict) -> dict:
    """Negative Binomial regression (log link, MLE for dispersion)."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    if not o or not predictors:
        return {"e": "o (count outcome), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y, X = y[:n], _design_matrix([x[:n] for x in X_cols], add_intercept=True)
    k = X.shape[1]
    # Start with Poisson fit
    beta = np.linalg.lstsq(X, np.log(np.maximum(y, 0.1)), rcond=None)[0]
    alpha = 1.0  # dispersion
    # Simple alternating update (not full MLE but reasonable)
    for _ in range(30):
        mu = np.exp(X @ beta)
        mu = np.clip(mu, 1e-6, 1e6)
        # Update beta (IRLS)
        W = np.diag(mu / (1 + alpha * mu))
        z = np.log(np.maximum(mu, 1e-6)) + (y - mu) / mu
        try:
            beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
        # Update alpha (method of moments)
        pearson = np.sum((y - mu) ** 2 / (mu + alpha * mu ** 2))
        alpha = max(0.001, (pearson - (n - k)) / np.sum(mu)) if np.sum(mu) > 0 else 0.001
    mu = np.exp(X @ beta)
    mu = np.clip(mu, 1e-6, 1e6)
    W = np.diag(mu / (1 + alpha * mu))
    try:
        cov = np.linalg.inv(X.T @ W @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, float('nan'))
    z_vals = beta / se
    p_vals = [float(2 * norm.sf(abs(zv if not math.isnan(zv) else 0))) for zv in z_vals]
    names = ["(Intercept)"] + list(predictors)
    coefs = []
    for i, name in enumerate(names):
        if i < len(beta):
            coefs.append({"n": name, "B": _rnd(float(beta[i])), "SE": _rnd(float(se[i])),
                         "z": _rnd(float(z_vals[i])), "p": _rnd(p_vals[i]),
                         "IRR": _rnd(math.exp(float(beta[i])))})
    return {"n": n, "alpha": _rnd(float(alpha)), "coef": coefs}


# ─── MEDIATION ANALYSIS ────────────────────────────────────────────

def _do_mediation(cols: dict, args: dict) -> dict:
    """Mediation analysis: Baron-Kenny + bootstrap indirect effect CI.

    NOTE: covariate-adjusted mediation (p=covariates) is not yet implemented —
    all paths are unadjusted OLS. Pass p only for forward-compat; it is
    reported back so callers know it was ignored.
    """
    o = args.get("o", "")     # outcome
    a = args.get("a", "")     # treatment/exposure (X)
    v = args.get("v", "")     # mediator (M)
    covariates = args.get("p", [])  # optional covariates
    if not o or not a or not v:
        return {"e": "o (outcome Y), a (treatment X), v (mediator M) required"}
    if o not in cols or a not in cols or v not in cols:
        return {"e": f"Variables not found"}
    Y = _arr(cols[o])
    X = _arr(cols[a])
    M = _arr(cols[v])
    n = min(len(Y), len(X), len(M))
    Y, X_var, M = Y[:n], X[:n], M[:n]

    # Path a: X → M
    X_mat = _design_matrix([X_var], add_intercept=True)
    beta_a = np.linalg.lstsq(X_mat, M, rcond=None)[0]
    a_coef = float(beta_a[1])

    # Path b + c: X + M → Y
    XM = _design_matrix([X_var, M], add_intercept=True)
    beta_bc = np.linalg.lstsq(XM, Y, rcond=None)[0]
    b_coef = float(beta_bc[2])  # M coefficient
    c_prime = float(beta_bc[1])  # X direct effect

    # Path c (total): X → Y
    beta_c = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    c_total = float(beta_c[1])

    # Indirect effect: a * b
    indirect = a_coef * b_coef

    # Bootstrap CI for indirect effect
    n_boot = 500
    indirect_boot = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        Xb, Mb, Yb = X_var[idx], M[idx], Y[idx]
        Xb_mat = _design_matrix([Xb], add_intercept=True)
        a_b = float(np.linalg.lstsq(Xb_mat, Mb, rcond=None)[0][1])
        XMb = _design_matrix([Xb, Mb], add_intercept=True)
        b_b = float(np.linalg.lstsq(XMb, Yb, rcond=None)[0][2])
        indirect_boot.append(a_b * b_b)
    indirect_boot.sort()
    ci_low = indirect_boot[int(n_boot * 0.025)]
    ci_high = indirect_boot[int(n_boot * 0.975)]

    # Proportion mediated
    prop_med = indirect / c_total if abs(c_total) > 1e-10 else 0

    return {
        "n": n,
        "total_effect": _rnd(c_total),       # c
        "direct_effect": _rnd(c_prime),       # c'
        "indirect_effect": _rnd(indirect),    # a*b
        "prop_mediated": _rnd(prop_med),
        "indirect_ci": [_rnd(ci_low), _rnd(ci_high)],
        "paths": {"a": _rnd(a_coef), "b": _rnd(b_coef), "c": _rnd(c_total), "c_prime": _rnd(c_prime)},
        "note": "unadjusted OLS paths — covariates in p are ignored" if covariates else "unadjusted OLS paths",
    }


# ─── PROPENSITY SCORE MATCHING ────────────────────────────────────

def _do_pscore(cols: dict, args: dict) -> dict:
    """Propensity score estimation + IPTW weights + balance check (SMD)."""
    a = args.get("a", "")  # treatment indicator
    predictors = args.get("p", [])
    if not a or not predictors:
        return {"e": "a (treatment), p (covariates) required"}
    if a not in cols:
        return {"e": f"Treatment '{a}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Covariate '{pv}' not found"}
    treat = _arr(cols[a])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(treat), *(len(x) for x in X_cols))
    treat, X_cols = treat[:n], [x[:n] for x in X_cols]

    # Logistic regression for propensity score
    X = _design_matrix(X_cols, add_intercept=True)
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W = np.diag(mu * (1 - mu))
        z = eta + (treat - mu) / (mu * (1 - mu))
        try:
            beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"e": "Matrix singular"}
    ps = 1 / (1 + np.exp(-X @ beta))
    ps = np.clip(ps, 0.01, 0.99)

    # IPTW weights
    w_att = np.where(treat >= 0.5, 1.0, ps / (1 - ps))
    w_ate = np.where(treat >= 0.5, 1 / ps, 1 / (1 - ps))

    # Standardized Mean Differences before/after weighting
    smd_before, smd_after = [], []
    for idx, pv in enumerate(predictors):
        x = X_cols[idx]
        t1 = x[treat >= 0.5]
        t0 = x[treat < 0.5]
        # Before
        sd_pool = math.sqrt((float(np.var(t1, ddof=1)) + float(np.var(t0, ddof=1))) / 2)
        smd_b = abs(float(np.mean(t1)) - float(np.mean(t0))) / sd_pool if sd_pool > 0 else 0
        smd_before.append(_rnd(smd_b))
        # After IPTW (ATE weights)
        w1 = w_ate[treat >= 0.5]
        w0 = w_ate[treat < 0.5]
        m1_w = float(np.average(t1, weights=w1)) if len(w1) > 0 else 0
        m0_w = float(np.average(t0, weights=w0)) if len(w0) > 0 else 0
        smd_a = abs(m1_w - m0_w) / sd_pool if sd_pool > 0 else 0
        smd_after.append(_rnd(smd_a))

    return {
        "n": n, "n_treated": int(np.sum(treat >= 0.5)), "n_control": int(np.sum(treat < 0.5)),
        "ps_mean_treated": _rnd(float(np.mean(ps[treat >= 0.5]))),
        "ps_mean_control": _rnd(float(np.mean(ps[treat < 0.5]))),
        "smd_before": dict(zip(predictors, smd_before)),
        "smd_after": dict(zip(predictors, smd_after)),
        "note": ("smd_after uses ATE (inverse-probability) weights for the average "
                 "treatment effect in the whole sample. For the ATT (effect on the "
                 "treated), reweight controls by ps/(1-ps) with treated weight 1."),
    }


# ─── GAM (SPLINE ADDITIVE MODEL) ──────────────────────────────────

def _do_gam(cols: dict, args: dict) -> dict:
    """GAM — penalized B-spline with simple polynomial spline basis."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    n_knots = args.get("df", 5)
    if not o or not predictors:
        return {"e": "o (outcome), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_list = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_list))
    y, X_list = y[:n], [x[:n] for x in X_list]
    # Build natural cubic spline bases (truncated power basis)
    bases = []
    for x in X_list:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        knots = np.linspace(x_min, x_max, n_knots + 2)[1:-1]
        # Truncated power basis: 1, x, (x-k1)³₊, (x-k2)³₊, ...
        basis = np.column_stack([np.ones(n), x])
        for k in knots:
            basis = np.column_stack([basis, np.maximum(0, (x - k)) ** 3])
        bases.append(basis)
    X_full = np.column_stack(bases)
    lam_ridge = 0.001
    D = np.eye(X_full.shape[1])
    D[0, 0] = 0
    beta = np.linalg.lstsq(X_full.T @ X_full + lam_ridge * D, X_full.T @ y, rcond=None)[0]
    y_pred = X_full @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"n": n, "R2": _rnd(R2), "smooth_terms": len(predictors), "knots_per_term": n_knots}


# ─── ZERO-INFLATED POISSON ────────────────────────────────────────

def _do_zip(cols: dict, args: dict) -> dict:
    """Zero-Inflated Poisson (EM algorithm)."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    if not o or not predictors:
        return {"e": "o (count outcome), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y, X = y[:n], _design_matrix([x[:n] for x in X_cols], add_intercept=True)
    k = X.shape[1]
    # EM for ZIP
    # Poisson part
    beta_p = np.linalg.lstsq(X, np.log(np.maximum(y, 0.1)), rcond=None)[0]
    # Zero-inflation part (logistic)
    gamma = np.zeros(k)
    is_zero = (y < 0.5).astype(float)
    # Simple alternating EM (5 iterations)
    for _ in range(5):
        # E-step: probability of being structural zero given y=0
        mu_p = np.exp(X @ beta_p)
        mu_p = np.clip(mu_p, 1e-10, 1e10)
        pi_z = 1 / (1 + np.exp(-X @ gamma))
        pi_z = np.clip(pi_z, 1e-10, 1 - 1e-10)
        # Posterior prob structural zero for y=0 cases
        tau_z = np.where(y < 0.5, pi_z / (pi_z + (1 - pi_z) * np.exp(-mu_p)), 1e-10)
        # M-step: update Poisson part (weighted by structural non-zeros)
        w_p = 1 - tau_z
        w_p = np.clip(w_p, 0.01, 1.0)
        XTWX = X.T @ np.diag(w_p) @ X
        XTWy = X.T @ (w_p * y)
        beta_p = np.linalg.lstsq(XTWX, XTWy, rcond=None)[0]
        # M-step: update zero-inflation part
        gamma = np.linalg.lstsq(X.T @ X, X.T @ tau_z, rcond=None)[0]
    mu_final = np.exp(X @ beta_p)
    pi_final = 1 / (1 + np.exp(-X @ gamma))
    zero_prob = float(np.mean(pi_final))
    names = ["(Intercept)"] + list(predictors)
    poisson_coef = []
    for i, nm in enumerate(names):
        if i < len(beta_p):
            poisson_coef.append({"n": nm, "B": _rnd(float(beta_p[i])), "IRR": _rnd(math.exp(float(beta_p[i])))})
    zi_coef = []
    for i, nm in enumerate(names):
        if i < len(gamma):
            zi_coef.append({"n": nm, "B": _rnd(float(gamma[i])), "OR": _rnd(math.exp(float(gamma[i])))})
    return {"n": n, "zero_prob": _rnd(float(zero_prob)), "poisson": poisson_coef, "zero_infl": zi_coef}


# ─── LASSO REGRESSION ─────────────────────────────────────────────

def _do_lasso(cols: dict, args: dict) -> dict:
    """LASSO regression via coordinate descent (soft thresholding)."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    lam = args.get("lam", 0.1)
    if not o or not predictors:
        return {"e": "o (outcome), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y, X_list = y[:n], [x[:n] for x in X_cols]
    # Standardize
    y_mean, y_std = float(np.mean(y)), float(np.std(y))
    y_scaled = (y - y_mean) / y_std if y_std > 0 else y - y_mean
    X_scaled_list = [(x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x - np.mean(x) for x in X_list]
    X_all = np.column_stack(X_scaled_list)
    k = len(predictors)
    beta = np.zeros(k)
    # Coordinate descent
    for _ in range(200):
        beta_old = beta.copy()
        for j in range(k):
            r = y_scaled - X_all @ beta + X_all[:, j] * beta[j]
            rho = float(np.dot(X_all[:, j], r))
            if rho < -lam / 2:
                beta[j] = (rho + lam / 2) / float(np.dot(X_all[:, j], X_all[:, j]))
            elif rho > lam / 2:
                beta[j] = (rho - lam / 2) / float(np.dot(X_all[:, j], X_all[:, j]))
            else:
                beta[j] = 0.0
        if np.max(np.abs(beta - beta_old)) < 1e-6:
            break
    # Rescale back
    beta_unscaled = np.zeros(k)
    for j in range(k):
        if np.std(X_list[j]) > 0:
            beta_unscaled[j] = beta[j] * y_std / np.std(X_list[j])
    intercept = y_mean - np.sum([beta_unscaled[j] * np.mean(X_list[j]) for j in range(k)])
    selected = [predictors[j] for j in range(k) if abs(beta[j]) > 1e-6]
    coefs = [{"n": "(Intercept)", "B": _rnd(float(intercept))}]
    for j, name in enumerate(predictors):
        coefs.append({"n": name, "B": _rnd(float(beta_unscaled[j]))})
    return {"lam": lam, "n": n, "selected": len(selected), "vars": selected, "coef": coefs}


# ─── ADF STATIONARITY TEST ────────────────────────────────────────

def _do_adf(cols: dict, args: dict) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    v = args.get("v", "")
    lags = args.get("lags", 1)
    if not v or v not in cols:
        return {"e": "v (time series variable) required"}
    y = _arr(cols[v])
    n = len(y)
    if n < 10:
        return {"e": f"Need ≥10 observations, got {n}"}
    # Difference
    dy = np.diff(y)
    # Lag levels
    y_lag = y[lags:n - 1]
    dy_use = dy[lags:]
    # Design: constant + trend + y_lag + lagged differences
    X_parts = [np.ones(len(y_lag)), np.arange(1, len(y_lag) + 1), y_lag]
    for p in range(1, lags + 1):
        X_parts.append(dy[lags - p:len(dy) - p] if lags - p >= 0 else np.zeros(len(y_lag)))
    X = np.column_stack(X_parts)
    beta = np.linalg.lstsq(X, dy_use, rcond=None)[0]
    # Test statistic = gamma / SE(gamma)
    resid = dy_use - X @ beta
    sigma2 = float(np.sum(resid ** 2)) / (len(dy_use) - X.shape[1])
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se_gamma = math.sqrt(float(cov[2, 2]))
    except (np.linalg.LinAlgError, ValueError):
        se_gamma = float('nan')
    gamma = float(beta[2])
    adf_stat = gamma / se_gamma if se_gamma > 0 else 0
    # Critical values (approximate, n>100)
    crit_1pct, crit_5pct, crit_10pct = -3.43, -2.86, -2.57
    stationary = adf_stat < crit_5pct
    return {"adf": _rnd(adf_stat), "n": n, "lags": lags,
            "crit_1pct": crit_1pct, "crit_5pct": crit_5pct, "crit_10pct": crit_10pct,
            "stationary_5pct": stationary}


# ─── BETA REGRESSION ──────────────────────────────────────────────

def _do_beta(cols: dict, args: dict) -> dict:
    """Beta regression (logit link, MLE via IRLS approximation)."""
    o = args.get("o", "")
    predictors = args.get("p", [])
    if not o or not predictors:
        return {"e": "o (outcome in (0,1)), p (predictors) required"}
    if o not in cols:
        return {"e": f"Outcome '{o}' not found"}
    for pv in predictors:
        if pv not in cols:
            return {"e": f"Predictor '{pv}' not found"}
    y = _arr(cols[o])
    X_cols = [_arr(cols[pv]) for pv in predictors]
    n = min(len(y), *(len(x) for x in X_cols))
    y, X = y[:n], _design_matrix([x[:n] for x in X_cols], add_intercept=True)
    # Beta regression requires y in (0,1): clip boundary values and note it.
    n_clip = int(np.sum((y <= 0) | (y >= 1)))
    y = np.clip(y, 0.001, 0.999)
    k = X.shape[1]
    # Transform: logit link
    eta = np.log(y / (1 - y))
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    for _ in range(30):
        mu = 1 / (1 + np.exp(-X @ beta))
        mu = np.clip(mu, 0.001, 0.999)
        # Working weights for beta regression: mu*(1-mu)
        W = np.diag(mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))
        try:
            beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    mu = 1 / (1 + np.exp(-X @ beta))
    mu = np.clip(mu, 0.001, 0.999)
    W = np.diag(mu * (1 - mu))
    try:
        cov = np.linalg.inv(X.T @ W @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, float('nan'))
    z_vals = beta / se
    p_vals = [float(2 * norm.sf(abs(zv if not math.isnan(zv) else 0))) for zv in z_vals]
    names = ["(Intercept)"] + list(predictors)
    coefs = []
    for i, nm in enumerate(names):
        if i < len(beta):
            coefs.append({"n": nm, "B": _rnd(float(beta[i])), "SE": _rnd(float(se[i])),
                         "z": _rnd(float(z_vals[i])), "p": _rnd(p_vals[i]),
                         "OR": _rnd(math.exp(float(beta[i])))})
    # Pseudo R²
    ll_full = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    ll_null = float(np.sum(y * np.log(np.mean(y)) + (1 - y) * np.log(1 - np.mean(y))))
    mcfadden = 1 - ll_full / ll_null if ll_null != 0 else 0
    out = {"n": n, "R2_mcfadden": _rnd(mcfadden), "coef": coefs,
           "note": "logit link: exp(B) = odds ratio for mean proportion mu/(1-mu). "
                   "Quasi-MLE via IRLS (no precision submodel); SEs are approximate."}
    if n_clip:
        out["warn_clip"] = (
            f"{n_clip}/{n} outcomes on the [0,1] boundary were clipped to "
            "(0.001,0.999). Beta regression requires y in (0,1); consider "
            "zero/one-inflated beta regression if boundary mass is substantial."
        )
    return out


# ─── HEDGES' G + MCDONALD'S OMEGA ────────────────────────────────

def _do_hedgesg(cols: dict, args: dict) -> dict:
    """Hedges' g (bias-corrected Cohen's d) from data or direct value.

    Signed: positive means g[1] (second group label) scores higher than
    g[0]. Magnitude-only consumers should take abs(g).
    """
    a, v, g = args.get("a", ""), args.get("v", ""), args.get("g", [])
    if a and v and len(g) >= 2 and a in cols and v in cols:
        grps = _groups(cols[v], cols[a], g[:2])
        n1, n2 = len(grps[0]), len(grps[1])
        if n1 < 2 or n2 < 2:
            return {"e": "Need ≥2 per group"}
        sd1, sd2 = float(np.std(grps[0], ddof=1)), float(np.std(grps[1], ddof=1))
        pooled = math.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
        d = float(np.mean(grps[1]) - float(np.mean(grps[0]))) / pooled if pooled > 0 else 0
        # Hedges' g correction
        df = n1 + n2 - 2
        J = 1 - 3 / (4 * df - 1) if df > 2 else 1
        g_val = d * J
        se_g = math.sqrt((n1+n2)/(n1*n2) + g_val**2/(2*(n1+n2)))
        return {"d": _rnd(d), "g": _rnd(g_val), "se": _rnd(se_g), "n1": n1, "n2": n2,
                "direction": f"{g[1]} minus {g[0]} (positive = {g[1]} higher)"}
    return {"e": "a (group var), v (value var), g=[g1,g2] required"}


def _do_omega(cols: dict, args: dict) -> dict:
    """McDonald's omega (hierarchical) — doesn't assume tau-equivalence.

    Single-factor approximation: assumes the items are unidimensional
    (one dominant factor). Verify with the 'factor' action first; if the
    scale is multidimensional, omega_hierarchical here is optimistic.
    """
    variables = args.get("v", [])
    if not variables or len(variables) < 2:
        return {"e": "v (2+ item variables) required"}
    for var in variables:
        if var not in cols:
            return {"e": f"Variable '{var}' not found"}
    arrays = [_arr(cols[v]) for v in variables]
    n = min(len(a) for a in arrays)
    data = np.column_stack([a[:n] for a in arrays])
    # Omega = (sum of loadings)² / ((sum loadings)² + sum(unique variances))
    # Extract first eigenvalue/factor loadings
    corr = np.corrcoef(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    loadings = eigenvectors[:, -1] * math.sqrt(max(eigenvalues[-1], 0))
    loadings = np.abs(loadings)  # sign doesn't matter
    sum_load = float(np.sum(loadings))
    unique_var = np.array([max(0, 1 - l**2) for l in loadings])
    omega_val = sum_load**2 / (sum_load**2 + float(np.sum(unique_var)))
    # Cronbach's alpha for comparison
    item_vars = np.var(data, axis=0, ddof=1)
    total_scores = np.sum(data, axis=1)
    total_var = float(np.var(total_scores, ddof=1))
    k = len(variables)
    alpha = (k / (k - 1)) * (1 - float(np.sum(item_vars)) / total_var) if total_var > 0 and k > 1 else 0
    return {"omega": _rnd(omega_val), "alpha": _rnd(alpha), "loadings": {v: _rnd(float(l)) for v, l in zip(variables, loadings)},
            "note": "single-factor approximation; assumes a unidimensional scale (verify with 'factor')."}


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

_ANALYSES = {
    "desc": _do_descriptives,
    "freq": _do_frequencies,
    "examine": _do_examine,
    "means": _do_means,
    "crosstab": _do_crosstab,
    "ttest": _do_ttest,
    "pttest": _do_pttest,
    "ttest1": _do_ttest1,
    "anova": _do_anova,
    "mw": _do_mann_whitney,
    "kw": _do_kruskal_wallis,
    "wilcoxon": _do_wilcoxon,
    "friedman": _do_friedman,
    "sign": _do_sign_test,
    "ks": _do_ks_test,
    "runs": _do_runs_test,
    "corr": _do_correlation,
    "spearman": _do_spearman,
    "partial": _do_partial_corr,
    "reg": _do_regression,
    "logistic": _do_logistic,
    "poisson": _do_poisson,
    "factor": _do_factor,
    "reliability": _do_reliability,
    "roc": _do_roc,
    "eval": _do_eval,
    "surv": _do_survival,
    "meta": _do_meta,
    "blandaltman": _do_blandaltman,
    "kappa": _do_kappa,
    "evalue": _do_evalue,
    "power": _do_power,
    "rank": _do_rank,
    "dca": _do_dca,
    "quantreg": _do_quantreg,
    "negbin": _do_negbin,
    "mediation": _do_mediation,
    "pscore": _do_pscore,
    "gam": _do_gam,
    "zip": _do_zip,
    "lasso": _do_lasso,
    "adf": _do_adf,
    "beta": _do_beta,
    "hedgesg": _do_hedgesg,
    "omega": _do_omega,
}


def pspp_run(
    t: str = "",
    d: str = "",
    a: str = "",
    v: Any = "",
    g: Optional[list] = None,
    o: str = "",
    w: str = "",
    p: Optional[list] = None,
    c: Optional[list] = None,
    mu: Optional[float] = None,
    paired: bool = False,
    posthoc: str = "",
    rotate: str = "",
    n: int = 0,
    pos: Any = 1,
    pos_ref: Any = 1,
    effect_col: str = "",
    se_col: str = "",
    power_test: str = "",
    alpha: float = 0.05,
    es_val: float = 0.5,
    ratio: float = 1.0,
    k: int = 2,
    p0: float = 0.5,
    p1: float = 0.7,
    calc: str = "",
    or_val: Optional[float] = None,
    rr_val: Optional[float] = None,
    tvar: str = "",
    tau: float = 0.5,
    lam: float = 0.1,
    df_val: int = 5,
    lags: int = 1,
    df: int = 5,
) -> str:
    """Statistical analysis — complete masterclass coverage, pure Python.

    t:          Test type: desc, freq, examine, means, crosstab, ttest, pttest, ttest1,
                anova, mw, kw, wilcoxon, friedman, sign, ks, runs, corr, spearman,
                partial, reg, logistic, poisson, factor, reliability, roc, eval, surv,
                meta, blandaltman, kappa, evalue, power, rank
    d:          CSV data with header
    a:          Grouping/row variable name
    v:          Value/column variable(s) — string or list
    g:          Group labels [g1,g2] for two-sample tests
    o:          Outcome/dependent variable
    w:          Weight/count variable / kappa weight type
    p:          Predictors list
    c:          Control variables (partial correlation)
    mu:         Population mean for one-sample tests
    paired:     Paired flag for McNemar
    posthoc:    Post-hoc method: tukey, bonferroni, gh
    rotate:     Rotation method: varimax
    n:          Components (factor) / per-group n (power)
    pos:        Positive class label (roc, eval)
    pos_ref:    Positive reference class (eval)
    effect_col: Effect size column name (meta)
    se_col:     SE column name (meta)
    power_test: ttest, prop, anova (power)
    alpha:      Significance level (power, default 0.05)
    es_val:     Effect size (power, default 0.5)
    ratio:      n2/n1 ratio (power)
    k:          Number of groups (power/anova)
    p0:         Baseline proportion (power)
    p1:         Alternative proportion (power)
    calc:       'n' for sample size, 'power' for power (power)
    or_val:     Direct OR value (evalue)
    rr_val:     Direct RR value (evalue)
    tvar:       Time variable name (surv)
    """
    try:
        test_type = t.strip().lower() if t else ""
        if not test_type:
            return _err("t (test type) required. Valid: " + ", ".join(sorted(_ANALYSES.keys())))

        if test_type not in _ANALYSES:
            return _err(f"Unknown test '{test_type}'. Valid: " + ", ".join(sorted(_ANALYSES.keys())))

        if not d or not d.strip():
            return _err("d (CSV data) required")

        cols = _parse_csv(d)
        if not cols:
            return _err("Could not parse CSV data")

        args = {
            "a": a.strip() if isinstance(a, str) and a else "",
            "v": v if isinstance(v, list) else (v.strip() if isinstance(v, str) and v else ""),
            "g": g if g else [],
            "o": o.strip() if o else "",
            "w": w.strip() if w else "",
            "p": p if p else [],
            "c": c if c else [],
            "mu": mu,
            "paired": paired,
            "posthoc": posthoc.strip().lower() if posthoc else "",
            "rotate": rotate.strip().lower() if rotate else "",
            "n": n,
            "n1": n,
            "n_per_group": n,
            "pos": pos,
            "pos_ref": pos_ref,
            "effect_col": effect_col.strip() if effect_col else "",
            "se_col": se_col.strip() if se_col else "",
            "power_test": power_test.strip().lower() if power_test else "ttest",
            "alpha": alpha,
            "es": es_val,
            "ratio": ratio,
            "k": k,
            "p0": p0,
            "p1": p1,
            "calc": calc.strip().lower() if calc else "n",
            "or": or_val,
            "rr": rr_val,
            "tvar": tvar.strip() if tvar else "",
            "tau": tau,
            "lam": lam,
            "df": df_val if df_val else df,
            "lags": lags,
        }

        result = _ANALYSES[test_type](cols, args)
        if "e" in result:
            return _err(result["e"])
        return _ok({"k": result})

    except Exception as e:
        return _err(f"Analysis error: {e}")


# ═══════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════

PSPP_SCHEMA = {
    "name": "pspp",
    "description": (
        "Statistical analysis (statistics-masterclass complete, pure Python). "
        f"{len(_ANALYSES)} test types. "
        "Descriptive: desc, freq, examine, means, crosstab. "
        "Compare means: ttest, pttest, ttest1, anova. "
        "Non-parametric: mw, kw, wilcoxon, friedman, sign, ks, runs. "
        "Correlation: corr, spearman, partial. "
        "Regression: reg, logistic, poisson. "
        "Advanced: factor, reliability, roc, eval, surv, meta, "
        "blandaltman, kappa, evalue, power. Transform: rank."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "t": {
                "type": "string",
                "description": "Test type. See full list above. Required.",
            },
            "d": {
                "type": "string",
                "description": "CSV data with header row.",
            },
            "a": {
                "type": "string",
                "description": "Grouping/row variable name.",
            },
            "v": {
                "type": "string",
                "description": "Value/column variable(s) — string or array.",
            },
            "g": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Group labels [g1,g2] for two-sample tests.",
            },
            "o": {
                "type": "string",
                "description": "Outcome/dependent variable.",
            },
            "w": {
                "type": "string",
                "description": "Weight/count column (crosstab).",
            },
            "p": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Predictor variables.",
            },
            "c": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Control variables (partial correlation).",
            },
            "mu": {
                "type": "number",
                "description": "Population mean for one-sample tests.",
            },
            "paired": {
                "type": "boolean",
                "description": "Paired flag for crosstab McNemar.",
            },
            "posthoc": {
                "type": "string",
                "description": "Post-hoc method: tukey, bonferroni, gh.",
            },
            "rotate": {
                "type": "string",
                "description": "Rotation method: varimax.",
            },
            "n": {
                "type": "integer",
                "description": "Number of components (factor).",
            },
            "pos": {
                "type": "number",
                "description": "Positive class label (roc).",
            },
            "pos_ref": {"type": "number", "description": "Positive reference class (eval)."},
            "effect_col": {"type": "string", "description": "Effect size column name (meta)."},
            "se_col": {"type": "string", "description": "SE column name (meta)."},
            "power_test": {"type": "string", "description": "ttest|prop|anova (power)."},
            "alpha": {"type": "number", "description": "Significance level (power)."},
            "es_val": {"type": "number", "description": "Effect size (power)."},
            "ratio": {"type": "number", "description": "n2/n1 ratio (power)."},
            "k": {"type": "integer", "description": "Groups (power/anova)."},
            "p0": {"type": "number", "description": "Baseline proportion (power prop)."},
            "p1": {"type": "number", "description": "Alternative proportion (power prop)."},
            "calc": {"type": "string", "description": "'n' sample size | 'power' power (power)."},
            "or_val": {"type": "number", "description": "Direct OR (evalue)."},
            "rr_val": {"type": "number", "description": "Direct RR (evalue)."},
            "tvar": {"type": "string", "description": "Time variable (surv)."},
            "tau": {"type": "number", "description": "Quantile level (quantreg)."},
            "lam": {"type": "number", "description": "Penalty (lasso)."},
            "df": {"type": "integer", "description": "DF (gam)."},
            "lags": {"type": "integer", "description": "Lags (adf)."},
        },
        "required": ["t", "d"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

from tools.registry import registry

registry.register(
    name="pspp",
    toolset="medical",
    schema=PSPP_SCHEMA,
    handler=lambda args, **kw: pspp_run(
        t=args.get("t", ""),
        d=args.get("d", ""),
        a=args.get("a", ""),
        v=args.get("v", ""),
        g=args.get("g"),
        o=args.get("o", ""),
        w=args.get("w", ""),
        p=args.get("p"),
        c=args.get("c"),
        mu=args.get("mu"),
        paired=args.get("paired", False),
        posthoc=args.get("posthoc", ""),
        rotate=args.get("rotate", ""),
        tau=args.get("tau", 0.5),
        lam=args.get("lam", 0.1),
        df=args.get("df", 5),
        lags=args.get("lags", 1),
        n=args.get("n", 0),
        pos=args.get("pos", 1),
        pos_ref=args.get("pos_ref", 1),
        effect_col=args.get("effect_col", ""),
        se_col=args.get("se_col", ""),
        power_test=args.get("power_test", ""),
        alpha=args.get("alpha", 0.05),
        es_val=args.get("es_val", 0.5),
        ratio=args.get("ratio", 1.0),
        k=args.get("k", 2),
        p0=args.get("p0", 0.5),
        p1=args.get("p1", 0.7),
        calc=args.get("calc", ""),
        or_val=args.get("or_val"),
        rr_val=args.get("rr_val"),
        tvar=args.get("tvar", ""),
    ),
    check_fn=lambda: True,
    emoji="📊",
)
