"""Complete analysis pipeline for
"Predicting Course Non-Participation from Moodle Gradebook Traces".

Every number reported in the manuscript is written to a single JSON file.

Usage:
    python src/analysis.py --data-dir data --out results/results.json
    python src/analysis.py --data-dir data --out results/smoke.json --fast   # quick smoke test

Inputs (not distributed, see data/README.md):
    udk_moodle_all_grades.csv   Moodle gradebook export
    logs_*.csv                  four Moodle event logs of student-designed courses (case study)
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
T0 = pd.Timestamp("2022-11-24", tz="UTC")          # primary cutoff, immediately before the batch
T1 = pd.Timestamp("2022-12-01", tz="UTC")          # sensitivity cutoff, after the batch
BATCH_DAY = pd.Timestamp("2022-11-24", tz="UTC")
TAU = 0.50                                         # routing risk gate (prototype parameter)
G5_SHARE = 0.50                                    # explanation-reliability rule (prototype parameter)
OBSERVER = "AMAL YALID"                            # account that exported the event logs; excluded

BEHAVIOR = ["has_prior_record", "n_prior_records", "n_prior_mod", "prior_mod_na_rate",
            "prior_mean_grade", "n_prior_courses", "n_active_days", "days_since_last_record",
            "n_prior_mod_target_course", "prior_course_total_na"]
CONTEXT = ["course_size", "ctx_n_prior_records", "ctx_n_active_users", "ctx_na_rate"]

CONFIGS = [("Dummy", "Dummy", []),
           ("Context-only, LR", "LR", CONTEXT),
           ("Behavior-only, LR", "LR", BEHAVIOR),
           ("Behavior-only, RF", "RF", BEHAVIOR),
           ("Behavior-only, XGBoost", "XGB", BEHAVIOR),
           ("Behavior+Context, LR", "LR", BEHAVIOR + CONTEXT),
           ("Behavior+Context, RF", "RF", BEHAVIOR + CONTEXT),
           ("Behavior+Context, XGBoost", "XGB", BEHAVIOR + CONTEXT)]
LR_CONFIGS = [c for c in CONFIGS if c[1] in ("Dummy", "LR")]

GRIDS = {
    "LR": {"clf__C": [0.1, 1, 10]},
    "RF": {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 5, 10],
           "clf__min_samples_leaf": [1, 5, 10]},
    "XGB": {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
            "clf__learning_rate": [0.05, 0.1]},
}
FAST_RF_GRID = {"clf__n_estimators": [300], "clf__max_depth": [None, 10], "clf__min_samples_leaf": [5]}
THRESH_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)

MONTHS = {"janv": 1, "janvier": 1, "févr": 2, "février": 2, "mars": 3, "avr": 4, "avril": 4,
          "mai": 5, "juin": 6, "juil": 7, "juillet": 7, "août": 8, "sept": 9, "septembre": 9,
          "oct": 10, "octobre": 10, "nov": 11, "novembre": 11, "déc": 12, "décembre": 12}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Data, audit and features
# ----------------------------------------------------------------------------------------------
def load_grades(data_dir):
    g = pd.read_csv(os.path.join(data_dir, "udk_moodle_all_grades.csv"))
    g["ts"] = pd.to_datetime(g["timemodified"], unit="s", utc=True)
    g["day"] = g["ts"].dt.floor("D")
    g["na"] = g["finalgrade"] == -1
    return g


def describe_and_audit(grades):
    is_batch = grades["day"] == BATCH_DAY
    is_mod = grades["itemtype"] == "mod"
    batch = grades[is_batch]
    data_summary = {
        "n_records": len(grades), "n_users": grades.userid.nunique(),
        "n_courses": grades.courseid.nunique(),
        "itemtype_counts": grades.itemtype.value_counts().to_dict(),
        "date_min": str(grades.ts.min().date()), "date_max": str(grades.ts.max().date()),
    }
    audit = {
        "batch_records": len(batch), "batch_users": batch.userid.nunique(),
        "batch_courses": batch.courseid.nunique(),
        "batch_top_hour_share": batch.ts.dt.floor("h").value_counts().iloc[0] / len(batch),
        "n_na_total": int(grades.na.sum()), "n_na_batch": int((grades.na & is_batch).sum()),
        "share_na_in_batch": (grades.na & is_batch).sum() / grades.na.sum(),
        "mod_na_rate_batch_day": (grades.na & is_mod & is_batch).sum() / (is_mod & is_batch).sum(),
        "mod_na_rate_other_days": (grades.na & is_mod & ~is_batch).sum() / (is_mod & ~is_batch).sum(),
    }
    return data_summary, audit


def build_dataset(grades, cutoff):
    """Analytic pairs = course totals written at/after cutoff; features use records strictly before it."""
    ct = grades[grades.itemtype == "course"]
    pop = ct[ct.ts >= cutoff][["userid", "courseid", "finalgrade", "day"]].copy()
    pop["y"] = (pop.finalgrade == -1).astype(int)
    pop = pop.reset_index(drop=True)
    pre = grades[grades.ts < cutoff]
    pre_mod = pre[pre.itemtype == "mod"]

    g = pre.groupby("userid")
    u = pd.DataFrame({
        "n_prior_records": g.size(),
        "n_prior_courses": g.courseid.nunique(),
        "n_active_days": g.day.nunique(),
        "days_since_last_record": (cutoff - g.ts.max()).dt.total_seconds() / 86400,
        "prior_course_total_na": pre[pre.itemtype == "course"].groupby("userid").na.sum(),
    })
    gm = pre_mod.groupby("userid")
    u["n_prior_mod"] = gm.size()
    u["prior_mod_na_rate"] = gm.na.mean()
    u["prior_mean_grade"] = pre_mod[~pre_mod.na].groupby("userid").finalgrade.mean()

    X = pop.join(u, on="userid")
    X["has_prior_record"] = X.userid.isin(pre.userid).astype(int)
    tc = pre_mod.groupby(["userid", "courseid"]).size().rename("n_prior_mod_target_course")
    X = X.join(tc, on=["userid", "courseid"])
    for col in ["n_prior_records", "n_prior_courses", "n_active_days", "n_prior_mod",
                "prior_course_total_na", "n_prior_mod_target_course"]:
        X[col] = X[col].fillna(0)          # structural zeros: verified absence of records
    # rate / grade / recency stay NaN when undefined -> median-imputed inside training folds

    # Course context, leave-self-out: the student's own records never describe their context
    X["course_size"] = X.courseid.map(pop.groupby("courseid").size()) - 1
    c_rec = pre.groupby("courseid").size()
    c_na = pre.groupby("courseid").na.sum()
    c_users = pre.groupby("courseid").userid.nunique()
    key = pd.MultiIndex.from_frame(X[["courseid", "userid"]])
    s_rec = pre.groupby(["courseid", "userid"]).size().reindex(key).fillna(0).values
    s_na = pre.groupby(["courseid", "userid"]).na.sum().reindex(key).fillna(0).values
    X["ctx_n_prior_records"] = X.courseid.map(c_rec).fillna(0).values - s_rec
    X["ctx_n_active_users"] = X.courseid.map(c_users).fillna(0).values - (s_rec > 0)
    na_others = X.courseid.map(c_na).fillna(0).values - s_na
    X["ctx_na_rate"] = np.where(X.ctx_n_prior_records > 0,
                                na_others / X.ctx_n_prior_records.replace(0, np.nan), np.nan)
    return X


def describe_population(grades, data):
    hist = data[data.has_prior_record == 1]
    return {
        "n_pairs": len(data), "n_users": data.userid.nunique(), "n_courses": data.courseid.nunique(),
        "n_excluded_finalized_before_T0": int((grades[grades.itemtype == "course"].ts < T0).sum()),
        "n_positive": int(data.y.sum()), "prevalence": data.y.mean(),
        "n_no_history": int((data.has_prior_record == 0).sum()),
        "share_no_history": (data.has_prior_record == 0).mean(),
        "prior_mean_grade_by_y": hist.groupby("y").prior_mean_grade.mean().to_dict(),
        "days_since_last_by_y": hist.groupby("y").days_since_last_record.mean().to_dict(),
        "n_active_days_by_y": data.groupby("y").n_active_days.mean().to_dict(),
        "median_days_since_last_with_history": hist.days_since_last_record.median(),
        "n_batch_day_outcomes": int((data.day == BATCH_DAY).sum()),
    }


# ----------------------------------------------------------------------------------------------
# Models and nested cross-validation
# ----------------------------------------------------------------------------------------------
def make_estimator(kind):
    if kind == "LR":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])
    if kind == "RF":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                                        n_jobs=1))])
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss",
                                           tree_method="hist"))])


def macro_f1(y, p, t=0.5):
    return f1_score(y, (p >= t).astype(int), average="macro")


def fit_tuned(kind, X, y, groups, grids):
    """Grouped 3-fold inner search by Macro-F1; returns best estimator and inner OOF probabilities."""
    if kind == "Dummy":
        return DummyClassifier(strategy="most_frequent").fit(X, y), None
    est = make_estimator(kind)
    if kind == "XGB":
        est.set_params(clf__scale_pos_weight=(y == 0).sum() / (y == 1).sum())
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y, groups))
    gs = GridSearchCV(est, grids[kind], scoring="f1_macro", cv=inner, n_jobs=-1, refit=True).fit(X, y)
    oof = np.zeros(len(y))     # inner out-of-fold predictions of the selected configuration
    for tr, te in inner:
        m = clone(gs.best_estimator_).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return gs.best_estimator_, oof


def best_threshold(y, p):
    """Post hoc threshold: maximizes Macro-F1 on inner out-of-fold predictions (grid 0.05-0.95)."""
    return THRESH_GRID[int(np.argmax([macro_f1(y, p, t) for t in THRESH_GRID]))]


def nested_cv(df, configs, grids):
    y, users = df.y.values, df.userid.values
    outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, users))
    out = {}
    for name, kind, feats in configs:
        Xall = df[feats] if feats else pd.DataFrame({"const": np.zeros(len(df))})
        oof = np.zeros(len(df))
        f1s, bas, f1t, thr, models = [], [], [], [], []
        for tr, te in outer:
            m, inner_oof = fit_tuned(kind, Xall.iloc[tr], y[tr], users[tr], grids)
            p = m.predict_proba(Xall.iloc[te])[:, 1]
            oof[te] = p
            f1s.append(macro_f1(y[te], p))
            bas.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
            if inner_oof is not None:
                t = best_threshold(y[tr], inner_oof)
                thr.append(float(t))
                f1t.append(macro_f1(y[te], p, t))
            models.append(m)
        out[name] = {
            "macro_f1_mean": np.mean(f1s), "macro_f1_sd": np.std(f1s, ddof=1),
            "bal_acc_mean": np.mean(bas), "bal_acc_sd": np.std(bas, ddof=1),
            "roc_auc": roc_auc_score(y, oof), "pr_auc": average_precision_score(y, oof),
            "brier": brier_score_loss(y, oof),
            "macro_f1_tuned_mean": np.mean(f1t) if f1t else None,
            "macro_f1_tuned_sd": np.std(f1t, ddof=1) if f1t else None,
            "tuned_thresholds": thr,
            "lr_C_by_fold": [m.get_params()["clf__C"] for m in models] if kind == "LR" else None,
            "oof": oof, "models": models,
        }
        log(f"{name:28s} F1={out[name]['macro_f1_mean']:.3f}  AUC={out[name]['roc_auc']:.3f}")
    return out, outer


def strip(cv):
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "models")} for k, v in cv.items()}


# ----------------------------------------------------------------------------------------------
# Inference, transfer and sensitivity
# ----------------------------------------------------------------------------------------------
def clustered_bootstrap(df, oof_a, oof_b=None, n=1000, seed=SEED):
    """User-clustered bootstrap: resample users with all their pairs; 95% percentile intervals."""
    rng = np.random.default_rng(seed)
    y = df.y.values
    idx_by_user = df.groupby("userid").indices
    ulist = np.array(sorted(idx_by_user))
    f1d, aucd = [], []
    for _ in range(n):
        idx = np.concatenate([idx_by_user[u] for u in rng.choice(ulist, len(ulist), replace=True)])
        yy = y[idx]
        fa, aa = macro_f1(yy, oof_a[idx]), roc_auc_score(yy, oof_a[idx])
        if oof_b is not None:
            fa -= macro_f1(yy, oof_b[idx])
            aa -= roc_auc_score(yy, oof_b[idx])
        f1d.append(fa)
        aucd.append(aa)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"macro_f1": ci(f1d), "roc_auc": ci(aucd)}


def course_groups(df, n_groups=5):
    """Deterministic, library-version-independent partition of courses into disjoint groups:
    courses sorted by number of pairs (desc, ties by id) are assigned greedily to the group
    with the fewest pairs so far."""
    sizes = df.groupby("courseid").size().reset_index(name="n")
    sizes = sizes.sort_values(["n", "courseid"], ascending=[False, True])
    load, assign = np.zeros(n_groups), {}
    for cid, n in zip(sizes.courseid, sizes.n):
        g = int(np.argmin(load))
        assign[cid] = g
        load[g] += n
    return df.courseid.map(assign).values


def leave_course_group_out(df, grids):
    y, users = df.y.values, df.userid.values
    grp = course_groups(df)
    rows = []
    for g in range(5):
        tr, te = np.where(grp != g)[0], np.where(grp == g)[0]
        row = {"group": g + 1, "n_courses": int(df.courseid.iloc[te].nunique()), "n": len(te),
               "prev": y[te].mean(), "Dummy_F1": macro_f1(y[te], np.zeros(len(te)))}
        for kind in ["LR", "RF", "XGB"]:
            m, _ = fit_tuned(kind, df[BEHAVIOR].iloc[tr], y[tr], users[tr], grids)
            p = m.predict_proba(df[BEHAVIOR].iloc[te])[:, 1]
            row[f"{kind}_F1"] = macro_f1(y[te], p)
            row[f"{kind}_AUC"] = roc_auc_score(y[te], p)
        rows.append(row)
        log(f"LCGO group {g + 1}: n={row['n']} prev={row['prev']:.3f} "
            f"LR {row['LR_F1']:.3f}/{row['LR_AUC']:.3f} RF {row['RF_F1']:.3f}/{row['RF_AUC']:.3f} "
            f"XGB {row['XGB_F1']:.3f}/{row['XGB_AUC']:.3f}")
    return rows


# ----------------------------------------------------------------------------------------------
# Explanations, redundancy and routing
# ----------------------------------------------------------------------------------------------
def explanation_audit(data, cv, outer):
    """Exact linear SHAP (independent masker, training-fold background), out-of-fold:
    phi_ij = beta_j * (z_ij - mean_train(z_j))."""
    phi = np.zeros((len(data), len(BEHAVIOR)))
    coefs, fold_imp = [], []
    for (tr, te), m in zip(outer, cv["Behavior-only, LR"]["models"]):
        Ztr = m[:-1].transform(data[BEHAVIOR].iloc[tr])
        Zte = m[:-1].transform(data[BEHAVIOR].iloc[te])
        beta = m[-1].coef_[0]
        phi[te] = beta * (Zte - Ztr.mean(axis=0))
        coefs.append(beta)
        fold_imp.append(np.abs(phi[te]).mean(axis=0))
    coefs = np.array(coefs)
    tab = pd.DataFrame({
        "feature": BEHAVIOR, "mean_abs_shap": np.abs(phi).mean(axis=0),
        "mean_beta": coefs.mean(axis=0),
        "n_pos": (coefs > 0).sum(axis=0), "n_neg": (coefs < 0).sum(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    tab["sign_stable"] = (tab.n_pos == 5) | (tab.n_neg == 5)
    fold_rhos = [spearmanr(fold_imp[a], fold_imp[b])[0] for a in range(5) for b in range(a + 1, 5)]

    out = {"fold_rank_rho_mean": float(np.mean(fold_rhos)), "fold_rank_rho_min": float(np.min(fold_rhos)),
           "fold_rank_rho_max": float(np.max(fold_rhos)), "n_sign_stable": int(tab.sign_stable.sum()),
           "coefficients_by_fold": coefs.tolist()}

    import shap
    tr, te = outer[0]
    m = cv["Behavior-only, LR"]["models"][0]
    Ztr = m[:-1].transform(data[BEHAVIOR].iloc[tr])
    Zte = m[:-1].transform(data[BEHAVIOR].iloc[te])
    sv = shap.LinearExplainer(m[-1], shap.maskers.Independent(Ztr, max_samples=len(Ztr))).shap_values(Zte)
    out["shap_library_max_abs_diff"] = float(np.abs(sv - phi[te]).max())

    # redundancy audit: |Pearson r| or |Spearman rho| >= .60 (pairwise-complete observations)
    pr = data[BEHAVIOR].corr(method="pearson").abs().values
    sr = data[BEHAVIOR].corr(method="spearman").abs().values
    iu = np.triu_indices(len(BEHAVIOR), 1)
    flag = (pr[iu] >= 0.60) | (sr[iu] >= 0.60)
    pairs = [(BEHAVIOR[i], BEHAVIOR[j]) for i, j, f in zip(*iu, flag) if f]
    out.update({"n_redundant_pairs": len(pairs), "n_pairs_total": len(iu[0]),
                "redundant_pairs": pairs, "max_abs_pearson": float(np.nanmax(pr[iu])),
                "activity_cluster": sorted({f for p in pairs for f in p})})

    # RF TreeSHAP ranking (out-of-fold, Behavior-only RF) against the LR ranking
    rf_imp = np.zeros(len(BEHAVIOR))
    for (tr, te), m in zip(outer, cv["Behavior-only, RF"]["models"]):
        Zte = m[:-1].transform(data[BEHAVIOR].iloc[te])
        s = shap.TreeExplainer(m[-1]).shap_values(Zte, check_additivity=False)
        s = s[1] if isinstance(s, list) else (s[..., 1] if s.ndim == 3 else s)
        rf_imp += np.abs(s).sum(axis=0)
    out["rho_lr_vs_rf_treeshap"] = float(spearmanr(np.abs(phi).mean(axis=0), rf_imp)[0])
    out["table"] = tab.to_dict(orient="records")
    return out, phi, tab


def routing(data, p, phi, tab):
    hist = data[data.has_prior_record == 1]
    cut = hist.days_since_last_record.median()
    uidx = [BEHAVIOR.index(f) for f in tab.loc[~tab.sign_stable, "feature"]]
    a = np.abs(phi)
    tot = a.sum(axis=1)
    unstable_share = a[:, uidx].sum(axis=1) / np.where(tot == 0, 1, tot)
    g5 = (p >= TAU) & (unstable_share > G5_SHARE)
    route = np.where(p < TAU, "R0",
             np.where(g5, "RH",
              np.where(data.has_prior_record.values == 0, "R1",
               np.where(data.days_since_last_record.values > cut, "R2", "RH"))))
    rt = pd.DataFrame({"route": route, "y": data.y.values})
    t = rt.groupby("route").agg(n=("y", "size"), observed_rate=("y", "mean"))
    t["share"] = t.n / len(rt)
    return {"table": t.reset_index().to_dict(orient="records"), "g5_fired": int(g5.sum()),
            "inactivity_cut_days": float(cut), "tau": TAU}


# ----------------------------------------------------------------------------------------------
# Design-side case study
# ----------------------------------------------------------------------------------------------
def parse_fr_date(s):
    """Moodle French export, e.g. '6 oct. 26, 17:11' or '3 avril 26, 09:15'."""
    d, mon, rest = s.split(" ", 2)
    return pd.Timestamp(2000 + int(rest.split(",")[0]), MONTHS[mon.rstrip(".").lower()], int(d))


def case_study(data_dir):
    profiles = []
    for f in sorted(glob.glob(os.path.join(data_dir, "logs_*.csv"))):
        lg = pd.read_csv(f, encoding="utf-8-sig")
        lg = lg[lg["Nom complet"] != OBSERVER]
        ev = lg["Nom de l'événement"]
        created = lg[ev == "Module de cours créé"]
        authoring = ev.str.contains(r"créé|créée|modifié|modifiée|supprimé|supprimée", regex=True)
        profiles.append({
            "events": len(lg), "days": lg.Heure.map(parse_fr_date).nunique(),
            "modules_created": len(created),
            "module_types": created.Description.str.extract(r"the '(\w+)' activity")[0].value_counts().to_dict(),
            "questions": int((ev == "Question créée").sum()),
            "components": lg.Composant.nunique(), "authoring_share": authoring.mean()})
    if len(profiles) != 4:
        return {}
    # anonymized labels as in the paper: C1=541, C2=317, C3=48, C4=171 designer events
    profiles = sorted(profiles, key=lambda r: -r["events"])
    return {f"C{i + 1}": profiles[k] for i, k in enumerate([0, 1, 3, 2])}


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def save_figures(grades, cv, tab, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    daily = grades.groupby("day").agg(records=("id", "size"), na=("na", "sum"))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(daily.index, daily.records, width=1, color="#9bb7d4", label="All grade records")
    ax.bar(daily.index, daily.na, width=1, color="#c0392b", label="Non-attempt markers (-1)")
    ax.axvline(T0, color="k", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_ylabel("Records per day (log)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig1_daily_records.png"), dpi=200); plt.close(fig)

    names = [c[0] for c in CONFIGS]
    colors = ["grey", "tab:orange"] + ["tab:blue"] * 3 + ["tab:green"] * 3
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], [cv[n]["macro_f1_mean"] for n in names][::-1],
            xerr=[cv[n]["macro_f1_sd"] for n in names][::-1], color=colors[::-1])
    ax.set_xlabel("Macro-F1 at 0.50 (mean ± SD over outer folds)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_macrof1.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(tab.feature[::-1], tab.mean_abs_shap[::-1],
            color=["tab:blue" if s else "silver" for s in tab.sign_stable][::-1])
    ax.set_xlabel("Mean |SHAP| (log-odds), out-of-fold, Behavior-only LR")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig4_shap.png"), dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------------------------
def to_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--fast", action="store_true", help="reduced RF grid and 200 bootstrap resamples")
    args = ap.parse_args()

    grids = dict(GRIDS, RF=FAST_RF_GRID) if args.fast else GRIDS
    n_boot = 200 if args.fast else 1000
    import sklearn, xgboost, shap
    R = {"meta": {"fast": args.fast, "seed": SEED, "versions": {
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "shap": shap.__version__}}}

    log("Loading data and auditing timestamps")
    grades = load_grades(args.data_dir)
    R["data"], R["audit"] = describe_and_audit(grades)
    data = build_dataset(grades, T0)
    R["population"] = describe_population(grades, data)

    log("Nested cross-validation (Table 3)")
    cv, outer = nested_cv(data, CONFIGS, grids)
    R["nested_cv"] = strip(cv)
    R["outer_fold_sizes"] = [len(te) for _, te in outer]
    R["outer_fold_positives"] = [int(data.y.values[te].sum()) for _, te in outer]

    log("User-clustered bootstrap (Table 4)")
    pairs = [("Behavior vs. Dummy", "Behavior-only, LR", "Dummy"),
             ("Behavior vs. Context", "Behavior-only, LR", "Context-only, LR"),
             ("Beh.+Context vs. Context", "Behavior+Context, LR", "Context-only, LR"),
             ("Beh.+Context vs. Behavior", "Behavior+Context, LR", "Behavior-only, LR")]
    R["bootstrap"] = {lab: clustered_bootstrap(data, cv[a]["oof"], cv[b]["oof"], n_boot) for lab, a, b in pairs}
    R["bootstrap"]["Behavior-only LR (single)"] = clustered_bootstrap(data, cv["Behavior-only, LR"]["oof"], n=n_boot)
    R["bootstrap"]["Context-only LR (single)"] = clustered_bootstrap(data, cv["Context-only, LR"]["oof"], n=n_boot)

    log("Leave-course-group-out (Table 5)")
    R["lcgo"] = leave_course_group_out(data, grids)

    log("Sensitivity analyses (Table 6)")
    sens = {"Primary T0": {k: [cv[k]["macro_f1_mean"], cv[k]["roc_auc"]] for k, _, _ in LR_CONFIGS}}
    data_excl = data[data.day != BATCH_DAY].reset_index(drop=True)
    cv_excl, _ = nested_cv(data_excl, LR_CONFIGS, grids)
    sens["Excl. batch-day outcomes"] = {k: [v["macro_f1_mean"], v["roc_auc"]] for k, v in cv_excl.items()}
    data_t1 = build_dataset(grades, T1)
    cv_t1, _ = nested_cv(data_t1, LR_CONFIGS, grids)
    sens["Cutoff T1"] = {k: [v["macro_f1_mean"], v["roc_auc"]] for k, v in cv_t1.items()}
    R["sensitivity"] = {"table": sens,
                        "n": {"excl": len(data_excl), "t1": len(data_t1)},
                        "prevalence": {"excl": data_excl.y.mean(), "t1": data_t1.y.mean()},
                        "t1_share_no_history": (data_t1.has_prior_record == 0).mean()}

    log("Explanation and stability audit (Table 7)")
    R["shap"], phi, tab = explanation_audit(data, cv, outer)

    log("Retrospective routing (Table 8)")
    R["routing"] = routing(data, cv["Behavior-only, LR"]["oof"], phi, tab)

    log("Design-side case study (Table 9)")
    R["case_study"] = case_study(args.data_dir)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, default=to_json, indent=1, ensure_ascii=False)
    save_figures(grades, cv, tab, os.path.join(out_dir, "figures"))
    log(f"Wrote {args.out} and figures")


if __name__ == "__main__":
    main()
