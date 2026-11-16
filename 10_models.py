"""Baseline, two models, slices and the clustering check, on the split and metric locked in the proposal.

Split     complaints ordered by received week; the first 70% are train, the rest are test (cut on a week boundary).
          For tuning only, the last 20% of the train weeks are held out as validation; the chosen settings are then
          refit on all of train. The test weeks are scored once.
Metric    recall at the top 25% of each week's labeled pool, averaged over weeks. Precision at the same cut is secondary.
Baseline  most problems first (ties broken at random, fixed seed).
Models    readable: L1 logistic regression on one-hot features (numeric features cut into named bins)
          flexible: histogram gradient boosting on the raw features
Intervals 95%, from resampling whole test weeks (2,000 draws), because complaints in one week share a cut.
LINE OF TRAVEL  removed from train and test (declared change). These complaints are entered by HPD after an inspector
          sees a condition in the parts of the building on the way to the complainant's apartment: none carries a
          311 request number, none is closed for no access or a tenant-reported fix, and 99% end in a violation.
          The week boundaries of the split are set before the removal, so they match the proposal's split.
          `--keep-lot` reproduces the earlier run with them kept (writes outputs/model_with_lot/ and
          data/model/test_scores_with_lot.parquet), which 11_figures.py uses for the with/without comparison.

Run 09_features.py first. Writes outputs/model/*.csv and data/model/test_scores.parquet.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from hpd_api import DATA, OUTPUTS

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
SEED, K, DRAWS = 0, 0.25, 2000
KEEP_LOT = "--keep-lot" in sys.argv
OUT = OUTPUTS / ("model_with_lot" if KEEP_LOT else "model")
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ data and split
c = pd.read_parquet(DATA / "model" / "complaints_features.parquet").sort_values("rec")
c["tie"] = np.random.default_rng(SEED).random(len(c))            # one fixed tie-break per complaint, for every ranking
share = c.groupby("week").size().cumsum() / len(c)
CUT = share.index[share.values >= 0.70][0] + pd.Timedelta(days=7)   # first test week, set before any removal
if not KEEP_LOT:
    lot = c.maj_line_of_travel > 0
    print(f"removed {lot.sum():,} LINE OF TRAVEL complaints ({c.label[lot].mean():.1%} violations)")
    c = c[~lot].drop(columns="maj_line_of_travel")
MAJ = [k for k in c.columns if k.startswith("maj_")]
train, test = c[c.week < CUT], c[c.week >= CUT]
tw = np.sort(train.week.unique())
VCUT = tw[int(len(tw) * 0.8)]
fit, val = train[train.week < VCUT], train[train.week >= VCUT]
print(f"train {len(train):,} complaints, {train.week.nunique()} weeks, {train.rec.min():%Y-%m-%d} to {train.rec.max():%Y-%m-%d}, "
      f"positive rate {train.label.mean():.3f}")
print(f"  of which validation {len(val):,}, {val.week.nunique()} weeks from {pd.Timestamp(VCUT):%Y-%m-%d}")
print(f"test  {len(test):,} complaints, {test.week.nunique()} weeks, {test.rec.min():%Y-%m-%d} to {test.rec.max():%Y-%m-%d}, "
      f"positive rate {test.label.mean():.3f}; share of complaints in test {len(test) / len(c):.3f}")
seen = test.building_id.isin(train.building_id)
print(f"  test complaints at a building with a labeled train complaint: {seen.mean():.1%}")


# ------------------------------------------------------------------ metric
def select(frame, score):
    """Boolean Series: is the complaint in the top 25% of its week's pool."""
    d = pd.DataFrame({"week": frame.week.values, "s": np.asarray(score, float), "tie": frame.tie.values}, index=frame.index)
    d = d.sort_values(["week", "s", "tie"], ascending=[True, False, False])
    rank = d.groupby("week").cumcount()
    k = np.maximum(1, (K * d.groupby("week").s.transform("size")).round())
    return (rank < k).reindex(frame.index)


def weekly(frame, sel):
    g = pd.DataFrame({"week": frame.week, "y": frame.label, "sel": sel, "hit": sel & (frame.label == 1)}).groupby("week")
    w = g.agg(n=("y", "size"), pos=("y", "sum"), sel=("sel", "sum"), hit=("hit", "sum"))
    w["recall"], w["precision"] = w.hit / w.pos, w.hit / w.sel
    w["ceiling"] = np.minimum(1, w.sel / w.pos)                  # recall if every selected complaint were a violation
    return w


def recall(frame, score):
    return weekly(frame, select(frame, score)).recall.mean()


BOOT = np.random.default_rng(SEED).integers(0, test.week.nunique(), (DRAWS, test.week.nunique()))
def ci(values):
    """95% interval for the mean of one value per test week, resampling weeks."""
    m = np.asarray(values, float)[BOOT].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


# ------------------------------------------------------------------ features for each model
CAT = ["minor", "type", "borough", "cb", "unit_type", "space_type"]
NUM = ["n_problems", "anonymous", "month", "weekday", "hour", "pall", "nall", "p6", "p12", "n12", "rate12",
       "months_since_p", "c7", "c30", "days_since_c"] + MAJ
HGB_CAT = CAT + ["zip"]

def bins(frame):
    """Readable version of every feature: categories as they are, numbers cut into named bins."""
    b = frame[CAT].copy()
    b["month"] = frame.month.astype(str).str.zfill(2)
    b["weekend"] = np.where(frame.weekday >= 5, "yes", "no")
    b["hours"] = pd.cut(frame.hour, [-1, 7, 17, 23], labels=["night 0-7", "day 8-17", "evening 18-23"]).astype(str)
    b["anonymous"] = np.where(frame.anonymous == 1, "yes", "no")
    b["problems"] = pd.cut(frame.n_problems, [0, 1, 2, 3, 5, 99], labels=["1", "2", "3", "4-5", "6+"]).astype(str)
    cnt = lambda s: pd.cut(s, [-1, 0, 2, 9, 29, 1e9], labels=["0", "1-2", "3-9", "10-29", "30+"]).astype(str)
    for k in ["p6", "p12", "n12", "pall", "c7", "c30"]:
        b[k] = cnt(frame[k])
    b["rate12"] = pd.cut(frame.rate12, [0, .2, .35, .5, .65, 1], labels=["<0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", ">0.65"]).astype(str)
    b["months_since_p"] = pd.cut(frame.months_since_p.fillna(999), [0, 1, 3, 12, 36, 998, 1e4],
                                 labels=["1", "2-3", "4-12", "13-36", "37+", "never"]).astype(str)
    b["days_since_c"] = pd.cut(frame.days_since_c, [-1, 1, 7, 30, 59.99, 61], labels=["<1", "1-7", "7-30", "30-60", "60+ or none"]).astype(str)
    for k in MAJ:                                                # does the complaint include this kind of problem
        b["has_" + k[4:]] = np.where(frame[k] > 0, "yes", "no")
    return b


def logistic(fit_frame, C, extra=None):
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200)
    B = bins(fit_frame) if extra is None else bins(fit_frame).assign(cluster=fit_frame[extra].astype(str))
    m = LogisticRegression(penalty="l1", C=C, solver="liblinear", random_state=SEED).fit(enc.fit_transform(B), fit_frame.label)
    def score(frame):
        Bf = bins(frame) if extra is None else bins(frame).assign(cluster=frame[extra].astype(str))
        return m.predict_proba(enc.transform(Bf))[:, 1]
    return m, enc, score


def boosting(fit_frame, params, extra=None):
    num = NUM + ([extra] if extra else [])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=250)
    X = lambda frame: np.hstack([enc.transform(frame[HGB_CAT]), frame[num].to_numpy(float)])
    enc.fit(fit_frame[HGB_CAT])
    m = HistGradientBoostingClassifier(categorical_features=list(range(len(HGB_CAT))), early_stopping=False,
                                       random_state=SEED, **params).fit(X(fit_frame), fit_frame.label)
    return m, X, lambda frame: m.predict_proba(X(frame))[:, 1]


# ------------------------------------------------------------------ tuning on the validation weeks (the locked metric)
rows = []
for C in [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]:
    m, _, score = logistic(fit, C)
    rows.append({"model": "logistic", "setting": f"C={C}", "nonzero": int((m.coef_ != 0).sum()),
                 "val_recall": recall(val, score(val)), "val_auc": roc_auc_score(val.label, score(val))})
GRID = [dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, max_iter=it, l2_regularization=1.0)
        for lr, it in [(0.05, 400), (0.1, 200)] for leaves in [15, 63] for msl in [50, 500]]
for p in GRID:
    _, _, score = boosting(fit, p)
    rows.append({"model": "boosting", "setting": str({k: v for k, v in p.items() if k != "l2_regularization"}),
                 "val_recall": recall(val, score(val)), "val_auc": roc_auc_score(val.label, score(val)), "params": p})
tune = pd.DataFrame(rows)
tune["val_baseline"] = recall(val, val.n_problems)
tune.drop(columns="params").round(4).to_csv(OUT / "tuning.csv", index=False)
print("\n=== tuning (validation weeks) ===")
print(tune.drop(columns="params").round(4).to_string(index=False))
lg = tune[tune.model == "logistic"]
best_C = float(lg.loc[lg.val_recall.idxmax(), "setting"][2:])
best_P = tune.loc[tune[tune.model == "boosting"].val_recall.idxmax(), "params"]
print(f"chosen: logistic C={best_C}; boosting {best_P}")

# ------------------------------------------------------------------ refit on all of train, score the test weeks once
lr_model, lr_enc, lr_score = logistic(train, best_C)
gb_model, gb_X, gb_score = boosting(train, best_P)
S = {"Random": np.zeros(len(test)), "Baseline: most problems first": test.n_problems.values,
     "Logistic (readable)": lr_score(test), "Gradient boosting (flexible)": gb_score(test)}
W = {name: weekly(test, select(test, s)) for name, s in S.items()}
base = W["Baseline: most problems first"]
head = []
for name, w in W.items():
    lo, hi = ci(w.recall); plo, phi = ci(w.precision); dlo, dhi = ci(w.recall - base.recall)
    head.append({"ranking": name, "recall@25%": w.recall.mean(), "lo": lo, "hi": hi, "precision@25%": w.precision.mean(),
                 "p_lo": plo, "p_hi": phi, "minus baseline": (w.recall - base.recall).mean(), "d_lo": dlo, "d_hi": dhi,
                 "weeks ahead of baseline": (w.recall > base.recall).mean(),
                 "violations found per week": w.hit.mean(), "AUC": roc_auc_score(test.label, S[name]) if name != "Random" else 0.5})
head = pd.DataFrame(head)
d = W["Gradient boosting (flexible)"].recall - W["Logistic (readable)"].recall
print("\n=== headline (test weeks) ===")
print(head.round(3).to_string(index=False))
print(f"ceiling: a perfect ranking would reach recall {base.ceiling.mean():.3f} (25% of the pool against a positive rate of "
      f"{test.label.mean():.3f}); per week: pool {base.n.mean():.0f}, selected {base.sel.mean():.0f}, violations {base.pos.mean():.0f}")
print(f"boosting minus logistic: {d.mean():+.4f}, 95% interval {ci(d)[0]:+.4f} to {ci(d)[1]:+.4f}, ahead in {(d > 0).mean():.0%} of weeks")
head.round(4).to_csv(OUT / "headline.csv", index=False)
pd.concat({k: w for k, w in W.items()}, names=["ranking"]).round(4).to_csv(OUT / "weekly.csv")

# what the readable model says
names = lr_enc.get_feature_names_out()
coef = pd.DataFrame({"feature": names, "coefficient": lr_model.coef_[0]})
coef = coef[coef.coefficient != 0].assign(odds_ratio=lambda t: np.exp(t.coefficient)).sort_values("coefficient")
coef.round(3).to_csv(OUT / "logistic_coefficients.csv", index=False)
print(f"\nlogistic: {len(coef)} of {len(names)} coefficients nonzero, intercept {lr_model.intercept_[0]:+.3f}")
print("lowest:");  print(coef.head(12).round(3).to_string(index=False))
print("highest:"); print(coef.tail(12).round(3).to_string(index=False))

# what the flexible model leans on: drop in AUC when one feature is shuffled (on a test sample; nothing is chosen from it)
samp = test.sample(40000, random_state=SEED)
pi = permutation_importance(gb_model, gb_X(samp), samp.label, scoring="roc_auc", n_repeats=3, random_state=SEED)
imp = pd.DataFrame({"feature": HGB_CAT + NUM, "auc_drop": pi.importances_mean}).sort_values("auc_drop", ascending=False)
imp.round(4).to_csv(OUT / "boosting_importance.csv", index=False)
print("\nboosting, AUC lost when a feature is shuffled (top 12):"); print(imp.head(12).round(4).to_string(index=False))

# ------------------------------------------------------------------ slices
SEL = {name: select(test, s) for name, s in S.items() if name != "Random"}
def slice_table(col, values, kind):
    out = []
    for v in sorted(pd.unique(values)):
        mask = (values == v).values
        row = {"variable": col, "slice": v, "kind": kind, "complaints": int(mask.sum()), "violations": int(test.label[mask].sum()),
               "positive rate": test.label[mask].mean()}
        for name, sel in SEL.items():
            hit = pd.Series((sel & (test.label == 1)).values & mask).groupby(test.week.values).sum()
            pos = pd.Series(((test.label == 1).values & mask)).groupby(test.week.values).sum()
            took = pd.Series(sel.values & mask).groupby(test.week.values).sum()
            h, p = hit.values[BOOT].sum(axis=1), pos.values[BOOT].sum(axis=1)
            lo, hi = np.percentile(h / np.maximum(p, 1), [2.5, 97.5])
            short = name.split(" ")[0].strip(":").lower()
            row.update({f"{short} recall": hit.sum() / max(pos.sum(), 1), f"{short} lo": lo, f"{short} hi": hi,
                        f"{short} share selected": took.sum() / mask.sum()})
        out.append(row)
    return out

heat = np.where(test.maj_heat_hot_water > 0, "heat / hot water", "no heat problem")
multi = np.where(test.n_problems > 1, "2+ problems", "1 problem")
pre = (slice_table("borough", test.borough, "named in proposal") + slice_table("heat", pd.Series(heat, index=test.index), "named in proposal")
       + slice_table("HPD triage type", test.type, "named in proposal") + slice_table("problems", pd.Series(multi, index=test.index), "named in proposal"))
hist_bin = pd.cut(test.p12, [-1, 0, 9, 1e9], labels=["0 violations, 12 m", "1-9", "10+"]).astype(str)
quarter = test.rec.dt.to_period("Q").astype(str)
expl = (slice_table("first problem's major category", test.major, "exploratory")
        + slice_table("building violations, prior 12 months", hist_bin, "exploratory")
        + slice_table("received quarter", quarter, "exploratory") + slice_table("unit type", test.unit_type, "exploratory")
        + slice_table("building had a labeled train complaint", pd.Series(np.where(seen, "yes", "no"), index=test.index), "exploratory"))
sl = pd.DataFrame(pre + expl)
sl.round(4).to_csv(OUT / "slices.csv", index=False)
show = ["variable", "slice", "complaints", "violations", "positive rate", "baseline recall", "logistic recall", "logistic lo",
        "logistic hi", "gradient recall", "gradient lo", "gradient hi", "gradient share selected"]
print("\n=== slices named in the proposal (recall = share of the slice's violations that reach the weekly top 25%) ===")
print(sl.loc[sl.kind == "named in proposal", show].round(3).to_string(index=False))
print("\n=== exploratory slices (found by looking; not named in advance) ===")
print(sl.loc[sl.kind == "exploratory", show].round(3).to_string(index=False))

# ------------------------------------------------------------------ unsupervised check: building-history clusters
HIST = ["pall", "nall", "p6", "p12", "n12", "c7", "c30"]
def profile(frame):
    return np.column_stack([np.log1p(frame[HIST].to_numpy(float)), frame.rate12, np.log1p(frame.months_since_p.fillna(300))])
sc = StandardScaler().fit(profile(fit))
Zs = sc.transform(profile(fit.sample(20000, random_state=SEED)))
sil = {k: silhouette_score(Zs, KMeans(k, n_init=10, random_state=SEED).fit_predict(Zs)) for k in range(2, 9)}
k_best = max((k for k in sil if k >= 3), key=sil.get)           # at least 3 groups, or the split says only "history / none"
print("\n=== clustering check: k-means on the building-history features ===")
print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> k =", k_best)
km = KMeans(k_best, n_init=10, random_state=SEED).fit(sc.transform(profile(fit)))
order = np.argsort(np.argsort(km.cluster_centers_[:, HIST.index("p12")]))   # renumber: 0 = least violation history
for frame in (fit, val, train, test):
    frame.insert(len(frame.columns), "cluster", order[km.predict(sc.transform(profile(frame)))])
desc = train.groupby("cluster")[["p12", "n12", "pall", "c30", "rate12", "label"]].median().join(train.groupby("cluster").size().rename("train complaints"))
desc["label"] = train.groupby("cluster").label.mean()
print("cluster medians on train (label = positive rate):"); print(desc.round(2).to_string())
_, _, s1 = logistic(fit, best_C, extra="cluster"); _, _, s2 = boosting(fit, best_P, extra="cluster")
gain = {"logistic": recall(val, s1(val)) - lg.val_recall.max(), "boosting": recall(val, s2(val)) - tune[tune.model == "boosting"].val_recall.max()}
print(f"validation recall change when the cluster is added as a feature: logistic {gain['logistic']:+.4f}, boosting {gain['boosting']:+.4f}")
cl = pd.DataFrame(slice_table("building-history cluster", test.cluster, "unsupervised"))
cl = cl.join(desc[["p12", "n12", "c30"]].add_prefix("train median ").reset_index(drop=True))
cl["validation gain, logistic"], cl["validation gain, boosting"] = gain["logistic"], gain["boosting"]
cl.round(4).to_csv(OUT / "clusters.csv", index=False)
print(cl[show[1:]].round(3).to_string(index=False))

test.assign(**{"score_logistic": S["Logistic (readable)"], "score_boosting": S["Gradient boosting (flexible)"],
               "sel_baseline": SEL["Baseline: most problems first"].values, "sel_logistic": SEL["Logistic (readable)"].values,
               "sel_boosting": SEL["Gradient boosting (flexible)"].values}).to_parquet(DATA / "model" / ("test_scores_with_lot.parquet" if KEEP_LOT else "test_scores.parquet"))
