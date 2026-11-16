"""What-if runs for the headline: change one value and see what it does to recall at the budget, in seconds.

Same split, metric, tie-break, bootstrap and model code as 10_models.py, without the tuning grid, permutation
importance, slices or clustering. Each model's test scores are cached by setting, so changing the budget, or
returning to a setting already fitted, needs no refit.

    python headline.py                          # chosen settings, 25% budget (matches outputs/model/headline.csv)
    python headline.py --budget 0.10            # another visit budget (instant: cached scores)
    python headline.py --C 0.01                 # refit the logistic model with another penalty
    python headline.py --lr 0.1 --trees 200 --leaves 63 --min-leaf 50
    python headline.py --keep-lot               # LINE OF TRAVEL kept (the earlier run's settings)
    python headline.py --models baseline,logistic
    python headline.py --check                  # assert the defaults reproduce outputs/model/headline.csv
    python headline.py --warm                   # pre-fit the whole tuning grid once (about 5 min), so every grid
                                                # value is instant afterwards

Times: a budget change or a cached setting takes a few seconds; a new boosting setting 5-15 s; a new logistic C
up to a minute (liblinear converges slowly when the penalty is weak, e.g. 48 s at C=0.1, 5 s at C=0.01).

Run 09_features.py first. Cache: data/model/cache/, keyed by the setting and the feature table's size and
modification time, so editing a feature in 09_features.py and rerunning it (15 s) forces fresh fits.
"""
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from hpd_api import DATA, OUTPUTS

warnings.filterwarnings("ignore", category=FutureWarning)
SEED, DRAWS = 0, 2000
# settings 10_models.py chose on the validation weeks (LINE OF TRAVEL removed / kept)
CHOSEN = {False: dict(C=0.1, lr=0.05, leaves=15, min_leaf=500, trees=400),
          True: dict(C=0.03, lr=0.05, leaves=15, min_leaf=50, trees=400)}
CACHE = DATA / "model" / "cache"


def load(keep_lot):
    """Complaint table and the locked split, built exactly as in 10_models.py."""
    c = pd.read_parquet(DATA / "model" / "complaints_features.parquet").sort_values("rec")
    c["tie"] = np.random.default_rng(SEED).random(len(c))
    share = c.groupby("week").size().cumsum() / len(c)
    cut = share.index[share.values >= 0.70][0] + pd.Timedelta(days=7)
    if not keep_lot:
        c = c[~(c.maj_line_of_travel > 0)].drop(columns="maj_line_of_travel")
    return c[c.week < cut], c[c.week >= cut]


def select(frame, score, k_share):
    d = pd.DataFrame({"week": frame.week.values, "s": np.asarray(score, float), "tie": frame.tie.values}, index=frame.index)
    d = d.sort_values(["week", "s", "tie"], ascending=[True, False, False])
    rank = d.groupby("week").cumcount()
    k = np.maximum(1, (k_share * d.groupby("week").s.transform("size")).round())
    return (rank < k).reindex(frame.index)


def weekly(frame, sel):
    g = pd.DataFrame({"week": frame.week, "y": frame.label, "sel": sel, "hit": sel & (frame.label == 1)}).groupby("week")
    w = g.agg(n=("y", "size"), pos=("y", "sum"), sel=("sel", "sum"), hit=("hit", "sum"))
    w["recall"], w["precision"] = w.hit / w.pos, w.hit / w.sel
    w["ceiling"] = np.minimum(1, w.sel / w.pos)
    return w


# ------------------------------------------------------------------ the two models, as in 10_models.py
CAT = ["minor", "type", "borough", "cb", "unit_type", "space_type"]


def bins(frame, maj):
    b = frame[CAT].copy()
    b["month"] = frame.month.astype(str).str.zfill(2)
    b["weekend"] = np.where(frame.weekday >= 5, "yes", "no")
    b["hours"] = pd.cut(frame.hour, [-1, 7, 17, 23], labels=["night 0-7", "day 8-17", "evening 18-23"]).astype(str)
    b["anonymous"] = np.where(frame.anonymous == 1, "yes", "no")
    b["problems"] = pd.cut(frame.n_problems, [0, 1, 2, 3, 5, 99], labels=["1", "2", "3", "4-5", "6+"]).astype(str)
    cnt = lambda s: pd.cut(s, [-1, 0, 2, 9, 29, 1e9], labels=["0", "1-2", "3-9", "10-29", "30+"]).astype(str)  # noqa: E731
    for k in ["p6", "p12", "n12", "pall", "c7", "c30"]:
        b[k] = cnt(frame[k])
    b["rate12"] = pd.cut(frame.rate12, [0, .2, .35, .5, .65, 1], labels=["<0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", ">0.65"]).astype(str)
    b["months_since_p"] = pd.cut(frame.months_since_p.fillna(999), [0, 1, 3, 12, 36, 998, 1e4],
                                 labels=["1", "2-3", "4-12", "13-36", "37+", "never"]).astype(str)
    b["days_since_c"] = pd.cut(frame.days_since_c, [-1, 1, 7, 30, 59.99, 61], labels=["<1", "1-7", "7-30", "30-60", "60+ or none"]).astype(str)
    for k in maj:
        b["has_" + k[4:]] = np.where(frame[k] > 0, "yes", "no")
    return b


def logistic_scores(train, test, maj, C):
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200)
    m = LogisticRegression(penalty="l1", C=C, solver="liblinear", random_state=SEED).fit(enc.fit_transform(bins(train, maj)), train.label)
    return m.predict_proba(enc.transform(bins(test, maj)))[:, 1]


def boosting_scores(train, test, maj, lr, leaves, min_leaf, trees):
    num = ["n_problems", "anonymous", "month", "weekday", "hour", "pall", "nall", "p6", "p12", "n12", "rate12",
           "months_since_p", "c7", "c30", "days_since_c"] + maj
    hgb_cat = CAT + ["zip"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=250).fit(train[hgb_cat])
    X = lambda f: np.hstack([enc.transform(f[hgb_cat]), f[num].to_numpy(float)])  # noqa: E731
    m = HistGradientBoostingClassifier(categorical_features=list(range(len(hgb_cat))), early_stopping=False, random_state=SEED,
                                       learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf, max_iter=trees,
                                       l2_regularization=1.0).fit(X(train), train.label)
    return m.predict_proba(X(test))[:, 1]


def cached(name, params, fit):
    """Test scores for one model setting, from the cache or by fitting once."""
    feats = DATA / "model" / "complaints_features.parquet"
    st = feats.stat()   # rebuilding the feature table (09_features.py) changes the key, so stale scores are never reused
    key = hashlib.sha1(json.dumps([name, params, st.st_size, st.st_mtime_ns], sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f"{name}-{key}.npy"
    if path.exists():
        return np.load(path), "cached"
    t0 = time.time()
    s = fit()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, s)
    return s, f"fitted in {time.time() - t0:.0f} s"


def warm():
    """Fit every setting in 10_models.py's tuning grid, with LINE OF TRAVEL removed and kept, into the cache."""
    grid = [(lr, it, leaves, msl) for lr, it in [(0.05, 400), (0.1, 200)] for leaves in [15, 63] for msl in [50, 500]]
    for keep in (False, True):
        train, test = load(keep)
        maj = [k for k in train.columns if k.startswith("maj_")]
        tag = "kept" if keep else "removed"
        for C in [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]:
            print(f"logistic {tag} C={C}:", cached("logistic", [tag, C], lambda: logistic_scores(train, test, maj, C))[1], flush=True)
        for lr, it, leaves, msl in grid:
            bp = [lr, leaves, msl, it]
            print(f"boosting {tag} {bp}:", cached("boosting", [tag] + bp, lambda: boosting_scores(train, test, maj, *bp))[1], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=0.25, help="share of each week's pool visited (locked: 0.25)")
    ap.add_argument("--keep-lot", action="store_true", help="keep LINE OF TRAVEL complaints")
    ap.add_argument("--models", default="baseline,logistic,boosting")
    ap.add_argument("--C", type=float, help="logistic: inverse L1 penalty")
    ap.add_argument("--lr", type=float, help="boosting: learning rate")
    ap.add_argument("--trees", type=int, help="boosting: number of trees (max_iter)")
    ap.add_argument("--leaves", type=int, help="boosting: max_leaf_nodes")
    ap.add_argument("--min-leaf", type=int, help="boosting: min_samples_leaf")
    ap.add_argument("--check", action="store_true", help="assert the defaults reproduce the saved headline")
    ap.add_argument("--warm", action="store_true", help="pre-fit the tuning grid (both LINE OF TRAVEL choices) into the cache")
    a = ap.parse_args()
    if a.warm:
        return warm()
    p = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in CHOSEN[a.keep_lot].items()}
    models = a.models.split(",")

    train, test = load(a.keep_lot)
    maj = [k for k in train.columns if k.startswith("maj_")]
    scores = {"Random": (np.zeros(len(test)), ""), "Baseline: most problems first": (test.n_problems.values, "")}
    tag = "kept" if a.keep_lot else "removed"
    if "logistic" in models:
        scores["Logistic (readable)"] = cached("logistic", [tag, p["C"]], lambda: logistic_scores(train, test, maj, p["C"]))
    if "boosting" in models:
        bp = [p["lr"], p["leaves"], p["min_leaf"], p["trees"]]
        scores["Gradient boosting (flexible)"] = cached("boosting", [tag] + bp, lambda: boosting_scores(train, test, maj, *bp))

    W = {name: weekly(test, select(test, s, a.budget)) for name, (s, _) in scores.items()}
    base = W["Baseline: most problems first"]
    boot = np.random.default_rng(SEED).integers(0, len(base), (DRAWS, len(base)))
    ci = lambda v: np.percentile(np.asarray(v, float)[boot].mean(axis=1), [2.5, 97.5])  # noqa: E731
    rows = []
    for name, w in W.items():
        lo, hi = ci(w.recall); d = w.recall - base.recall; dlo, dhi = ci(d)
        rows.append({"ranking": name, "recall": w.recall.mean(), "lo": lo, "hi": hi, "lead": d.mean(), "lead_lo": dlo,
                     "lead_hi": dhi, "precision": w.precision.mean(), "viol_week": w.hit.mean(),
                     "weeks ahead": f"{(w.recall > base.recall).sum()}/{len(w)}", "note": scores[name][1]})
    out = pd.DataFrame(rows)

    print(f"LINE OF TRAVEL {tag}; train {len(train):,}, test {len(test):,} complaints in {len(base)} weeks; budget {a.budget:.0%}")
    print(f"settings: logistic C={p['C']}; boosting lr={p['lr']}, leaves={p['leaves']}, min leaf={p['min_leaf']}, trees={p['trees']}")
    print(f"per week: pool {base.n.mean():.0f}, visits {base.sel.mean():.0f}, violations {base.pos.mean():.0f}; "
          f"best possible recall {base.ceiling.mean():.3f}\n")
    for r in out.itertuples():
        lead = "" if r.ranking.startswith(("Random", "Baseline")) else f"  lead {r.lead:+.3f} ({r.lead_lo:+.3f} to {r.lead_hi:+.3f})"
        print(f"  {r.ranking:30s} recall {r.recall:.3f} ({r.lo:.3f} to {r.hi:.3f})  precision {r.precision:.3f}  "
              f"{r.viol_week:5.0f} viol/week{lead}  {r.note}")
    if len(scores) == 4:
        d = W["Gradient boosting (flexible)"].recall - W["Logistic (readable)"].recall
        print(f"\n  boosting minus logistic {d.mean():+.4f} ({ci(d)[0]:+.4f} to {ci(d)[1]:+.4f}), ahead in {(d > 0).mean():.0%} of weeks")

    if a.check:
        saved = pd.read_csv(OUTPUTS / ("model_with_lot" if a.keep_lot else "model") / "headline.csv").set_index("ranking")
        for r in out.itertuples():
            for mine, theirs in [("recall", "recall@25%"), ("lo", "lo"), ("hi", "hi"), ("lead", "minus baseline")]:
                assert abs(getattr(r, mine) - saved.loc[r.ranking, theirs]) < 1e-4, (r.ranking, mine)
        print("\ncheck passed: matches", "outputs/model_with_lot/headline.csv" if a.keep_lot else "outputs/model/headline.csv")


if __name__ == "__main__":
    main()
