"""Candidate baselines and N on the proposed split: train received 2024-10-01 to 2025-09-30, test 2026-01-01 to
2026-07-31. Complaint positive if any problem is P, negative if any is N and none is P. Rates fitted on train only.
Also compares taking a complaint's category from its lowest problem_id with the row-order `first` used in 06_inspect.py,
where rows are concatenated by outcome class. Writes outputs/inspect/baseline_split.csv.
"""
import numpy as np
import pandas as pd
from hpd_api import DATA, OUTPUTS

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
rng = np.random.default_rng(0)
files = [f for f in sorted((DATA / "rows").glob("*.parquet")) if f.name != "complaints.parquet"]
rows = pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)   # same order as 06_inspect.py
rows["received_date"] = pd.to_datetime(rows.received_date)
rows["problem_id"] = rows.problem_id.astype("int64")
legacy = rows.groupby("complaint_id").major_category.first()
rows = rows.sort_values("problem_id")
rows = rows[rows.group.isin(["P", "N"]) | rows.complaint_id.isin(rows.loc[rows.group.isin(["P", "N"]), "complaint_id"])]
rows["minor"] = rows.major_category.fillna("") + " | " + rows.minor_category.fillna("")
g = rows.groupby("complaint_id")
c = g.agg(received=("received_date", "min"), cat=("major_category", "first"), minor=("minor", "first"),
          type=("type", "first"), n_problems=("group", "size"))
c["has_P"] = (rows.group == "P").groupby(rows.complaint_id).any()
c["has_N"] = (rows.group == "N").groupby(rows.complaint_id).any()
c = c[c.has_P | c.has_N].copy()
c["label"] = c.has_P.astype(int)
c["legacy_cat"] = legacy.reindex(c.index)
c["week"] = c.received.dt.to_period("W-SUN").dt.start_time
print(f"category differs between row-order and lowest problem_id: {(c.cat != c.legacy_cat).mean():.2%} of labeled complaints")

tr = c[(c.received >= "2024-10-01") & (c.received < "2025-10-01")].copy()
te = c[(c.received >= "2026-01-05") & (c.received < "2026-07-27")].copy()
print(f"train {len(tr):,} (P rate {tr.label.mean():.3f}); test {len(te):,} (P rate {te.label.mean():.3f}), {te.week.nunique()} weeks; "
      f"test pool per week median {te.groupby('week').size().median():.0f}, min {te.groupby('week').size().min()}")

def rate(key, shrink=20):
    base = tr.label.mean()
    a = tr.groupby(key).label.agg(["sum", "count"])
    return (a["sum"] + shrink * base) / (a["count"] + shrink)
def mapped(frame, key, r):
    return frame[key].map(r).fillna(tr.label.mean())

# highest rate across a complaint's problems (a complaint with several problems is scored by its worst one)
def max_over_problems(col, r, ids):
    s = rows[rows.complaint_id.isin(ids)]
    return s[col].map(r).fillna(tr.label.mean()).groupby(s.complaint_id).max().reindex(ids)

R_major, R_minor, R_type = rate("cat"), rate("minor"), rate("type")
tr["nb"] = tr.n_problems.clip(upper=5); te["nb"] = te.n_problems.clip(upper=5)
R_n = rate("nb")
scores = {
    "random": pd.Series(0.0, index=te.index),
    "HPD triage type rate": mapped(te, "type", R_type),
    "problems in complaint": mapped(te, "nb", R_n),
    "major category, row-order (06_inspect.py)": te.legacy_cat.map(rate("legacy_cat")).fillna(tr.label.mean()),
    "major category, first problem": mapped(te, "cat", R_major),
    "major category, highest across problems": max_over_problems("major_category", R_major, te.index),
    "minor category, first problem": mapped(te, "minor", R_minor),
    "minor category, highest across problems": max_over_problems("minor", R_minor, te.index),
}
def evaluate(score, frac, frame=te):
    prec, rec = [], []
    d = frame.assign(s=score.reindex(frame.index) + 1e-6 * rng.random(len(frame)))
    for _, w in d.groupby("week"):
        k = max(1, round(frac * len(w)))
        top = w.nlargest(k, "s")
        prec.append(top.label.mean()); rec.append(top.label.sum() / max(1, w.label.sum()))
    return np.mean(prec), np.mean(rec)
out = {}
for name, s in scores.items():
    r = {}
    for f in (.05, .10, .20):
        p, re_ = evaluate(s, f)
        r[f"P@{int(f*100)}%"] = round(p, 3); r[f"R@{int(f*100)}%"] = round(re_, 3)
    out[name] = r
tab = pd.DataFrame(out).T
tab.to_csv(OUTPUTS / "inspect" / "baseline_split.csv")
print(tab)

lot = "LINE OF TRAVEL"
print("\nwithout LINE OF TRAVEL complaints in the test weeks:")
nolot = te[te.cat != lot]
for name in ["random", "major category, first problem", "minor category, first problem"]:
    print(f"  {name}: P@10% {evaluate(scores[name], .10, nolot)[0]:.3f}")
print("\nLINE OF TRAVEL")
l = rows[rows.major_category == lot]
print(l.groupby("minor_category").group.value_counts().unstack(fill_value=0).to_string())
print(l.groupby(l.received_date.dt.to_period("Q")).complaint_id.nunique().to_string())
print("type:", l.type.value_counts().to_dict(), " unit_type:", l.unit_type.value_counts().head(3).to_dict())
print(f"labeled complaints per test week {te[te.cat == lot].groupby('week').size().median():.0f}; P rate train {tr[tr.cat == lot].label.mean():.3f}, test {te[te.cat == lot].label.mean():.3f}")
top = te.assign(s=scores["minor category, first problem"] + 1e-6 * rng.random(len(te)))
slots = top.groupby("week", group_keys=False).apply(lambda w: w.nlargest(max(1, round(.1 * len(w))), "s"))
print("\ntop-10% slots by major category (minor-category rule):", slots.cat.value_counts(normalize=True).head(6).round(3).to_dict())
print("top minor categories by train rate (n >= 200):")
a = tr.groupby("minor").label.agg(["mean", "count"]); print(a[a["count"] >= 200].sort_values("mean", ascending=False).head(10).round(3).to_string())
best = scores["minor category, highest across problems"]
print(f"\nhighest-minor rule without LINE OF TRAVEL complaints: P@10% {evaluate(best, .10, te[~te.index.isin(rows.loc[rows.major_category == lot, 'complaint_id'])])[0]:.3f}")
one = te[te.n_problems == 1]
print(f"single-problem complaints only ({len(one):,}): random P@10% {evaluate(scores['random'], .10, one)[0]:.3f}, highest-minor rule {evaluate(best, .10, one)[0]:.3f}")
weekly = te.assign(s=best + 1e-6 * rng.random(len(te))).groupby("week").apply(lambda w: w.nlargest(max(1, round(.1 * len(w))), "s").label.mean())
print(f"highest-minor rule, weekly P@10%: min {weekly.min():.3f}, median {weekly.median():.3f}, max {weekly.max():.3f}")
allc = pd.concat(pd.read_parquet(f, columns=["complaint_id", "received_date"]) for f in files)
allc["received_date"] = pd.to_datetime(allc.received_date)
allc = allc[(allc.received_date >= "2026-01-05") & (allc.received_date < "2026-07-27")].drop_duplicates("complaint_id")
k = te.groupby("week").size().median() * .10
print(f"all complaints per test week, median {allc.groupby(allc.received_date.dt.to_period('W-SUN')).size().median():.0f}; top 10% of pool is about {k:.0f}")
