"""Descriptive statistics and pre-test validation for the baseline choice, at the top 25% of each week's labeled pool.
Nothing received on or after 2025-10-01 is used. Writes outputs/inspect/baseline_validation.csv and feature_spread.csv.
  Check A, off-season: rates fitted on 2024-10-01..2025-03-31, scored on 2025-04-01..2025-09-30.
  Check B, heat season: rates fitted on 2024-01-01..2024-09-30, scored on 2024-10-01..2025-03-31 (crosses the Oct 2024 shift).
"""
import numpy as np
import pandas as pd
from hpd_api import DATA, OUTPUTS

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200, "display.max_colwidth", 60)
END = pd.Timestamp("2025-10-01")
rng = np.random.default_rng(0)
K = .25
fs = [f for f in sorted((DATA / "rows").glob("*.parquet")) if f.name != "complaints.parquet"]
r = pd.concat(pd.read_parquet(f) for f in fs).reset_index(drop=True)
r["received_date"] = pd.to_datetime(r.received_date)
r["problem_status_date"] = pd.to_datetime(r.problem_status_date)
r["problem_id"] = r.problem_id.astype("int64"); r["building_id"] = r.building_id.astype("int64")
r = r[r.received_date < END].sort_values("problem_id")
keep = r.loc[r.group.isin(["P", "N"]), "complaint_id"].unique()
r = r[r.complaint_id.isin(keep)].copy()
r["minor"] = r.major_category.fillna("") + " | " + r.minor_category.fillna("")
g = r.groupby("complaint_id")
c = g.agg(rec=("received_date", "min"), major=("major_category", "first"), minor=("minor", "first"), type=("type", "first"),
          borough=("borough", "first"), unit=("unit_type", "first"), building_id=("building_id", "first"), n=("group", "size"))
c["label"] = (r.group == "P").groupby(r.complaint_id).any().astype(int)
pn = r[r.group.isin(["P", "N"])]
c["days"] = (pn.groupby("complaint_id").problem_status_date.min().dt.normalize() - c.rec.dt.normalize()).dt.days
c["wk"] = c.rec.dt.to_period("W-SUN").dt.start_time
c["nb"] = c.n.clip(upper=5)
c["midx"] = (c.rec.dt.year - 2003) * 12 + c.rec.dt.month - 1

def hist(tag):
    h = pd.read_parquet(DATA / "history" / f"{tag}_by_building_month.parquet")
    h["hm"] = (h.m.dt.year - 2003) * 12 + h.m.dt.month - 1
    h["hm"] = h.hm.astype("int64"); h = h.sort_values(["building_id", "hm"]); h["cum"] = h.groupby("building_id").n.cumsum()
    return h[["building_id", "hm", "cum"]].sort_values("hm")
def asof(h, upto):
    key = pd.DataFrame({"building_id": c.building_id.values, "hm": (upto - 1).astype("int64").values, "i": c.index}).sort_values("hm")
    return pd.merge_asof(key, h, on="hm", by="building_id", direction="backward").set_index("i").cum.fillna(0).reindex(c.index)
hp, hn = hist("p"), hist("n")
c["p6"] = asof(hp, c.midx) - asof(hp, c.midx - 6)
c["p12"] = asof(hp, c.midx) - asof(hp, c.midx - 12)
c["n12"] = asof(hn, c.midx) - asof(hn, c.midx - 12)
c["p6b"] = pd.cut(c.p6, [-1, 0, 2, 9, 1e9], labels=["0", "1-2", "3-9", "10+"]).astype(str)

win = c[c.rec >= "2024-10-01"]
print(f"training window: {len(win):,} labeled complaints, positive rate {win.label.mean():.3f}; "
      f"{win.minor.nunique()} minor categories, {win.major.nunique()} major")

# ---- 1. how much each field separates positives from negatives
def spread(col):
    a = win.groupby(col).label.agg(["mean", "size"])
    big = a[a["size"] >= 200]
    p = win.label.mean()
    between = (a["size"] * (a["mean"] - p) ** 2).sum() / a["size"].sum()
    return {"groups (n>=200)": len(big), "lowest rate": big["mean"].min(), "highest rate": big["mean"].max(),
            "variance explained": between / (p * (1 - p))}
sp = pd.DataFrame({k: spread(v) for k, v in {"borough": "borough", "unit type": "unit", "HPD triage type": "type",
     "prior violations 6m": "p6b", "problems in complaint": "nb", "major category": "major", "minor category": "minor"}.items()}).T.round(3)
sp.to_csv(OUTPUTS / "inspect" / "feature_spread.csv"); print("\n", sp)
print("\npositive rate by problems in complaint:"); print(win.groupby("nb").label.agg(["mean", "size"]).round(3).T.to_string())
print("\npositive rate by unit type:"); print(win.groupby("unit").label.agg(["mean", "size"]).round(3).to_string())
print("unit type share by major category (top):"); print(pd.crosstab(win.unit, win.major).idxmax(axis=1).to_string())
print("\npositive rate by HPD triage type:"); print(win.groupby("type").label.agg(["mean", "size"]).round(3).to_string())
a = win.groupby("minor").label.agg(["mean", "size"]); a = a[a["size"] >= 200].sort_values("mean")
print("\nlowest minor categories:"); print(a.head(5).round(3).to_string()); print("highest:"); print(a.tail(8).round(3).to_string())
print("\nhow current speed relates to outcome (training window, days to verdict):")
print(win.groupby(pd.cut(win.days, [-1, 2, 7, 30, 1e5], labels=["0-2", "3-7", "8-30", "31+"])).label.agg(["mean", "size"]).round(3).to_string())

# ---- 2. pre-test validation
def weeks(frame, a, b):
    w = frame[(frame.rec >= a) & (frame.rec < b)]
    n = w.groupby("wk").rec.agg(["min", "max"])
    ok = n.index[(n.index >= pd.Timestamp(a)) & (n.index + pd.Timedelta(days=7) <= pd.Timestamp(b))]
    return w[w.wk.isin(ok)]
def run(fit_a, fit_b, ev_a, ev_b):
    tr, ev = c[(c.rec >= fit_a) & (c.rec < fit_b)], weeks(c, ev_a, ev_b)
    base = tr.label.mean()
    def rate(col, frame=tr, s=20):
        x = frame.groupby(col).label.agg(["sum", "count"])
        return (x["sum"] + s * base) / (x["count"] + s)
    rm, rj = rate("major"), rate("minor")
    prob = r[r.complaint_id.isin(ev.index)]
    def hi(col, m):
        return prob[col].map(m).fillna(base).groupby(prob.complaint_id).max().reindex(ev.index)
    R = {"Random": pd.Series(0., index=ev.index),
         "HPD current order (fastest verdict first)": -ev.days.astype(float),
         "HPD triage type": ev.type.map(rate("type")).fillna(base),
         "Prior violations at building, 6m": ev.p6.astype(float),
         "Building violation rate, 12m": (ev.p12 + 2 * base) / (ev.p12 + ev.n12 + 2),
         "Unit type": ev.unit.map(rate("unit")).fillna(base),
         "Problems in complaint": ev.nb.map(rate("nb")).fillna(base),
         "Major category, first problem": ev.major.map(rm).fillna(base),
         "Major category, highest across problems": hi("major_category", rm),
         "Minor category, first problem": ev.minor.map(rj).fillna(base),
         "Minor category, highest across problems": hi("minor", rj)}
    res, wk = {}, {}
    for name, s in R.items():
        d = ev.assign(s=s + 1e-6 * rng.random(len(ev)))
        top = d.groupby("wk", group_keys=False).apply(lambda w: w.nlargest(max(1, round(K * len(w))), "s"))
        wp = top.groupby("wk").label.mean(); wk[name] = wp
        pos = ev.groupby("wk").label.sum()
        res[name] = {"P@25%": wp.mean(), "Recall@25%": (top.groupby("wk").label.sum() / pos).mean()}
    ev_m = ev.minor.isin(rj.index).mean()
    return pd.DataFrame(res).T, pd.DataFrame(wk), ev, ev_m, rj, ev
out = []
for tag, args in {"A off-season": ("2024-10-01", "2025-04-01", "2025-04-01", "2025-10-01"),
                  "B heat season": ("2024-01-01", "2024-10-01", "2024-10-01", "2025-04-01")}.items():
    t, wk, ev, cover, rj, _ = run(*args)
    print(f"\n=== Check {tag}: {len(ev):,} complaints, {ev.wk.nunique()} weeks, positive rate {ev.label.mean():.3f}; "
          f"minor category seen in fit period {cover:.1%}")
    print(t.round(3))
    dif = wk["Minor category, highest across problems"] - wk["Major category, highest across problems"]
    d2 = wk["Minor category, highest across problems"] - wk["Problems in complaint"]
    d3 = wk["Minor category, highest across problems"] - wk["HPD current order (fastest verdict first)"]
    for nm, d in [("minor-major", dif), ("minor-problems", d2), ("minor-HPD order", d3)]:
        print(f"  {nm}: mean {d.mean():+.3f}, SE {d.std() / np.sqrt(len(d)):.3f}, weeks ahead {(d > 0).mean():.0%}")
    m_eval = ev.groupby("minor").label.agg(["mean", "size"]); m_eval = m_eval[m_eval["size"] >= 100]
    both = m_eval.join(rj.rename("fit"), how="inner")
    print(f"  stability: Spearman of minor-category rates, fit vs scored, {both['fit'].rank().corr(both['mean'].rank()):.3f} over {len(both)} categories")
    t.round(3).assign(check=tag).to_csv(OUTPUTS / "inspect" / f"baseline_validation_{tag[0]}.csv")
one = c[(c.n == 1) & (c.rec >= "2024-10-01")]
print(f"\nsingle-problem share of training window: {(win.n == 1).mean():.3f}; positive rate {one.label.mean():.3f}; multi {win[win.n > 1].label.mean():.3f}")
