"""Data inspection for the proposal revisions. Reads data/rows/*.parquet (04), data/history (05) and the
aggregate CSVs, writes outputs/inspect/*.csv and prints the numbers each decision turns on.

Complaint-level label: a complaint is positive if any of its problems is class P, negative if it has an N
problem and no P, otherwise unlabeled (the verdict-only population).  Inspected = any problem with a
verdict (P, N, N_partial, P_existing, P_lead) or a no-access outcome.
"""
import json
import numpy as np
import pandas as pd

from hpd_api import DATA, OUTPUTS
from hpd_labels import group_of

OUT = OUTPUTS / "inspect"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
VERDICT = {"P", "N", "N_partial", "P_existing", "P_lead"}
rng = np.random.default_rng(0)


def h(title):
    print(f"\n{'#' * 3} {title}")


def save(df, name):
    df.to_csv(OUT / f"{name}.csv")
    return df


# ------------------------------------------------------------------ load
rows = pd.concat(pd.read_parquet(f) for f in sorted((DATA / "rows").glob("*.parquet")) if f.name != "complaints.parquet").reset_index(drop=True)
for c in ("received_date", "problem_status_date", "complaint_status_date"):
    rows[c] = pd.to_datetime(rows[c])
rows["complaint_id"] = rows.complaint_id.astype("int64")
rows["building_id"] = rows.building_id.astype("int64")
window = json.loads((DATA / "rows" / "window.json").read_text())
LAST_FULL_WEEK = (pd.Timestamp(window["end"]) - pd.Timedelta(days=14)).to_period("W-SUN").start_time
rows["week"] = rows.received_date.dt.to_period("W-SUN").dt.start_time   # Monday-start weeks
rows["month"] = rows.received_date.dt.to_period("M")
rows["year"] = rows.received_date.dt.year
print(f"rows {len(rows):,} problems, {rows.complaint_id.nunique():,} complaints, received {window['start']} to {window['end']}")

# complaint-level table
g = rows.groupby("complaint_id")
ct = pd.crosstab(rows.complaint_id, rows.group) > 0
comp = g.agg(received=("received_date", "min"), week=("week", "min"), month=("month", "min"), year=("year", "min"),
             n_problems=("group", "size"), n_classes=("group", "nunique"), building_id=("building_id", "first"),
             borough=("borough", "first"), category=("major_category", "first"), type=("type", "first"),
             status_date=("problem_status_date", "max"))
comp["dup"] = (rows.problem_duplicate_flag == "Y").groupby(rows.complaint_id).any()
for name, grp in [("has_P", "P"), ("has_N", "N"), ("has_noaccess", "U_noaccess"), ("has_fixed", "U_fixed"),
                  ("has_open", "U_open"), ("has_Pex", "P_existing"), ("has_Npart", "N_partial")]:
    comp[name] = ct[grp] if grp in ct else False
comp["has_verdict"] = ct[[c for c in ct.columns if c in VERDICT]].any(axis=1)
comp["label"] = np.where(comp.has_P, 1, np.where(comp.has_N, 0, -1))
comp["inspected"] = comp.has_verdict | comp.has_noaccess
comp["pool"] = comp.label >= 0

comp["cls"] = np.select([comp.has_P, comp.has_N, comp.has_Pex, comp.has_Npart, comp.has_noaccess, comp.has_fixed, comp.has_open],
                        ["P", "N", "P_existing", "N_partial", "U_noaccess", "U_fixed", "U_open"], "other")

# ------------------------------------------------------------------ 1. capacity and pool per week
h("1a/1b/1c  Complaints per week: all, inspected (verdict or no-access), verdict-only, labeled pool (P or N)")
full = comp[comp.week < LAST_FULL_WEEK]
wk = full.groupby("week").agg(complaints=("label", "size"), inspected=("inspected", "sum"),
                              verdict=("has_verdict", "sum"), pool=("pool", "sum"),
                              positives=("label", lambda s: (s == 1).sum()))
wk["top10"] = (wk.pool * 0.10).round().astype(int)
wk["pos_rate"] = wk.positives / wk.pool
wk["season"] = np.where(wk.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5]), "heat (Oct-May)", "summer (Jun-Sep)")
save(wk, "weekly")
def q(s):
    return pd.Series({"min": s.min(), "p10": s.quantile(.1), "median": s.median(), "p90": s.quantile(.9), "max": s.max()}).round(0).astype(int)
for y, sub in wk.groupby(wk.index.year):
    print(f"\n{y} ({len(sub)} full weeks)")
    print(pd.DataFrame({c: q(sub[c]) for c in ["complaints", "inspected", "verdict", "pool", "top10"]}).T)
print("\nseason medians (all years):")
print(wk.groupby([wk.index.year, "season"])[["complaints", "pool", "top10", "pos_rate"]].median().round(3))
print("\nfixed count vs percent: if the inspector budget were the 2024 median top-10% count every week,")
m24 = int(wk.loc[wk.index.year == 2024, "top10"].median())
print(f"  fixed N = {m24}; share of the pool that N covers: min {m24 / wk.pool.max():.1%}, max {m24 / wk.pool.min():.1%}")

h("1d  Grain: do problems in one complaint get different outcomes?")
multi = comp[comp.n_problems > 1]
print(f"complaints with 1 problem: {(comp.n_problems == 1).mean():.1%};  2+: {(comp.n_problems > 1).mean():.1%} "
      f"(mean problems per complaint {comp.n_problems.mean():.2f}, max {comp.n_problems.max()})")
print(f"among 2+ problem complaints: same class for all problems {(multi.n_classes == 1).mean():.1%}; "
      f"mixed {(multi.n_classes > 1).mean():.1%}")
mix = multi[multi.n_classes > 1]
print(f"  P and N in the same complaint: {(mix.has_P & mix.has_N).sum():,} complaints = "
      f"{(mix.has_P & mix.has_N).sum() / len(comp):.2%} of all complaints, {(mix.has_P & mix.has_N).sum() / len(multi):.1%} of multi-problem ones")
print(f"  verdict mixed with an unlabeled class (fixed/no-access/open): {(mix.has_verdict & (mix.has_noaccess | mix.has_fixed | mix.has_open)).sum() / len(multi):.1%} of multi-problem")
pr = rows[rows.group.isin(["P", "N"])]
print(f"problem-level P rate {(pr.group == 'P').mean():.3f} vs complaint-level (any-P rule) {(comp.label[comp.pool] == 1).mean():.3f}")
print(f"P rows per positive complaint: {rows[rows.group == 'P'].groupby('complaint_id').size().mean():.2f}")

# ------------------------------------------------------------------ 2. rule-based baseline
h("2a  Building ID coverage (aggregate, all years, from the API)")
print("building_id non-null on 100.0% of rows every year 2013-2026 and never 0; bbl and bin non-null on 99.6-99.8%.")
print(f"window: {rows.building_id.nunique():,} distinct buildings, bbl missing on {rows.bbl.isna().mean():.2%} of rows")
hist = pd.read_parquet(DATA / "history" / "p_by_building_month.parquet")
print(f"history: {len(hist):,} building-months with a violation, {hist.building_id.nunique():,} buildings, {hist.n.sum():,} P rows 2003-2026")

h("2b  Which date marks the violation? problem_status_date vs complaint_status_date vs received_date (P rows)")
P = rows[rows.group == "P"].copy()
P["lag_prob"] = (P.problem_status_date - P.received_date).dt.days
P["lag_comp"] = (P.complaint_status_date - P.received_date).dt.days
print(f"P rows: problem_status_date missing {P.problem_status_date.isna().mean():.2%}, before received_date {(P.lag_prob < 0).mean():.2%}, "
      f"equal to complaint_status_date {(P.problem_status_date == P.complaint_status_date).mean():.1%}")
print("days from received to problem_status_date (P rows):", P.lag_prob.quantile([.5, .9, .95, .99]).round(0).to_dict())
print("problem_status values on P rows:", P.problem_status.value_counts().to_dict())
print("complaint_status values on P rows:", P.complaint_status.value_counts().to_dict())

h("2c  Prior violations at the building (verdict-only complaints), by lookback")
# cumulative P count per building by month index
hist["midx"] = (hist.m.dt.year - 2003) * 12 + hist.m.dt.month - 1
hist = hist.sort_values(["building_id", "midx"])
hist["cum"] = hist.groupby("building_id").n.cumsum()
hist["cumc"] = hist.groupby("building_id").c.cumsum()
lab = comp[comp.pool].copy()
lab["midx"] = (lab.month.dt.year - 2003) * 12 + lab.month.dt.month - 1
hs = hist[["building_id", "midx", "cum"]].rename(columns={"midx": "hm"})

def prior(upto_exclusive):
    """P rows at the building in months strictly before `upto_exclusive` (a Series of month indexes)."""
    key = pd.DataFrame({"building_id": lab.building_id.values, "hm": (upto_exclusive - 1).values, "i": lab.index})
    key = key.sort_values("hm")
    m = pd.merge_asof(key, hs.sort_values("hm"), on="hm", by="building_id", direction="backward")
    return m.set_index("i").cum.fillna(0).astype(int).reindex(lab.index)

cum_now = prior(lab.midx)                      # everything before the received month
for name, k in [("6m", 6), ("12m", 12), ("24m", 24), ("all", None)]:
    lab[f"prior_{name}"] = cum_now if k is None else cum_now - prior(lab.midx - k)
# last violation month for recency
last = hs.sort_values("hm")
key = pd.DataFrame({"building_id": lab.building_id.values, "hm": (lab.midx - 1).values, "i": lab.index}).sort_values("hm")
mm = pd.merge_asof(key, hist[["building_id", "midx"]].rename(columns={"midx": "hm"}).assign(lastm=lambda d: d.hm).sort_values("hm"),
                   on="hm", by="building_id", direction="backward").set_index("i").reindex(lab.index)
lab["months_since"] = lab.midx - mm.lastm   # NaN when never
zero = pd.DataFrame({w: {"zero prior": (lab[f"prior_{w}"] == 0).mean(), "1": (lab[f"prior_{w}"] == 1).mean(),
                         "2-5": lab[f"prior_{w}"].between(2, 5).mean(), "6+": (lab[f"prior_{w}"] >= 6).mean(),
                         "P rate | zero": lab.label[lab[f"prior_{w}"] == 0].mean(),
                         "P rate | 1+": lab.label[lab[f"prior_{w}"] >= 1].mean()} for w in ["6m", "12m", "24m", "all"]}).T
print(save(zero.round(3), "prior_violations"))
print("(the received month itself is excluded from every lookback, so a violation from this same complaint cannot count)")

h("2d  Baseline precision at top 10% per week, by lookback and tie-break (2024 = training; 2025 shown for comparison)")
nh_f = DATA / "history" / "n_by_building_month.parquet"
if nh_f.exists():
    nh = pd.read_parquet(nh_f)
    nh["midx"] = (nh.m.dt.year - 2003) * 12 + nh.m.dt.month - 1
    nh = nh.sort_values(["building_id", "midx"]); nh["cum"] = nh.groupby("building_id").n.cumsum()
    nhs = nh[["building_id", "midx", "cum"]].rename(columns={"midx": "hm"}).sort_values("hm")
    def prior_n(upto_exclusive):
        key = pd.DataFrame({"building_id": lab.building_id.values, "hm": (upto_exclusive - 1).values, "i": lab.index}).sort_values("hm")
        return pd.merge_asof(key, nhs, on="hm", by="building_id", direction="backward").set_index("i").cum.fillna(0).astype(int).reindex(lab.index)
    n_now = prior_n(lab.midx)
    for name, k in [("12m", 12), ("all", None)]:
        lab[f"priorN_{name}"] = n_now if k is None else n_now - prior_n(lab.midx - k)
def p_at_k(df, score, frac=0.10):
    out = []
    for w, s in df.groupby("week"):
        k = max(1, int(round(frac * len(s))))
        top = s.assign(_s=score.loc[s.index].values).sort_values("_s", ascending=False).head(k)
        out.append(top.label.mean())
    return np.mean(out)
lab["rand"] = rng.random(len(lab))
res = {}
for y in [2024, 2025]:
    d = lab[(lab.year == y) & (lab.week < LAST_FULL_WEEK)]
    tr = lab[lab.year == y - 1] if y == 2025 else lab[(lab.year == 2024) & (lab.month.dt.month <= 6)]
    base = tr.label.mean()
    cat_rate = tr.groupby("category").label.mean()
    d_eval = d if y == 2025 else d[d.month.dt.month >= 7]
    cr = d_eval.category.map(cat_rate).fillna(base)
    rec = -d_eval.months_since.fillna(9999)
    jit = 1e-6 * d_eval.rand
    r = {"random": p_at_k(d_eval, d_eval.rand), "base rate (mean label)": d_eval.label.mean(),
         "category rate (from training period)": p_at_k(d_eval, cr + jit)}
    for w in ["6m", "12m", "24m", "all"]:
        s_ = d_eval[f"prior_{w}"]
        r[f"prior violations {w}, random ties"] = p_at_k(d_eval, s_ + 1e-3 * d_eval.rand)
        r[f"prior violations {w}, category tie-break"] = p_at_k(d_eval, s_ + 0.5 * cr + jit)
    r["prior all, recency tie-break"] = p_at_k(d_eval, d_eval.prior_all + 1e-6 * rec + 1e-9 * d_eval.rand)
    r["recency only (months since last violation)"] = p_at_k(d_eval, rec + jit)
    if "priorN_all" in lab:
        for w in ["12m", "all"]:
            a, b = d_eval[f"prior_{w}"], d_eval[f"priorN_{w}"]
            brate = (a + 2 * base) / (a + b + 2)           # shrunk building violation rate P/(P+N)
            r[f"building violation rate {w} (shrunk)"] = p_at_k(d_eval, brate + jit)
            r[f"building rate {w} x category rate"] = p_at_k(d_eval, brate * cr / base + jit)
    r["category rate, prior-6m tie-break"] = p_at_k(d_eval, cr + 1e-4 * np.minimum(d_eval.prior_6m, 20) + jit)
    res[f"{y} ({'Jul-Dec' if y == 2024 else 'full year'}, {len(d_eval):,} labeled complaints)"] = r
print(save(pd.DataFrame(res).round(3), "baseline_p_at_10"))
print("category P rates in 2024 H1 (top and bottom):")
cat24 = lab[(lab.year == 2024) & (lab.month.dt.month <= 6)].groupby("category").label.agg(["mean", "size"]).sort_values("mean")
print(save(cat24.round(3), "category_rates_2024h1").to_string())

# ------------------------------------------------------------------ 3. information at arrival, selection
h("3b  problem_duplicate_flag")
dup = pd.read_csv(DATA / "dupflag_by_wording_2024plus.csv")
print(f"rows flagged Y: {(rows.problem_duplicate_flag == 'Y').mean():.1%}; Y rows whose wording starts 'More than one complaint...': "
      f"{dup[(dup.problem_duplicate_flag == 'Y') & dup.dup_word].n.sum() / dup[dup.problem_duplicate_flag == 'Y'].n.sum():.2%}")
print("flag by class (rows):"); print(rows.groupby("group").problem_duplicate_flag.apply(lambda s: (s == "Y").mean()).round(4))
print(f"flag Y among verdict-only (P/N) rows: {(rows[rows.group.isin(['P', 'N'])].problem_duplicate_flag == 'Y').sum()} rows")
print(f"flag Y among still-open rows: {(rows[rows.group == 'U_open'].problem_duplicate_flag == 'Y').mean():.1%} (the flag exists before closure)")
# is a flagged complaint preceded by another complaint at the same building within 7 days?
cs = comp.sort_values("received")[["building_id", "received", "dup"]].copy()
cs["prev"] = cs.groupby("building_id").received.shift(1)
cs["gap"] = (cs.received - cs.prev).dt.days
print(f"flagged complaints with an earlier complaint at the same building within 7 days: {(cs[cs.dup].gap <= 7).mean():.1%}; "
      f"unflagged: {(cs[~cs.dup].gap <= 7).mean():.1%}")

h("3c  Time to resolution: received to problem_status_date, labeled (P/N) problems")
pn = rows[rows.group.isin(["P", "N"])].copy()
pn["days"] = (pn.problem_status_date - pn.received_date).dt.days
tt = pn[pn.received_date < pd.Timestamp(window["end"]) - pd.Timedelta(days=180)].groupby("year").days.describe(percentiles=[.5, .9, .95, .99])[["50%", "90%", "95%", "99%"]]
for d in [30, 60, 90]:
    tt[f"<= {d}d"] = pn.groupby("year").days.apply(lambda s: (s <= d).mean()).round(3)
print(save(tt, "time_to_resolution"))
print("(2026 rows received in the last 180 days are left out of the percentiles)")
print("\nshare of complaints still open, by received month (label maturity):")
op = comp.groupby("month").has_open.mean()
print(op[op.index >= pd.Period("2025-07", "M")].round(3).to_string())

h("3d  The 2025 shift, by month: verdict share, no-access share, P/(P+N)")
mo = pd.read_csv(DATA / "monthly_status_2019_2023.csv") if (DATA / "monthly_status_2019_2023.csv").exists() else None
mrows = rows.groupby(["month", "group"]).size().unstack(fill_value=0)
if mo is not None:
    mo["m"] = pd.to_datetime(mo.m).dt.to_period("M")
    old = mo.pivot_table(index="m", columns="group", values="n", aggfunc="sum", fill_value=0)
    mrows = pd.concat([old, mrows]).fillna(0).astype(int)
tot = mrows.sum(axis=1)
shift = pd.DataFrame({"rows": tot, "verdict": mrows[list(VERDICT & set(mrows.columns))].sum(axis=1) / tot,
                      "no_access": mrows["U_noaccess"] / tot, "fixed": mrows["U_fixed"] / tot,
                      "open": mrows.get("U_open", 0) / tot, "P/(P+N)": mrows["P"] / (mrows["P"] + mrows["N"])})
save(shift.round(3), "monthly_shift")
print(shift[shift.index >= pd.Period("2024-01", "M")].round(3).to_string())
if mo is not None:
    print("\nyearly means 2019-2023 for reference:"); print(shift[shift.index < pd.Period("2024-01", "M")].groupby(shift.index[shift.index < pd.Period("2024-01", "M")].year)[["verdict", "no_access", "P/(P+N)"]].mean().round(3))
print("\nno-access share by borough, 2024 vs 2025 vs 2026 (rows):")
print(rows.assign(na=rows.group == "U_noaccess").groupby(["year", "borough"]).na.mean().unstack(0).round(3))
print("\nno-access share by category (top 6), 2024 vs 2025 vs 2026:")
top6 = rows.major_category.value_counts().index[:6]
r6 = rows[rows.major_category.isin(top6)]
print(r6.assign(na=(r6.group == "U_noaccess")).groupby(["year", "major_category"]).na.mean().unstack(0).round(3))

h("3e  Who gets excluded: labeled vs U_fixed vs U_noaccess complaints (2024 to now)")
c2 = comp[comp.cls.isin(["P", "N", "U_fixed", "U_noaccess"])].copy()
c2["pop"] = np.where(c2.cls.isin(["P", "N"]), "labeled", c2.cls)
c2 = c2.join(lab[["prior_all", "prior_12m"]], how="left")
c2["prior12_bin"] = pd.cut(c2.prior_12m.fillna(-1), [-2, -1, 0, 2, 9, 1e9], labels=["(not computed)", "0", "1-2", "3-9", "10+"])
def dist(col, top=8):
    t = c2.groupby("pop")[col].value_counts(normalize=True).unstack(0).fillna(0)
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index[:top]]
    return t.round(3)
def tvd(col):
    t = c2.groupby("pop")[col].value_counts(normalize=True).unstack(0).fillna(0)
    return {k: round(0.5 * (t["labeled"] - t[k]).abs().sum(), 3) for k in ["U_fixed", "U_noaccess"]}
for col in ["category", "borough", "type"]:
    print(f"\n{col} (total variation distance from labeled: {tvd(col)})"); print(save(dist(col), f"excluded_{col}"))
# prior violations: recompute for all complaints
allc = comp[comp.cls.isin(["P", "N", "U_fixed", "U_noaccess"])].copy()
allc["midx"] = (allc.month.dt.year - 2003) * 12 + allc.month.dt.month - 1
key = pd.DataFrame({"building_id": allc.building_id.values, "hm": (allc.midx - 1).values, "i": allc.index}).sort_values("hm")
allc["prior_all"] = pd.merge_asof(key, hs.sort_values("hm"), on="hm", by="building_id", direction="backward").set_index("i").cum.fillna(0).astype(int).reindex(allc.index)
key["hm"] = key.hm - 12
allc["prior_12m"] = allc.prior_all - pd.merge_asof(key.sort_values("hm"), hs.sort_values("hm"), on="hm", by="building_id", direction="backward").set_index("i").cum.fillna(0).astype(int).reindex(allc.index)
allc["pop"] = np.where(allc.cls.isin(["P", "N"]), "labeled", allc.cls)
allc["prior12_bin"] = pd.cut(allc.prior_12m, [-1, 0, 2, 9, 1e9], labels=["0", "1-2", "3-9", "10+"])
print("\nprior violations in the 12 months before (share of complaints):")
print(save(allc.groupby("pop").prior12_bin.value_counts(normalize=True).unstack(0).round(3), "excluded_prior12"))
print("mean prior violations (all history):", allc.groupby("pop").prior_all.mean().round(2).to_dict())

# ------------------------------------------------------------------ label decisions
h("L1  P_existing and N_partial")
pex = comp[comp.has_Pex]
print(f"P_existing: {comp.has_Pex.mean():.2%} of complaints; of those, {(pex.has_P | pex.has_N).mean():.1%} also carry a P or N problem")
pex_only = comp[comp.cls == "P_existing"].copy()
pex_only["midx"] = (pex_only.month.dt.year - 2003) * 12 + pex_only.month.dt.month - 1
key = pd.DataFrame({"building_id": pex_only.building_id.values, "hm": (pex_only.midx - 1).values, "i": pex_only.index}).sort_values("hm")
pex_only["prior_all"] = pd.merge_asof(key, hs.sort_values("hm"), on="hm", by="building_id", direction="backward").set_index("i").cum.fillna(0).reindex(pex_only.index)
print(f"  complaints whose only verdict is P_existing: {len(pex_only):,}; share with >= 1 prior violation at the building (our history table): "
      f"{(pex_only.prior_all >= 1).mean():.1%} (labeled complaints overall: {(lab.prior_all >= 1).mean():.1%})")
print(f"  so counting P_existing as positive would reward the prior-violation baseline by construction")
npart = rows[rows.group == "N_partial"]
print(f"N_partial: {comp.has_Npart.mean():.2%} of complaints; by category: {npart.major_category.value_counts(normalize=True).head(3).round(3).to_dict()}")
print("  by month of year (share of all rows):")
print((rows.groupby(rows.received_date.dt.month).group.apply(lambda s: (s == 'N_partial').mean()) * 100).round(2).to_string())
print(f"  P rate of complaints that also had an N_partial problem: {comp[comp.has_Npart & comp.pool].label.mean():.3f}")

h("L2  Verdict-only population size")
print(f"complaints with a P/N label: {comp.pool.mean():.1%} of complaints (2024 {comp[comp.year == 2024].pool.mean():.1%}, "
      f"2025 {comp[comp.year == 2025].pool.mean():.1%}, 2026 {comp[comp.year == 2026].pool.mean():.1%})")
print(f"P rate in the pool: 2024 {comp[(comp.year == 2024) & comp.pool].label.mean():.3f}, 2025 {comp[(comp.year == 2025) & comp.pool].label.mean():.3f}, "
      f"2026 {comp[(comp.year == 2026) & comp.pool].label.mean():.3f}")
comp.drop(columns=["received", "status_date"]).to_parquet(DATA / "rows" / "complaints.parquet")
print("\nwrote outputs/inspect/*.csv and data/rows/complaints.parquet")
