"""Complaint-level table for modeling: one row per labeled complaint, with only information known when it arrives.

Label (the rule used throughout the project): positive if any problem is class P, negative if any is N
and none is P. Complaints with neither are not in the table.

Features
  from the complaint itself   n_problems, a count of problems per major category (maj_*), the first problem's
                              major | minor category, the most severe HPD triage type, borough, community board, zip,
                              unit type, space type, anonymous flag, month, weekday, hour
  building outcome history    p6, p12, pall (violation rows), n12, nall (clean-inspection rows), rate12, months_since_p.
                              Counted by the month the outcome was recorded (problem_status_date), and the complaint's
                              own received month is excluded, so no outcome that was still unknown at receipt is used.
  recent complaints           c7, c30: complaints of any outcome class received at the same building in the previous
                              7 and 30 days; days_since_c: days since the previous one (capped at 60)

Window: received weeks starting 2024-02-05 (so the 30-day lookback is complete; the row pull starts 2024-01-01)
through the week ending 2026-08-02 (90 days before the 2026-11-05 pull, section 8.5: 97-98% of labels are in).
Writes data/model/complaints_features.parquet.
"""
# Purpose: Build the complaint-level modeling table: one row per labeled complaint, intake-time features only.
# Input:   data/rows/*.parquet (04), data/history/{p,n}_by_building_month.parquet (05).
# Output:  data/model/complaints_features.parquet.
import numpy as np
import pandas as pd

from hpd_api import DATA

START, END = pd.Timestamp("2024-02-05"), pd.Timestamp("2026-08-03")   # Mondays; END is exclusive
SEVERITY = {"NON EMERGENCY": 0, "EMERGENCY": 1, "IMMEDIATE EMERGENCY": 2}


# Purpose: Load every pulled problem row in problem_id order (fixes which problem `first` picks).
# Input:   data/rows/<class>.parquet.
# Output:  DataFrame of problem rows sorted by problem_id.
def load_rows():
    files = [f for f in sorted((DATA / "rows").glob("*.parquet")) if f.name != "complaints.parquet"]
    r = pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)
    r["received_date"] = pd.to_datetime(r.received_date)
    r["problem_id"] = r.problem_id.astype("int64")
    r["building_id"] = r.building_id.astype("int64")
    return r.sort_values("problem_id")


# Purpose: Cumulative outcome rows per building by month index.
# Input:   tag 'p' or 'n'.
# Output:  DataFrame: building_id, hm, cum, sorted by hm.
def history(tag):
    """Cumulative outcome rows per building by month index (months since Jan 2003)."""
    h = pd.read_parquet(DATA / "history" / f"{tag}_by_building_month.parquet")
    h["hm"] = ((h.m.dt.year - 2003) * 12 + h.m.dt.month - 1).astype("int64")
    h = h.sort_values(["building_id", "hm"])
    h["cum"] = h.groupby("building_id").n.cumsum()
    return h[["building_id", "hm", "cum"]].sort_values("hm")


# Purpose: Look up a building's history value as of the month before a complaint's month.
# Input:   history table, building ids, month indexes, column to return.
# Output:  Series aligned to the complaints (NaN when no history).
def asof(h, building, upto, col="cum"):
    """Value of `col` at each complaint's building as of the end of month `upto - 1` (NaN if no history)."""
    key = pd.DataFrame({"building_id": building.values, "hm": (upto - 1).astype("int64").values, "i": building.index})
    got = pd.merge_asof(key.sort_values("hm"), h, on="hm", by="building_id", direction="backward")
    return got.set_index("i")[col].reindex(building.index)


# Purpose: Build the complaint table: label, intake features, building history, recent complaints.
# Input:   data/rows/*.parquet, data/history/{p,n}_by_building_month.parquet.
# Output:  data/model/complaints_features.parquet; one printed summary line.
def main():
    r = load_rows()

    # every complaint in the pull (any outcome class), for the recent-complaint counts
    allc = r.groupby("complaint_id").agg(rec=("received_date", "min"), building_id=("building_id", "first"))
    sec = (allc.rec - pd.Timestamp("2023-01-01")).dt.total_seconds().astype("int64")
    key_all = np.sort(allc.building_id.values * 10**10 + sec.values)

    labeled = r.loc[r.group.isin(["P", "N"]), "complaint_id"].unique()
    r = r[r.complaint_id.isin(labeled)].copy()
    r["minor"] = r.major_category + " | " + r.minor_category
    r["sev"] = r.type.map(SEVERITY)
    g = r.groupby("complaint_id")
    c = g.agg(rec=("received_date", "min"), building_id=("building_id", "first"), n_problems=("problem_id", "size"),
              major=("major_category", "first"), minor=("minor", "first"), sev=("sev", "max"),
              borough=("borough", "first"), cb=("community_board", "first"), zip=("post_code", "first"),
              unit_type=("unit_type", "first"), space_type=("space_type", "first"),
              anonymous=("complaint_anonymous_flag", "first"))
    c["label"] = (r.group == "P").groupby(r.complaint_id).any().astype(int)
    c = c[(c.rec >= START) & (c.rec < END)].copy()
    c["week"] = c.rec.dt.to_period("W-SUN").dt.start_time
    c["type"] = c.sev.map({v: k for k, v in SEVERITY.items()})
    c["cb"] = c.borough.str[:2] + c.cb.astype(str)
    c["zip"] = c.zip.fillna("none")
    c["anonymous"] = (c.anonymous == "Y").astype(int)
    c["month"], c["weekday"], c["hour"] = c.rec.dt.month, c.rec.dt.dayofweek, c.rec.dt.hour

    # problems per major category
    counts = pd.crosstab(r.complaint_id, r.major_category).reindex(c.index).fillna(0).astype(int)
    counts.columns = ["maj_" + k.lower().replace("/", "_").replace(" ", "_") for k in counts.columns]
    c = c.join(counts)

    # building outcome history, received month excluded
    midx = (c.rec.dt.year - 2003) * 12 + c.rec.dt.month - 1
    hp, hn = history("p"), history("n")
    cum = lambda h, upto: asof(h, c.building_id, upto).fillna(0)
    c["pall"], c["nall"] = cum(hp, midx), cum(hn, midx)
    c["p6"] = c.pall - cum(hp, midx - 6)
    c["p12"] = c.pall - cum(hp, midx - 12)
    c["n12"] = c.nall - cum(hn, midx - 12)
    c["rate12"] = (c.p12 + 1) / (c.p12 + c.n12 + 2)          # violation share of verdict rows, pulled toward 0.5
    last = hp.assign(last=hp.hm)[["building_id", "hm", "last"]]
    c["months_since_p"] = midx - asof(last, c.building_id, midx, "last")   # NaN: no violation on record

    # complaints at the same building in the previous 7 / 30 days
    t = (c.rec - pd.Timestamp("2023-01-01")).dt.total_seconds().astype("int64").values
    b = c.building_id.values * 10**10
    here = np.searchsorted(key_all, b + t, side="left")
    c["c7"] = here - np.searchsorted(key_all, b + t - 7 * 86400, side="left")
    c["c30"] = here - np.searchsorted(key_all, b + t - 30 * 86400, side="left")
    prev = key_all[np.maximum(here - 1, 0)]
    same = (here > 0) & (prev // 10**10 == c.building_id.values)
    c["days_since_c"] = np.where(same, np.minimum((b + t - prev) / 86400, 60), 60)

    out = DATA / "model"
    out.mkdir(exist_ok=True)
    c.drop(columns="sev").to_parquet(out / "complaints_features.parquet")
    print(f"{len(c):,} labeled complaints, {c.week.nunique()} weeks, {c.rec.min():%Y-%m-%d} to {c.rec.max():%Y-%m-%d}, "
          f"positive rate {c.label.mean():.3f}, {c.building_id.nunique():,} buildings")


if __name__ == "__main__":
    main()
