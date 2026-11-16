"""Outcome history per building: problem rows of one class (P by default: inspector issued a violation;
`--group N` for inspected-and-clean), counted per building_id and month of problem_status_date, over the whole table. This is the input for the
"prior violations at this building" baseline and its lookback windows.

Fetched one problem_status_date year at a time (aggregate queries, no row-level download).
Writes data/history/<group>_by_building_month.parquet (p_... for P, n_... for N).
"""
# Purpose: Per building and month, how many problem rows ended in one outcome class, over the whole table.
# Input:   The Open Data API; outputs/wording_groups.csv (02).
# Output:  data/history/<p|n>_<year>.csv (per-year cache) and data/history/<p|n>_by_building_month.parquet.
import argparse
import datetime as dt

import pandas as pd

from hpd_api import DATA, OUTPUTS, soda_csv


# Purpose: Quote a string as a SoQL literal.
# Input:   string.
# Output:  the string in single quotes, inner quotes doubled.
def q(s):
    return "'" + s.replace("'", "''") + "'"


# Purpose: Count one class's problem rows per building and month of problem_status_date, year by year.
# Input:   --group P or N; outputs/wording_groups.csv; the API.
# Output:  data/history/<p|n>_<year>.csv and <p|n>_by_building_month.parquet.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="P", choices=["P", "N"])
    args = ap.parse_args()
    tag = args.group.lower()
    out = DATA / "history"
    out.mkdir(parents=True, exist_ok=True)
    wg = pd.read_csv(OUTPUTS / "wording_groups.csv")
    pw = [w for w in wg.loc[wg.group == args.group, "status_description"]]
    inlist = ", ".join(q(w) for w in pw)
    parts = []
    for y in range(2003, dt.date.today().year + 1):
        f = out / f"{tag}_{y}.csv"
        if f.exists() and y < dt.date.today().year:
            parts.append(pd.read_csv(f, dtype=str)); continue
        where = (f"status_description in ({inlist}) and problem_status_date >= '{y}-01-01T00:00:00' "
                 f"and problem_status_date < '{y + 1}-01-01T00:00:00'")
        got = []
        off = 0
        while True:
            df = soda_csv(f"select building_id, date_trunc_ym(problem_status_date) as m, count(*) as n, "
                          f"count(distinct complaint_id) as c where {where} group by building_id, m "
                          f"order by building_id, m limit 100000 offset {off}")
            got.append(df); off += len(df)
            if len(df) < 100000:
                break
        df = pd.concat(got)
        df.to_csv(f, index=False)
        parts.append(df)
        print(f"{y}: {len(df):,} building-months, {df.n.astype(int).sum():,} {args.group} rows")
    h = pd.concat(parts)
    h["m"] = pd.to_datetime(h["m"]).dt.to_period("M")
    h = h.astype({"building_id": "int64", "n": "int64", "c": "int64"})
    h.to_parquet(out / f"{tag}_by_building_month.parquet", index=False)
    print("done", len(h))


if __name__ == "__main__":
    main()
