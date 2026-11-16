"""Fetch the small aggregate tables the analysis needs from NYC Open Data.

Writes to data/:
  metadata.json                    license fields, update times, fetch time (the provenance record)
  columns.csv                      the data dictionary (column name, type, description)
  by_year/status_<year>.csv        problem rows per status_description wording, per received year
  by_year/complaints_<year>.csv    distinct complaints and problem rows per received year

The full table is about 16M rows, so nothing here downloads it. Years before the current one are cached
and skipped on a rerun; the current year is always refetched. Use --refresh to refetch everything.
The table updates daily, so counts differ slightly between runs: keep metadata.json with any result.
"""
import argparse
import datetime as dt
import json

import pandas as pd

from hpd_api import DATA, FIRST_YEAR, soda, view_metadata, year_clause


def save_metadata():
    m = view_metadata()
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat() if t else None  # noqa: E731
    out = {
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": m.get("id"),
        "name": m.get("name"),
        "attribution": m.get("attribution"),
        "license_id": m.get("licenseId"),
        "license": m.get("license"),
        "rows_updated_at_utc": iso(m.get("rowsUpdatedAt")),
        "view_last_modified_utc": iso(m.get("viewLastModified")),
    }
    (DATA / "metadata.json").write_text(json.dumps(out, indent=2) + "\n")
    cols = pd.DataFrame([{"column": c.get("fieldName"), "type": c.get("dataTypeName"),
                          "description": (c.get("description") or "").strip()} for c in m.get("columns", [])])
    cols.to_csv(DATA / "columns.csv", index=False)
    print(f"metadata: license={out['license']!r}, rows updated {out['rows_updated_at_utc']}, {len(cols)} columns")


def fetch_year(year, refresh):
    this_year = dt.date.today().year
    status_f = DATA / "by_year" / f"status_{year}.csv"
    comp_f = DATA / "by_year" / f"complaints_{year}.csv"
    if status_f.exists() and comp_f.exists() and year < this_year and not refresh:
        return
    where = year_clause(year)
    rows = soda(f"select status_description, count(*) as n where {where} group by status_description limit 200")
    pd.DataFrame(rows, columns=["status_description", "n"]).astype({"n": "int64"}).to_csv(status_f, index=False)
    c = soda(f"select count(distinct complaint_id) as complaints, count(*) as problems where {where}")[0]
    pd.DataFrame([{"year": year, "complaints": int(c["complaints"]), "problems": int(c["problems"])}]).to_csv(
        comp_f, index=False)
    print(f"  {year}: {int(c['problems']):>9,} problems, {int(c['complaints']):>9,} complaints")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="refetch cached years too")
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    args = ap.parse_args()
    (DATA / "by_year").mkdir(parents=True, exist_ok=True)
    save_metadata()
    for year in range(args.first_year, dt.date.today().year + 1):
        fetch_year(year, args.refresh)
    print("done")


if __name__ == "__main__":
    main()
