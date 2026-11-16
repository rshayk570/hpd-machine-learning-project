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


# Save the dataset's provenance (license, update times, fetch time) and its column dictionary.
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


# Fetch one received year's two aggregate tables and save them as CSVs.
def fetch_year(year, refresh):
    this_year = dt.date.today().year
    status_f = DATA / "by_year" / f"status_{year}.csv"
    comp_f = DATA / "by_year" / f"complaints_{year}.csv"
    # Cache check: a past year is skipped if both files exist. The current year is still filling in, so it
    # is always refetched. --refresh refetches every year.
    if status_f.exists() and comp_f.exists() and year < this_year and not refresh:
        return
    where = year_clause(year)
    # Query 1: problem rows per outcome wording (status_description) for this year.
    rows = soda(f"select status_description, count(*) as n where {where} group by status_description limit 200")
    pd.DataFrame(rows, columns=["status_description", "n"]).astype({"n": "int64"}).to_csv(status_f, index=False)
    # Query 2: distinct complaints against problem rows. A complaint can hold several problems, one row each,
    # so problems >= complaints.
    c = soda(f"select count(distinct complaint_id) as complaints, count(*) as problems where {where}")[0]
    pd.DataFrame([{"year": year, "complaints": int(c["complaints"]), "problems": int(c["problems"])}]).to_csv(
        comp_f, index=False)
    print(f"  {year}: {int(c['problems']):>9,} problems, {int(c['complaints']):>9,} complaints")


# Summarize what earlier runs saved in data/. Reads files only: no network calls, nothing written.
def inspect_saved():
    meta_f = DATA / "metadata.json"
    if meta_f.exists():
        m = json.loads(meta_f.read_text())
        print(f"metadata: fetched {m.get('fetched_at_utc')}, dataset rows last updated {m.get('rows_updated_at_utc')}")
    else:
        print(f"missing: {meta_f}")

    cols_f = DATA / "columns.csv"
    if cols_f.exists():
        print(f"columns: {len(pd.read_csv(cols_f))}")
    else:
        print(f"missing: {cols_f}")

    # One line per received year from the complaints_<year>.csv files.
    comp_files = sorted((DATA / "by_year").glob("complaints_*.csv"))
    if comp_files:
        comp = pd.concat([pd.read_csv(f) for f in comp_files]).sort_values("year")
        print(f"\ncomplaints by received year ({comp.year.min()}-{comp.year.max()}):")
        print(f"  {'year':>4} {'complaints':>11} {'problems':>11} {'per complaint':>14}")
        for r in comp.itertuples():
            per = r.problems / r.complaints if r.complaints else float("nan")
            print(f"  {r.year:>4} {r.complaints:>11,} {r.problems:>11,} {per:>14.2f}")
    else:
        print(f"missing: {DATA / 'by_year' / 'complaints_*.csv'}")

    # Outcome wordings for the most recent complete year (the current year is still filling in).
    this_year = dt.date.today().year
    status_years = sorted(int(f.stem.split("_")[1]) for f in (DATA / "by_year").glob("status_*.csv"))
    complete = [y for y in status_years if y < this_year]
    if not complete:
        print(f"missing: no status_<year>.csv before {this_year} in {DATA / 'by_year'}")
        return
    year = complete[-1]
    s = pd.read_csv(DATA / "by_year" / f"status_{year}.csv")
    s["status_description"] = s.status_description.fillna("(blank)")
    total = s.n.sum()
    print(f"\nstatus wordings in {year}: {s.status_description.nunique()} distinct, {total:,} problem rows")
    print(f"  {'rows':>9} {'share':>6}  wording (first 90 characters)")
    for r in s.sort_values("n", ascending=False).head(10).itertuples():
        text = r.status_description if len(r.status_description) <= 90 else r.status_description[:87] + "..."
        print(f"  {r.n:>9,} {r.n / total:>6.1%}  {text}")


# Fetch metadata and every year's aggregates, or with --inspect only summarize what is already saved.
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --refresh: refetch past years even when their CSVs are already saved.
    ap.add_argument("--refresh", action="store_true", help="refetch cached years too")
    # --first-year: earliest received year to fetch (default 2001, the table's first rows).
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    # --inspect: print a summary of data/ and stop; no request is sent.
    ap.add_argument("--inspect", action="store_true", help="summarize saved files only, no network")
    args = ap.parse_args()
    if args.inspect:
        inspect_saved()
        return
    (DATA / "by_year").mkdir(parents=True, exist_ok=True)
    save_metadata()
    for year in range(args.first_year, dt.date.today().year + 1):
        fetch_year(year, args.refresh)
    print("done")


if __name__ == "__main__":
    main()
