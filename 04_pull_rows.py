"""Row-level pull of one received-date window, for the inspection items that aggregates cannot answer
(grain, weekly pools, time to resolution, duplicate-flag timing, who gets excluded).

The long status text is not downloaded: rows are fetched one outcome class at a time (the wordings of
each class from outputs/wording_groups.csv) and tagged with the class locally. Writes
data/rows/<class>.parquet and data/rows/window.json. Default window: 2024-01-01 to today.
"""
import argparse
import datetime as dt
import json

import pandas as pd

from hpd_api import DATA, OUTPUTS, keyset

COLS = ("complaint_id,problem_id,received_date,problem_status,problem_status_date,complaint_status,"
        "complaint_status_date,building_id,bbl,borough,major_category,minor_category,type,unit_type,space_type,"
        "problem_duplicate_flag,complaint_anonymous_flag,post_code,community_board")


def q(s):
    return "'" + s.replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="exclusive; default tomorrow")
    args = ap.parse_args()
    end = args.end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    out = DATA / "rows"
    out.mkdir(parents=True, exist_ok=True)
    wg = pd.read_csv(OUTPUTS / "wording_groups.csv")
    window = f"received_date >= '{args.start}T00:00:00' and received_date < '{end}T00:00:00'"
    total = 0
    for group, sub in wg.groupby("group"):
        f = out / f"{group}.parquet"
        if f.exists():
            print(f"{group}: cached"); continue
        words = [w for w in sub["status_description"] if isinstance(w, str) and w.strip()]
        if group == "U_missing":
            where = f"{window} and status_description is null"
        else:
            where = f"{window} and status_description in ({', '.join(q(w) for w in words)})"
        parts = []
        for df in keyset(COLS, where, log=lambda n, last: print(f"  {group}: +{n:,} (to problem_id {last})")):
            parts.append(df)
        df = pd.concat(parts) if parts else pd.DataFrame(columns=COLS.split(","))
        df["group"] = group
        df.to_parquet(f, index=False)
        total += len(df)
        print(f"{group}: {len(df):,} rows")
    (out / "window.json").write_text(json.dumps({"start": args.start, "end": end, "rows": total,
                                                 "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=2))
    print("done", total)


if __name__ == "__main__":
    main()
