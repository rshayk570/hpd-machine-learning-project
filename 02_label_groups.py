"""Group the status wordings into outcome classes and measure how the label population changes by year.

Reads data/by_year/status_*.csv (from 01_fetch_aggregates.py). Writes to outputs/:
  wording_groups.csv   every wording, its row count, and the class it was given (audit this file)
  group_counts.csv     rows and share per class over the chosen years
  label_by_year.csv    per year: rows, positive rate among verdicts, and the unlabeled shares

"Positive rate among verdicts" is P / (P + N): of the problems an inspector ruled on, the share where
a violation was issued. It is not the share of all complaints.
"""
import argparse

import pandas as pd

from hpd_api import DATA, OUTPUTS
from hpd_labels import GROUPS, add_group


def load_status():
    frames = []
    for f in sorted((DATA / "by_year").glob("status_*.csv")):
        d = pd.read_csv(f, keep_default_na=True)
        d["year"] = int(f.stem.split("_")[1])
        frames.append(d)
    if not frames:
        raise SystemExit("No data/by_year/status_*.csv found. Run 01_fetch_aggregates.py first.")
    df = pd.concat(frames, ignore_index=True)
    df["status_description"] = df["status_description"].fillna("")
    return add_group(df)


def per_year(df):
    t = df.pivot_table(index="year", columns="group", values="n", aggfunc="sum", fill_value=0)
    t = t.reindex(columns=GROUPS, fill_value=0)
    rows = t.sum(axis=1)
    verdict = t[["P", "N", "N_partial", "P_existing", "P_lead"]].sum(axis=1)
    out = pd.DataFrame({
        "rows": rows,
        "positive_rate_P_over_P_plus_N": t["P"] / (t["P"] + t["N"]),
        "verdict_share": verdict / rows,
        "tenant_fixed_share": t["U_fixed"] / rows,
        "no_access_share": t["U_noaccess"] / rows,
        "open_share": t["U_open"] / rows,
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-year", type=int, default=2013, help="first received year to include (default 2013)")
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)

    df = load_status()
    df = df[df["year"] >= args.from_year]

    wordings = (df.groupby(["status_description", "group"], as_index=False)["n"].sum()
                  .sort_values("n", ascending=False))
    wordings.to_csv(OUTPUTS / "wording_groups.csv", index=False)

    counts = df.groupby("group")["n"].sum().reindex(GROUPS, fill_value=0).to_frame("rows")
    counts["share"] = counts["rows"] / counts["rows"].sum()
    counts.to_csv(OUTPUTS / "group_counts.csv")

    by_year = per_year(df)
    by_year.to_csv(OUTPUTS / "label_by_year.csv")

    total = counts["rows"].sum()
    p, n = counts.loc["P", "rows"], counts.loc["N", "rows"]
    pd.options.display.float_format = "{:,.3f}".format
    print(f"{args.from_year}-latest: {total:,} problem rows, {df['status_description'].nunique()} distinct wordings\n")
    print(counts.to_string(formatters={"rows": "{:,.0f}".format, "share": "{:.1%}".format}))
    print(f"\npositive rate among verdicts, P/(P+N): {p / (p + n):.1%}\n")
    print(by_year.to_string(formatters={
        "rows": "{:,.0f}".format, "positive_rate_P_over_P_plus_N": "{:.1%}".format,
        "verdict_share": "{:.1%}".format, "tenant_fixed_share": "{:.1%}".format,
        "no_access_share": "{:.1%}".format, "open_share": "{:.1%}".format}))


if __name__ == "__main__":
    main()
