"""Complaint and problem volume per year and per week, from data/by_year/complaints_*.csv.

Writes outputs/volume_by_year.csv. The weekly figure is the yearly count divided by 52 (a rough average:
it hides the winter heat-season peak and, for the current year, only a partial year of data).
"""
import pandas as pd

from hpd_api import DATA, OUTPUTS


def main():
    files = sorted((DATA / "by_year").glob("complaints_*.csv"))
    if not files:
        raise SystemExit("No data/by_year/complaints_*.csv found. Run 01_fetch_aggregates.py first.")
    v = pd.concat([pd.read_csv(f) for f in files]).set_index("year").sort_index()
    v["problems_per_complaint"] = v["problems"] / v["complaints"]
    v["complaints_per_week"] = v["complaints"] / 52
    v["problems_per_week"] = v["problems"] / 52
    OUTPUTS.mkdir(exist_ok=True)
    v.to_csv(OUTPUTS / "volume_by_year.csv")
    print(v[v.index >= 2013].to_string(float_format="{:,.1f}".format))
    print("\nNote: the last row is a partial year if the run is before Dec 31.")


if __name__ == "__main__":
    main()
