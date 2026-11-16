"""Small client for the NYC Open Data (Socrata) HPD complaints dataset, plus shared paths."""
import time
from pathlib import Path

import requests

HOST = "https://data.cityofnewyork.us"
DATASET = "ygpa-z7cr"  # Housing Maintenance Code Complaints and Problems (one row per problem)
PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
FIRST_YEAR = 2001  # the table has two stray rows in 2001-02; real volume starts in 2003


def soda(query, timeout=240, tries=3):
    """Run a SoQL query against the dataset and return the result rows as a list of dicts."""
    url = f"{HOST}/resource/{DATASET}.json"
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params={"$query": query}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == tries:
                raise
            print(f"  retry {attempt}/{tries - 1} after: {e}")
            time.sleep(5 * attempt)


def view_metadata(timeout=60):
    """The dataset's portal metadata (name, license fields, columns, update times)."""
    r = requests.get(f"{HOST}/api/views/{DATASET}.json", timeout=timeout)
    r.raise_for_status()
    return r.json()


def year_clause(year):
    """SoQL filter for complaints received during one calendar year."""
    return f"received_date >= '{year}-01-01T00:00:00' and received_date < '{year + 1}-01-01T00:00:00'"


def soda_csv(query, timeout=300, tries=3):
    """Run a SoQL query and return a DataFrame (CSV transport: smaller than JSON for wide pulls)."""
    import io
    import pandas as pd
    url = f"{HOST}/resource/{DATASET}.csv"
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params={"$query": query}, timeout=timeout)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype=str)
        except (requests.RequestException, ValueError) as e:
            if attempt == tries:
                raise
            print(f"  retry {attempt}/{tries - 1} after: {e}")
            time.sleep(5 * attempt)


def keyset(select, where, order_col="problem_id", page=200000, log=None):
    """Fetch every row matching `where`, paging on a numeric key (problem_id by default). Yields DataFrames."""
    last = None
    while True:
        w = where if last is None else f"({where}) and {order_col} > {last}"
        df = soda_csv(f"select {select} where {w} order by {order_col} limit {page}")
        if df.empty:
            return
        last = int(df[order_col].astype("int64").max())
        if log:
            log(len(df), last)
        yield df
        if len(df) < page:
            return
