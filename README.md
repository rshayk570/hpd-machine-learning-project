# hpd-machine-learning-project

An investigation into improving NYC's Housing Preservation and Development complaint response with machine
learning techniques (CSCI 5622 project).

Ranks NYC HPD housing complaints by how likely an inspection is to find a violation, using only what is
known when the complaint arrives. Data: NYC Open Data `ygpa-z7cr` (source: NYC Department of Housing
Preservation and Development), fetched by script; the exact pull behind the report is attached to a release.

## Use the exact data from the report (no fetching)

The table updates daily, so a fresh pull gives slightly different counts. The data the report used is a
122 MB zip on the [data-2026-11-05 release](https://github.com/rshayk570/hpd-machine-learning-project/releases/tag/data-2026-11-05):
rows received 2024-01-01 to 2026-11-04 (pulled 2026-11-05 04:36 UTC), building history back to 2003, yearly
aggregates (2026-10-23), and the derived feature table and test scores.

```bash
curl -LO https://github.com/rshayk570/hpd-machine-learning-project/releases/download/data-2026-11-05/hpd-data-snapshot-2026-11-05.zip
shasum -a 256 hpd-data-snapshot-2026-11-05.zip   # 87153f96f4f07903f57b170a4b2903d3b8a7162a34d5e1f5d1aa597d3cbbe703
unzip hpd-data-snapshot-2026-11-05.zip           # in the repo root: creates data/ and SHA256SUMS
shasum -a 256 -c SHA256SUMS                      # every file should print OK
python 09_features.py && python 10_models.py && python 11_figures.py
```

Everything from `09_features.py` on, and `headline.py`, reads only `data/`; no network access is needed. The
zip leaves out `headline.py`'s score cache; `python headline.py --warm` rebuilds it.

## Reproduce from a fresh pull

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 02_label_groups.py                 # outcome wording -> classes (needs 01_fetch_aggregates.py once)
python 04_pull_rows.py                    # row-level pull from 2024-01-01 (about 10 min)
python 05_building_history.py             # violation history per building and month
python 05_building_history.py --group N   # clean-inspection history
python 09_features.py                     # complaint-level table
python 10_models.py                       # baseline, both models, slices, clustering check
python 10_models.py --keep-lot            # earlier run with LINE OF TRAVEL kept (for fig8 only)
python 11_figures.py                      # report figures into outputs/figures/
```

`10_models.py` prints the headline table (recall at the top 25% of each week's labeled pool, test weeks)
and writes it to `outputs/model/headline.csv`. Seed 0 throughout.

## What-if runs (seconds, not minutes)

`headline.py` reruns only the headline, with the same split, metric, tie-break, bootstrap and model code as
`10_models.py`, and caches each model's test scores by setting:

```bash
python headline.py --check                 # defaults reproduce outputs/model/headline.csv
python headline.py --budget 0.10           # another visit budget: instant, no refit
python headline.py --C 0.01                # another logistic penalty
python headline.py --lr 0.1 --trees 200 --leaves 63 --min-leaf 50
python headline.py --keep-lot              # LINE OF TRAVEL kept
python headline.py --warm                  # pre-fit the whole tuning grid once (about 5 min)
```

After `--warm`, every value in the tuning grid is instant; a new boosting setting takes 5-15 s and a new logistic
`C` up to a minute. Changing a feature means editing `09_features.py` and rerunning it (15 s); the cache is keyed
on the feature table, so the next `headline.py` run refits.

LINE OF TRAVEL complaints are removed from train and test (a declared change). HPD enters them after an
inspector sees a condition on the way to the complainant's apartment; none has a 311 request number. The
split's week boundaries are fixed before the removal. `--keep-lot` writes the earlier run to
`outputs/model_with_lot/`.

The source table updates daily. Counts shift slightly on a fresh pull; `data/rows/window.json` records the
pull used (2026-11-05).

Result tables are in `outputs/model/` (`headline.csv`, `weekly.csv`, `slices.csv`, `clusters.csv`, `tuning.csv`,
coefficients and importances; `outputs/model_with_lot/` for the run with LINE OF TRAVEL kept) and the report
figures in `outputs/figures/`. Scripts 01-08 are the data inspection that fixed the label, split and baseline.
