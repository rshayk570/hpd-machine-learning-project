# hpd-machine-learning-project

An investigation into improving NYC's Housing Preservation and Development complaint response with machine
learning techniques (CSCI 5622 project).

Ranks NYC HPD housing complaints by how likely an inspection is to find a violation, using only what is
known when the complaint arrives. Data: NYC Open Data `ygpa-z7cr`, fetched by script (nothing is committed).

## Reproduce the headline number

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
