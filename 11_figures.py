"""Report figures from the tables 10_models.py writes. Each figure is saved as PDF (vector, for the report) and PNG.

Run 09_features.py and 10_models.py first (and `10_models.py --keep-lot` for fig8). Writes outputs/figures/fig*.pdf
and .png, and fig_numbers.csv (the extra numbers computed here: capacity curve, LINE OF TRAVEL check, selection mix).
LINE OF TRAVEL complaints are left out everywhere, as in 10_models.py; fig8 compares against the run that kept them.

Colors: the two models take categorical slots 1 and 2 (blue, orange); the baseline is ink with an open
square marker; random and ceilings are muted gray reference lines. Every series is also labeled directly.
"""
# Purpose: Report figures.
# Input:   data/model/complaints_features.parquet, data/model/test_scores.parquet, outputs/model/*.csv, outputs/model_with_lot/headline.csv (fig8).
# Output:  outputs/figures/fig*.pdf and .png, outputs/figures/fig_numbers.csv.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from hpd_api import DATA, OUTPUTS

M, FIG = OUTPUTS / "model", OUTPUTS / "figures"
FIG.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = {"baseline": (INK, "s", "Baseline: most problems first"), "logistic": (BLUE, "o", "Logistic (readable)"),
          "gradient": (ORANGE, "D", "Gradient boosting (flexible)")}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "lines.linewidth": 1.6})
NUMBERS = []


# Purpose: Month tick labels on a date axis.
# Input:   axes, months between ticks.
# Output:  none (formats the axes).
def months(ax, step=3):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, step)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


# Purpose: Save a figure as PDF and PNG and close it.
# Input:   figure, file stem.
# Output:  outputs/figures/<name>.pdf and .png.
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


# Purpose: Top share of each week's pool for one score (same cut as select() in 10_models.py).
# Input:   complaint frame, scores, share of the week.
# Output:  boolean Series.
def week_select(frame, score, k):
    d = frame.assign(s=score).sort_values(["week", "s", "tie"], ascending=[True, False, False])
    sel = d.groupby("week").cumcount() < np.maximum(1, (k * d.groupby("week").s.transform("size")).round())
    return sel.reindex(frame.index)


# Purpose: Recall per week for one selection.
# Input:   complaint frame, boolean selection.
# Output:  Series of recall by week.
def week_recall(frame, sel):
    g = pd.DataFrame({"w": frame.week, "hit": sel & (frame.label == 1), "pos": frame.label}).groupby("w").sum()
    return g.hit / g.pos


c_all = pd.read_parquet(DATA / "model" / "complaints_features.parquet")
c = c_all[c_all.maj_line_of_travel == 0]
t = pd.read_parquet(DATA / "model" / "test_scores.parquet")
head = pd.read_csv(M / "headline.csv")
weekly = pd.read_csv(M / "weekly.csv", parse_dates=["week"])
slices = pd.read_csv(M / "slices.csv")
clusters = pd.read_csv(M / "clusters.csv")
coef = pd.read_csv(M / "logistic_coefficients.csv")
SCORES = {"baseline": t.n_problems.astype(float), "logistic": t.score_logistic, "gradient": t.score_boosting}
TEST0, VAL0 = t.week.min(), pd.Timestamp("2025-07-21")

# ---------------------------------------------------------------- fig1: the data and the split
w = c.groupby("week").agg(pool=("label", "size"), rate=("label", "mean"))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 3.4), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15})
for ax in (a1, a2):
    ax.axvspan(VAL0, TEST0, color="#f1f0ec", lw=0)
    ax.axvspan(TEST0, w.index.max() + pd.Timedelta(days=7), color="#e8f0fb", lw=0)
    ax.axvline(pd.Timestamp("2024-10-01"), color=MUTED, lw=0.8, ls=(0, (3, 3)))
a1.plot(w.index, w.pool, color=INK, lw=1.2)
a1.set_ylabel("labeled complaints\nper week")
a1.set_ylim(0, w.pool.max() * 1.25)
a1.set_title("Weekly labeled pool and violation rate, with the time split")
y = w.pool.max() * 1.13
for x, s in [(w.index.min() + pd.Timedelta(days=150), "train (fit)"), (VAL0 + pd.Timedelta(days=20), "validation"),
             (TEST0 + pd.Timedelta(days=60), "test (scored once)")]:
    a1.text(x, y, s, color=INK2, fontsize=8)
a1.text(pd.Timestamp("2024-10-08"), w.pool.max() * 0.08, "Oct 2024 shift", color=INK2, fontsize=7.5)
a2.plot(w.index, w.rate, color=INK, lw=1.2)
a2.set_ylabel("share that were\nviolations")
a2.set_ylim(0.3, 0.7)
a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
months(a2)
save(fig, "fig1_split_timeline")

# ---------------------------------------------------------------- fig2: headline with intervals
order = ["Random", "Baseline: most problems first", "Logistic (readable)", "Gradient boosting (flexible)"]
h = head.set_index("ranking").loc[order]
ceiling = weekly[weekly.ranking == order[1]].ceiling.mean()
fig, ax = plt.subplots(figsize=(6.5, 2.0))
colors = [MUTED, INK, BLUE, ORANGE]; marks = ["o", "s", "o", "D"]
for i, (name, r) in enumerate(h.iterrows()):
    yy = len(order) - 1 - i
    ax.plot([r.lo, r.hi], [yy, yy], color=colors[i], lw=2.4, solid_capstyle="butt")
    ax.plot(r["recall@25%"], yy, marks[i], color=colors[i], ms=7, mfc="white" if name.startswith("Baseline") else colors[i], mew=1.5)
    lift = "" if i < 2 else f"   +{r['minus baseline'] * 100:.1f} pts over baseline"
    ax.text(r["recall@25%"] + 0.006, yy, f"{r['recall@25%']:.3f}{lift}", va="center", fontsize=8, color=INK)
ax.axvline(ceiling, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(ceiling - 0.004, 3.45, f"best possible {ceiling:.3f}", ha="right", fontsize=7.5, color=INK2)
ax.set_yticks(range(4)); ax.set_yticklabels(order[::-1])
ax.set_xlim(0.2, 0.55); ax.set_ylim(-0.6, 3.8)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.set_xlabel("recall at the top 25% of each week's pool, mean of 35 test weeks")
ax.grid(axis="y", visible=False)
ax.set_title("Headline: both models beat the baseline; the gap between them is small")
save(fig, "fig2_headline")

# ---------------------------------------------------------------- fig3: weekly lift over baseline
pv = weekly.pivot(index="week", columns="ranking", values="recall")
fig, ax = plt.subplots(figsize=(6.5, 2.4))
ax.axhline(0, color=INK, lw=0.9)
for key, col in [("logistic", "Logistic (readable)"), ("gradient", "Gradient boosting (flexible)")]:
    d = (pv[col] - pv[order[1]]) * 100
    ax.plot(d.index, d, color=SERIES[key][0], marker=SERIES[key][1], ms=3.5, lw=1.4)
ax.text(pv.index[0], -0.35, "baseline", color=INK2, fontsize=8, va="top")
ax.set_ylabel("recall minus baseline\n(percentage points)")
ax.set_xlim(pv.index[0] - pd.Timedelta(days=4), pv.index[-1] + pd.Timedelta(days=4))
months(ax, 1)
ax.legend([plt.Line2D([], [], color=SERIES[k][0], marker=SERIES[k][1], ms=4) for k in ("logistic", "gradient")],
          [SERIES[k][2] for k in ("logistic", "gradient")], ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
ax.set_ylim(-1, max(5, ((pv.max(axis=1) - pv[order[1]]) * 100).max() + 0.5))
ax.set_title("Lift over the baseline in each of the 35 test weeks", pad=22)
save(fig, "fig3_weekly_lift")

# ---------------------------------------------------------------- fig4: capacity curve (recall at other visit budgets)
ks = np.round(np.arange(0.05, 0.51, 0.05), 2)
curve = {key: [week_recall(t, week_select(t, s, k)).mean() for k in ks] for key, s in SCORES.items()}
ceil = [np.minimum(1, (np.maximum(1, (k * t.groupby("week").size()).round())) / t.groupby("week").label.sum()).mean() for k in ks]
rand = [week_recall(t, week_select(t, np.zeros(len(t)), k)).mean() for k in ks]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.plot(ks, ceil, color=MUTED, ls=(0, (4, 3)), lw=1)
ax.plot(ks, rand, color=MUTED, lw=1)
ax.text(0.345, 0.75, "best possible", color=INK2, fontsize=8, ha="right")
ax.text(ks[-1] + 0.01, rand[-1], "random", color=INK2, fontsize=8, va="center")
for key, ys in curve.items():
    col, mk, lab = SERIES[key]
    ax.plot(ks, ys, color=col, marker=mk, ms=4, mfc="white" if key == "baseline" else col)
ax.text(ks[-1] + 0.01, curve["gradient"][-1] + 0.03, "boosting", color=INK, fontsize=8, va="center")
ax.text(ks[-1] + 0.01, curve["logistic"][-1] - 0.012, "logistic", color=INK, fontsize=8, va="center")
ax.text(ks[-1] + 0.01, curve["baseline"][-1] - 0.03, "baseline", color=INK, fontsize=8, va="center")
ax.axvline(0.25, color=INK2, lw=0.8)
ax.text(0.255, 0.04, "locked budget: 25%", color=INK2, fontsize=8)
ax.set_xlabel("share of each week's pool that gets a visit")
ax.set_ylabel("recall (mean over test weeks)")
ax.set_xlim(0.03, 0.57); ax.set_ylim(0, 1.0)
for f in (ax.xaxis, ax.yaxis):
    f.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.set_title("Recall at other visit budgets (test weeks)")
save(fig, "fig4_capacity_curve")
for i, k in enumerate(ks):
    NUMBERS.append({"figure": "fig4", "budget": k, "random": rand[i], "baseline": curve["baseline"][i], "logistic": curve["logistic"][i],
                    "boosting": curve["gradient"][i], "ceiling": ceil[i]})


# ---------------------------------------------------------------- slice dot plots (fig5, fig6, fig7)
# Purpose: Dot plot of slice recall with intervals for the three rankings.
# Input:   slice rows from slices.csv, file stem, title, optional highlighted slice, note, height, group per row.
# Output:  the figure, saved with save().
def slice_plot(rows, name, title, highlight=None, note=None, height=None, groups=None):
    rows = rows.reset_index(drop=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.5, height or 0.32 * n + 1.0))
    off = {"baseline": 0.22, "logistic": 0, "gradient": -0.22}
    for i, r in rows.iterrows():
        yy = n - 1 - i
        if highlight is not None and r["slice"] == highlight:
            ax.axhspan(yy - 0.45, yy + 0.45, color="#fbe6dc", lw=0)
        for key, (col, mk, _) in SERIES.items():
            ax.plot([r[f"{key} lo"], r[f"{key} hi"]], [yy + off[key]] * 2, color=col, lw=1.6, solid_capstyle="butt")
            ax.plot(r[f"{key} recall"], yy + off[key], mk, color=col, ms=4.5, mfc="white" if key == "baseline" else col, mew=1.2)
    labels = [f"{r['slice']}   n={r.complaints:,}, viol. {r.violations:,}" for _, r in rows.iterrows()]
    ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.02, 1.02 if rows[["gradient hi", "logistic hi", "baseline hi"]].max().max() > 0.7 else 0.72)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    if groups is not None:
        for i in range(1, n):
            if groups[i] != groups[i - 1]:
                ax.axhline(n - i - 0.5, color=MUTED, lw=0.6)
    ax.set_xlabel("recall within the slice at the overall weekly top-25% cut")
    hs = [plt.Line2D([], [], color=col, marker=mk, mfc="white" if key == "baseline" else col, lw=1.6, ms=5)
          for key, (col, mk, _) in SERIES.items()]
    ax.legend(hs, [s[2] for s in SERIES.values()], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    ax.set_title(title, pad=22)
    save(fig, name)


named = slices[slices.kind == "named in proposal"].copy()
named["slice"] = named.variable.map({"borough": "", "heat": "", "HPD triage type": "", "problems": ""}).fillna("") + named["slice"].str.title()
named["slice"] = named["slice"].replace({"Heat / Hot Water": "Heat or hot water", "No Heat Problem": "No heat problem", "2+ Problems": "2+ problems",
                                         "1 Problem": "1 problem", "Immediate Emergency": "Immediate emergency", "Non Emergency": "Non emergency"})
slice_plot(named, "fig5_named_slices", "Slices named in the proposal: single-problem complaints are the worst",
           highlight="1 problem", groups=list(named.variable))

ex = slices[slices.kind == "exploratory"]
pick = ex[((ex.variable == "unit type") & ex["slice"].isin(["APARTMENT", "BUILDING-WIDE", "PUBLIC AREA"]))
          | (ex.variable == "building violations, prior 12 months") | (ex.variable == "building had a labeled train complaint")].copy()
pick["slice"] = pick["slice"].replace({"APARTMENT": "Unit: apartment", "BUILDING-WIDE": "Unit: building-wide", "PUBLIC AREA": "Unit: public area",
                                       "0 violations, 12 m": "Building violations, 12 m: 0", "1-9": "Building violations, 12 m: 1-9",
                                       "10+": "Building violations, 12 m: 10+", "no": "Building new since train: yes", "yes": "Building seen in train"})
slice_plot(pick, "fig6_exploratory_slices", "Exploratory slices (found by looking, not named in advance)",
           highlight="Building violations, 12 m: 0", groups=list(pick.variable))

cl = clusters.copy()
cl["slice"] = cl["slice"].map({0: "Cluster 0: little history", 1: "Cluster 1: moderate history", 2: "Cluster 2: heavy history"})
slice_plot(cl, "fig7_clusters", "k-means clusters of building history: the misses gather in cluster 0",
           highlight="Cluster 0: little history", height=2.3)

# ---------------------------------------------------------------- fig8: LINE OF TRAVEL, kept against removed and refit
MW = OUTPUTS / "model_with_lot"
if (MW / "headline.csv").exists():
    hw, hn = pd.read_csv(MW / "headline.csv").set_index("ranking"), head.set_index("ranking")
    n_lot = int(((c_all.week >= TEST0) & (c_all.maj_line_of_travel > 0)).sum())
    fig, ax = plt.subplots(figsize=(6.5, 1.6))
    for i, (key, name) in enumerate([("gradient", "Gradient boosting (flexible)"), ("logistic", "Logistic (readable)")]):
        a_, b_ = hw.loc[name, "minus baseline"] * 100, hn.loc[name, "minus baseline"] * 100
        NUMBERS.append({"figure": "fig8", "model": key, "lift_kept_pts": a_, "lift_removed_refit_pts": b_})
        col, mk, lab = SERIES[key]
        ax.plot([b_, a_], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(a_, i, mk, color=col, ms=7, zorder=2)
        ax.plot(b_, i, mk, color=col, ms=7, mfc="white", mew=1.5, zorder=2)
        hi_, lo_ = (a_, b_) if a_ >= b_ else (b_, a_)
        ax.text(hi_ + 0.06, i, f"{hi_:.1f} {'kept' if hi_ == a_ else 'removed, refit'}", va="center", fontsize=8)
        ax.text(lo_ - 0.06, i, f"{'removed, refit' if lo_ == b_ else 'kept'} {lo_:.1f}", va="center", ha="right", fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Gradient boosting", "Logistic"])
    ax.set_xlim(0, 3.5); ax.set_ylim(-0.6, 1.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("recall minus baseline, percentage points (filled: category kept; open: removed and models refit)")
    ax.set_title(f"LINE OF TRAVEL ({n_lot:,} test-period complaints): lift with the category kept and removed")
    save(fig, "fig8_line_of_travel")
else:
    print("skipped fig8: run `python 10_models.py --keep-lot` first")

# ---------------------------------------------------------------- fig9: the readable model's largest effects
cf = coef.copy()
cf["name"] = (cf.feature.str.replace("minor_", "", regex=False).str.replace(" | ", ": ", regex=False).str.replace("has_", "includes ", regex=False)
              .str.replace("_yes", "", regex=False).str.replace("unit_type_", "unit: ", regex=False)
              .str.replace("space_type_", "space: ", regex=False).str.replace("problems_", "problems: ", regex=False)
              .str.replace("rate12_", "building violation rate 12 m: ", regex=False).str.replace("c7_", "complaints at building, 7 d: ", regex=False)
              .str.replace("month_", "month ", regex=False).str.replace("_", " ", regex=False))
sel = pd.concat([cf.nsmallest(8, "coefficient"), cf.nlargest(8, "coefficient").iloc[::-1]])
fig, ax = plt.subplots(figsize=(6.5, 3.6))
yy = np.arange(len(sel))[::-1]
ax.axvline(1, color=INK, lw=0.9)
for y0, (_, r) in zip(yy, sel.iterrows()):
    ax.plot([1, r.odds_ratio], [y0, y0], color=BLUE, lw=1.6)
    ax.plot(r.odds_ratio, y0, "o", color=BLUE, ms=4.5)
    ax.text(r.odds_ratio * (1.08 if r.odds_ratio > 1 else 0.93), y0, f"{r.odds_ratio:.2f}", va="center",
            ha="left" if r.odds_ratio > 1 else "right", fontsize=7.5)
ax.set_xscale("log")
right = max(6, sel.odds_ratio.max() * 1.6)
ticks = [x for x in [0.25, 0.5, 1, 2, 4, 8, 16, 32] if x <= right]
ax.set_xticks(ticks); ax.set_xticklabels([f"{x:g}" for x in ticks])
ax.set_xlim(0.3, right)
ax.set_yticks(yy); ax.set_yticklabels(sel.name.str.slice(0, 48), fontsize=7.5)
ax.grid(axis="y", visible=False)
ax.set_xlabel("odds ratio (log scale); each feature is a yes/no statement against its reference group")
ax.set_title("Readable model: the eight largest effects each way")
save(fig, "fig9_logistic_effects")

# ---------------------------------------------------------------- fig10: what each ranking sends inspectors to
mix = []
for key, s in SCORES.items():
    selected = t[week_select(t, s, 0.25)]
    per = selected.week.nunique()
    one = selected.n_problems == 1
    mix.append({"ranking": SERIES[key][2], "multi, violation": ((~one) & (selected.label == 1)).sum() / per,
                "single, violation": (one & (selected.label == 1)).sum() / per, "no violation": (selected.label == 0).sum() / per})
mix = pd.DataFrame(mix).set_index("ranking")
fig, ax = plt.subplots(figsize=(6.5, 1.9))
fills = {"multi, violation": "#1c5cab", "single, violation": "#86b6ef", "no violation": "#e4e3df"}
for i, (name, r) in enumerate(mix.iloc[::-1].iterrows()):
    left = 0
    for k, col in fills.items():
        ax.barh(i, r[k], left=left, color=col, height=0.6, edgecolor="white", linewidth=1.5)
        if r[k] > 40:
            ax.text(left + r[k] / 2, i, f"{r[k]:.0f}", ha="center", va="center", fontsize=8, color="white" if k == "multi, violation" else INK)
        left += r[k]
ax.set_yticks(range(3)); ax.set_yticklabels(mix.index[::-1], fontsize=8)
ax.set_xlim(0, mix.sum(axis=1).max() * 1.02); ax.grid(axis="y", visible=False)
ax.set_xlabel("visits per week (top 25% of the pool), mean over test weeks")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in fills.values()],
          ["violation, 2+ problems", "violation, 1 problem", "no violation"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
ax.set_title("What each ranking sends inspectors to", pad=22)
save(fig, "fig10_selection_mix")
for name, r in mix.iterrows():
    NUMBERS.append({"figure": "fig10", "model": name, **r.to_dict()})

pd.DataFrame(NUMBERS).round(4).to_csv(FIG / "fig_numbers.csv", index=False)
print("done")
