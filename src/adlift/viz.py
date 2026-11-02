"""Static figures for the report (matplotlib).

Colors follow a validated categorical order (blue, orange, aqua, yellow, magenta,
green, violet, red), assigned by entity and never cycled; truth/benchmark lines
are neutral ink so they never compete with series colors. One y-axis per panel.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e1"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "figure.dpi": 130,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


METHOD_ORDER = ["difference_in_means", "regression_adjustment", "ipw", "aipw_parametric", "aipw_ml"]
METHOD_LABEL = {"difference_in_means": "Naive diff.", "regression_adjustment": "Regression adj.",
                "ipw": "IPW", "aipw_parametric": "AIPW (logit)", "aipw_ml": "AIPW (GBM, cross-fit)"}


def incrementality_mc(summary: pd.DataFrame, path, outcome: str = "converted"):
    style()
    s = summary[summary.outcome == outcome]
    scen = list(dict.fromkeys(s["scenario"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    y = np.arange(len(scen))
    off = np.linspace(-0.3, 0.3, len(METHOD_ORDER))
    for k, m in enumerate(METHOD_ORDER):
        d = s[s.method == m].set_index("scenario").reindex(scen)
        rb = np.where(d["truth"].abs() > 1e-9, 100 * d["bias"] / d["truth"].abs(), np.nan)
        axes[0].scatter(np.clip(rb, -60, 200), y + off[k], s=28, color=SERIES[k], label=METHOD_LABEL[m], zorder=3)
        axes[1].scatter(100 * d["coverage"], y + off[k], s=28, color=SERIES[k], zorder=3)
    axes[0].axvline(0, color=INK, lw=1)
    axes[0].set_xlabel("Bias, % of true ATT (clipped at 200%)")
    axes[1].axvline(90, color=INK, lw=1)
    axes[1].set_xlabel("90% CI coverage (%)")
    axes[1].set_xlim(-2, 102)
    axes[0].set_yticks(y, [x.replace("_", " ") for x in scen])
    axes[0].invert_yaxis()
    axes[0].set_title(f"Bias by selection mechanism ({'conversion' if outcome == 'converted' else 'brand awareness'})")
    axes[1].set_title("Interval coverage (target 90%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    _save(fig, path)


def criteo(obs: pd.DataFrame, path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, o, lab in zip(axes, ["visit", "conversion"], ["Visit", "Conversion"]):
        d = obs[obs.outcome == o].reset_index(drop=True)
        tot, se = d["rct_tot"].iloc[0], d["rct_tot_se"].iloc[0]
        ax.axvspan(100 * (tot - 1.645 * se), 100 * (tot + 1.645 * se), color=GRID, zorder=0)
        ax.axvline(100 * tot, color=INK, lw=1.5, zorder=1)
        yy = np.arange(len(d))
        ax.text(100 * tot, len(d) - 0.45, " RCT benchmark (Wald), 90% CI shaded", color=INK2, fontsize=8, va="center")
        ax.errorbar(100 * d["estimate"], yy, xerr=100 * 1.645 * d["se"], fmt="o", color=SERIES[0], ms=6,
                    capsize=0, zorder=3)
        ax.set_yticks(yy, d["method"])
        ax.set_ylim(len(d) - 0.2, -0.6)
        ax.set_xlabel("Effect of exposure on exposed users (pp)")
        ax.set_title(f"Criteo: {lab}")
    _save(fig, path)


def ads_psa(by_bin: pd.DataFrame, path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(by_bin))
    axes[0].plot(x, 100 * by_bin["psa_rate"], "-o", color=SERIES[1], label="PSA arm (ads cannot work)", ms=5)
    axes[0].plot(x, 100 * by_bin["ad_rate"], "-o", color=SERIES[0], label="Ad arm", ms=5)
    axes[0].set_xticks(x, by_bin["ads_bin"])
    axes[0].set_xlabel("Impressions seen")
    axes[0].set_ylabel("Conversion rate (%)")
    axes[0].set_title("Conversion rises with exposure in BOTH arms")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, 100 * by_bin["lift"], color=SERIES[0], width=0.6, zorder=2)
    axes[1].errorbar(x, 100 * by_bin["lift"], yerr=100 * 1.645 * by_bin["se"], fmt="none", ecolor=INK2, zorder=3)
    axes[1].axhline(0, color=INK, lw=1)
    axes[1].set_xticks(x, by_bin["ads_bin"])
    axes[1].set_xlabel("Impressions seen")
    axes[1].set_ylabel("Ad - PSA (pp), 90% CI")
    axes[1].set_title("Causal lift within exposure bins")
    _save(fig, path)


def mmm_mc(summary: pd.DataFrame, path, channel: str = "search"):
    style()
    s = summary[(summary.channel == channel)].copy()
    s["cond"] = s["scenario"] + "\n" + s["calibration"].replace({"none": "no calibration"})
    conds = list(dict.fromkeys(s["cond"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(conds))
    for k, (metric, lab) in enumerate([("roas", "ROAS"), ("mroas", "Marginal ROAS")]):
        d = s[s.metric == metric].set_index("cond").reindex(conds)
        axes[k].barh(y, d["bias"], color=SERIES[k], height=0.55, zorder=2)
        axes[k].axvline(0, color=INK, lw=1)
        for i, (b, c) in enumerate(zip(d["bias"], d["coverage"])):
            axes[k].text(max(b, 0) + 0.03, i, f"coverage {100 * c:.0f}%", va="center", ha="left",
                         fontsize=8, color=INK2)
        axes[k].set_xlabel(f"Bias (posterior mean - truth), {lab}")
        axes[k].set_title(f"{channel.title()} {lab}: bias and 90% CrI coverage")
    axes[0].set_yticks(y, conds)
    axes[0].invert_yaxis()
    _save(fig, path)


def response_curves(curves: pd.DataFrame, path):
    style()
    ch = list(dict.fromkeys(curves["channel"]))
    fig, axes = plt.subplots(1, len(ch), figsize=(3 * len(ch), 3), sharey=False)
    for k, (ax, c) in enumerate(zip(axes, ch)):
        d = curves[curves.channel == c]
        ax.fill_between(d["spend_multiple"], d["low"] / 1e3, d["high"] / 1e3, color=SERIES[k], alpha=0.18, lw=0)
        ax.plot(d["spend_multiple"], d["mean"] / 1e3, color=SERIES[k])
        ax.axvline(1, color=MUTED, lw=1, ls="--")
        ax.set_title(c)
        ax.set_xlabel("x current weekly spend")
    axes[0].set_ylabel("Weekly contribution ($K)")
    _save(fig, path)


def auction_tradeoff(summary: pd.DataFrame, path):
    style()
    s = summary[summary.format == "second"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, (_, r) in enumerate(s.iterrows()):
        ax.errorbar(100 * r["fill_rate"], r["rpm_lift_pct"],
                    yerr=[[r["rpm_lift_pct"] - r["rpm_lift_ci_low_pct"]], [r["rpm_lift_ci_high_pct"] - r["rpm_lift_pct"]]],
                    fmt="o", color=SERIES[k], ms=8)
        off = (6, -12) if r["policy"] == "empirical_segment" else (6, 4)
        ax.annotate(r["policy"], (100 * r["fill_rate"], r["rpm_lift_pct"]), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("Fill rate (%)")
    ax.set_ylabel("RPM change vs. $2 floor (%), 90% CI")
    ax.set_title("Reserve policies: revenue vs. fill (14 held-out days)")
    _save(fig, path)


def viewer(weekly: pd.DataFrame, path, treat_weeks: int = 6):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, lab in zip(axes, ["revenue", "watch_minutes"], ["Revenue per viewer", "Watch time per viewer"]):
        for k, arm in enumerate(["B_optimized_floors", "C_double_pods"]):
            d = weekly[(weekly.arm == arm) & (weekly.metric == metric)]
            ax.plot(d["week"], d["true_relative_pct"], color=SERIES[k], label=f"{arm} (truth)")
            ax.errorbar(d["week"], d["relative_pct"], yerr=100 * 1.645 * d["se"] / d["control_mean"], fmt="o",
                        color=SERIES[k], ms=4, alpha=0.8)
        ax.axvspan(treat_weeks + 0.5, d["week"].max() + 0.5, color=GRID, zorder=0)
        ax.text(0.99, 0.03, "shaded: post-period,\nall viewers on status quo", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=INK2)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlabel("Week")
        ax.set_ylabel("% vs. status quo")
        ax.set_title(lab)
    axes[0].legend(fontsize=8, loc="center left")
    _save(fig, path)
