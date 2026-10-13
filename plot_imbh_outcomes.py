"""
Where our IMBHs end up at z = 0, by IMBH mass and host-galaxy mass.

Reads the same cached z = 0 positions as plot_imbh_radial_astrid.py
(<cache-dir>/imbh_z0_positions<suffix>_mbh<min>.csv; build them there first).
For each model (rows) and z = 0 stellar-mass bin of the host (columns), one
horizontal bar per IMBH-mass bin shows the fraction of those IMBHs in each end
state, with the number per galaxy written beside the bar:

  at the centre of the central galaxy   inspiraled into, or pinned in, the main progenitor
  r < R200c                             wandering inside the halo
  R200c - 3 R200c                       wandering in the outskirts
  > 3 R200c                             beyond 3 R200c (the 'escaped' status)
  unknown                               no z = 0 position (missing track / tree ends early)

r is the distance from the central galaxy, R200c the central's at z = 0. Only
IMBHs >= --mbh-min (default 500 Msun, the lightest kept in the cache) are
included. Also prints the counts (with imbh.py's own status for each end state)
and writes them to <output stem>.csv.

Example:
  python plot_imbh_outcomes.py --cache-dir ASTRID/z0_positions \\
      --model "Maximal" "--jitter-dex 0.1" \\
      --model "Conservative" "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12"
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_imbh_radial_astrid import build_parser, load_inputs, NUCLEAR, WANDERING, UNKNOWN
from rate_runs import DEFAULT_MASS_BINS

FONT_SIZE = 20
plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 3,
                     "ytick.labelsize": FONT_SIZE - 3, "legend.fontsize": FONT_SIZE - 3,
                     "legend.title_fontsize": FONT_SIZE - 3})

# end states, nearest -> farthest; one-hue ordinal ramp (dark = centre), validated light->dark,
# plus a neutral gray for 'unknown'
STATES = [
    ("centre", "at centre of central galaxy", "#0d366b"),
    ("in", r"wandering, $r<R_{200c}$", "#256abf"),
    ("out", r"wandering, $R_{200c}$–$3R_{200c}$", "#5598e7"),
    ("far", r"beyond $3R_{200c}$", "#86b6ef"),
    ("unknown", "unknown", "#c3c2b7"),
]
SURFACE = "#fcfcfb"
TEXT2 = "#52514e"


def mass_bins(mbh_min):
    edges = [mbh_min, 1e3, 1e4, 1e5, np.inf]
    labels = [rf"${mbh_min:g}$–$10^3$", r"$10^3$–$10^4$", r"$10^4$–$10^5$", r"$>10^5$"]
    return edges, labels


def end_state(df):
    x = df["r_kpc"] / df["r200_kpc"]
    state = pd.Series("unknown", index=df.index)
    state[df["location"] == NUCLEAR] = "centre"
    w = df["location"] == WANDERING
    state[w & (x < 1)] = "in"
    state[w & (x >= 1) & (x < 3)] = "out"
    state[w & (x >= 3)] = "far"
    return state


def main():
    p = build_parser(__doc__, "imbh_outcomes.pdf")
    args = p.parse_args()
    args.no_astrid, args.match_astrid_mass, args.mbh_max = True, False, None
    models, _ = load_inputs(args)
    edges, mlabels = mass_bins(args.mbh_min)
    global PLAIN
    PLAIN = [f"{args.mbh_min:g}-1e3", "1e3-1e4", "1e4-1e5", ">1e5"]

    rows = []
    for label, _, df, ngal in models:
        d = df[df["location"].isin([NUCLEAR, WANDERING, UNKNOWN])].copy()
        d["state"] = end_state(d)
        d["mbin"] = pd.cut(d["imbh_mass_msun"], edges, right=False, labels=mlabels)
        for tag, _, _, _ in DEFAULT_MASS_BINS:
            n_gal = ngal.get(tag, 0)
            for ml in mlabels:
                sel = d[(d["mass_bin"] == tag) & (d["mbin"] == ml)]
                row = {"model": label, "host_bin": tag, "imbh_mass_bin": ml, "n_galaxies": n_gal,
                       "n_imbh": len(sel), "per_galaxy": len(sel) / n_gal if n_gal else np.nan}
                for key, _, _ in STATES:
                    s = sel[sel["state"] == key]
                    row[f"n_{key}"] = len(s)
                    row[f"frac_{key}"] = len(s) / len(sel) if len(sel) else np.nan
                    for st, n in s["status"].value_counts().items():
                        row[f"n_{key}_{st}"] = int(n)
                rows.append(row)
    table = pd.DataFrame(rows).fillna({c: 0 for c in [] })
    stem = os.path.splitext(args.output)[0]
    table.to_csv(f"{stem}.csv", index=False)

    # ---------- printout ----------
    print(f"\n{'model':13s} {'host':10s} {'M_BH':>12s} {'N/gal':>8s}  " +
          "  ".join(f"{k:>7s}" for k, _, _ in STATES) + "   (fractions)")
    for r in rows:
        ml = PLAIN[mlabels.index(r["imbh_mass_bin"])]
        fr = "  ".join(f"{r[f'frac_{k}']:7.2f}" if r["n_imbh"] else f"{'-':>7s}" for k, _, _ in STATES)
        print(f"{r['model'][:13]:13s} {r['host_bin']:10s} {ml:>12s} {r['per_galaxy']:8.1f}  {fr}")
    print(f"\nSaved {stem}.csv")

    # ---------- figure ----------
    nrow, ncol = len(models), len(DEFAULT_MASS_BINS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.4 * nrow + 1.2), sharex=True, sharey=True,
                             squeeze=False, gridspec_kw={"wspace": 0.42, "hspace": 0.12})
    y = np.arange(len(mlabels))[::-1]          # lightest IMBHs on top
    for i, (label, _, _, _) in enumerate(models):
        for j, (tag, blabel, _, _) in enumerate(DEFAULT_MASS_BINS):
            ax = axes[i, j]
            for k, ml in enumerate(mlabels):
                r = next(r for r in rows if r["model"] == label and r["host_bin"] == tag
                         and r["imbh_mass_bin"] == ml)
                left = 0.0
                if r["n_imbh"]:
                    for key, _, color in STATES:
                        f = r[f"frac_{key}"]
                        if f > 0:
                            ax.barh(y[k], f, left=left, height=0.72, color=color,
                                    edgecolor=SURFACE, linewidth=1.5)
                        left += f
                    v = r["per_galaxy"]
                    txt = f"{v:,.0f}" if v >= 1000 else f"{v:.3g}"
                else:
                    txt = "0"
                ax.text(1.03, y[k], txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=FONT_SIZE - 4, color=TEXT2)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, len(mlabels) - 0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(mlabels)
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(axis="y", length=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(rf"$M_\star$: {blabel}", fontsize=FONT_SIZE - 2, pad=10)
            if j == ncol - 1:
                ax.text(1.32, 0.5, label, transform=ax.transAxes, rotation=270, ha="left", va="center",
                        fontsize=FONT_SIZE)
        axes[i, 0].set_ylabel(r"$M_{\rm BH}$ ($M_\odot$)")
    for j in range(ncol):
        axes[-1, j].set_xlabel("fraction of IMBHs")
    for j in range(ncol):
        axes[0, j].text(1.03, 1.0, "per gal.", transform=axes[0, j].transAxes, ha="left", va="bottom",
                        fontsize=FONT_SIZE - 6, color=TEXT2)
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for _, l, c in STATES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False,
               title="End state at $z=0$")
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    print(f"Saved {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
