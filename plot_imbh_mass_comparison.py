"""
Two-panel figure of IMBH mass vs. host-cluster stellar mass, sharing the x
axis: the top panel is one model run, the bottom panel another (by default
the default-mode run, labelled "Maximal model", over the observational run
with --n-relation bf20_seed --radius-relation marks_kroupa12, labelled
"Conservative model").

Each panel is the left-hand panel of plot_imbh_occupation.py, without the
title and the M_BH/M_cl ratio labels: every cluster with an IMBH as a point
(one colour/marker per z = 0 stellar-mass bin), plus each bin's running
median of M_BH in cluster-mass bins holding at least --min-per-bin clusters.
Both panels use the same cluster-mass bins and the same axis limits. A dotted
line marks --mbh-min (the TDE floor in analysis.py); --no-mbh-line removes it.
--ratio-lines adds the dashed constant-M_BH/M_cl guides (unlabelled).

The two runs are chosen with --top-flags / --bottom-flags, each a quoted
string of the usual run-selection flags (see run_naming.py), e.g.
    --bottom-flags "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12"
Files are read as <data-dir>/cluster_output_<ID>_<SNAP><suffix>.dat (or .csv),
exactly as in plot_imbh_occupation.py, which must sit next to this script
(together with run_naming.py).

Examples:
    python plot_imbh_mass_comparison.py --data-dir /u/scratch/c/clairewi/imbh-output
    python plot_imbh_mass_comparison.py --data-dir ... --top-label "Maximal" --bottom-label "Conservative"
"""
import argparse
import os
import shlex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from run_naming import add_run_args, run_suffix, run_description
from plot_imbh_occupation import DEFAULT_MASS_BINS, EXTRA_COLORS, MARKERS, load_halo, classify

FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
    "legend.title_fontsize": FONT_SIZE - 2,
})

DEFAULT_TOP_FLAGS = ""
DEFAULT_BOTTOM_FLAGS = "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12"


def parse_run_flags(flag_string):
    """Run-selection flags (a string) -> (suffix, description), via run_naming."""
    rp = argparse.ArgumentParser(add_help=False)
    add_run_args(rp)
    rargs = rp.parse_args(shlex.split(flag_string))
    return run_suffix(rargs), run_description(rargs)


def load_run(data_dir, snap, suffix, bins, mbh_min):
    """Per bin: (m_cl, m_bh) of usable clusters with M_BH > 0, or None if no files."""
    out = []
    for tag, label, color, ids in bins:
        frames = [f for f in (load_halo(data_dir, h, snap, suffix) for h in ids) if f is not None]
        if not frames:
            print(f"  WARNING: no cluster_output files for bin {tag}; it is left out of this panel")
            out.append(None)
            continue
        df = pd.concat(frames, ignore_index=True)
        valid, _ = classify(df, mbh_min, require_formed=False)
        m_cl = pd.to_numeric(df["cluster_mass_msun"], errors="coerce").to_numpy()
        m_bh = pd.to_numeric(df["IMBH_mass_msun"], errors="coerce").to_numpy()
        has_bh = valid & (m_bh > 0)
        print(f"  [{tag}] {len(frames)} halo(s), {len(df)} clusters, "
              f"{int(has_bh.sum())} with an IMBH mass > 0 plotted "
              f"({int((~valid).sum())} without an IMBH mass excluded)")
        out.append((m_cl[has_bh], m_bh[has_bh]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Base output directory ($OUTDIR) with the cluster_output files.")
    p.add_argument("--top-data-dir", default=None, help="Override --data-dir for the top run.")
    p.add_argument("--bottom-data-dir", default=None, help="Override --data-dir for the bottom run.")
    p.add_argument("--snap", default="99", help="Output snapshot in the cluster_output file names (default: 99).")
    p.add_argument("--top-flags", default=DEFAULT_TOP_FLAGS,
                   help='Run-selection flags for the top panel, as one quoted string (default: "" = default mode).')
    p.add_argument("--bottom-flags", default=DEFAULT_BOTTOM_FLAGS,
                   help=f'Run-selection flags for the bottom panel (default: "{DEFAULT_BOTTOM_FLAGS}").')
    p.add_argument("--top-label", default="Maximal model")
    p.add_argument("--bottom-label", default="Conservative model")
    p.add_argument("--legend-title", default=r"Stellar mass at $z = 0$")
    p.add_argument("--bin", nargs="+", action="append", metavar=("LABEL", "ID"),
                   help="A mass bin: label then subhalo IDs. Repeat per bin. "
                        "Default: the three bins used by the TDE-rate plots.")
    p.add_argument("--mbh-min", type=float, default=500.0,
                   help="Height of the dotted line [Msun] (default: 500, analysis.py's TDE floor).")
    p.add_argument("--no-mbh-line", action="store_true", help="Don't draw the dotted --mbh-min line.")
    p.add_argument("--ratio-lines", action="store_true",
                   help="Draw unlabelled dashed lines of M_BH/M_cl = 1e-3, 1e-2, 1e-1.")
    p.add_argument("--dlogm", type=float, default=0.25, help="Cluster-mass bin width for the medians [dex].")
    p.add_argument("--min-per-bin", type=int, default=5,
                   help="Only draw a median where a cluster-mass bin has at least this many clusters.")
    p.add_argument("--output", default=None,
                   help="Output image (default: imbh_mass_comparison<top suffix>_vs<bottom suffix>.png).")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if args.bin:
        bins = []
        for i, e in enumerate(args.bin):
            if len(e) < 2:
                p.error("--bin needs a label followed by at least one subhalo ID")
            tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in e[0])
            bins.append((tag, e[0], EXTRA_COLORS[i % len(EXTRA_COLORS)], e[1:]))
    else:
        bins = DEFAULT_MASS_BINS

    panels = []  # (label, per-bin data)
    for which, flags, label, ddir in (("top", args.top_flags, args.top_label, args.top_data_dir),
                                      ("bottom", args.bottom_flags, args.bottom_label, args.bottom_data_dir)):
        suffix, desc = parse_run_flags(flags)
        ddir = ddir or args.data_dir
        print(f"\n{which} panel ({label}): {ddir}  [{desc or 'default mode'}]")
        data = load_run(ddir, args.snap, suffix, bins, args.mbh_min)
        if all(d is None for d in data):
            raise SystemExit(f"No cluster_output files found for the {which} run -- "
                             f"check --data-dir, --snap and --{which}-flags.")
        panels.append((label, data, suffix))

    # common cluster-mass bins / limits over both runs
    all_m = np.concatenate([d[0] for _, data, _ in panels for d in data if d is not None])
    all_bh = np.concatenate([d[1] for _, data, _ in panels for d in data if d is not None])
    lo = np.floor(np.log10(all_m.min()) / args.dlogm) * args.dlogm
    hi = np.ceil(np.log10(all_m.max()) / args.dlogm) * args.dlogm
    edges = 10 ** np.arange(lo, hi + 0.5 * args.dlogm, args.dlogm)
    if len(edges) < 2:
        edges = np.array([all_m.min() * 0.9, all_m.max() * 1.1])
    centres = np.sqrt(edges[1:] * edges[:-1])
    xlim = (edges[0], edges[-1])
    ylim = (10 ** 2, 10 **6)

    fig, axes = plt.subplots(2, 1, figsize=(9, 11), sharex=True, sharey=True)
    for ax, (label, data, _) in zip(axes, panels):
        for i, ((tag, _, color, _), d) in enumerate(zip(bins, data)):
            if d is None:
                continue
            m_cl, m_bh = d
            marker = MARKERS[i % len(MARKERS)]
            ax.scatter(m_cl, m_bh, s=20, marker=marker, color=color, alpha=0.35,
                       linewidths=0, rasterized=True)
            idx = np.digitize(m_cl, edges) - 1
            med = np.array([np.median(m_bh[idx == j]) if (idx == j).sum() >= args.min_per_bin else np.nan
                            for j in range(len(centres))])
            ok = np.isfinite(med)
            ax.plot(centres[ok], med[ok], color=color, lw=2.5, marker=marker, ms=8,
                    markeredgecolor="white", markeredgewidth=1.2)

        if args.ratio_lines:
            xx = np.array(xlim)
            for ratio in (1e-3, 1e-2, 1e-1):
                ax.plot(xx, ratio * xx, ls="--", color="0.6", lw=1, zorder=0)
        if not args.no_mbh_line:
            ax.axhline(args.mbh_min, ls=":", color="0.45", lw=1.2, zorder=0)

        ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT_SIZE + 2, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="0.92", lw=0.6)
        ax.set_axisbelow(True)

    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[1].set_xlabel(r"Cluster stellar mass $M_{\rm cl}$ [$M_\odot$]")
    fig.supylabel(r"IMBH mass $M_{\rm BH}$ [$M_\odot$]", fontsize=FONT_SIZE + 2, x=0.02)

    handles = [Line2D([], [], color=color, lw=2.5, marker=MARKERS[i % len(MARKERS)], ms=8,
                      markeredgecolor="white", markeredgewidth=1.2, label=label)
               for i, (_, label, color, _) in enumerate(bins)]
    fig.tight_layout(rect=(0.03, 0, 1, 0.945), h_pad=0.0)
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(bins),
               bbox_to_anchor=(0.5, 0.94), title=args.legend_title, columnspacing=1.2,
               handlelength=1.8)

    out = args.output or f"imbh_mass_comparison{panels[0][2] or '_default'}_vs{panels[1][2] or '_default'}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
