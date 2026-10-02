"""
Four-panel version of plot_tde_rates_smoothed.py: smoothed, bin-averaged TDE
rates vs. redshift for two model runs (columns) and two cluster selections
(rows).

    rows    (shared x axis): top    = escaped clusters excluded ("Bound to galaxy")
                             bottom = escaped clusters only     ("Escaped")
    columns (shared y axis): left   = --left-flags  run (default: default mode, "Maximal model")
                             right  = --right-flags run (default: --mode observational
                                      --n-relation bf20_seed --radius-relation marks_kroupa12,
                                      "Conservative model")

Each panel is exactly what plot_tde_rates_smoothed.py draws for that run and
--escaped selection (same smoothing, band and mass bins -- the functions are
imported from it), minus the title. Default-mode columns keep the grey
z = 7 -> 0 shading, labelled --shade-text. y limits are shared along each row
(the two rows have their own scale); x is shared by all four panels.

Reads <data-dir>/tde_rates_<ID>_alpha<alpha>.csv for a default-mode run, or
<data-dir>/analysis<suffix>/... for any other run (see run_naming.py); the
files must carry tde_rate_escaped_msunyr (analysis.py with the escaped split).
plot_tde_rates_smoothed.py and run_naming.py must sit next to this script.

Examples:
    python plot_tde_rates_quad.py --data-dir /u/scratch/c/clairewi/imbh-output
    python plot_tde_rates_quad.py --data-dir ... --logy --smooth-myr 100
    python plot_tde_rates_quad.py --data-dir ... --left-flags "--jitter-dex 0.3"
"""
import argparse
import shlex

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from run_naming import add_run_args, run_suffix, analysis_dir, run_description
from plot_tde_rates_smoothed import (DEFAULT_MASS_BINS, EXTRA_COLORS, Z_MAX, load_rates,
                                     smooth, band, age_to_redshift)

FONT_SIZE = 20
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2,
    "legend.fontsize": FONT_SIZE - 2,
    "legend.title_fontsize": FONT_SIZE - 2,
})

DEFAULT_LEFT_FLAGS = ""
DEFAULT_RIGHT_FLAGS = "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12"
ROWS = [("exclude", "Bound to galaxy"), ("only", "Escaped")]
SHADE_Z = (7, 0)  # no new clusters form below z = 7 in the default (simulation-calibrated) mode


def wrap_words(text, words_per_line):
    """Break text into lines of at most words_per_line words (so it fits the z = 7 -> 0 strip)."""
    w = text.split()
    return "\n".join(" ".join(w[i:i + words_per_line]) for i in range(0, len(w), words_per_line))


def parse_run_flags(flag_string):
    """Run-selection flags (a string) -> parsed args, via run_naming."""
    rp = argparse.ArgumentParser(add_help=False)
    add_run_args(rp)
    return rp.parse_args(shlex.split(flag_string))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Base output directory ($OUTDIR).")
    p.add_argument("--left-flags", default=DEFAULT_LEFT_FLAGS,
                   help='Run-selection flags for the left column, one quoted string (default: "" = default mode).')
    p.add_argument("--right-flags", default=DEFAULT_RIGHT_FLAGS,
                   help=f'Run-selection flags for the right column (default: "{DEFAULT_RIGHT_FLAGS}").')
    p.add_argument("--left-label", default="Maximal model")
    p.add_argument("--right-label", default="Conservative model")
    p.add_argument("--top-label", default=ROWS[0][1])
    p.add_argument("--bottom-label", default=ROWS[1][1])
    p.add_argument("--shade-text", default="no further star cluster formation",
                   help="Text written in the grey z = 7 -> 0 region of default-mode panels ('' for none).")
    p.add_argument("--shade-text-y", type=float, default=0.95,
                   help="Height of the top of --shade-text, as a fraction of the panel height (default: 0.95).")
    p.add_argument("--legend-title", default=r"Stellar mass at $z = 0$")
    p.add_argument("--alpha", type=float, default=1.2, help="Power-law index of the rates (default: 1.2).")
    p.add_argument("--bin", nargs="+", action="append", metavar=("LABEL", "ID"),
                   help="A mass bin: label then subhalo IDs. Repeat per bin. Default: DEFAULT_MASS_BINS.")
    p.add_argument("--smooth-myr", type=float, default=50.0, help="Smoothing width in Myr (default: 50).")
    p.add_argument("--kernel", choices=["boxcar", "gaussian"], default="boxcar")
    p.add_argument("--band", choices=["minmax", "percentile", "std"], default="minmax")
    p.add_argument("--logy", action="store_true", help="Log-scale y axes.")
    p.add_argument("--output", default=None,
                   help="Output image (default: tde_rate_smoothed_quad_alpha<a>_<s>Myr_<left>_vs_<right>.png).")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if args.bin:
        mass_bins = []
        for i, e in enumerate(args.bin):
            if len(e) < 2:
                p.error("--bin needs a label followed by at least one subhalo ID")
            tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in e[0])
            mass_bins.append((tag, e[0], EXTRA_COLORS[i % len(EXTRA_COLORS)], e[1:]))
    else:
        mass_bins = DEFAULT_MASS_BINS

    columns = []  # (label, run args, suffix, data dir)
    for flags, label in ((args.left_flags, args.left_label), (args.right_flags, args.right_label)):
        rargs = parse_run_flags(flags)
        suffix = run_suffix(rargs)
        ddir = analysis_dir(args.data_dir, suffix)
        print(f"{label}: {ddir}  [{run_description(rargs) or 'default mode'}]")
        columns.append((label, rargs, suffix, ddir))

    row_labels = [args.top_label, args.bottom_label]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey="row")
    row_hi = [[], []]  # positive band tops per row, for log y-limits
    band_desc = None
    for c, (col_label, rargs, _, ddir) in enumerate(columns):
        for r, (escaped, _) in enumerate(ROWS):
            ax = axes[r, c]
            print(f"\n[{col_label} / {row_labels[r]}]")
            for bin_tag, label, color, halo_ids in mass_bins:
                loaded = load_rates(ddir, halo_ids, args.alpha, escaped)
                if loaded is None:
                    print(f"WARNING: no tde_rates files for bin {bin_tag}; skipping it in this panel")
                    continue
                time_yr, rates, used_ids = loaded
                dt_yr = np.median(np.diff(time_yr))
                smoothed = smooth(rates, dt_yr, args.smooth_myr, args.kernel)
                mean = smoothed.mean(axis=0)
                lo, hi, band_desc = band(smoothed, mean, args.band)
                step = max(1, int(args.smooth_myr * 1e6 / dt_yr / 20))
                sl = slice(None, None, step)
                z = age_to_redshift(time_yr[sl])
                ax.fill_between(z, lo[sl], hi[sl], color=color, alpha=0.2, lw=0)
                ax.plot(z, mean[sl], color=color, lw=2)
                h = hi[sl][(z <= Z_MAX) & (hi[sl] > 0)]
                row_hi[r].append(h)
                print(f"  {bin_tag}: {len(used_ids)} halos")

            if rargs.mode != "observational":
                ax.axvspan(*SHADE_Z, color="grey", alpha=0.3, lw=0)
                if args.shade_text:
                    ax.text(np.mean(SHADE_Z), args.shade_text_y, wrap_words(args.shade_text, 2),
                            transform=ax.get_xaxis_transform(), ha="center", va="top",
                            multialignment="center", fontsize=FONT_SIZE - 4, color="0.2",
                            linespacing=1.1, zorder=3, clip_on=True)

    for ax in axes.flat:
        ax.set_xlim(Z_MAX, 0)  # redshift decreases left to right: time runs forward
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for r in range(2):
        if args.logy:
            axes[r, 0].set_yscale("log")
            pos = np.concatenate(row_hi[r]) if row_hi[r] else np.array([])
            if pos.size:
                axes[r, 0].set_ylim(max(pos.max() * 1e-4, pos.min()), pos.max() * 1.5)
        else:
            axes[r, 0].set_ylim(bottom=0)
        # row label on the right-hand edge of the row
        axes[r, 1].text(1.03, 0.5, row_labels[r], transform=axes[r, 1].transAxes, rotation=270,
                        ha="left", va="center", fontsize=FONT_SIZE + 2, fontweight="bold")
    for c, (col_label, _, _, _) in enumerate(columns):
        axes[0, c].set_title(col_label, fontsize=FONT_SIZE + 2, fontweight="bold", pad=12)

    fig.supxlabel("Redshift", fontsize=FONT_SIZE + 2, y=0.035)
    fig.supylabel(r"TDE rate ($M_\odot$/yr)", fontsize=FONT_SIZE + 2, x=0.015)

    handles = [Line2D([], [], color=color, lw=2.5, label=label) for _, label, color, _ in mass_bins]
    fig.tight_layout(rect=(0.01, 0.01, 0.97, 0.90), h_pad=1.0, w_pad=1.0)
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.905), title=args.legend_title)

    tags = [s.lstrip("_") or "default" for _, _, s, _ in columns]
    out = args.output or (f"tde_rate_smoothed_quad_alpha{args.alpha}_{args.smooth_myr:g}Myr_"
                          f"{tags[0]}_vs_{tags[1]}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}  (shaded bands: {band_desc})")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
