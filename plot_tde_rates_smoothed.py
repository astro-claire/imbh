"""
Average the TDE rates of several halos in each of a few halo-mass bins and plot
one smoothed curve per bin, each with a shaded band showing the halo-to-halo
spread within that bin.

Reads the per-halo outputs of analysis.py:

    <data-dir>/tde_rates_<subhaloid>_alpha<alpha>.csv

Steps (done separately for each mass bin):
  1. Each halo's rate is smoothed in cosmic time with a kernel of width
     --smooth-myr (default 50 Myr). The default top-hat (boxcar) kernel
     conserves the total mass lost: the integral of rate over time is the
     same before and after smoothing (apart from edge effects).
  2. At each time, the mean over the bin's halos is the central curve, and the
     band is either the full min-max range of those halos (default), the
     16-84th percentiles, or mean +/- 1 standard deviation.
  3. All bins are plotted together against redshift, and each bin is also
     written to its own CSV.

Mass bins default to the three in DEFAULT_MASS_BINS below. Override them with
one --bin per bin, giving a label and the subhalo IDs:

    --bin "gt1e11" 685512 697044 588075 467548 665702 \
    --bin "1e10-1e11" 801308 753345 8 745415 826784

Example:
    python plot_tde_rates_smoothed.py --data-dir /u/scratch/c/clairewi/imbh-output
    python plot_tde_rates_smoothed.py --smooth-myr 100 --band percentile --logy
    python plot_tde_rates_smoothed.py --escaped exclude   # drop escaped clusters (-> *_noescaped.png)
    python plot_tde_rates_smoothed.py --escaped only      # escaped clusters alone (-> *_escapedonly.png)

Observational runs: pass the same flags as submit_halos.pl / submit_analysis.pl.
--data-dir stays the base output directory; the run's analysis<suffix>/ folder
is found automatically and every output file name gets the same suffix, so the
default-mode plots are never overwritten:

    python plot_tde_rates_smoothed.py --data-dir /u/scratch/c/clairewi/imbh-output \\
        --mode observational --n-relation bf20_seed
    -> tde_rate_smoothed_massbins_alpha1.2_50Myr_obs_bf20_seed_gclf_brown_gnedin21_analytic_hostown_boost1.0.png
"""
import argparse
import os

from run_naming import (add_run_args, run_suffix, analysis_dir, run_description,
                        add_escaped_arg, escaped_suffix, escaped_description)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

cosmo = FlatLambdaCDM(71, 0.27, Ob0=0.044, Tcmb0=2.726 * u.K)

# TNG50-1-Dark subhalo IDs, grouped by halo mass.
# Each entry: (short tag for file names, legend label, color, subhalo IDs)
DEFAULT_MASS_BINS = [
    ("gt1e11", r"$>10^{11}\,M_\odot$", "#0072B2",
     ["685512", "697044", "588075", "467548", "665702"]),
    ("1e10-1e11", r"$10^{10}$–$10^{11}\,M_\odot$", "#D55E00",
     ["801308", "753345", "8", "745415", "826784"]),
    ("1e9-1e10", r"$10^{9}$–$10^{10}\,M_\odot$", "#009E73",
     ["1235585", "1117358", "1136724", "1044309", "939095"]),
]
# Colors for bins given on the command line (Okabe-Ito, colorblind-safe)
EXTRA_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

Z_MAX = 20  # left edge of the plot

# 20 pt for all text: axis labels, tick labels, legend and title
FONT_SIZE = 20
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'legend.title_fontsize': FONT_SIZE,
})


def load_rates(data_dir, halo_ids, alpha, escaped="include"):
    """
    Returns (time_yr, rates, used_ids): rates has shape (n_halos, n_times).
    All halos from analysis.py share the same time grid; if one doesn't,
    it's interpolated onto the first halo's grid. Returns None if none of the
    halos' files exist.

    escaped: 'include' (total rate), 'exclude' (total minus the escaped
    clusters' share) or 'only' (just the escaped share) -- see --escaped.
    """
    time_yr, rates, used_ids = None, [], []
    for halo_id in halo_ids:
        path = os.path.join(data_dir, f"tde_rates_{halo_id}_alpha{alpha}.csv")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping halo {halo_id}")
            continue
        df = pd.read_csv(path)
        t = df['time'].to_numpy(dtype=float)
        r = df['tde_rate_array_msunyr'].to_numpy(dtype=float)
        if escaped != "include":
            if 'tde_rate_escaped_msunyr' not in df.columns:
                raise SystemExit(f"{path} has no tde_rate_escaped_msunyr column -- re-run "
                                 f"analysis.py (with the escaped split) to use --escaped {escaped}.")
            r_esc = df['tde_rate_escaped_msunyr'].to_numpy(dtype=float)
            # clip: float round-off in total - escaped can leave tiny negatives
            r = np.clip(r - r_esc, 0.0, None) if escaped == "exclude" else r_esc
        if time_yr is None:
            time_yr = t
        elif len(t) != len(time_yr) or not np.allclose(t, time_yr):
            print(f"  note: halo {halo_id} is on a different time grid; interpolating")
            r = np.interp(time_yr, t, r, left=0.0, right=0.0)
        rates.append(r)
        used_ids.append(halo_id)
    if not rates:
        return None
    return time_yr, np.vstack(rates), used_ids


def smooth(rates, dt_yr, smooth_myr, kernel):
    """Smooth each row of `rates` along time with a kernel of width smooth_myr."""
    width_bins = smooth_myr * 1e6 / dt_yr
    if kernel == "boxcar":
        size = max(1, int(round(width_bins)))
        return uniform_filter1d(rates, size=size, axis=1, mode="constant", cval=0.0)
    # gaussian: treat smooth_myr as the FWHM
    sigma_bins = width_bins / (2 * np.sqrt(2 * np.log(2)))
    return gaussian_filter1d(rates, sigma=sigma_bins, axis=1, mode="constant", cval=0.0)


def band(smoothed, mean, kind):
    """Lower and upper edges of the shaded band, plus a description of it."""
    if kind == "minmax":
        return smoothed.min(axis=0), smoothed.max(axis=0), "halo-to-halo range"
    if kind == "percentile":
        lo, hi = np.percentile(smoothed, [16, 84], axis=0)
        return lo, hi, "16–84th percentile"
    std = smoothed.std(axis=0)
    return np.clip(mean - std, 0, None), mean + std, r"mean $\pm1\sigma$"


def age_to_redshift(t_yr):
    """Fast age -> redshift via interpolation on a precomputed grid."""
    z_grid = np.concatenate([[0.0], np.logspace(-4, np.log10(1000), 4000)])
    age_grid = cosmo.age(z_grid).to(u.yr).value  # decreasing with z
    return np.interp(t_yr, age_grid[::-1], z_grid[::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".",
                        help="Base output directory. Default-mode runs: holds the "
                             "tde_rates_<ID>_alpha<alpha>.csv files. Observational/labelled "
                             "runs: their files are read from <data-dir>/analysis<suffix>/ "
                             "(see the run selection flags).")
    parser.add_argument("--alpha", type=float, default=1.2,
                        help="Power-law index the rates were computed with (default: 1.2).")
    parser.add_argument("--bin", nargs="+", action="append", metavar=("LABEL", "ID"),
                        help="A mass bin: a label followed by its subhalo IDs. Repeat for "
                             "each bin. Default: the three bins in DEFAULT_MASS_BINS.")
    parser.add_argument("--smooth-myr", type=float, default=50.0,
                        help="Smoothing width in Myr (default: 50). For --kernel gaussian "
                             "this is the FWHM.")
    parser.add_argument("--kernel", choices=["boxcar", "gaussian"], default="boxcar",
                        help="Smoothing kernel (default: boxcar, which conserves mass lost).")
    parser.add_argument("--band", choices=["minmax", "percentile", "std"], default="minmax",
                        help="Shaded band: full halo-to-halo range (minmax, default), "
                             "16-84th percentiles, or mean +/- 1 std.")
    parser.add_argument("--logy", action="store_true", help="Log-scale y axis.")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: "
                             "tde_rate_smoothed_massbins_alpha<alpha>_<smooth>Myr<suffix>.png).")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open an interactive window (e.g. on a cluster node).")
    add_run_args(parser)
    add_escaped_arg(parser)
    args = parser.parse_args()

    suffix = run_suffix(args)
    data_dir = analysis_dir(args.data_dir, suffix)
    run_desc = run_description(args)
    print(f"Reading analysis outputs from {data_dir}"
          + (f"  [{run_desc}]" if run_desc else "  [default mode]"))
    # run + escaped-selection text for the title ('' if neither applies)
    run_desc = "; ".join(x for x in (run_desc, escaped_description(args)) if x)
    if args.escaped != "include":
        print(f"Escaped clusters: {args.escaped}")

    if args.bin:
        mass_bins = []
        for i, entry in enumerate(args.bin):
            if len(entry) < 2:
                parser.error("--bin needs a label followed by at least one subhalo ID")
            label, ids = entry[0], entry[1:]
            tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
            mass_bins.append((tag, label, EXTRA_COLORS[i % len(EXTRA_COLORS)], ids))
    else:
        mass_bins = DEFAULT_MASS_BINS

    tag = f"alpha{args.alpha}_{args.smooth_myr:g}Myr{suffix}{escaped_suffix(args)}"
    fig, ax = plt.subplots(figsize=(8, 7))
    band_desc = None
    all_hi = []  # (z, hi) per bin, for setting log y-limits

    for bin_tag, label, color, halo_ids in mass_bins:
        print(f"\n[{bin_tag}]")
        loaded = load_rates(data_dir, halo_ids, args.alpha, args.escaped)
        if loaded is None:
            print(f"WARNING: no tde_rates files found for bin {bin_tag}; skipping it")
            continue
        time_yr, rates, used_ids = loaded
        dt_yr = np.median(np.diff(time_yr))
        print(f"Loaded {len(used_ids)} halos, {len(time_yr)} time bins of {dt_yr/1e6:.3g} Myr")

        smoothed = smooth(rates, dt_yr, args.smooth_myr, args.kernel)
        mean = smoothed.mean(axis=0)
        lo, hi, band_desc = band(smoothed, mean, args.band)

        # The smoothed curves vary on ~smooth_myr scales, so plotting every
        # 1e5-yr bin is overkill: keep ~20 points per smoothing width.
        step = max(1, int(args.smooth_myr * 1e6 / dt_yr / 20))
        sl = slice(None, None, step)
        z = age_to_redshift(time_yr[sl])

        csv_path = f"tde_rate_smoothed_{bin_tag}_{tag}.csv"
        pd.DataFrame({
            'time_yr': time_yr[sl], 'redshift': z,
            'mean_rate_msunyr': mean[sl], 'band_lo_msunyr': lo[sl], 'band_hi_msunyr': hi[sl],
            **{f'rate_{h}_msunyr': smoothed[i][sl] for i, h in enumerate(used_ids)},
        }).to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        ax.fill_between(z, lo[sl], hi[sl], color=color, alpha=0.2, lw=0)
        # ax.plot(z, mean[sl], color=color, lw=2, label=f"{label} (mean of {len(used_ids)})")
        ax.plot(z, mean[sl], color=color, lw=2, label=f"{label}")
        all_hi.append((z, hi[sl]))

    if not all_hi:
        raise SystemExit(f"No tde_rates files found for any bin in {data_dir} -- check "
                         "--data-dir, --alpha and the run selection flags.")

    ax.set_xlabel('Redshift', fontsize=20)
    ax.set_ylabel(r'TDE Rate ($M_\odot$/yr)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlim([Z_MAX, 0])  # redshift decreases left-to-right: time flows forward
    if args.logy:
        ax.set_yscale('log')
        positive = np.concatenate([h[(z <= Z_MAX) & (h > 0)] for z, h in all_hi])
        if positive.size:
            ax.set_ylim(bottom=max(positive.max() * 1e-4, positive.min()))
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(rf'$\alpha = {args.alpha}$, {args.smooth_myr:g} Myr {args.kernel} smoothing'
                 f'\n(shaded: {band_desc})'
                 + (f'\n{run_desc}' if run_desc else ''), fontsize=16 if not run_desc else 13)
    ax.legend(frameon=False, title="Stellar mass at z=0", fontsize=20, loc = "upper right")
    plt.tight_layout()
    # Legend outside the axes, to the right. Added after tight_layout so the
    # axes keep their size; bbox_inches="tight" below widens the saved image
    # to include it.
    ax.legend(frameon=False, title="Halo mass", fontsize=16,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    output = args.output or f"tde_rate_smoothed_massbins_{tag}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved {output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()