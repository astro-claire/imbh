"""
Average the TDE rates of several halos and plot a single smoothed curve with
a shaded band showing the halo-to-halo spread.

Reads the per-halo outputs of analysis.py:

    <data-dir>/tde_rates_<subhaloid>_alpha<alpha>.csv

Steps:
  1. Each halo's rate is smoothed in cosmic time with a kernel of width
     --smooth-myr (default 50 Myr). The default top-hat (boxcar) kernel
     conserves the total mass lost: the integral of rate over time is the
     same before and after smoothing (apart from edge effects).
  2. At each time, the mean over halos is the central curve, and the band
     is either the full min-max range of the halos (default), the 16-84th
     percentiles, or mean +/- 1 standard deviation.
  3. Plotted against redshift and also written to a CSV.

Example:
    python plot_tde_rates_smoothed.py --data-dir /u/scratch/c/clairewi/imbh-output
    python plot_tde_rates_smoothed.py --smooth-myr 100 --band percentile --logy
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

cosmo = FlatLambdaCDM(71, 0.27, Ob0=0.044, Tcmb0=2.726 * u.K)

# TNG50-1-Dark subhalo IDs of the 5 selected ellipticals
# (from tng_download/selected_ellipticals.json)
DEFAULT_HALO_IDS = ["685512", "697044", "588075", "467548", "665702"]

Z_MAX = 20  # left edge of the plot
LINE_COLOR = "#0072B2"


def load_rates(data_dir, halo_ids, alpha):
    """
    Returns (time_yr, rates, used_ids): rates has shape (n_halos, n_times).
    All halos from analysis.py share the same time grid; if one doesn't,
    it's interpolated onto the first halo's grid.
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
        if time_yr is None:
            time_yr = t
        elif len(t) != len(time_yr) or not np.allclose(t, time_yr):
            print(f"  note: halo {halo_id} is on a different time grid; interpolating")
            r = np.interp(time_yr, t, r, left=0.0, right=0.0)
        rates.append(r)
        used_ids.append(halo_id)
    if not rates:
        raise SystemExit("No tde_rates files found -- check --data-dir and --alpha.")
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


def age_to_redshift(t_yr):
    """Fast age -> redshift via interpolation on a precomputed grid."""
    z_grid = np.concatenate([[0.0], np.logspace(-4, np.log10(1000), 4000)])
    age_grid = cosmo.age(z_grid).to(u.yr).value  # decreasing with z
    return np.interp(t_yr, age_grid[::-1], z_grid[::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".",
                        help="Directory holding the tde_rates_<ID>_alpha<alpha>.csv files.")
    parser.add_argument("--alpha", type=float, default=1.2,
                        help="Power-law index the rates were computed with (default: 1.2).")
    parser.add_argument("--halo-ids", nargs="+", default=DEFAULT_HALO_IDS,
                        help="Subhalo IDs to include (default: the 5 selected ellipticals).")
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
                             "tde_rate_smoothed_alpha<alpha>_<smooth>Myr.png).")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open an interactive window (e.g. on a cluster node).")
    args = parser.parse_args()

    time_yr, rates, used_ids = load_rates(args.data_dir, args.halo_ids, args.alpha)
    dt_yr = np.median(np.diff(time_yr))
    print(f"Loaded {len(used_ids)} halos, {len(time_yr)} time bins of {dt_yr/1e6:.3g} Myr")

    smoothed = smooth(rates, dt_yr, args.smooth_myr, args.kernel)
    mean = smoothed.mean(axis=0)
    if args.band == "minmax":
        lo, hi = smoothed.min(axis=0), smoothed.max(axis=0)
        band_label = f"range of {len(used_ids)} halos"
    elif args.band == "percentile":
        lo, hi = np.percentile(smoothed, [16, 84], axis=0)
        band_label = "16–84th percentile"
    else:
        std = smoothed.std(axis=0)
        lo, hi = np.clip(mean - std, 0, None), mean + std
        band_label = r"mean $\pm1\sigma$"

    # The smoothed curves vary on ~smooth_myr scales, so plotting every
    # 1e5-yr bin is overkill: keep ~20 points per smoothing width.
    step = max(1, int(args.smooth_myr * 1e6 / dt_yr / 20))
    sl = slice(None, None, step)
    z = age_to_redshift(time_yr[sl])

    tag = f"alpha{args.alpha}_{args.smooth_myr:g}Myr"
    csv_path = f"tde_rate_smoothed_{tag}.csv"
    pd.DataFrame({
        'time_yr': time_yr[sl], 'redshift': z,
        'mean_rate_msunyr': mean[sl], 'band_lo_msunyr': lo[sl], 'band_hi_msunyr': hi[sl],
        **{f'rate_{h}_msunyr': smoothed[i][sl] for i, h in enumerate(used_ids)},
    }).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(z, lo[sl], hi[sl], color=LINE_COLOR, alpha=0.25, lw=0, label=band_label)
    ax.plot(z, mean[sl], color=LINE_COLOR, lw=2, label=f"mean of {len(used_ids)} halos")
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'TDE Rate ($M_\odot$/yr)')
    ax.set_xlim([Z_MAX, 0])  # redshift decreases left-to-right: time flows forward
    if args.logy:
        ax.set_yscale('log')
        positive = hi[sl][(z <= Z_MAX) & (hi[sl] > 0)]
        if positive.size:
            ax.set_ylim(bottom=max(positive.max() * 1e-4, positive.min()))
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(rf'$\alpha = {args.alpha}$, {args.smooth_myr:g} Myr {args.kernel} smoothing')
    ax.legend(frameon=False)
    plt.tight_layout()

    output = args.output or f"tde_rate_smoothed_{tag}.png"
    plt.savefig(output, dpi=200)
    print(f"Saved {output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
