"""
Plot the TDE rate vs. redshift for several halos on one set of axes, one
colour per halo. Reads the per-halo outputs of analysis.py:

    <data-dir>/tde_rates_<subhaloid>_alpha<alpha>.csv

Example:
    python plot_tde_rates_combined.py --data-dir /u/scratch/c/clairewi/imbh-output
    python plot_tde_rates_combined.py --alpha 1.2 --halo-ids 685512 467548 --logy
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value

cosmo = FlatLambdaCDM(71, 0.27, Ob0=0.044, Tcmb0=2.726 * u.K)

# TNG50-1-Dark subhalo IDs of the 5 selected ellipticals
# (from tng_download/selected_ellipticals.json)
DEFAULT_HALO_IDS = ["685512", "697044", "588075", "467548", "665702"]

# Colour-blind-safe categorical palette (Okabe-Ito), one per halo
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
          "#56B4E9", "#F0E442", "#000000"]

Z_MAX = 20  # left edge of the plot


def reduce_zero_runs(time_arr, rate_arr):
    """
    Keep all nonzero points, plus the immediate zero neighbors on either side
    of each nonzero run (so the line still touches down to zero exactly where
    it should), plus the first/last points of the array for axis bounds.
    Drops all other redundant interior zeros.
    """
    nz = rate_arr != 0

    keep = nz.copy()
    keep[1:] |= nz[:-1]   # zero immediately AFTER a nonzero point
    keep[:-1] |= nz[1:]   # zero immediately BEFORE a nonzero point

    keep[0] = True   # preserve left edge of plot
    keep[-1] = True  # preserve right edge of plot

    return time_arr[keep], rate_arr[keep]


def load_halo(path):
    """Read one analysis.py output and return (redshift, rate) arrays."""
    df = pd.read_csv(path)
    time_reduced, rate_reduced = reduce_zero_runs(
        df['time'].to_numpy(), df['tde_rate_array_msunyr'].to_numpy())
    print(f"  {os.path.basename(path)}: {len(df)} -> {len(time_reduced)} points")
    z_reduced = z_at_value(cosmo.age, time_reduced * u.yr).value
    return z_reduced, rate_reduced


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".",
                        help="Directory holding the tde_rates_<ID>_alpha<alpha>.csv files "
                             "(default: current directory).")
    parser.add_argument("--alpha", type=float, default=1.2,
                        help="Power-law index the rates were computed with (default: 1.2).")
    parser.add_argument("--halo-ids", nargs="+", default=DEFAULT_HALO_IDS,
                        help="Subhalo IDs to plot (default: the 5 selected ellipticals).")
    parser.add_argument("--logy", action="store_true",
                        help="Log-scale y axis (useful if the halos' rates differ by orders "
                             "of magnitude).")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: "
                             "tde_rate_vs_redshift_combined_alpha<alpha>.png).")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open an interactive window (e.g. on a cluster node).")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    n_plotted = 0
    for i, halo_id in enumerate(args.halo_ids):
        path = os.path.join(args.data_dir, f"tde_rates_{halo_id}_alpha{args.alpha}.csv")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping halo {halo_id}")
            continue
        z, rate = load_halo(path)
        ax.plot(z, rate, color=COLORS[i % len(COLORS)], lw=1.2, label=f"Subhalo {halo_id}")
        n_plotted += 1

    if n_plotted == 0:
        raise SystemExit("No tde_rates files found -- check --data-dir and --alpha.")

    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'TDE Rate ($M_\odot$/yr)')
    ax.set_xlim([Z_MAX, 0])  # redshift decreases left-to-right: time flows forward
    if args.logy:
        ax.set_yscale('log')
    ax.set_title(rf'$\alpha = {args.alpha}$')
    ax.legend(frameon=False)
    plt.tight_layout()

    output = args.output or f"tde_rate_vs_redshift_combined_alpha{args.alpha}.png"
    plt.savefig(output, dpi=200)
    print(f"Saved {output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
