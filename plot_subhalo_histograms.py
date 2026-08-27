"""
Plot histograms of delta_t_gyr, halfmass_rad_kpc, and dm_mass_msun from the
CSV produced by subhalo_formation_table.py.

Requires: numpy, pandas, matplotlib
    pip install numpy pandas matplotlib

Usage:
    python plot_subhalo_histograms.py <csv_path> [--out histograms.png] [--bins 40]
                                       [--mass-scale log|linear] [--radius-scale log|linear]
                                       [--deltat-scale log|linear]

Example:
    python plot_subhalo_histograms.py subhalo_formation_467548.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_hist(ax, values, label, unit, scale="linear", bins=40, color="steelblue"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n_dropped_nonfinite = None  # tracked for the log-scale message below
    if scale == "log":
        n_before = len(values)
        values = values[values > 0]
        n_dropped_nonfinite = n_before - len(values)

    if len(values) == 0:
        ax.set_title(f"{label} (no valid data)")
        return

    if scale == "log":
        bin_edges = np.logspace(np.log10(values.min()), np.log10(values.max()), bins + 1)
        ax.set_xscale("log")
    else:
        bin_edges = np.linspace(values.min(), values.max(), bins + 1)

    ax.hist(values, bins=bin_edges, color=color, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"{label}" + (f" [{unit}]" if unit else ""))
    ax.set_ylabel("Number of subhalos")
    ax.set_title(label)

    if n_dropped_nonfinite:
        print(f"  note: dropped {n_dropped_nonfinite} non-positive value(s) from '{label}' for log-scale binning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the subhalo_formation_<ID>.csv file")
    parser.add_argument("--out", default=None, help="Output image path (default: <csv_basename>_histograms.png)")
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins per panel (default: 40)")
    parser.add_argument("--mass-scale", choices=["log", "linear"], default="log",
                         help="X-axis scale for dm_mass_msun (default: log, since masses span many orders of magnitude)")
    parser.add_argument("--radius-scale", choices=["log", "linear"], default="log",
                         help="X-axis scale for halfmass_rad_kpc (default: log)")
    parser.add_argument("--deltat-scale", choices=["log", "linear"], default="linear",
                         help="X-axis scale for delta_t_gyr (default: linear; some subhalos merge within "
                              "the same snapshot, giving delta_t=0, which log-scale can't show)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        sys.exit(f"Could not find {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    required = {"delta_t_gyr", "halfmass_rad_kpc", "dm_mass_msun"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing expected column(s): {missing}")

    print(f"Loaded {len(df)} subhalo branches from {args.csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    make_hist(axes[0], df["delta_t_gyr"], "delta_t", "Gyr",
              scale=args.deltat_scale, bins=args.bins, color="steelblue")
    make_hist(axes[1], df["halfmass_rad_kpc"], "Half-mass radius", "kpc",
              scale=args.radius_scale, bins=args.bins, color="seagreen")
    make_hist(axes[2], df["dm_mass_msun"], "DM mass", r"$M_\odot$",
              scale=args.mass_scale, bins=args.bins, color="indianred")

    fig.suptitle(f"Subhalo formation-time properties  (N = {len(df)})", y=1.02)
    fig.tight_layout()

    out_path = args.out or (os.path.splitext(os.path.basename(args.csv_path))[0] + "_histograms.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()