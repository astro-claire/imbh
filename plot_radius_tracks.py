"""
Plot separation-from-central-galaxy vs. time for every cluster's radius
track saved by imbh.py's --save-radius-tracks (see imbh.py's
save_radius_track / TRACK_MIN_IMBH_MASS_MSUN).

Each track file is a small per-cluster CSV named
radius_track_h<halo_idx>_c<cluster_idx>.csv, with columns
'time_since_formation_gyr' and 'separation_from_host_kpc' (see imbh.py's
trace_cluster_to_snapshot docstring for exactly what "separation from the
central galaxy" means -- the tree's own main-branch/root position, not
just the cluster's immediate local host). With potentially thousands of
these files, each up to ~1e5 rows, this script stays tractable by
(a) capping how many CLUSTERS get plotted (randomly subsampled if more
are available -- see --max-clusters) and (b) sparsely, evenly subsampling
each individual track's own rows (see --max-points-per-track), rather than
reading and plotting every single 1e5-yr-resolution point.

Pass --cluster-csv (the cluster_output_<ID>_snap<N>.csv table imbh.py's
save_cluster_output writes) to color each track by its cluster's fate --
either whether it actually reached the center ('status': inspiraled vs
outskirts, --color-by status, the default once --cluster-csv is given) or
how far it still is from its FINAL host's own virial radius at the traced
snapshot ('final_r_over_rvir', --color-by r_over_rvir). This is what lets
you tell apart clusters that are still just falling in for the first time
(large, still-growing separation from a not-yet-reached host) from ones
that have genuinely settled into a bound, oscillating orbit. Without
--cluster-csv, every track is drawn in a single uniform color as before.

Pass --tree-path (the same raw sublink_full_<ID>.hdf5 tree imbh.py uses) to
additionally overlay the FINAL host's own R_200(t) growth curve -- directly
showing whether/when a cluster's separation drops inside its eventual
host's own (growing) virial radius. R_200(t) is naturally a function of
COSMIC time, not "time since any one cluster's formation" -- and different
clusters form at different cosmic times -- so using --tree-path switches
the plot's x-axis to absolute cosmic age and shifts EACH track by its OWN
cluster's formation age (from --cluster-csv's initial_subhalo_formation_redshift,
which --tree-path therefore requires) rather than approximating everything
with one shared reference time. Clusters missing a matched formation
redshift can't be placed on this axis and are excluded, with a count printed.

Requires: numpy, pandas, matplotlib (plus h5py/astropy, only if --tree-path
is used -- imported lazily so plotting without it never needs them)
    pip install numpy pandas matplotlib

Usage:
    python plot_radius_tracks.py <track_dir> [--cluster-csv cluster_output_ID_snapN.csv]
                                  [--color-by status|r_over_rvir|none]
                                  [--tree-path sublink_full_ID.hdf5]
                                  [--snap-redshift-path tng_download/snapshot_redshifts.json]
                                  [--box-size 35000]
                                  [--out radius_tracks.png]
                                  [--max-clusters 500] [--max-points-per-track 200]
                                  [--yscale log|linear] [--xscale log|linear]
                                  [--alpha 0.15] [--seed 0]

Example:
    python plot_radius_tracks.py cluster_tracks --cluster-csv cluster_output_467548_snap99.csv \\
        --tree-path tng_download/sublink_full_467548.hdf5
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_sparse_track(path, max_points):
    """
    Load one track CSV, subsampled to at most `max_points` rows (evenly
    strided, not random -- preserves the track's own shape/trend rather
    than scattering gaps through it). Returns (time_gyr, radius_kpc) as
    plain numpy arrays, or (None, None) if the file is missing/empty/
    malformed (skipped with a warning rather than aborting the whole plot).
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  WARNING: could not read {path} ({e}) -- skipping.")
        return None, None
    required = {"time_since_formation_gyr", "separation_from_host_kpc"}
    if len(df) == 0 or not required.issubset(df.columns):
        print(f"  WARNING: {path} is empty or missing expected columns -- skipping.")
        return None, None
    stride = max(1, len(df) // max_points)
    sub = df.iloc[::stride]
    return sub["time_since_formation_gyr"].to_numpy(), sub["separation_from_host_kpc"].to_numpy()


def load_cluster_lookup(cluster_file):
    """Reads the main per-cluster output table (imbh.py's save_cluster_output)

    and builds {basename(radius_track_path): row} for every row that actually
    has one -- most rows won't (only clusters clearing TRACK_MIN_IMBH_MASS_MSUN
    get a track saved at all). Matched by FILENAME rather than the full stored
    path, so this still works if the tracks were moved to a different directory
    since the run that made them. Supports both .csv and .dat formats.
    """
    ext = os.path.splitext(cluster_file)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(cluster_file)
    elif ext == ".dat":
        # Reads space- or tab-delimited files; adjust sep if your .dat file uses a specific delimiter
        df = pd.read_pickle(cluster_file)
    else:
        sys.exit(
            f"Error: Unsupported file format '{ext}' for {cluster_file}. Expected .csv or .dat."
        )

    if "radius_track_path" not in df.columns:
        sys.exit(
            f"{cluster_file} has no 'radius_track_path' column -- was it produced by a run "
            f"with --save-radius-tracks enabled?"
        )

    has_track = df["radius_track_path"].notna() & (
        df["radius_track_path"] != ""
    )
    lookup = {
        os.path.basename(p): row
        for p, (_, row) in zip(
            df.loc[has_track, "radius_track_path"],
            df.loc[has_track].iterrows(),
        )
    }
    print(
        f"Loaded {len(df)} cluster rows from {cluster_file} ({len(lookup)} with a saved radius track)"
    )
    return lookup


def load_host_rvir_history(tree_path, snap_redshift_path, box_size):
    """
    Builds a MergerTreeNavigator over tree_path and returns
    (navigator, ages_gyr, r200_kpc) -- the final host's OWN growth history
    (see MergerTreeNavigator.main_branch_growth_history). Imports
    tree_navigator (and its own h5py/astropy dependencies) lazily, so
    plotting without --tree-path never requires them.
    """
    from tree_navigator import MergerTreeNavigator
    navigator = MergerTreeNavigator(tree_path, snap_redshift_path, box_size_ckpc_h=box_size)
    ages_gyr, r200_kpc = navigator.main_branch_growth_history()
    return navigator, ages_gyr, r200_kpc


STATUS_COLORS = {"inspiraled": "indianred", "outskirts": "steelblue"}
DEFAULT_COLOR = "gray"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("track_dir", help="Directory of radius_track_h<halo>_c<cluster>.csv files "
                                           "(see imbh.py's --save-radius-tracks / --track-dir)")
    parser.add_argument("--cluster-csv", default=None,
                         help="Path to the cluster_output_<ID>_snap<N>.csv table (imbh.py's "
                              "save_cluster_output) to join against for --color-by. Optional -- "
                              "without it, --color-by is forced to 'none'.")
    parser.add_argument("--color-by", choices=["status", "r_over_rvir", "none"], default=None,
                         help="How to color each track (default: 'status' if --cluster-csv is given, "
                              "else 'none'). 'status': inspiraled (reached the center) vs outskirts "
                              "(survived to the traced snapshot without merging), discrete legend. "
                              "'r_over_rvir': continuous colormap on final_r_over_rvir (separation "
                              "from the FINAL host at the traced snapshot, in units of that host's own "
                              "virial radius) -- 0 for inspiraled clusters, otherwise how far outside "
                              "the eventual host's virial radius the cluster still is.")
    parser.add_argument("--tree-path", default=None,
                         help="Path to the raw sublink_full_<ID>.hdf5 tree (same file imbh.py uses). "
                              "If given, overlays the FINAL host's own R_200(t) growth curve on the "
                              "plot -- a direct reference for when/whether a cluster's separation "
                              "drops inside its eventual host's own virial radius.")
    parser.add_argument("--snap-redshift-path", default="tng_download/snapshot_redshifts.json",
                         help="Path to snapshot_redshifts.json (only used with --tree-path; default "
                              "matches imbh.py's own default)")
    parser.add_argument("--box-size", type=float, default=35000.0,
                         help="Simulation box size, comoving ckpc/h (only used with --tree-path; "
                              "default: 35000, TNG50's box, matching imbh.py's own default)")
    parser.add_argument("--out", default=None,
                         help="Output image path (default: <track_dir_basename>_radius_tracks.png)")
    parser.add_argument("--max-clusters", type=int, default=500,
                         help="Max number of cluster tracks to plot (default: 500). If more files are "
                              "found, a random subset of this size is drawn (see --seed) -- plotting "
                              "every cluster in a run with many thousands of tracks would both be slow "
                              "and produce an unreadably dense/over-plotted figure.")
    parser.add_argument("--max-points-per-track", type=int, default=200,
                         help="Max number of (evenly-strided) points read per track (default: 200). "
                              "Tracks are saved at ~1e5 yr resolution and can have up to ~1e5 rows each "
                              "-- this keeps the total number of plotted points bounded regardless of "
                              "how long any individual track is.")
    parser.add_argument("--yscale", choices=["log", "linear"], default="log",
                         help="Y-axis (separation) scale (default: log, since orbits typically span a "
                              "wide range of radii)")
    parser.add_argument("--xscale", choices=["log", "linear"], default="linear",
                         help="X-axis (time since cluster formation) scale (default: linear)")
    parser.add_argument("--alpha", type=float, default=0.15,
                         help="Per-line transparency (default: 0.15) -- with many overlapping tracks, "
                              "low alpha is what actually makes the DENSITY of trajectories visible, "
                              "rather than a solid mass of overlapping opaque lines.")
    parser.add_argument("--r-over-rvir-cap", type=float, default=None,
                         help="Clip final_r_over_rvir at this value for the --color-by r_over_rvir "
                              "colormap (default: the 98th percentile of the plotted clusters' own "
                              "values) -- keeps a handful of extreme outliers from washing out the "
                              "color scale for everyone else.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed used for --max-clusters subsampling (default: 0)")
    args = parser.parse_args()

    if not os.path.isdir(args.track_dir):
        sys.exit(f"Not a directory: {args.track_dir}")

    if args.tree_path is not None and args.cluster_csv is None:
        sys.exit("--tree-path requires --cluster-csv too (needed to look up each cluster's own "
                  "formation redshift, so every track can be placed correctly on the shared "
                  "absolute-cosmic-time axis --tree-path switches the plot to).")

    color_by = args.color_by
    lookup = None
    if args.cluster_csv is not None:
        lookup = load_cluster_lookup(args.cluster_csv)
        if color_by is None:
            color_by = "status"
    else:
        if color_by not in (None, "none"):
            sys.exit(f"--color-by {color_by} requires --cluster-csv")
        color_by = "none"

    paths = sorted(glob.glob(os.path.join(args.track_dir, "radius_track_h*_c*.csv")))
    if not paths:
        sys.exit(f"No radius_track_h*_c*.csv files found in {args.track_dir}")
    n_total = len(paths)
    print(f"Found {n_total} radius track file(s) in {args.track_dir}")

    if n_total > args.max_clusters:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_total, size=args.max_clusters, replace=False)
        paths = [paths[i] for i in idx]
        print(f"Randomly sampled {args.max_clusters} of them (--max-clusters; seed={args.seed})")

    # For r_over_rvir coloring, need the cap and colormap set up before the
    # plotting loop -- collect matched values from just the clusters we're
    # actually about to plot, not the whole table, so the cap adapts to
    # whatever subsample --max-clusters landed on.
    cmap = norm = None
    if color_by == "r_over_rvir":
        matched_vals = []
        for path in paths:
            row = lookup.get(os.path.basename(path))
            if row is not None and np.isfinite(row.get("final_r_over_rvir", np.nan)):
                matched_vals.append(row["final_r_over_rvir"])
        if not matched_vals:
            sys.exit("No plotted tracks matched a row with a finite final_r_over_rvir in --cluster-csv.")
        cap = args.r_over_rvir_cap if args.r_over_rvir_cap is not None else np.percentile(matched_vals, 98)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0.0, vmax=cap)

    # --tree-path switches the x-axis to absolute cosmic time (see the
    # module docstring) -- each track needs its OWN cluster's formation age
    # (from --cluster-csv, via the SAME cosmology the tree/tracks themselves
    # were built with -- tree_navigator.TNG_COSMO, not imbh.py's separate
    # star-cluster-model cosmology) added to its 'time since formation'
    # values before plotting. A track with no matched row, or a row missing
    # a usable formation redshift, can't be placed on this axis and is
    # skipped entirely (counted and reported) rather than plotted misaligned.
    navigator = ages_gyr = r200_kpc = None
    cosmic_time_mode = args.tree_path is not None
    if cosmic_time_mode:
        import astropy.units as u
        navigator, ages_gyr, r200_kpc = load_host_rvir_history(
            args.tree_path, args.snap_redshift_path, args.box_size,
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    n_plotted = 0
    n_unmatched = 0
    n_no_formation_z = 0
    status_counts = {}
    for path in paths:
        t, r = load_sparse_track(path, args.max_points_per_track)
        if t is None:
            continue

        row = lookup.get(os.path.basename(path)) if lookup is not None else None

        if cosmic_time_mode:
            if row is None:
                n_unmatched += 1
                continue
            z_form = row.get("initial_subhalo_formation_redshift", np.nan)
            if not np.isfinite(z_form):
                n_no_formation_z += 1
                continue
            t = t + navigator.cosmo.age(z_form).to(u.Gyr).value

        if color_by == "none":
            color = "steelblue"
        elif row is None:
            color = DEFAULT_COLOR
            n_unmatched += 1
        elif color_by == "status":
            color = STATUS_COLORS.get(row["status"], DEFAULT_COLOR)
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        else:  # r_over_rvir
            val = row.get("final_r_over_rvir", np.nan)
            color = cmap(norm(np.clip(val, 0, norm.vmax))) if np.isfinite(val) else DEFAULT_COLOR

        ax.plot(t, r, color=color, alpha=args.alpha, linewidth=0.8)
        n_plotted += 1

    if n_plotted == 0:
        sys.exit("No usable track files -- nothing to plot.")

    legend_handles = []
    if color_by == "status":
        legend_handles += [Line2D([0], [0], color=c, lw=2, label=label) for label, c in STATUS_COLORS.items()]
        if n_unmatched:
            legend_handles.append(Line2D([0], [0], color=DEFAULT_COLOR, lw=2, label="no match in --cluster-csv"))
        print("Colored by status: " + ", ".join(f"{k}={v}" for k, v in status_counts.items())
              + (f", unmatched={n_unmatched}" if n_unmatched else ""))
    elif color_by == "r_over_rvir":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f"Final separation / final host $R_{{vir}}$ (capped at {norm.vmax:.1f})")
        print(f"Colored by final_r_over_rvir (cap={norm.vmax:.2f}); unmatched={n_unmatched}")

    if cosmic_time_mode:
        if n_unmatched or n_no_formation_z:
            print(f"Excluded {n_unmatched} track(s) with no matching --cluster-csv row and "
                  f"{n_no_formation_z} with no usable formation redshift (can't place them on "
                  f"the absolute cosmic-time axis).")
        r200_line, = ax.plot(ages_gyr, r200_kpc, color="black", linestyle="--", linewidth=1.8,
                              label="Final host $R_{200}(t)$", zorder=10)
        legend_handles.append(r200_line)

    ax.set_xlabel("Cosmic age of the universe [Gyr]" if cosmic_time_mode else "Time since cluster formation [Gyr]")
    ax.set_ylabel("Separation from central galaxy [kpc]")
    if args.yscale == "log":
        ax.set_yscale("log")
    if args.xscale == "log":
        ax.set_xscale("log")
    ax.set_title(f"Cluster separation vs. time  (N = {n_plotted} of {n_total} tracks)")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9)

    fig.tight_layout()

    out_path = args.out or (os.path.basename(os.path.normpath(args.track_dir)) + "_radius_tracks.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path} ({n_plotted} clusters plotted)")


if __name__ == "__main__":
    main()