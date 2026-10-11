"""
Model comparison for referee point 4: the conservative (empirically calibrated)
model is turned into the maximal (simulation-calibrated) one ONE INGREDIENT AT A
TIME, so each step shows how much that ingredient changes the TDE rate.

Default ladder (each step adds to the previous one; flags as given to
submit_halos.pl / submit_analysis.pl, plus --loss-cone/--coulomb-log below):
  0 Conservative           --mode observational --n-relation bf20_seed --radius-relation marks_kroupa12
  1 + same z/mass cuts     ... --resolution-cuts on
  2 + sim. positions/vel.  ... --phase-space simulation
  3 + sim. sizes           ... --radius-relation simulation --jitter-dex 0.1
  4 + sim. masses          ... --mass-function simulation
  5 + sim. numbers         ... --n-relation simulation   (all ingredients from the simulation)
  6 Maximal                --jitter-dex 0.1               (should agree with step 5)
plus a REVERSE test, drawn separately: the maximal model with the empirical
(Marks & Kroupa) sizes.

Outputs (prefix --output-prefix, default model_ladder_<selection>_alpha<a>_<s>Myr):
  <prefix>_rates.png    smoothed TDE rate per galaxy vs redshift, one panel per
                        z=0 stellar-mass bin, one curve per ladder step
                        (mean over the bin's halos)
  <prefix>_summary.png  per galaxy, vs ladder step: IMBH-hosting clusters
                        (M_BH >= 500 Msun), time-integrated TDEs, and peak
                        smoothed rate (points = bin mean, bars = halo range)
  <prefix>_summary.csv  the same numbers plus cluster statistics (median mass,
                        r_h, IMBH mass), TDEs before/after --z-split, discarded
                        orbits (trace_failed) and the step-to-step TDE ratio

--selection total (default) uses all TDEs; within / beyond split them by the
cluster's separation from the main progenitor at the time of each event
(see distance_split.py; needs the tde_radial npz files and R200c histories).
Runs whose analysis folder doesn't exist yet are skipped with a warning, so
the script can be re-run as jobs finish.

Examples:
  python plot_model_ladder.py --data-dir /u/scratch/c/clairewi/imbh-output --loss-cone full
  python plot_model_ladder.py --data-dir ... --loss-cone full --selection within
  python plot_model_ladder.py --data-dir ... --run "Conservative" "--mode observational ..." --run ...
"""
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rate_runs import (Run, RateLoader, DEFAULT_MASS_BINS, SELECTIONS, selection_label, smooth,
                       age_to_redshift, on_grid, plot_step)

OBS = "--mode observational"
CONS = f"{OBS} --n-relation bf20_seed --radius-relation marks_kroupa12"
DEFAULT_LADDER = [
    ("Conservative", CONS),
    ("+ same z/mass cuts", f"{CONS} --resolution-cuts on"),
    ("+ sim. positions/velocities", f"{CONS} --resolution-cuts on --phase-space simulation"),
    ("+ sim. sizes", f"{OBS} --n-relation bf20_seed --radius-relation simulation --phase-space simulation "
                     f"--resolution-cuts on --jitter-dex 0.1"),
    ("+ sim. masses", f"{OBS} --n-relation bf20_seed --mass-function simulation --radius-relation simulation "
                      f"--phase-space simulation --resolution-cuts on --jitter-dex 0.1"),
    ("+ sim. numbers (all sim.)", f"{OBS} --n-relation simulation --mass-function simulation "
                                  f"--radius-relation simulation --phase-space simulation --resolution-cuts on "
                                  f"--jitter-dex 0.1"),
    ("Maximal", "--jitter-dex 0.1"),
]
DEFAULT_REVERSE = ("Maximal, empirical sizes",
                   f"{OBS} --n-relation simulation --mass-function simulation --radius-relation marks_kroupa12 "
                   f"--phase-space simulation --resolution-cuts on --jitter-dex 0.1")

FONT_SIZE = 22
plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 3,
                     "xtick.labelsize": FONT_SIZE - 2, "ytick.labelsize": FONT_SIZE - 2})
Z_MAX = 20


def step_colors(n):
    """Ordered steps conservative -> maximal: one hue, light -> dark (sequential)."""
    cmap = plt.get_cmap("Blues")
    return [cmap(x) for x in np.linspace(0.35, 1.0, n)]


def analyse_run(run, loader, mass_bins, smooth_myr, kernel, z_split):
    """Per mass bin: per-halo integrals/peaks, the bin-mean smoothed curve, and cluster stats."""
    out = {}
    t_ref = None
    for tag, _, _, halo_ids in mass_bins:
        curves, per_halo = [], []
        for hid in halo_ids:
            loaded = loader.load(run.analysis_dir, hid)
            if loaded is None:
                continue
            t, rate = loaded
            if t_ref is None:
                t_ref, dt = t, float(np.median(np.diff(t)))
                z_ref = age_to_redshift(t_ref)
            rate = on_grid(t_ref, t, rate)
            s = smooth(rate[None, :], dt, smooth_myr, kernel)[0]
            k = int(np.argmax(s))
            row = {"halo": hid, "n_tde": rate.sum() * dt,
                   "n_tde_early": rate[z_ref > z_split].sum() * dt,
                   "n_tde_late": rate[z_ref <= z_split].sum() * dt,
                   "peak_rate": s[k], "z_peak": z_ref[k]}
            stats = run.cluster_stats(hid)
            if stats:
                row.update(stats)
            per_halo.append(row)
            curves.append(s)
        if per_halo:
            out[tag] = {"per_halo": per_halo, "mean_curve": np.mean(curves, axis=0)}
    return out, t_ref


def bin_summary(per_halo, key):
    vals = np.array([h.get(key, np.nan) for h in per_halo], dtype=float)
    if np.all(np.isnan(vals)):
        return np.nan, np.nan, np.nan
    return np.nanmean(vals), np.nanmin(vals), np.nanmax(vals)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Base output directory ($OUTDIR).")
    p.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "FLAGS"),
                   help="A ladder step (repeat, in order). Replaces the default ladder.")
    p.add_argument("--reverse", nargs=2, metavar=("LABEL", "FLAGS"), default=None,
                   help=f"Reverse-test run drawn separately (default: '{DEFAULT_REVERSE[0]}').")
    p.add_argument("--no-reverse", action="store_true", help="Don't show the reverse-test run.")
    p.add_argument("--loss-cone", default="full", help="Appended to every run's flags (default: full).")
    p.add_argument("--coulomb-log", default="bh", help="Appended to every run's flags (default: bh).")
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--snap", type=int, default=99, help="Snapshot in the cluster_output names (default 99).")
    p.add_argument("--selection", choices=SELECTIONS, default="total")
    p.add_argument("--r200-factor", type=float, default=3.0)
    p.add_argument("--tree-dir", default="tng_download")
    p.add_argument("--snap-redshift-path", default="tng_download/snapshot_redshifts.json")
    p.add_argument("--r200-cache-dir", default=None, help="Where r200_history_<ID>.csv live (default: --data-dir).")
    p.add_argument("--smooth-myr", type=float, default=50.0)
    p.add_argument("--kernel", choices=["boxcar", "gaussian"], default="boxcar")
    p.add_argument("--z-split", type=float, default=7.0,
                   help="Redshift dividing 'early' and 'late' TDE counts in the table (default 7).")
    p.add_argument("--linear", action="store_true", help="Linear y axes (default: log).")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    extra = f" --loss-cone {args.loss_cone} --coulomb-log {args.coulomb_log}"
    ladder = [Run(lbl, fl + extra, args.data_dir, args.snap) for lbl, fl in (args.run or DEFAULT_LADDER)]
    reverse = None
    if not args.no_reverse:
        lbl, fl = args.reverse or DEFAULT_REVERSE
        reverse = Run(lbl, fl + extra, args.data_dir, args.snap)
    loader = RateLoader(args.alpha, args.selection, args.r200_factor, args.tree_dir,
                        args.snap_redshift_path, args.r200_cache_dir or args.data_dir)
    mass_bins = DEFAULT_MASS_BINS

    results, t_ref = {}, None
    for run in ladder + ([reverse] if reverse else []):
        print(f"{run.label:30s} -> {run.analysis_dir}")
        if not run.exists:
            print("    WARNING: analysis folder not found -- skipping (not finished yet?)")
            continue
        res, t = analyse_run(run, loader, mass_bins, args.smooth_myr, args.kernel, args.z_split)
        if not res:
            print("    WARNING: no rate files for any halo -- skipping")
            continue
        for tag, _, _, ids in mass_bins:
            n = len(res.get(tag, {}).get("per_halo", []))
            if n < len(ids):
                print(f"    note: {tag}: {n} of {len(ids)} halos found")
        results[run.label] = res
        t_ref = t if t_ref is None else t_ref
    if not results:
        raise SystemExit("No runs found -- check --data-dir and the run flags.")

    prefix = args.output_prefix or (f"model_ladder_{args.selection}_alpha{args.alpha}_{args.smooth_myr:g}Myr"
                                    f"_lc{args.loss_cone}")
    sel_text = selection_label(args.selection, args.r200_factor)
    present = [r for r in ladder if r.label in results]
    colors = dict(zip([r.label for r in ladder], step_colors(len(ladder))))

    # ---------------- figure 1: rates vs redshift ----------------
    dt = float(np.median(np.diff(t_ref)))
    z = age_to_redshift(t_ref)
    sl = slice(None, None, plot_step(dt, args.smooth_myr))
    fig, axes = plt.subplots(1, len(mass_bins), figsize=(6.5 * len(mass_bins), 6), sharey=True)
    ymax = []
    for ax, (tag, blabel, _, _) in zip(np.atleast_1d(axes), mass_bins):
        for run in present + ([reverse] if reverse and reverse.label in results else []):
            res = results[run.label].get(tag)
            if res is None:
                continue
            is_rev = run is reverse
            curve = res["mean_curve"]
            ax.plot(z[sl], curve[sl], color="0.15" if is_rev else colors[run.label],
                    lw=2, ls="--" if is_rev else "-")
            ymax.append(curve[z <= Z_MAX].max())
        if not any(results[r.label].get(tag) for r in present):
            ax.text(0.5, 0.5, "no runs found for this bin", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
        ax.set_title(blabel)
        ax.set_xlim(Z_MAX, 0)
        ax.set_xlabel("Redshift")
    axes0 = np.atleast_1d(axes)[0]
    axes0.set_ylabel(r"TDE rate per galaxy (yr$^{-1}$)")
    if not args.linear and ymax:
        axes0.set_yscale("log")
        top = max(ymax)
        axes0.set_ylim(top * 1e-4, top * 2)
    handles = [Line2D([], [], color=colors[r.label], lw=2.5, label=f"{i}  {r.label}")
               for i, r in enumerate(ladder) if r.label in results]
    if reverse and reverse.label in results:
        handles.append(Line2D([], [], color="0.15", lw=2.5, ls="--", label=reverse.label))
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=min(3, len(handles)), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(f"{prefix}_rates.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {prefix}_rates.png")

    # ---------------- table ----------------
    rows = []
    for run in ladder + ([reverse] if reverse else []):
        if run.label not in results:
            continue
        for tag, blabel, _, _ in mass_bins:
            res = results[run.label].get(tag)
            if res is None:
                continue
            ph = res["per_halo"]
            row = {"run": run.label, "mass_bin": tag, "n_halos": len(ph), "flags": run.flags}
            for key in ("n_clusters", "n_imbh_ge500", "n_tde", "n_tde_early", "n_tde_late", "peak_rate",
                        "n_trace_failed", "n_imbh_model_failed", "n_escaped_z0"):
                row[f"{key}_mean"], row[f"{key}_min"], row[f"{key}_max"] = bin_summary(ph, key)
            for key in ("median_cluster_mass", "median_r_h_pc", "median_m_imbh_ge500", "z_peak"):
                row[f"{key}_mean"] = bin_summary(ph, key)[0]
            rows.append(row)
    # step-to-step ratio of time-integrated TDEs (ladder order only)
    for tag, _, _, _ in mass_bins:
        prev = None
        for run in ladder:
            r = next((x for x in rows if x["run"] == run.label and x["mass_bin"] == tag), None)
            if r is None:
                continue
            r["n_tde_ratio_to_previous_step"] = (r["n_tde_mean"] / prev if prev else np.nan)
            prev = r["n_tde_mean"]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{prefix}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {prefix}_summary.csv")

    print(f"\n{'run':30s} {'bin':10s} {'clusters':>9s} {'IMBH>=500':>9s} {'N_TDE':>9s} {'xprev':>6s} "
          f"{'peak/yr':>9s} {'z_pk':>5s} {'r_h pc':>7s} {'failed':>7s}   (per galaxy, bin mean)")
    for r in rows:
        ratio = r.get("n_tde_ratio_to_previous_step", np.nan)
        print(f"{r['run'][:30]:30s} {r['mass_bin']:10s} {r['n_clusters_mean']:9.1f} {r['n_imbh_ge500_mean']:9.1f} "
              f"{r['n_tde_mean']:9.3g} {ratio:6.2f} {r['peak_rate_mean']:9.2e} {r['z_peak_mean']:5.1f} "
              f"{r['median_r_h_pc_mean']:7.3f} {r['n_trace_failed_mean']:7.1f}")

    # ---------------- figure 2: summary vs ladder step ----------------
    panels = [("n_imbh_ge500", r"IMBH-hosting clusters per galaxy" "\n" r"($M_{\rm BH}\geq500\,M_\odot$)"),
              ("n_tde", "TDEs per galaxy\n(time-integrated)"),
              ("peak_rate", "Peak smoothed TDE rate\nper galaxy (yr$^{-1}$)")]
    labels = [r.label for r in ladder] + ([reverse.label] if reverse else [])
    xpos = {lbl: i for i, lbl in enumerate(r.label for r in ladder)}
    if reverse:
        xpos[reverse.label] = len(ladder) + 0.5
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7.5))
    offsets = np.linspace(-0.15, 0.15, len(mass_bins))
    for ax, (key, title) in zip(axes, panels):
        for (tag, blabel, color, _), dx in zip(mass_bins, offsets):
            pts = [(xpos[r["run"]] + dx, r[f"{key}_mean"], r[f"{key}_min"], r[f"{key}_max"])
                   for r in rows if r["mass_bin"] == tag and np.isfinite(r.get(f"{key}_mean", np.nan))]
            if not pts:
                continue
            x, y, lo, hi = map(np.array, zip(*pts))
            main = x < len(ladder)
            ax.errorbar(x[main], y[main], yerr=[y[main] - lo[main], hi[main] - y[main]], color=color,
                        marker="o", ms=8, lw=2, capsize=3, label=blabel)
            if (~main).any():
                ax.errorbar(x[~main], y[~main], yerr=[y[~main] - lo[~main], hi[~main] - y[~main]],
                            color=color, marker="D", ms=8, mfc="white", lw=0, elinewidth=2, capsize=3)
        if not args.linear:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.set_xticks([xpos[l] for l in labels if l in xpos])
        ax.set_xticklabels([l for l in labels if l in xpos], rotation=55, ha="right", fontsize=FONT_SIZE - 4)
        if reverse:
            ax.axvline(len(ladder) - 0.25, color="0.7", lw=1)
        ax.grid(axis="y", color="0.9", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    # one legend centred above all three panels (clear of the panel titles)
    leg_handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(leg_handles, leg_labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(leg_handles),
               title=r"Stellar mass at $z=0$", frameon=False, title_fontsize=FONT_SIZE - 3)
    fig.savefig(f"{prefix}_summary.png", dpi=200, bbox_inches="tight")
    print(f"Saved {prefix}_summary.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
