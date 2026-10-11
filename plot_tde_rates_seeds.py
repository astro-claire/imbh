"""
Smoothed TDE rate per galaxy vs redshift for the maximal model, showing how
much it varies between independent random realizations (referee point 10).

Realizations: the run given by --base-flags (default "--jitter-dex 0.1", the
original unseeded run) plus its submit_halos.pl --seeds copies, labelled
seed1 .. seedN (--seeds "1 2 3 4 5"). --no-base drops the original run.

Main figure, one row per --rows selection (default: within / beyond 3 R200c
of the main progenitor at the time of each event, as in the quad figure;
'total' = all TDEs), one curve per z=0 stellar-mass bin:
  line         mean over realizations of the bin-mean rate
  dark band    spread of the bin-mean rate across realizations (--band)
  light band   (--show-halo-range) halo-to-halo range of the realization-averaged rate
  thin lines   (--show-seeds) each realization's bin-mean rate
--per-halo adds one figure per selection with a panel per halo (rows = mass
bins), showing that halo's realization spread -- the place to check whether
a feature (like the z~10 bump in 697044) survives a new draw.

Also writes <prefix>_summary.csv and prints, per selection and mass bin, the
time-integrated TDEs and peak rate in each realization with their scatter,
plus per halo the realization scatter of the TDE count and the number of
orbits discarded at the integration limits (status trace_failed).

Examples:
  python plot_tde_rates_seeds.py --data-dir /u/scratch/c/clairewi/imbh-output --loss-cone full
  python plot_tde_rates_seeds.py --data-dir ... --loss-cone full --rows total --per-halo --show-seeds
"""
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rate_runs import (Run, RateLoader, DEFAULT_MASS_BINS, SELECTIONS, selection_label, smooth,
                       age_to_redshift, on_grid, band, plot_step, parse_run_flags)

FONT_SIZE = 20
plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE - 3,
                     "legend.title_fontsize": FONT_SIZE - 3, "xtick.labelsize": FONT_SIZE - 3,
                     "ytick.labelsize": FONT_SIZE - 3})
Z_MAX = 20
SHADE_Z = (7, 0)   # no new clusters below z = 7 in the maximal model


def realizations(args):
    base = " ".join(args.base_flags.split())
    extra = f" --loss-cone {args.loss_cone} --coulomb-log {args.coulomb_log}"
    base_label = parse_run_flags(base).label
    runs = [] if args.no_base else [Run("original", base + extra, args.data_dir, args.snap)]
    for s in args.seeds.replace(",", " ").split():
        label = "_".join(x for x in (base_label, f"seed{s}") if x)
        flags = base.replace(f"--label {base_label}", "") if base_label else base
        runs.append(Run(f"seed{s}", f"{flags} --label {label}" + extra, args.data_dir, args.snap))
    return runs


def load_cube(runs, loader, halo_ids, smooth_myr, kernel, t_ref=None):
    """
    For one mass bin: raw and smoothed rates with shape (n_realizations, n_halos, n_t),
    using only halos present in EVERY realization (so the spread isn't from missing halos).
    """
    data = {}
    for run in runs:
        for hid in halo_ids:
            loaded = loader.load(run.analysis_dir, hid)
            if loaded is not None:
                data[(run.label, hid)] = loaded
    common = [h for h in halo_ids if all((r.label, h) in data for r in runs)]
    missing = sorted({h for h in halo_ids if h not in common})
    if not common:
        return None
    if t_ref is None:
        t_ref = data[(runs[0].label, common[0])][0]
    dt = float(np.median(np.diff(t_ref)))
    raw = np.array([[on_grid(t_ref, *data[(r.label, h)]) for h in common] for r in runs])
    sm = smooth(raw.reshape(-1, raw.shape[-1]), dt, smooth_myr, kernel).reshape(raw.shape)
    return {"t": t_ref, "dt": dt, "halos": common, "missing": missing, "raw": raw, "smoothed": sm}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Base output directory ($OUTDIR).")
    p.add_argument("--base-flags", default="--jitter-dex 0.1",
                   help='Run-selection flags of the model (default: "--jitter-dex 0.1").')
    p.add_argument("--seeds", default="1 2 3 4 5", help='Seed numbers, as given to --seeds (default "1 2 3 4 5").')
    p.add_argument("--no-base", action="store_true", help="Leave out the original (unlabelled) run.")
    p.add_argument("--loss-cone", default="full")
    p.add_argument("--coulomb-log", default="bh")
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--rows", nargs="+", choices=SELECTIONS, default=["within", "beyond"])
    p.add_argument("--r200-factor", type=float, default=3.0)
    p.add_argument("--tree-dir", default="tng_download")
    p.add_argument("--snap-redshift-path", default="tng_download/snapshot_redshifts.json")
    p.add_argument("--r200-cache-dir", default=None, help="Where r200_history_<ID>.csv live (default: --data-dir).")
    p.add_argument("--smooth-myr", type=float, default=50.0)
    p.add_argument("--kernel", choices=["boxcar", "gaussian"], default="boxcar")
    p.add_argument("--band", choices=["minmax", "percentile", "std"], default="minmax",
                   help="Spread across realizations (default: full range).")
    p.add_argument("--show-halo-range", action="store_true", help="Also shade the halo-to-halo range.")
    p.add_argument("--show-seeds", action="store_true", help="Draw each realization's curve.")
    p.add_argument("--per-halo", action="store_true", help="Also make a per-halo figure per selection.")
    p.add_argument("--linear", action="store_true", help="Linear y axes (default: log).")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    runs = realizations(args)
    found = []
    for r in runs:
        ok = r.exists
        print(f"{r.label:10s} -> {r.analysis_dir}{'' if ok else '   (MISSING -- skipped)'}")
        if ok:
            found.append(r)
    if len(found) < 2:
        raise SystemExit("Need at least two realizations with analysis output.")
    runs = found
    mass_bins = DEFAULT_MASS_BINS
    base_run = Run("original", " ".join(args.base_flags.split())
                   + f" --loss-cone {args.loss_cone} --coulomb-log {args.coulomb_log}", args.data_dir, args.snap)
    prefix = args.output_prefix or (f"tde_rate_seeds_alpha{args.alpha}_{args.smooth_myr:g}Myr"
                                    f"{base_run.analysis_suffix or '_default'}")
    cubes = {}
    t_ref = None
    for sel in args.rows:
        loader = RateLoader(args.alpha, sel, args.r200_factor, args.tree_dir, args.snap_redshift_path,
                            args.r200_cache_dir or args.data_dir)
        for tag, _, _, ids in mass_bins:
            cube = load_cube(runs, loader, ids, args.smooth_myr, args.kernel, t_ref)
            if cube is None:
                print(f"WARNING: {sel}/{tag}: no halo present in every realization -- skipped")
                continue
            if cube["missing"]:
                print(f"note: {sel}/{tag}: halos {cube['missing']} missing from some realization -- left out")
            t_ref = cube["t"]
            cubes[(sel, tag)] = cube
    if not cubes:
        raise SystemExit("No rate files found.")
    dt = float(np.median(np.diff(t_ref)))
    z = age_to_redshift(t_ref)
    sl = slice(None, None, plot_step(dt, args.smooth_myr))

    # ---------------- main figure ----------------
    nrow = len(args.rows)
    fig, axes = plt.subplots(nrow, 1, figsize=(10, 4.6 * nrow + 1), sharex=True, squeeze=False)
    for i, sel in enumerate(args.rows):
        ax = axes[i, 0]
        tops = []
        for tag, blabel, color, _ in mass_bins:
            c = cubes.get((sel, tag))
            if c is None:
                continue
            bin_mean = c["smoothed"].mean(axis=1)          # (n_real, n_t)
            centre = bin_mean.mean(axis=0)
            lo, hi = band(bin_mean, args.band, axis=0)
            if args.show_halo_range:
                per_halo = c["smoothed"].mean(axis=0)       # realization-averaged, (n_halos, n_t)
                ax.fill_between(z[sl], per_halo.min(0)[sl], per_halo.max(0)[sl], color=color, alpha=0.12, lw=0)
            ax.fill_between(z[sl], lo[sl], hi[sl], color=color, alpha=0.35, lw=0)
            if args.show_seeds:
                for curve in bin_mean:
                    ax.plot(z[sl], curve[sl], color=color, lw=0.7, alpha=0.6)
            ax.plot(z[sl], centre[sl], color=color, lw=2.2)
            tops.append(hi[z <= Z_MAX].max())
        ax.axvspan(*SHADE_Z, color="grey", alpha=0.3, lw=0)
        ax.text(1.01, 0.5, selection_label(sel, args.r200_factor), transform=ax.transAxes, rotation=270,
                ha="left", va="center")
        if not args.linear and tops:
            ax.set_yscale("log")
            ax.set_ylim(max(tops) * 1e-4, max(tops) * 2)
        ax.set_xlim(Z_MAX, 0)
    axes[-1, 0].set_xlabel("Redshift")
    fig.supylabel(r"TDE rate per galaxy (yr$^{-1}$)", fontsize=FONT_SIZE)
    band_txt = {"minmax": "full range", "percentile": "16–84%", "std": r"$\pm1\sigma$"}[args.band]
    handles = [Line2D([], [], color=c, lw=2.5, label=l) for tag, l, c, _ in mass_bins
               if any((sel, tag) in cubes for sel in args.rows)]
    handles += [Patch(color="0.4", alpha=0.35, label=f"{len(runs)} realizations ({band_txt})")]
    if args.show_halo_range:
        handles += [Patch(color="0.4", alpha=0.12, label="halo-to-halo range")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               title=r"Stellar mass at $z=0$")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(f"{prefix}.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved {prefix}.png")

    # ---------------- per-halo figures ----------------
    if args.per_halo:
        for sel in args.rows:
            bins_here = [b for b in mass_bins if (sel, b[0]) in cubes]
            if not bins_here:
                continue
            ncol = max(len(cubes[(sel, b[0])]["halos"]) for b in bins_here)
            nrow = len(bins_here)
            # Panels touch: shared x and y over the whole grid, halo ID written inside each panel.
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow + 0.8),
                                     sharex=True, sharey=True, squeeze=False,
                                     gridspec_kw={"wspace": 0, "hspace": 0})
            gmax = 0.0
            for i, (tag, blabel, color, ids) in enumerate(bins_here):
                c = cubes.get((sel, tag))
                for j in range(ncol):
                    ax = axes[i, j]
                    if c is None or j >= len(c["halos"]):
                        ax.axis("off")
                        continue
                    s = c["smoothed"][:, j]                   # (n_real, n_t)
                    lo, hi = band(s, args.band, axis=0)
                    ax.fill_between(z[sl], lo[sl], hi[sl], color=color, alpha=0.35, lw=0)
                    for curve in s:
                        ax.plot(z[sl], curve[sl], color=color, lw=0.6, alpha=0.6)
                    ax.plot(z[sl], s.mean(0)[sl], color=color, lw=1.8)
                    ax.axvspan(*SHADE_Z, color="grey", alpha=0.3, lw=0)
                    gmax = max(gmax, float(hi[z <= Z_MAX].max()))
                    ax.text(0.95, 0.93, f"{c['halos'][j]}", transform=ax.transAxes, ha="right", va="top",
                            fontsize=FONT_SIZE - 6)
                    ax.tick_params(labelsize=FONT_SIZE - 7, direction="in", top=True, right=True, which="both")
                axes[i, -1].yaxis.set_label_position("right")
                axes[i, -1].set_ylabel(blabel, fontsize=FONT_SIZE - 6, rotation=270, labelpad=22)
            axes[0, 0].set_xlim(Z_MAX, 0)
            axes[0, 0].set_xticks([15, 10, 5])             # no labels on the shared panel edges
            if not args.linear and gmax > 0:
                axes[0, 0].set_yscale("log")
                axes[0, 0].set_ylim(gmax * 1e-4, gmax * 2)
            fig.supxlabel("Redshift", fontsize=FONT_SIZE - 2, y=0.02)
            fig.supylabel(r"TDE rate (yr$^{-1}$)", fontsize=FONT_SIZE - 2, x=0.06)
            fig.suptitle(selection_label(sel, args.r200_factor), fontsize=FONT_SIZE - 2, y=0.94)
            out = f"{prefix}_perhalo_{sel}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved {out}")

    # ---------------- summary table ----------------
    rows = []
    print(f"\n{'selection':9s} {'bin':10s}  {'N_TDE/galaxy: mean  std/mean  [min, max]':44s} "
          f"{'peak/yr: mean  std/mean':26s} {'z_peak range':12s}")
    for (sel, tag), c in cubes.items():
        n_real = c["raw"].sum(axis=2) * c["dt"]            # (n_real, n_halos) TDEs per halo
        n_bin = n_real.mean(axis=1)                         # bin mean per realization
        bm = c["smoothed"].mean(axis=1)
        pk = bm.max(axis=1)
        zpk = z[np.argmax(bm, axis=1)]
        print(f"{sel:9s} {tag:10s}  {n_bin.mean():10.3g}  {n_bin.std() / n_bin.mean():7.1%}  "
              f"[{n_bin.min():.3g}, {n_bin.max():.3g}]      {pk.mean():9.2e}  {pk.std() / pk.mean():7.1%}   "
              f"{zpk.min():.1f}-{zpk.max():.1f}")
        for k, run in enumerate(runs):
            rows.append({"selection": sel, "mass_bin": tag, "halo": "bin_mean", "realization": run.label,
                         "n_tde": n_bin[k], "peak_rate": pk[k], "z_peak": zpk[k]})
        for j, hid in enumerate(c["halos"]):
            sh = c["smoothed"][:, j]
            for k, run in enumerate(runs):
                row = {"selection": sel, "mass_bin": tag, "halo": hid, "realization": run.label,
                       "n_tde": n_real[k, j], "peak_rate": sh[k].max(), "z_peak": z[np.argmax(sh[k])]}
                if sel == args.rows[0]:
                    stats = run.cluster_stats(hid)
                    if stats:
                        row.update(stats)
                rows.append(row)

    first = args.rows[0]
    print(f"\nPer halo ({first}): TDEs per realization, scatter, and discarded orbits (trace_failed / clusters)")
    for tag, _, _, ids in mass_bins:
        for hid in ids:
            rr = [r for r in rows if r["selection"] == first and r["halo"] == hid]
            if not rr:
                continue
            n = np.array([r["n_tde"] for r in rr])
            failed = [f"{r.get('n_trace_failed', '?')}/{r.get('n_clusters', '?')}" for r in rr]
            print(f"  {tag:10s} {hid:>8s}: N_TDE mean {n.mean():9.3g}, std/mean {n.std() / n.mean() if n.mean() else 0:6.1%};"
                  f"  discarded: {', '.join(failed)}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{prefix}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {prefix}_summary.csv")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
