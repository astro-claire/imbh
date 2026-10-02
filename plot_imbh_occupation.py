"""
IMBH mass vs. host-cluster mass, and IMBH occupation fraction vs. cluster
stellar mass, from imbh.py's per-cluster output tables.

Reads, for every halo in each mass bin (same bins as the TDE-rate plots):

    <data-dir>/cluster_output_<ID>_<SNAP><suffix>.dat   (or .csv)

where <suffix> comes from the run selection flags (same flags as
submit_halos.pl / submit_analysis.pl; empty for an unlabelled default-mode run
-- see run_naming.py). All clusters of all halos in a bin are pooled.

Left panel  -- M_BH vs M_cl: every cluster with an IMBH (IMBH_mass_msun > 0),
               one colour/marker per halo-mass bin, with each bin's running
               median of M_BH in cluster-mass bins (only where a bin holds at
               least --min-per-bin clusters). Dashed lines mark constant
               M_BH / M_cl; the dotted line marks --mbh-min.
Right panel -- occupation fraction f_BH(M_cl): in log-spaced cluster-mass bins
               (--dlogm dex), the fraction of clusters that host an IMBH, with
               the 68% Wilson binomial interval shaded. A cluster "hosts an
               IMBH" if IMBH_mass_msun >= --mbh-min (default 500 Msun, the same
               floor analysis.py uses for TDEs) AND the IMBH finished forming
               before the cluster's trace ended (IMBH_final_formation_time_gyr <
               total_time_gyr, as in analysis.py) -- --ignore-formation-time
               drops the second condition.

Cluster mass is the cluster's stellar mass at formation (cluster_mass_msun).
Clusters with no IMBH mass (status 'trace_failed', a failed/timed-out
timescales call, or a zero-length trace -- e.g. clusters born at the final
snapshot, which have total_time_gyr = 0 and IMBH_mass_msun = NaN) are left out
of both panels; the printout counts each kind.

Also writes the occupation fractions (and their intervals) to a CSV next to
the figure.

Examples:
    python plot_imbh_occupation.py --data-dir /u/scratch/c/clairewi/imbh-output
    python plot_imbh_occupation.py --data-dir /u/scratch/c/clairewi/imbh-output \\
        --mode observational --n-relation bf20_seed --radius-relation marks_kroupa12
    python plot_imbh_occupation.py --data-dir ... --pool --mbh-min 100
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_naming import add_run_args, run_suffix, run_description

# (tag, legend label, colour, subhalo IDs) -- same bins/colours as plot_tde_*_smoothed.py
DEFAULT_MASS_BINS = [
    ("gt1e11", r"$>10^{11}\,M_\odot$", "#0072B2",
     ["685512", "697044", "588075", "467548", "665702"]),
    ("1e10-1e11", r"$10^{10}$–$10^{11}\,M_\odot$", "#D55E00",
     ["801308", "753345", "8", "745415", "826784"]),
    ("1e9-1e10", r"$10^{9}$–$10^{10}\,M_\odot$", "#009E73",
     ["1235585", "1117358", "1136724", "1044309", "939095"]),
]
EXTRA_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
# second encoding so bins stay distinguishable in grayscale / for colour-blind readers
MARKERS = ["o", "s", "^", "D", "v", "P"]
POOLED_COLOR = "#0072B2"

FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE + 2,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
    "legend.title_fontsize": FONT_SIZE - 2,
})


def load_halo(data_dir, halo_id, snap, suffix):
    """One halo's cluster table (pickled .dat as written by imbh.py, else .csv), or None."""
    stem = os.path.join(data_dir, f"cluster_output_{halo_id}_{snap}{suffix}")
    for ext, reader in ((".dat", pd.read_pickle), (".csv", pd.read_csv)):
        if os.path.exists(stem + ext):
            df = reader(stem + ext)
            df["halo_id"] = str(halo_id)
            return df
    print(f"  WARNING: {stem}.dat/.csv not found, skipping halo {halo_id}")
    return None


def wilson_interval(k, n, z=1.0):
    """Wilson score interval for k successes in n trials (z=1 -> ~68%)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def classify(df, mbh_min, require_formed):
    """
    Returns (valid, hosts): boolean arrays over df's rows. valid = usable
    clusters (trace and IMBH model both succeeded); hosts = valid clusters
    that host an IMBH by the module docstring's definition.
    """
    m_bh = pd.to_numeric(df["IMBH_mass_msun"], errors="coerce").to_numpy()
    m_cl = pd.to_numeric(df["cluster_mass_msun"], errors="coerce").to_numpy()
    status = df["status"].astype(str).to_numpy() if "status" in df else np.full(len(df), "")
    valid = np.isfinite(m_bh) & np.isfinite(m_cl) & (m_cl > 0) & (status != "trace_failed")
    hosts = valid & (m_bh >= mbh_min)
    if require_formed:
        t_form = pd.to_numeric(df["IMBH_final_formation_time_gyr"], errors="coerce").to_numpy()
        t_tot = pd.to_numeric(df["total_time_gyr"], errors="coerce").to_numpy()
        with np.errstate(invalid="ignore"):
            hosts &= t_form < t_tot
    return valid, hosts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Base output directory ($OUTDIR) with the cluster_output files.")
    p.add_argument("--snap", default="99", help="Output snapshot in the cluster_output file names (default: 99).")
    p.add_argument("--bin", nargs="+", action="append", metavar=("LABEL", "ID"),
                   help="A halo-mass bin: label then subhalo IDs. Repeat per bin. "
                        "Default: the three bins used by the TDE-rate plots.")
    p.add_argument("--pool", action="store_true",
                   help="Pool every halo into a single population instead of one curve per bin.")
    p.add_argument("--mbh-min", type=float, default=500.0,
                   help="Minimum IMBH mass [Msun] for a cluster to count as hosting an IMBH "
                        "(default: 500, analysis.py's TDE floor).")
    p.add_argument("--ignore-formation-time", action="store_true",
                   help="Count an IMBH even if it would only finish forming after the cluster's "
                        "trace ended (default: it must form before the trace ends, as in analysis.py).")
    p.add_argument("--dlogm", type=float, default=0.25,
                   help="Width of the cluster-mass bins in dex (default: 0.25).")
    p.add_argument("--min-per-bin", type=int, default=5,
                   help="Only show occupation fractions / medians for cluster-mass bins with at "
                        "least this many clusters (default: 5).")
    p.add_argument("--output", default=None,
                   help="Output image (default: imbh_occupation_mbh<mbh-min><suffix>.png); "
                        "the CSV goes next to it.")
    p.add_argument("--no-show", action="store_true")
    add_run_args(p)
    args = p.parse_args()

    suffix = run_suffix(args)
    run_desc = run_description(args)
    print(f"Reading cluster_output files from {args.data_dir}"
          + (f"  [{run_desc}]" if run_desc else "  [default mode]"))

    if args.bin:
        bins = []
        for i, e in enumerate(args.bin):
            if len(e) < 2:
                p.error("--bin needs a label followed by at least one subhalo ID")
            tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in e[0])
            bins.append((tag, e[0], EXTRA_COLORS[i % len(EXTRA_COLORS)], e[1:]))
    else:
        bins = DEFAULT_MASS_BINS
    if args.pool:
        bins = [("all", "All halos", POOLED_COLOR, [h for _, _, _, ids in bins for h in ids])]

    require_formed = not args.ignore_formation_time
    populations = []  # (tag, label, colour, marker, df, valid, hosts)
    for i, (tag, label, color, ids) in enumerate(bins):
        print(f"\n[{tag}]")
        frames = [f for f in (load_halo(args.data_dir, h, args.snap, suffix) for h in ids) if f is not None]
        if not frames:
            print(f"  WARNING: no cluster_output files for bin {tag}; skipping it")
            continue
        df = pd.concat(frames, ignore_index=True)
        valid, hosts = classify(df, args.mbh_min, require_formed)
        n_bad = int((~valid).sum())
        failed = (df["status"].astype(str) == "trace_failed").to_numpy() if "status" in df else np.zeros(len(df), bool)
        t_tot = pd.to_numeric(df["total_time_gyr"], errors="coerce").to_numpy()
        with np.errstate(invalid="ignore"):
            zero_t = ~valid & ~failed & (t_tot <= 0)
        print(f"  {len(frames)} halo(s), {len(df)} clusters; excluded {n_bad}: "
              f"{int(failed.sum())} trace_failed, {int(zero_t.sum())} with zero trace time "
              f"(no IMBH mass computed), {n_bad - int(failed.sum()) - int(zero_t.sum())} other NaN")
        print(f"  "
              f"{int(hosts.sum())} host an IMBH >= {args.mbh_min:g} Msun "
              f"({hosts.sum() / max(valid.sum(), 1):.1%} of usable clusters)")
        populations.append((tag, label, color, MARKERS[i % len(MARKERS)], df, valid, hosts))
    if not populations:
        raise SystemExit("No cluster_output files found -- check --data-dir, --snap and the run flags.")

    # common log-spaced cluster-mass bins over every usable cluster
    all_m = np.concatenate([pd.to_numeric(df["cluster_mass_msun"], errors="coerce").to_numpy()[v]
                            for _, _, _, _, df, v, _ in populations])
    lo = np.floor(np.log10(all_m.min()) / args.dlogm) * args.dlogm
    hi = np.ceil(np.log10(all_m.max()) / args.dlogm) * args.dlogm
    edges = 10 ** np.arange(lo, hi + 0.5 * args.dlogm, args.dlogm)
    if len(edges) < 2:
        edges = np.array([all_m.min() * 0.9, all_m.max() * 1.1])
    centres = np.sqrt(edges[1:] * edges[:-1])

    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(15, 6.5))
    rows = []
    for tag, label, color, marker, df, valid, hosts in populations:
        m_cl = pd.to_numeric(df["cluster_mass_msun"], errors="coerce").to_numpy()
        m_bh = pd.to_numeric(df["IMBH_mass_msun"], errors="coerce").to_numpy()

        # ---- left: M_BH vs M_cl (clusters with any IMBH mass, so it can go on a log axis)
        has_bh = valid & (m_bh > 0)
        ax_m.scatter(m_cl[has_bh], m_bh[has_bh], s=14, marker=marker, color=color, alpha=0.35,
                     linewidths=0, rasterized=True)
        idx = np.digitize(m_cl[has_bh], edges) - 1
        med = np.full(len(centres), np.nan)
        for j in range(len(centres)):
            sel = idx == j
            if sel.sum() >= args.min_per_bin:
                med[j] = np.median(m_bh[has_bh][sel])
        ok = np.isfinite(med)
        ax_m.plot(centres[ok], med[ok], color=color, lw=2.5, marker=marker, ms=8,
                  markeredgecolor="white", markeredgewidth=1.2, label=f"{label}")

        # ---- right: occupation fraction
        idx_all = np.digitize(m_cl[valid], edges) - 1
        n_tot = np.bincount(idx_all[(idx_all >= 0) & (idx_all < len(centres))],
                            minlength=len(centres))[:len(centres)]
        idx_host = np.digitize(m_cl[hosts], edges) - 1
        n_host = np.bincount(idx_host[(idx_host >= 0) & (idx_host < len(centres))],
                             minlength=len(centres))[:len(centres)]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = n_host / n_tot
        f_lo, f_hi = wilson_interval(n_host, n_tot)
        show = n_tot >= args.min_per_bin
        ax_f.fill_between(centres[show], f_lo[show], f_hi[show], color=color, alpha=0.2, lw=0, step=None)
        ax_f.plot(centres[show], frac[show], color=color, lw=2.5, marker=marker, ms=8,
                  markeredgecolor="white", markeredgewidth=1.2, label=f"{label}")
        for j in range(len(centres)):
            rows.append({"bin": tag, "m_cl_lo_msun": edges[j], "m_cl_hi_msun": edges[j + 1],
                         "m_cl_centre_msun": centres[j], "n_clusters": int(n_tot[j]),
                         "n_with_imbh": int(n_host[j]),
                         "occupation_fraction": frac[j] if n_tot[j] else np.nan,
                         "frac_lo_68": f_lo[j] if n_tot[j] else np.nan,
                         "frac_hi_68": f_hi[j] if n_tot[j] else np.nan,
                         "median_mbh_msun": med[j]})

    # reference lines of constant M_BH / M_cl, and the --mbh-min floor
    xlim = (edges[0], edges[-1])
    xx = np.array(xlim)
    for ratio in (1e-3, 1e-2, 1e-1):
        ax_m.plot(xx, ratio * xx, ls="--", color="0.6", lw=1, zorder=0)
        ax_m.text(xx[1], ratio * xx[1], f" {ratio:g}", color="0.45", fontsize=FONT_SIZE - 4,
                  va="center", ha="left", clip_on=False)
    ax_m.axhline(args.mbh_min, ls=":", color="0.45", lw=1.2, zorder=0)
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlim(*xlim)
    ax_m.set_xlabel(r"Cluster stellar mass $M_{\rm cl}$ [$M_\odot$]")
    ax_m.set_ylabel(r"IMBH mass $M_{\rm BH}$ [$M_\odot$]")
    ax_m.set_title(r"lines: median $M_{\rm BH}$; dashed: $M_{\rm BH}/M_{\rm cl}$", fontsize=FONT_SIZE - 3,
                   color="0.35")

    ax_f.set_xscale("log")
    ax_f.set_xlim(*xlim)
    ax_f.set_ylim(-0.02, 1.02)
    ax_f.set_xlabel(r"Cluster stellar mass $M_{\rm cl}$ [$M_\odot$]")
    ax_f.set_ylabel("IMBH occupation fraction")
    ax_f.set_title(rf"$M_{{\rm BH}}\geq{args.mbh_min:g}\,M_\odot$"
                   + ("" if require_formed else ", formation time ignored")
                   + "; shaded: 68% binomial interval", fontsize=FONT_SIZE - 3, color="0.35")

    for ax in (ax_m, ax_f):
        ax.grid(True, which="major", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    legend_title = "Halo mass" if not args.pool else None
    if run_desc:
        legend_title = (legend_title + "\n" if legend_title else "") + run_desc
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(*ax_f.get_legend_handles_labels(), frameon=False, loc="lower center",
               ncol=len(populations), bbox_to_anchor=(0.5, 0.87), title=legend_title)

    tag = f"mbh{args.mbh_min:g}" + ("_pooled" if args.pool else "") + \
          ("_anytime" if not require_formed else "") + suffix
    out = args.output or f"imbh_occupation_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    csv = os.path.splitext(out)[0] + ".csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nSaved {out} and {csv}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
