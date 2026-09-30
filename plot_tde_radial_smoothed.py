"""
Radial distribution of TDEs about the central galaxy, averaged over the halos
in each of a few halo-mass bins.

"Radius" here is each TDE-producing cluster's separation from the central
galaxy (the tree's main-branch/root position -- the same quantity
plot_radius_tracks.py plots). A cluster's own size is << that separation, so
all of a cluster's TDEs at a given time sit at one radius.

Reads the per-halo files analysis.py writes:

    <data-dir>/tde_radial_<subhaloid>_alpha<alpha>.npz

each holding the TDE rate [Msun/yr] on analysis.py's 1e5-yr time grid x a
log-spaced separation grid (hist_msunyr), plus the rate from clusters that
have already inspiraled to the center (inspiraled_msunyr) and from clusters
with no radius track (no_track_msunyr, excluded here and reported).

Main figure: one panel per redshift window (--z-windows). In each, for every
mass bin, each halo's rate is time-averaged over the window, turned into a
rate per dex of separation, smoothed in log r (Gaussian, --smooth-dex FWHM),
and the curve is the mean over the bin's halos with a shaded band for the
halo-to-halo spread (--band). --normalize plots each halo's distribution as a
fraction per dex before averaging (so every halo counts equally and bins of
very different total rate can be compared by shape). The share of the
window's rate coming from clusters already at the center is printed on each
panel (it has no finite radius to plot).

--evolution also writes a second figure: the TDE-weighted median separation
(and its 16-84th percentile range) versus redshift for each mass bin,
computed from the bin's mean distribution after --smooth-myr time smoothing.
Clusters already at the center are left out of those percentiles.

Mass bins default to DEFAULT_MASS_BINS; override with one --bin per bin:

    python plot_tde_radial_smoothed.py --data-dir /u/scratch/c/clairewi/imbh-output --logy \\
        --bin ">1e11" 685512 697044 588075 467548 665702 \\
        --bin "1e10-1e11" 801308 753345 8 745415 826784 \\
        --bin "1e9-1e10" 1235585 1117358 1136724 1044309 939095

    # different redshift windows, shape only, plus the median-vs-z figure
    python plot_tde_radial_smoothed.py --z-windows 20,12,9,7,0 --normalize --evolution
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

# Same cosmology as analysis.py (the time grid in the npz files is built with it)
cosmo = FlatLambdaCDM(71, 0.27, Ob0=0.044, Tcmb0=2.726 * u.K)

# (tag for file names, legend label, color, subhalo IDs) -- matches plot_tde_rates_smoothed.py
DEFAULT_MASS_BINS = [
    ("gt1e11", r"$>10^{11}\,M_\odot$", "#0072B2",
     ["685512", "697044", "588075", "467548", "665702"]),
    ("1e10-1e11", r"$10^{10}$–$10^{11}\,M_\odot$", "#D55E00",
     ["801308", "753345", "8", "745415", "826784"]),
    ("1e9-1e10", r"$10^{9}$–$10^{10}\,M_\odot$", "#009E73",
     ["1235585", "1117358", "1136724", "1044309", "939095"]),
]
EXTRA_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
Z_MAX = 20


def age_to_redshift(t_yr):
    z_grid = np.concatenate([[0.0], np.logspace(-4, np.log10(1000), 4000)])
    age_grid = cosmo.age(z_grid).to(u.yr).value
    return np.interp(t_yr, age_grid[::-1], z_grid[::-1])


def redshift_to_age_yr(z):
    return cosmo.age(z).to(u.yr).value


def load_bin(data_dir, halo_ids, alpha):
    """Returns (time_yr, edges_kpc, halos) with halos = list of (id, npz dict), or None."""
    time_yr = edges = None
    halos = []
    for hid in halo_ids:
        path = os.path.join(data_dir, f"tde_radial_{hid}_alpha{alpha}.npz")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping halo {hid}")
            continue
        d = dict(np.load(path))
        if edges is None:
            time_yr, edges = d["time_yr"], d["edges_kpc"]
        else:
            if not np.allclose(d["edges_kpc"], edges):
                raise SystemExit(f"{path}: separation grid differs from the other halos' "
                                 "(re-run analysis.py with the same --r-min-kpc/--r-max-kpc/--n-rbins)")
            if len(d["time_yr"]) != len(time_yr) or not np.allclose(d["time_yr"], time_yr):
                raise SystemExit(f"{path}: time grid differs from the other halos'")
        lost = lambda k: float(np.sum(d[k])) * np.median(np.diff(d["time_yr"]))
        tot = lost("hist_msunyr") + lost("inspiraled_msunyr") + lost("no_track_msunyr")
        if tot > 0:
            print(f"  {hid}: {tot:.3g} Msun lost; at center {lost('inspiraled_msunyr')/tot:.1%}, "
                  f"no radius track {lost('no_track_msunyr')/tot:.1%} (excluded), "
                  f"clipped to grid edge {float(d['clipped_msun'])/tot:.1%}")
        else:
            print(f"  {hid}: no TDEs")
        halos.append((hid, d))
    if not halos:
        return None
    return time_yr, edges, halos


def coarsen(edges, rebin):
    """Merge every `rebin` adjacent log-r bins (drops a trailing partial group)."""
    n = (len(edges) - 1) // rebin * rebin
    return edges[:n + 1:rebin], n


def band(stack, mean, kind):
    if stack.shape[0] == 1:
        return mean, mean
    if kind == "minmax":
        return stack.min(0), stack.max(0)
    if kind == "percentile":
        return tuple(np.percentile(stack, [16, 84], axis=0))
    s = stack.std(0)
    return np.clip(mean - s, 0, None), mean + s


BAND_DESC = {"minmax": "halo-to-halo range", "percentile": "16–84th percentile",
             "std": r"mean $\pm1\sigma$"}


def window_profiles(time_yr, halos, n_r, rebin, t_lo, t_hi, sigma_bins, dlog, normalize):
    """
    Per halo: time-averaged rate over [t_lo, t_hi) per dex of separation,
    smoothed in log r. Returns (profiles (n_halo, n_r'), center_frac (n_halo,)).
    Halos with no TDEs in the window get a zero profile (dropped if --normalize).
    """
    sel = (time_yr >= t_lo) & (time_yr < t_hi)
    profs, fc = [], []
    for _, d in halos:
        h = d["hist_msunyr"][sel, :n_r].astype(float).mean(0) if sel.any() else np.zeros(n_r)
        h = h.reshape(-1, rebin).sum(1)
        center = float(d["inspiraled_msunyr"][sel].mean()) if sel.any() else 0.0
        placed = h.sum() + center
        if sigma_bins > 0:
            h = gaussian_filter1d(h, sigma_bins, mode="constant")
        dens = h / dlog  # Msun/yr per dex
        if normalize:
            if placed <= 0:
                continue
            dens = dens / placed  # fraction per dex (integrates to 1 - center fraction)
        profs.append(dens)
        fc.append(center / placed if placed > 0 else np.nan)
    if not profs:
        return None, None
    return np.vstack(profs), np.array(fc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Directory with tde_radial_<ID>_alpha<alpha>.npz files.")
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--bin", nargs="+", action="append", metavar=("LABEL", "ID"),
                   help="A mass bin: label then subhalo IDs. Repeat per bin.")
    p.add_argument("--z-windows", default="20,12,10,8,6,0",
                   help="Comma-separated redshift window EDGES; one panel per window "
                        "(default: 20,12,10,8,6,0 -- most TDEs happen at z~12-7).")
    p.add_argument("--smooth-dex", type=float, default=0.2,
                   help="Gaussian FWHM in log10(r) for smoothing each profile (default: 0.2; 0 = none).")
    p.add_argument("--rebin", type=int, default=1,
                   help="Merge this many adjacent separation bins before smoothing (default: 1).")
    p.add_argument("--normalize", action="store_true",
                   help="Plot fraction of TDE rate per dex (each halo normalized before averaging).")
    p.add_argument("--band", choices=["minmax", "percentile", "std"], default="minmax")
    p.add_argument("--logy", action="store_true", help="Log y axis.")
    p.add_argument("--r-range", default=None, help="x-axis limits in kpc, e.g. 0.1,500.")
    p.add_argument("--evolution", action="store_true",
                   help="Also plot the TDE-weighted median separation vs redshift.")
    p.add_argument("--smooth-myr", type=float, default=50.0,
                   help="Time smoothing (boxcar, Myr) for --evolution (default: 50).")
    p.add_argument("--output", default=None)
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

    z_edges = sorted({float(z) for z in args.z_windows.split(",")}, reverse=True)
    windows = list(zip(z_edges[:-1], z_edges[1:]))  # (z_hi, z_lo)
    if not windows:
        p.error("--z-windows needs at least two edges")

    loaded = []
    for tag, label, color, ids in mass_bins:
        print(f"\n[{tag}]")
        res = load_bin(args.data_dir, ids, args.alpha)
        if res is None:
            print(f"  WARNING: no files for bin {tag}; skipping it")
            continue
        loaded.append((tag, label, color, res))
    if not loaded:
        raise SystemExit("No tde_radial files found -- check --data-dir / --alpha, and that "
                         "analysis.py was re-run with the radial output.")

    edges0 = loaded[0][3][1]
    edges, n_r = coarsen(edges0, args.rebin)
    dlog = np.diff(np.log10(edges))[0]
    centers = np.sqrt(edges[1:] * edges[:-1])
    sigma_bins = args.smooth_dex / (2 * np.sqrt(2 * np.log(2))) / dlog if args.smooth_dex > 0 else 0

    ncol = min(len(windows), 3)
    nrow = int(np.ceil(len(windows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    rows, ymax, ypos = [], 0.0, []

    for w, (z_hi, z_lo) in enumerate(windows):
        ax = axes_flat[w]
        t_lo, t_hi = redshift_to_age_yr(z_hi), redshift_to_age_yr(z_lo)
        notes = []
        for tag, label, color, (time_yr, e, halos) in loaded:
            if not np.allclose(e, edges0):
                raise SystemExit(f"bin {tag}: separation grid differs from the first bin's")
            prof, fc = window_profiles(time_yr, halos, n_r, args.rebin, t_lo, t_hi,
                                       sigma_bins, dlog, args.normalize)
            if prof is None or not np.any(prof > 0):
                continue
            mean = prof.mean(0)
            lo, hi = band(prof, mean, args.band)
            ax.fill_between(centers, lo, hi, color=color, alpha=0.2, lw=0)
            ax.plot(centers, mean, color=color, lw=2,
                    label=f"{label} ({prof.shape[0]})" if w == 0 else None)
            ymax = max(ymax, hi.max())
            ypos.append(hi[hi > 0])
            if np.isfinite(fc).any():
                notes.append((color, np.nanmean(fc)))
            for k in range(len(centers)):
                rows.append({"bin": tag, "z_hi": z_hi, "z_lo": z_lo, "r_kpc": centers[k],
                             "mean": mean[k], "band_lo": lo[k], "band_hi": hi[k],
                             "n_halos": prof.shape[0],
                             "center_frac_mean": np.nanmean(fc) if np.isfinite(fc).any() else np.nan})
        ax.set_title(rf"${z_lo:g} < z < {z_hi:g}$", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, which="major", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if notes:
            ax.text(0.03, 0.97, "at center:", transform=ax.transAxes, va="top", fontsize=8, color="0.35")
            for i, (c, f) in enumerate(notes):
                ax.text(0.03, 0.97 - 0.07 * (i + 1), f"{f:.0%}", transform=ax.transAxes,
                        va="top", fontsize=8, color=c)
        else:
            ax.text(0.5, 0.5, "no TDEs", transform=ax.transAxes, ha="center", color="0.5")
    for ax in axes_flat[len(windows):]:
        ax.set_visible(False)

    ylabel = (r"fraction of TDE rate per dex" if args.normalize
              else r"$d\dot M_{\rm TDE}/d\log_{10} r$  [$M_\odot$ yr$^{-1}$ dex$^{-1}$]")
    for ax in axes[-1]:
        ax.set_xlabel("Separation from central galaxy [kpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if args.r_range:
        axes_flat[0].set_xlim(*[float(x) for x in args.r_range.split(",")])
    if args.logy and ypos:
        pos = np.concatenate(ypos)
        axes_flat[0].set_yscale("log")
        axes_flat[0].set_ylim(max(pos.max() * 1e-4, pos.min()), pos.max() * 2)
    else:
        axes_flat[0].set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
    top = 0.84 if nrow == 1 else 0.91
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(frameon=False, loc="lower center", ncol=len(loaded), bbox_to_anchor=(0.5, top),
               title=(rf"Halo mass (N halos)   ·   $\alpha={args.alpha}$, {args.smooth_dex:g} dex "
                      f"smoothing, shaded: {BAND_DESC[args.band]}"))

    tagstr = f"alpha{args.alpha}" + ("_norm" if args.normalize else "")
    out = args.output or f"tde_radial_massbins_{tagstr}.png"
    fig.savefig(out, dpi=200)
    csv = os.path.splitext(out)[0] + ".csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nSaved {out} and {csv}")

    if args.evolution:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for tag, label, color, (time_yr, e, halos) in loaded:
            dt = np.median(np.diff(time_yr))
            k = max(1, int(round(args.smooth_myr * 1e6 / dt / 10)))  # coarse step = smooth/10
            nt = len(time_yr) // k * k
            H = np.mean([d["hist_msunyr"][:nt].astype(float).reshape(-1, k, len(edges0) - 1).mean(1)
                         for _, d in halos], axis=0)
            H = uniform_filter1d(H, size=10, axis=0, mode="constant")
            t_c = time_yr[:nt].reshape(-1, k).mean(1)
            tot = H.sum(1)
            ok = tot > 1e-3 * tot.max()
            # quantiles of log r from the rate-weighted CDF; NaN where no TDEs
            # (so the line breaks rather than bridging quiet periods)
            cdf = np.cumsum(H, 1) / np.where(ok, tot, 1)[:, None]
            logc = np.log10(np.sqrt(edges0[1:] * edges0[:-1]))
            q = {qq: np.where(ok, 10 ** np.array([np.interp(qq, c, logc) for c in cdf]), np.nan)
                 for qq in (0.16, 0.5, 0.84)}
            z = age_to_redshift(t_c)
            ax2.fill_between(z, q[0.16], q[0.84], color=color, alpha=0.2, lw=0)
            ax2.plot(z, q[0.5], color=color, lw=2, label=f"{label} ({len(halos)})")
        ax2.set_xlim(Z_MAX, 0)
        ax2.set_yscale("log")
        ax2.set_xlabel("Redshift")
        ax2.set_ylabel("TDE-weighted separation from central [kpc]")
        ax2.set_title(f"median (line) and 16–84th percentile (shaded) of the bin-mean distribution; "
                      f"{args.smooth_myr:g} Myr smoothing", fontsize=9, color="0.3")
        ax2.grid(True, color="0.9", lw=0.6)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        ax2.legend(frameon=False, title="Halo mass (N halos)")
        fig2.tight_layout()
        out2 = os.path.splitext(out)[0] + "_vs_z.png"
        fig2.savefig(out2, dpi=200)
        print(f"Saved {out2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
