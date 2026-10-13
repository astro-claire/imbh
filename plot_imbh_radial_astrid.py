"""
Radial distribution of IMBHs at z = 0 in our 15 TNG50 galaxies, overplotted on
the wandering black holes of ASTRID's quenched, dispersion-dominated centrals
(the 2-panel figure of ASTRID/wandering_bh_radial_2panel.ipynb):

  left   number of BHs per galaxy per dex of distance r from the central galaxy
  right  the same per dex of r / R200 (R200c of the central at z = 0)

one colour per z = 0 stellar-mass bin; ASTRID shaded, each of our models as a
line (default: maximal solid, conservative dashed).

Where our IMBHs are at z = 0 (from imbh.py's per-cluster table):
  pinned / outskirts / escaped  -- traced to z = 0; r is the last point of the
      cluster's radius track (separation from the central galaxy, i.e. the
      tree root's main branch; see imbh.save_radius_track). Tracks exist for
      IMBHs >= 500 Msun, so --mbh-min must be >= 500.
  inspiraled -- the trace stops when the cluster sinks to the centre of its
      host; the IMBH is then followed forward with that host through the
      merger tree to z = 0 (as for pinned clusters).
An IMBH that ends at the centre of the central galaxy itself (pinned in the
main branch, or inspiraled into a host that is or becomes the main branch) is counted as NUCLEAR and left out: ASTRID's
wandering sample likewise excludes the most massive (central) BH of each
galaxy. IMBHs whose z = 0 position can't be determined (missing track, tree
ends early) are counted as UNKNOWN and left out; both counts are printed.
Only IMBHs with M_BH >= --mbh-min that finished forming before their trace
ended (IMBH_final_formation_time_gyr < total_time_gyr, as in analysis.py) are
used.

The positions need the cluster tables, radius tracks and merger trees, so run
the first pass where those live (e.g. Hoffman2). Each model's z = 0 positions
are cached in <cache-dir>/imbh_z0_positions<suffix>_mbh<min>.csv; copy those
small files to another machine and re-run there to plot from the cache
(the .dat / tracks / trees are then not needed). --refresh rebuilds them.

--bin-width W sets the histogram bin width in dex (default 0.2); --nbins N
instead splits each panel's range into N equal bins. Counts are always per dex,
so the normalization doesn't change with the binning.

--max-r200 X keeps only BHs within X * R200 of the central (ours and ASTRID's),
e.g. --max-r200 3 to compare like with like (ASTRID's BHs are FoF-group members,
while ours include escaped IMBHs at any distance).

The shaded band in the right panel marks r < R_STOP_FRAC * R200 (0.01 R200): an
orbit that decays inside it is ended as 'inspiraled', so our model has no
wandering IMBHs there by construction (a resolution limit, not a prediction).
The triangles on the left panel mark the same radius in kpc for each bin's
median R200.

ASTRID selection (same as the notebook): galaxies with dispersion_dominated;
BHs with dispersion_dominated, rank_mass_in_gal > 0 (not the galaxy's most
massive BH) and bh_mass < --astrid-mbh-max (default 3e5/h Msun).

Examples:
  python plot_imbh_radial_astrid.py --data-dir /u/scratch/c/clairewi/imbh-output
  python plot_imbh_radial_astrid.py --cache-dir z0_positions          # plot from cached CSVs
  python plot_imbh_radial_astrid.py --data-dir ... --model "Maximal, seed 1" "--jitter-dex 0.1 --label seed1"
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rate_runs import Run, DEFAULT_MASS_BINS

H_ASTRID = 0.6774
ASTRID_MASSBIN = {"1e9-1e10": 1, "1e10-1e11": 2, "gt1e11": 3}   # our bin tag -> ASTRID 'massbin'
DEFAULT_MODELS = [
    ("Maximal", "--jitter-dex 0.1"),
    ("Conservative", "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12"),
]
LINESTYLES = ["-", "--", ":", "-."]
DLOG = 0.2           # default bin width in dex (as in the notebook); see --bin-width / --nbins
Z0_TOL_GYR = 0.02    # a track must end within this of the output snapshot's age
NUCLEAR, WANDERING, UNKNOWN, BEYOND_CUT = "nuclear", "wandering", "unknown", "beyond_cut"
try:   # the radius at which imbh.py ends an orbit as 'inspiraled' (in units of the host's R200)
    from dynamical_friction import R_STOP_FRAC
except Exception:
    R_STOP_FRAC = 0.01

FONT_SIZE = 20
plt.rcParams.update({
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE - 2, "ytick.labelsize": FONT_SIZE - 2,
    "legend.fontsize": FONT_SIZE - 4, "legend.title_fontsize": FONT_SIZE - 4,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 1.5,
})


# ---------------------------------------------------------------------------
# our IMBHs at z = 0
# ---------------------------------------------------------------------------
def last_track_point(path):
    """(time_since_formation_gyr, separation_kpc) of the last row of a radius track."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4096))
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    t, r = lines[-1].decode().split(",")[:2]
    return float(t), float(r)        # raises ValueError if only the header is there


def load_navigator(halo_id, tree_dir, snap_redshift_path):
    from tree_navigator import MergerTreeNavigator   # needs h5py; only when building positions
    tree_path = os.path.join(tree_dir, f"sublink_full_{halo_id}.hdf5")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"merger tree {tree_path} not found (set --tree-dir)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MergerTreeNavigator(tree_path, snap_redshift_path, box_size_ckpc_h=35000.0)


def on_main_branch(nav, subhalo_id):
    """True if subhalo_id is the root's main-branch progenitor at its own snapshot. (A pinned
    cluster's final_subhalo_id is its host at the last snapshot BEFORE the output one, e.g.
    snap 98, so comparing with the z=0 root itself would miss it.)"""
    sid = int(subhalo_id)
    return nav.is_virtual(sid) or nav.main_branch_id_at_snap(nav.snap_of(sid)) == sid


def follow_to_snap(nav, subhalo_id, root_id, snap):
    """Where a subhalo (an inspiraled cluster's host) ends up at `snap`: (location, r_kpc)."""
    sid = int(subhalo_id)
    if nav.is_virtual(sid):              # a stand-in main-branch row: it is the main branch
        return NUCLEAR, 0.0
    while True:
        nxt, _ = nav.step_forward(sid)
        if nxt is None:
            break
        sid = nxt
    if sid == root_id:
        return NUCLEAR, 0.0
    if nav.snap_of(sid) != snap:         # tree stops before the output snapshot
        return UNKNOWN, np.nan
    rel_pos, _ = nav.relative_state(sid, root_id)
    return WANDERING, float(np.linalg.norm(rel_pos.to("kpc").value))


def halo_positions(run, halo_id, tree_dir, snap_redshift_path, mbh_min):
    """One row per IMBH >= mbh_min in this halo: status, mass, location, r_kpc, r200_kpc."""
    path = run.cluster_output_path(halo_id)
    if not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    cols = ["halo", "status", "imbh_mass_msun", "location", "r_kpc", "r200_kpc"]
    if df.empty or "IMBH_mass_msun" not in df.columns:
        return pd.DataFrame(columns=cols)
    m_bh = pd.to_numeric(df["IMBH_mass_msun"], errors="coerce")
    sel = m_bh >= mbh_min
    if "IMBH_final_formation_time_gyr" in df.columns:
        sel &= df["IMBH_final_formation_time_gyr"] < df["total_time_gyr"]
    df = df[sel]

    nav = load_navigator(halo_id, tree_dir, snap_redshift_path)
    root_id = nav.main_branch_id_at_snap(run.snap)
    r200 = nav.host_properties(root_id)[1]
    t_out = nav.age_at_snap(run.snap).to("Gyr").value
    track_dir = os.path.join(run.data_dir, f"cluster_tracks_{halo_id}{run.model_suffix}")

    rows = []
    for _, c in df.iterrows():
        status = str(c["status"])
        loc, r = UNKNOWN, np.nan
        if status == "inspiraled":
            loc, r = follow_to_snap(nav, c["final_subhalo_id"], root_id, run.snap)
        elif status in ("pinned", "outskirts", "escaped"):
            if status == "pinned" and on_main_branch(nav, c["final_subhalo_id"]):
                loc, r = NUCLEAR, 0.0
            else:
                tpath = c.get("radius_track_path")
                if isinstance(tpath, str) and tpath and not os.path.exists(tpath):
                    tpath = os.path.join(track_dir, os.path.basename(tpath))
                if isinstance(tpath, str) and tpath and os.path.exists(tpath):
                    try:
                        t_last, r_last = last_track_point(tpath)
                        if abs(c["cluster_formation_age_gyr"] + t_last - t_out) < Z0_TOL_GYR:
                            loc, r = WANDERING, r_last
                    except (ValueError, IndexError):
                        pass
        rows.append((halo_id, status, float(c["IMBH_mass_msun"]), loc, r, r200))
    return pd.DataFrame(rows, columns=cols)


def model_positions(run, mass_bins, args):
    """All halos of one model, from the cache if present; adds 'mass_bin'. Also returns
    {bin_tag: number of galaxies with output}."""
    os.makedirs(args.cache_dir, exist_ok=True)
    cache = os.path.join(args.cache_dir,
                         f"imbh_z0_positions{run.model_suffix or '_default'}_mbh{args.mbh_min:g}.csv")
    if os.path.exists(cache) and not args.refresh:
        df = pd.read_csv(cache, dtype={"halo": str})
        print(f"  {run.label}: read {cache}")
    else:
        parts = []
        for tag, _, _, ids in mass_bins:
            for hid in ids:
                p = halo_positions(run, hid, args.tree_dir, args.snap_redshift_path, args.mbh_min)
                if p is None:
                    print(f"    {run.label}: no cluster output for {hid} -- skipped")
                    continue
                p["mass_bin"] = tag
                # a halo with no IMBHs still counts as a galaxy: keep a placeholder row
                parts.append(p if len(p) else pd.DataFrame(
                    [{"halo": hid, "mass_bin": tag, "location": "none"}]))
        if not parts:
            return None, {}
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(cache, index=False)
        print(f"  {run.label}: wrote {cache}")
    ngal = df.groupby("mass_bin")["halo"].nunique().to_dict()
    return df, ngal


# ---------------------------------------------------------------------------
# ASTRID
# ---------------------------------------------------------------------------
def load_astrid(astrid_dir, prefix, mbh_max):
    load = lambda what: np.load(os.path.join(astrid_dir, f"{prefix}_{what}.npy"))
    cat, bhs, halo = load("catalog"), load("bhs"), load("halo_r200")
    gal = cat[cat["dispersion_dominated"]]
    wand = bhs[bhs["dispersion_dominated"] & (bhs["rank_mass_in_gal"] > 0) & (bhs["bh_mass"] < mbh_max)]
    order = np.argsort(halo["subidx"])
    idx = order[np.searchsorted(halo["subidx"], wand["gal_subidx"], sorter=order)]
    r200 = halo["r200_kpc"][idx]
    ngal = {b: int(np.sum(gal["massbin"] == b)) for b in (1, 2, 3)}
    return wand, r200, ngal


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def per_dex(values, edges, ngal):
    """Histogram of log10(values) as number per galaxy per dex; empty bins -> NaN (gaps on log axes)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    n, _ = np.histogram(np.log10(v), bins=edges)
    out = n / max(ngal, 1) / np.diff(edges)     # per dex, whatever the bin width
    return np.where(n > 0, out, np.nan)


def edges_covering(arrays, lo, hi, width=DLOG, nbins=None, top=None):
    """
    Bin edges in log10 spanning at least [lo, hi] and all the data. Either fixed-width bins
    (width dex, snapped to multiples of width) or, with nbins, exactly nbins equal bins over
    that range. With top (log10 of a cut), the edges end exactly at top instead.
    """
    vals = np.concatenate([np.log10(a[np.isfinite(a) & (a > 0)]) for a in arrays if len(a)] or [[lo]])
    lo = min(lo, np.floor(vals.min() / width) * width)
    hi = max(hi, np.ceil(vals.max() / width) * width) if top is None else top
    if nbins is not None:
        return np.linspace(lo, hi, nbins + 1)
    if top is not None:   # count down from the cut so the last bin isn't only partly filled
        return top - width * np.arange(int(np.ceil((top - lo) / width - 1e-9)), -1, -1)
    return np.arange(lo, hi + width / 2, width)


def log_ticks(ax, edges, margin=0.0):
    """Decade ticks labelled in linear units; with margin (dex), skip ticks that close to an edge
    (so labels of touching panels don't collide)."""
    ticks = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
    ticks = ticks[(ticks - edges[0] >= margin) & (edges[-1] - ticks >= margin)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10 ** t:g}" for t in ticks])
    ax.set_xlim(edges[0], edges[-1])


# ---------------------------------------------------------------------------
# building blocks shared with plot_imbh_radial_astrid_quad.py
# ---------------------------------------------------------------------------
def build_parser(description, default_output):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="imbh.py output directory ($OUTDIR).")
    p.add_argument("--model", nargs=2, action="append", metavar=("LABEL", "FLAGS"),
                   help="A model to plot (repeatable; replaces the default maximal + conservative).")
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--mbh-min", type=float, default=500.0,
                   help="Minimum IMBH mass (default 500 Msun; radius tracks exist only above 500).")
    p.add_argument("--tree-dir", default="tng_download")
    p.add_argument("--snap-redshift-path", default="tng_download/snapshot_redshifts.json")
    p.add_argument("--cache-dir", default=".", help="Where the z=0 position CSVs are read/written.")
    p.add_argument("--refresh", action="store_true", help="Rebuild the position CSVs from the run output.")
    p.add_argument("--astrid-dir", default="ASTRID")
    p.add_argument("--astrid-prefix", default="ellip_centrals_z0_PIG817_sfr200")
    p.add_argument("--astrid-mbh-max", type=float, default=3e5 / H_ASTRID,
                   help="ASTRID wandering-BH mass cut in Msun (default 3e5/h, as in the notebook).")
    p.add_argument("--no-astrid", action="store_true")
    p.add_argument("--bin-width", type=float, default=DLOG,
                   help=f"Histogram bin width in dex (default {DLOG}, as in the ASTRID notebook).")
    p.add_argument("--nbins", type=int, default=None,
                   help="Number of histogram bins per panel (overrides --bin-width; each panel's "
                        "range is split into this many equal bins).")
    p.add_argument("--noise-floor", type=float, default=None,
                   help="Grey horizontal line marking our noise floor. Default: one IMBH in one bin, "
                        "1 / (N_galaxies x bin width) per galaxy per dex; give a value to override, 0 for none.")
    p.add_argument("--max-r200", type=float, default=None,
                   help="Only keep BHs within this many R200 of the central (ours and ASTRID's), e.g. 3.")
    p.add_argument("--output", default=default_output)
    p.add_argument("--no-show", action="store_true")
    return p


def load_inputs(args):
    """(models, astrid): models = [(label, linestyle, positions df, {bin: n_galaxies})]."""
    if args.mbh_min < 500:
        print("WARNING: radius tracks are only saved for IMBHs >= 500 Msun; lighter ones will be 'unknown'.")
    models = []
    for (label, flags), ls in zip(args.model or DEFAULT_MODELS, LINESTYLES * 4):
        run = Run(label, flags, args.data_dir, args.snap)
        df, ngal = model_positions(run, DEFAULT_MASS_BINS, args)
        if df is None:
            print(f"WARNING: no output for model '{label}' -- skipped")
            continue
        models.append((label, ls, df, ngal))
    astrid = None if args.no_astrid else load_astrid(args.astrid_dir, args.astrid_prefix, args.astrid_mbh_max)
    return models, astrid


def apply_cut(models, astrid, max_r200):
    """Copies of (models, astrid) keeping only BHs within max_r200 * R200 (None = no cut)."""
    if max_r200 is None:
        return models, astrid
    out = []
    for label, ls, df, ngal in models:
        df = df.copy()
        far = (df["location"] == WANDERING) & (df["r_kpc"] / df["r200_kpc"] > max_r200)
        df.loc[far, "location"] = BEYOND_CUT
        print(f"  r < {max_r200:g} R200: dropped {far.sum()} of {label}'s wandering IMBHs")
        out.append((label, ls, df, ngal))
    if astrid:
        wand, r200a, ngal_a = astrid
        keep = wand["dist_kpc"] / r200a <= max_r200
        print(f"  r < {max_r200:g} R200: dropped {np.sum(~keep)} of {len(wand)} ASTRID BHs")
        astrid = (wand[keep], r200a[keep], ngal_a)
    return out, astrid


def print_summary(models, astrid, mass_bins=DEFAULT_MASS_BINS):
    print(f"\n{'model':14s} {'bin':10s} {'gal':>4s} {'IMBH/gal':>9s} {'wander':>7s} {'nuclear':>8s} "
          f"{'unknown':>8s} {'median r':>9s} {'>R200':>6s} {'>3R200':>7s}   (per galaxy; r in kpc)")
    for label, _, df, ngal in models:
        for tag, _, _, _ in mass_bins:
            d = df[df["mass_bin"] == tag]
            n = ngal.get(tag, 0)
            if not n:
                continue
            w = d[d["location"] == WANDERING]
            x = w["r_kpc"] / w["r200_kpc"]
            print(f"{label[:14]:14s} {tag:10s} {n:4d} "
                  f"{d['location'].isin([WANDERING, NUCLEAR, UNKNOWN, BEYOND_CUT]).sum() / n:9.1f} "
                  f"{len(w) / n:7.1f} {(d['location'] == NUCLEAR).sum() / n:8.1f} "
                  f"{(d['location'] == UNKNOWN).sum() / n:8.1f} {np.nanmedian(w['r_kpc']) if len(w) else np.nan:9.1f} "
                  f"{np.mean(x > 1) if len(w) else np.nan:6.2f} {np.mean(x > 3) if len(w) else np.nan:7.2f}")
    if astrid:
        wand, r200a, ngal_a = astrid
        for tag, _, _, _ in mass_bins:
            b = ASTRID_MASSBIN[tag]
            s = wand["massbin"] == b
            x = wand["dist_kpc"][s] / r200a[s]
            print(f"{'ASTRID':14s} {tag:10s} {ngal_a[b]:4d} {'':9s} {s.sum() / ngal_a[b]:7.1f} {'':8s} {'':8s} "
                  f"{np.median(wand['dist_kpc'][s]):9.1f} {np.mean(x > 1):6.2f} {np.mean(x > 3):7.2f}")


def compute_edges(models, astrid, args, max_r200=None):
    """(r edges, r/R200 edges) in log10, covering all plotted BHs."""
    r_all, x_all = [], []
    for _, _, df, _ in models:
        w = df[df["location"] == WANDERING]
        r_all.append(w["r_kpc"].to_numpy(float))
        x_all.append((w["r_kpc"] / w["r200_kpc"]).to_numpy(float))
    if astrid:
        r_all.append(np.asarray(astrid[0]["dist_kpc"], float))
        x_all.append(np.asarray(astrid[0]["dist_kpc"] / astrid[1], float))
    lr = edges_covering(r_all, -1.0, 3.2, args.bin_width, args.nbins)
    lx = edges_covering(x_all, -3.0, 1.0, args.bin_width, args.nbins,
                        top=None if max_r200 is None else np.log10(max_r200))
    return lr, lx


def noise_floor(models, edges, value=None):
    """Per galaxy per dex value of ONE IMBH in one bin: 1 / (N_galaxies * bin width). With value,
    that value instead (<= 0: no line)."""
    if value is not None:
        return value if value > 0 else None
    n = min([v for _, _, _, ngal in models for v in ngal.values()] or [0])
    return 1.0 / (n * float(np.median(np.diff(edges)))) if n else None


def draw_pair(ax_r, ax_x, models, astrid, lr_edges, lx_edges, mass_bins=DEFAULT_MASS_BINS,
              annotate=True, triangles=True, floor_value=None):
    """Histograms in r (ax_r) and r/R200 (ax_x), plus the inspiral-radius shading and the R200 line.
    Returns every plotted (positive) value, for setting y limits."""
    values = []
    for tag, _, color, _ in mass_bins:
        if astrid:
            wand, r200a, ngal_a = astrid
            b = ASTRID_MASSBIN[tag]
            s = wand["massbin"] == b
            for ax, vals, edges in ((ax_r, wand["dist_kpc"][s], lr_edges),
                                    (ax_x, wand["dist_kpc"][s] / r200a[s], lx_edges)):
                h = per_dex(vals, edges, ngal_a[b])
                ax.stairs(np.nan_to_num(h), edges, fill=True, color=color, alpha=0.15, lw=0)
                ax.stairs(h, edges, color=color, lw=1.0, alpha=0.6, baseline=None)
                values.append(h)
        for label, ls, df, ngal in models:
            w = df[(df["mass_bin"] == tag) & (df["location"] == WANDERING)]
            n = ngal.get(tag, 0)
            if not n:
                continue
            for ax, vals, edges in ((ax_r, w["r_kpc"], lr_edges),
                                    (ax_x, w["r_kpc"] / w["r200_kpc"], lx_edges)):
                h = per_dex(vals, edges, n)
                ax.stairs(h, edges, color=color, lw=2.5, ls=ls, baseline=None)
                values.append(h)

    # our noise floor: one IMBH in a bin
    for ax, edges in ((ax_r, lr_edges), (ax_x, lx_edges)):
        f = noise_floor(models, edges, floor_value)
        if f:
            ax.axhline(f, color="0.55", lw=1.5, zorder=1)

    # region where our orbits are ended as 'inspiraled' (no wandering IMBHs by construction)
    lstop = np.log10(R_STOP_FRAC)
    ax_x.axvspan(-10, lstop, color="0.85", lw=0, zorder=0)
    ax_x.axvline(0, color="0.4", lw=2, ls=":")
    if annotate:
        ax_x.text((lx_edges[0] + lstop) / 2, 0.97, "inspiral\nradius", color="0.35", ha="center", va="top",
                  fontsize=FONT_SIZE - 6, transform=ax_x.get_xaxis_transform())
    if triangles:   # the same radius in kpc, at each bin's median R200 (from our galaxies)
        for tag, _, color, _ in mass_bins:
            r200s = pd.concat([df.loc[df["mass_bin"] == tag, ["halo", "r200_kpc"]] for _, _, df, _ in models])
            r200s = r200s.dropna().drop_duplicates("halo")["r200_kpc"]
            if len(r200s):
                ax_r.plot(np.log10(R_STOP_FRAC * np.median(r200s)), 1.0, marker="v", ms=12, color=color,
                          transform=ax_r.get_xaxis_transform(), clip_on=False, zorder=5)
    v = np.concatenate([np.ravel(a) for a in values]) if values else np.array([])
    return v[np.isfinite(v) & (v > 0)]


def legend_handles(models, astrid, args, triangles=True, edges=None):
    e = int(np.floor(np.log10(args.astrid_mbh_max)))
    mtxt = rf"{args.astrid_mbh_max / 10 ** e:.1f}\times10^{{{e}}}"
    style = []
    if astrid:
        style.append(Patch(facecolor="0.6", alpha=0.4, edgecolor="0.4",
                           label=rf"ASTRID, $M_{{\rm BH}}<{mtxt}\,M_\odot$"))
    for label, ls, _, _ in models:
        style.append(Line2D([], [], color="0.2", lw=2.5, ls=ls,
                            label=rf"{label}, $M_{{\rm BH}}\geq{args.mbh_min:g}\,M_\odot$"))
    if edges is not None:
        f = noise_floor(models, edges, args.noise_floor)
        if f:
            n = min([v for _, _, _, ngal in models for v in ngal.values()] or [0])
            style.append(Line2D([], [], color="0.55", lw=1.5,
                                label=f"one IMBH per bin ({n} galaxies)" if args.noise_floor is None
                                else f"noise floor ({f:g})"))
    if triangles:
        style.append(Line2D([], [], color="0.3", marker="v", ms=10, ls="none",
                            label=rf"${R_STOP_FRAC:g}\,R_{{200}}$ (median per bin)"))
    colors = [Line2D([], [], color=c, lw=4, label=l) for _, l, c, _ in DEFAULT_MASS_BINS]
    return colors, style


def add_legends(fig, color_handles, style_handles, style_title=None, gap=0.01, y=1.0):
    """The two legends side by side, centred above the figure."""
    fig.legend(handles=color_handles, loc="lower right", bbox_to_anchor=(0.5 - gap, y), ncol=1,
               frameon=False, title=r"Host stellar mass at $z=0$")
    fig.legend(handles=style_handles, loc="lower left", bbox_to_anchor=(0.5 + gap, y), ncol=1,
               frameon=False, title=style_title)


def main():
    args = build_parser(__doc__, "imbh_radial_z0_astrid.pdf").parse_args()
    models, astrid = load_inputs(args)
    models, astrid = apply_cut(models, astrid, args.max_r200)
    print_summary(models, astrid)
    lr_edges, lx_edges = compute_edges(models, astrid, args, args.max_r200)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_pair(axes[0], axes[1], models, astrid, lr_edges, lx_edges, floor_value=args.noise_floor)
    log_ticks(axes[0], lr_edges)
    log_ticks(axes[1], lx_edges)
    axes[0].set_xlabel(r"$r_{\rm BH}$ (kpc)")
    axes[1].set_xlabel(r"$r_{\rm BH}\,/\,R_{200}$")
    axes[0].set_ylabel("BHs per galaxy per dex")
    for ax in axes:
        ax.set_yscale("log")
    fig.tight_layout()
    cut_title = rf"only $r_{{\rm BH}}<{args.max_r200:g}\,R_{{200}}$" if args.max_r200 is not None else None
    add_legends(fig, *legend_handles(models, astrid, args, edges=lx_edges), style_title=cut_title)
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    print(f"\nSaved {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
