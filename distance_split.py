"""
Split each halo's TDE rate by the cluster's separation from the main
progenitor AT THE TIME OF EACH EVENT, relative to a multiple of the main
progenitor's R200c at that same time (default 3 R200c).

This replaces the z = 0 'escaped' status (beyond 3 R200c of the z = 0 host
at the last snapshot) as the way to separate "near" from "far" events: a
cluster that ends up far away at z = 0 may have been well inside the
assembling galaxy when its TDEs went off, and vice versa.

Inputs (no re-run of imbh.py or analysis.py needed):
  * <analysis dir>/tde_radial_<ID>_alpha<alpha>.npz from analysis.py: the
    TDE rate on the analysis time grid, binned in separation from the main
    progenitor (hist_msunyr), plus the parts with no separation
    (inspiraled_msunyr: after inspiral, at the centre; no_track_msunyr:
    clusters without a radius track).
  * the TNG merger tree (sublink_full_<ID>.hdf5), for the main progenitor's
    R200c at every snapshot -- the same main-branch reference the orbit
    integration's background potential and the radius tracks use
    (MergerTreeNavigator.main_branch_growth_history). The result is cached
    to r200_history_<ID>.csv so the tree is only read once.

Bookkeeping:
  far  = rate in separation bins beyond f * R200c(t). A bin that straddles
         the threshold contributes the fraction of its width (in log r)
         that lies beyond it.
  near = everything else: hist inside the threshold + inspiraled (at the
         centre) + no_track (separation unknown -- counted as near, and
         reported so you can check it is ~0).
  near + far == the total rate in tde_rates_<ID>_alpha<alpha>.csv.

Caveat: analysis.py clips separations beyond its r_max (default 1 Mpc) into
the last bin, so where f * R200c(t) reaches the last bin the split there is
ambiguous. The mass involved is reported per halo; if it is not negligible,
re-run analysis.py with a larger --r-max-kpc.
"""
import os
import warnings

import numpy as np
import pandas as pd


def main_branch_r200(halo_id, tree_dir, snap_redshift_path, cache_dir=None):
    """
    (z_snap, r200_kpc) of the main progenitor at every snapshot, sorted by
    increasing redshift. Read from r200_history_<ID>.csv in cache_dir if
    present, else built from the merger tree and written there.
    """
    cache_path = os.path.join(cache_dir, f"r200_history_{halo_id}.csv") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        from tree_navigator import MergerTreeNavigator  # needs h5py; only imported if required
        tree_path = os.path.join(tree_dir, f"sublink_full_{halo_id}.hdf5")
        if not os.path.exists(tree_path):
            raise FileNotFoundError(f"merger tree {tree_path} not found (set --tree-dir)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nav = MergerTreeNavigator(tree_path, snap_redshift_path, box_size_ckpc_h=35000.0)
            snaps = sorted(nav._main_branch_by_snap.keys())
            _, r200 = nav.main_branch_growth_history()  # same order as snaps
        z = np.array([nav._snap_to_redshift[s] for s in snaps])
        df = pd.DataFrame({"snap": snaps, "redshift": z, "r200c_kpc": r200})
        if cache_path:
            df.to_csv(cache_path, index=False)
    df = df.sort_values("redshift")
    return df["redshift"].to_numpy(float), df["r200c_kpc"].to_numpy(float)


def r200_at_redshift(z, z_snap, r200_snap):
    """Interpolate log R200c linearly in log(1+z); clamped outside the main branch's range."""
    return 10 ** np.interp(np.log10(1 + np.asarray(z)), np.log10(1 + z_snap), np.log10(r200_snap))


def split_rates(npz_path, age_to_redshift, z_snap, r200_snap, factor=3.0):
    """
    Returns (time_yr, near, far, diag) for one halo, rates in Msun/yr
    (= events/yr for 1 Msun stars). diag holds Msun totals for the log.
    """
    d = np.load(npz_path, allow_pickle=True)
    t = d["time_yr"]
    hist = d["hist_msunyr"].astype(float)
    edges = d["edges_kpc"]
    dt = float(np.median(np.diff(t)))
    total = hist.sum(axis=1) + d["inspiraled_msunyr"] + d["no_track_msunyr"]

    far = np.zeros_like(total)
    rows = np.nonzero(hist.any(axis=1))[0]
    r_cut = factor * r200_at_redshift(age_to_redshift(t[rows]), z_snap, r200_snap)
    le = np.log10(edges)
    lc = np.log10(r_cut)[:, None]
    frac_beyond = np.clip((le[1:] - lc) / (le[1:] - le[:-1]), 0.0, 1.0)
    far[rows] = (hist[rows] * frac_beyond).sum(axis=1)
    near = np.clip(total - far, 0.0, None)

    ambiguous = r_cut > edges[-2]  # threshold reaches the last (clipped-into) bin
    diag = {
        "total_msun": float(total.sum() * dt),
        "far_msun": float(far.sum() * dt),
        "no_track_msun": float(np.sum(d["no_track_msunyr"]) * dt),
        "inspiraled_msun": float(np.sum(d["inspiraled_msunyr"]) * dt),
        "ambiguous_msun": float(hist[rows[ambiguous], -1].sum() * dt),
    }
    return t, near, far, diag


def load_rates_by_distance(data_dir, halo_ids, alpha, age_to_redshift, tree_dir,
                           snap_redshift_path, factor=3.0, cache_dir=None):
    """
    Like plot_tde_rates_smoothed.load_rates, but returns
    (time_yr, near, far, used_ids), with near/far of shape (n_halos, n_times),
    or None if none of the halos' .npz files exist.
    """
    time_yr, nears, fars, used = None, [], [], []
    for halo_id in halo_ids:
        path = os.path.join(data_dir, f"tde_radial_{halo_id}_alpha{alpha}.npz")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping halo {halo_id}")
            continue
        z_snap, r200_snap = main_branch_r200(halo_id, tree_dir, snap_redshift_path, cache_dir)
        t, near, far, diag = split_rates(path, age_to_redshift, z_snap, r200_snap, factor)
        tot = diag["total_msun"]
        print(f"    {halo_id}: {diag['far_msun']:.3g} of {tot:.3g} Msun beyond {factor:g} R200c "
              f"({diag['far_msun'] / tot if tot > 0 else 0:.1%}); no track {diag['no_track_msun']:.3g}, "
              f"ambiguous (last radial bin) {diag['ambiguous_msun']:.3g}")
        if time_yr is None:
            time_yr = t
        elif len(t) != len(time_yr) or not np.allclose(t, time_yr):
            near = np.interp(time_yr, t, near, left=0.0, right=0.0)
            far = np.interp(time_yr, t, far, left=0.0, right=0.0)
        nears.append(near)
        fars.append(far)
        used.append(halo_id)
    if not used:
        return None
    return time_yr, np.vstack(nears), np.vstack(fars), used
