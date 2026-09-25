"""
Find the N (default 5) most dispersion-dominated elliptical galaxies in TNG50-1
(full physics), match each to its dark-matter-only counterpart halo in
TNG50-1-Dark, and download the merger trees for those halos only (no
full-catalog / full-simulation tree downloads).

Morphology is measured directly from each candidate's stellar particle cutout
(kappa_rot and the circularity distribution), because the TNG API does not
expose the stellar circularity supplementary catalog for TNG50-1 subhalos.

Requires: requests, h5py, numpy (>= 1.20)
    pip install requests h5py numpy

You need a free TNG API key: register at https://www.tng-project.org/users/register/
then find your key at https://www.tng-project.org/users/profile/
"""

import os
import json
import requests
import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
def load_api_key(path="apikey.txt"):
    """
    Reads your TNG API key from a local text file (not committed to git).
    The file should contain nothing but the key itself, e.g.:
        echo "abc123yourkeyhere" > apikey.txt
    Get a key at https://www.tng-project.org/users/profile/
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"Could not find '{path}'. Create it and put your TNG API key inside "
            f"(https://www.tng-project.org/users/profile/), e.g.:\n"
            f'  echo "your_key_here" > {path}'
        )
    with open(path) as f:
        key = f.read().strip()
    if not key:
        raise SystemExit(f"'{path}' is empty -- put your API key inside it.")
    return key


API_KEY = load_api_key()
BASE_URL = "https://www.tng-project.org/api/"
HEADERS = {"API-Key": API_KEY}

SIM_BARYONIC = "TNG50-1"
SIM_DARK = "TNG50-1-Dark"
SNAP_Z0 = 99  # z=0 (scale factor a = 1, so comoving == physical)
LITTLE_H = 0.6774  # TNG's H0/100

# Morphology measurement settings
APERTURE_KPC = 30.0       # stars within this 3D radius (physical kpc) are used
JCIRC_WINDOW = 101        # number of energy-neighbours used to estimate j_circ(E)
KAPPA_ROT_ELLIPTICAL = 0.5  # kappa_rot below this ~ dispersion-dominated

OUT_DIR = "tng_download"
os.makedirs(OUT_DIR, exist_ok=True)

# Stellar particle cutouts are large (up to ~100 MB each), so they go to
# scratch instead of OUT_DIR. Override with --cutout-dir or the
# TNG_CUTOUT_DIR environment variable.
CUTOUT_DIR = os.environ.get("TNG_CUTOUT_DIR", "/u/scratch/c/clairewi/tng_cutouts")


def api_get(path, params=None):
    """GET request against the TNG API, returns parsed JSON."""
    r = requests.get(BASE_URL + path, params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def api_download(path, out_path, params=None):
    """Stream-download a file (e.g. a merger tree or supplementary catalog) from the API."""
    if os.path.exists(out_path):
        print(f"  already have {out_path}")
        return out_path
    r = requests.get(BASE_URL + path, params=params, headers=HEADERS, stream=True)
    r.raise_for_status()
    content_type = r.headers.get("content-type", "")
    if "hdf5" not in content_type and "octet-stream" not in content_type:
        raise RuntimeError(
            f"Expected a binary file from {path!r} but got content-type "
            f"'{content_type}'. This usually means the URL/endpoint is wrong. "
            f"First bytes: {next(r.iter_content(chunk_size=200))!r}"
        )
    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    os.replace(tmp_path, out_path)  # only keep complete downloads
    print(f"  downloaded {out_path}")
    return out_path


# ----------------------------------------------------------------------
# STEP 1a -- stellar kinematic morphology from a particle cutout
# ----------------------------------------------------------------------
def download_star_cutout(sub_id, snap=SNAP_Z0):
    """Download the stellar particles of ONE subhalo (only the fields we need)."""
    os.makedirs(CUTOUT_DIR, exist_ok=True)
    path = os.path.join(CUTOUT_DIR, f"cutout_stars_{SIM_BARYONIC}_{snap}_{sub_id}.hdf5")
    api_download(
        f"{SIM_BARYONIC}/snapshots/{snap}/subhalos/{sub_id}/cutout.hdf5",
        path,
        params={"stars": "Coordinates,Velocities,Masses,Potential,GFM_StellarFormationTime"},
    )
    return path


def kinematic_morphology(pos_ckpc_h, vel_kms, mass, potential, center_ckpc_h,
                         box_size_ckpc_h, h=LITTLE_H, aperture_kpc=APERTURE_KPC,
                         window=JCIRC_WINDOW):
    """
    Kinematic morphology of a stellar system at z=0 (a=1).

    Inputs are raw snapshot units: positions in ckpc/h, velocities in km/s,
    potential in (km/s)^2. Stars within `aperture_kpc` of `center_ckpc_h` are used.

    Returns dict with:
      kappa_rot      -- Sales+2012: fraction of kinetic energy in ordered rotation
                        about the net angular momentum axis. Low => elliptical.
      f_circ07       -- mass fraction with circularity eps = j_z / j_circ(E) > 0.7
                        (same quantity as CircAbove07Frac). Low => elliptical.
      spheroid_frac  -- 2 x mass fraction with eps < 0 (counter-rotating
                        stars mirrored), a standard bulge/spheroid mass fraction.
      n_stars        -- number of stars used.
    """
    # periodic wrap, then to physical kpc (a = 1 at z = 0)
    dx = (pos_ckpc_h - center_ckpc_h + box_size_ckpc_h / 2.0) % box_size_ckpc_h \
        - box_size_ckpc_h / 2.0
    x = dx / h
    r = np.linalg.norm(x, axis=1)
    sel = r < aperture_kpc
    x, v, m, phi = x[sel], vel_kms[sel], mass[sel], potential[sel]
    if len(m) < 100:
        raise RuntimeError(f"Only {len(m)} stars inside {aperture_kpc} kpc.")

    # stellar bulk velocity (not the subhalo velocity, which includes DM)
    v = v - (m[:, None] * v).sum(axis=0) / m.sum()

    # specific angular momentum and rotation axis
    J = np.cross(x, v)
    L = (m[:, None] * J).sum(axis=0)
    zhat = L / np.linalg.norm(L)
    jz = J @ zhat

    # kappa_rot (Sales et al. 2012)
    z = x @ zhat
    R = np.sqrt(np.maximum((x ** 2).sum(axis=1) - z ** 2, 0.0))
    ok = R > 0
    kin_rot = (m[ok] * (jz[ok] / R[ok]) ** 2).sum()
    kin_tot = (m * (v ** 2).sum(axis=1)).sum()
    kappa_rot = kin_rot / kin_tot

    # circularity eps = jz / j_circ(E). j_circ(E) is estimated as the maximum
    # |j| among stars of similar binding energy (circular orbits maximise j at
    # fixed E), made monotonic in E.
    E = 0.5 * (v ** 2).sum(axis=1) + phi
    order = np.argsort(E)
    jmag_sorted = np.linalg.norm(J, axis=1)[order]
    window = min(window, len(jmag_sorted) | 1)  # odd, not larger than N
    half = window // 2
    padded = np.pad(jmag_sorted, half, mode="edge")
    jcirc_sorted = sliding_window_view(padded, window).max(axis=1)
    jcirc_sorted = np.maximum.accumulate(jcirc_sorted)
    jcirc = np.empty_like(jcirc_sorted)
    jcirc[order] = jcirc_sorted
    eps = jz / np.where(jcirc > 0, jcirc, np.inf)

    mtot = m.sum()
    return {
        "kappa_rot": float(kappa_rot),
        "f_circ07": float(m[eps > 0.7].sum() / mtot),
        "spheroid_frac": float(min(2.0 * m[eps < 0].sum() / mtot, 1.0)),
        "n_stars": int(len(m)),
    }


def measure_subhalo_morphology(sub, box_size_ckpc_h):
    """Download one subhalo's stellar cutout and compute its kinematic morphology."""
    path = download_star_cutout(sub["id"])
    with h5py.File(path, "r") as f:
        if "PartType4" not in f:
            raise RuntimeError(f"Cutout for subhalo {sub['id']} has no star particles.")
        s = f["PartType4"]
        pos = s["Coordinates"][()].astype(np.float64)
        vel = s["Velocities"][()].astype(np.float64)
        mass = s["Masses"][()].astype(np.float64)
        pot = s["Potential"][()].astype(np.float64)
        tform = s["GFM_StellarFormationTime"][()]

    real_stars = tform > 0  # drop wind-phase cells, which share PartType4
    center = np.array([sub["pos_x"], sub["pos_y"], sub["pos_z"]], dtype=np.float64)
    return kinematic_morphology(
        pos[real_stars], vel[real_stars], mass[real_stars], pot[real_stars],
        center, box_size_ckpc_h,
    )


# ----------------------------------------------------------------------
# STEP 1 -- find a prototypical elliptical in TNG50-1 at z=0
# ----------------------------------------------------------------------
# "Prototypical elliptical" here = massive, quenched (low SFR), central, and
# kinematically hot / dispersion-supported. The last criterion is measured
# from each candidate's star particles (see kinematic_morphology), and the
# candidates are ranked from lowest to highest kappa_rot.
def find_prototypical_elliptical(mass_min_msun=1e11, mass_max_msun=None):
    """
    Search TNG50-1 at z=0 for elliptical candidates with total stellar
    mass in [mass_min_msun, mass_max_msun] (mass_max_msun=None means no
    upper bound).

    Returns a list of (subhalo_id, mass_stars_msun, sfr, morph_dict) tuples,
    sorted from most to least dispersion-dominated (ascending kappa_rot).
    """
    lo_str = f"{mass_min_msun:.3e}"
    hi_str = f"{mass_max_msun:.3e}" if mass_max_msun is not None else "no upper bound"
    print(f"Searching TNG50-1 subhalo catalog for a prototypical elliptical "
          f"with M_star in [{lo_str}, {hi_str}] Msun...")

    # Group catalog masses are in code units of 1e10 Msun/h
    mass_min_code = mass_min_msun * LITTLE_H / 1e10

    search_params = {
        "mass_stars__gt": mass_min_code,
        "sfr__lt": 0.1,                                # essentially quenched
        "primary_flag": 1,                             # central galaxy, not a satellite
        "order_by": "-mass_stars",
        "limit": 20,
    }
    if mass_max_msun is not None:
        mass_max_code = mass_max_msun * LITTLE_H / 1e10
        search_params["mass_stars__lt"] = mass_max_code

    result = api_get(f"{SIM_BARYONIC}/snapshots/{SNAP_Z0}/subhalos/", params=search_params)
    candidates = result["results"]
    if not candidates:
        raise RuntimeError(
            "No candidates found in that mass range -- widen mass_min_msun/mass_max_msun."
        )

    box_size = api_get(SIM_BARYONIC)["boxsize"]  # ckpc/h

    print(f"Measuring stellar kinematics for {len(candidates)} candidates "
          f"(downloads one star cutout per candidate into {CUTOUT_DIR}; "
          f"the most massive can be ~100 MB)...")
    rows = []
    for cand in candidates:
        sub_id = cand["id"]
        # the search endpoint only returns id/mass_log_msun/url, so fetch the detail
        sub = api_get(f"{SIM_BARYONIC}/snapshots/{SNAP_Z0}/subhalos/{sub_id}/")
        mass_stars_msun = sub["mass_stars"] * 1e10 / LITTLE_H
        try:
            morph = measure_subhalo_morphology(sub, box_size)
        except Exception as e:  # keep going if one cutout fails
            print(f"  subhalo {sub_id}: morphology failed ({e}); skipping")
            continue
        rows.append((sub_id, mass_stars_msun, sub.get("sfr", float("nan")), morph))
        print(f"  subhalo {sub_id}: M_star={mass_stars_msun:.3e} Msun, "
              f"SFR={sub.get('sfr', float('nan')):.3f}, "
              f"kappa_rot={morph['kappa_rot']:.3f}, "
              f"f(eps>0.7)={morph['f_circ07']:.3f}, "
              f"spheroid_frac={morph['spheroid_frac']:.3f}, "
              f"N_star={morph['n_stars']}")

    if not rows:
        raise RuntimeError("Could not measure morphology for any candidate.")

    # rank from most dispersion-dominated (lowest kappa_rot) to least
    rows.sort(key=lambda r: r[3]["kappa_rot"])

    n_ell = sum(r[3]["kappa_rot"] < KAPPA_ROT_ELLIPTICAL for r in rows)
    print(f"  {n_ell}/{len(rows)} candidates have kappa_rot < {KAPPA_ROT_ELLIPTICAL} "
          f"(dispersion-dominated).")

    # save the full ranked table so the selection is reproducible / citable
    table_path = os.path.join(OUT_DIR, "elliptical_candidates_ranked.json")
    with open(table_path, "w") as f:
        json.dump([{"subhalo_id": int(sid), "mass_stars_msun": float(ms),
                    "sfr_msun_yr": float(sfr), **morph}
                   for sid, ms, sfr, morph in rows], f, indent=2)
    print(f"  saved ranked candidate table to {table_path}")

    print("  Ranking by kappa_rot (lowest = most elliptical):")
    for rank, (sid, ms, _, morph) in enumerate(rows, start=1):
        flag = "" if morph["kappa_rot"] < KAPPA_ROT_ELLIPTICAL else "  (rotation-supported!)"
        print(f"    #{rank}: subhalo {sid}, M_star={ms:.3e} Msun, "
              f"kappa_rot={morph['kappa_rot']:.3f}, f(eps>0.7)={morph['f_circ07']:.3f}{flag}")

    return rows


# ----------------------------------------------------------------------
# STEP 2 -- match this subhalo to its counterpart in TNG50-1-Dark
# ----------------------------------------------------------------------
def match_to_dark(subhalo_id_baryonic, snap=SNAP_Z0):
    """
    Downloads the 'subhalo_matching_to_dark.hdf5' supplementary catalog for
    TNG50-1 (once; it is reused/cached on disk) and looks up the matched
    subhalo index in TNG50-1-Dark for the given full-physics subhalo ID.
    """
    print("Downloading/using the baryonic<->dark matching catalog...")
    match_path = os.path.join(OUT_DIR, "subhalo_matching_to_dark.hdf5")
    api_download(f"{SIM_BARYONIC}/files/subhalo_matching_to_dark.hdf5", match_path)

    with h5py.File(match_path, "r") as f:
        # TNG (not original-Illustris) layout: group "Snapshot_N" containing
        # two arrays, SubhaloIndexDark_LHaloTree and SubhaloIndexDark_SubLink.
        # We use the SubLink-based one for consistency with the SubLink trees
        # we're downloading elsewhere in this script.
        dark_inds = f[f"Snapshot_{snap}/SubhaloIndexDark_SubLink"][()]

    dark_id = int(dark_inds[subhalo_id_baryonic])
    if dark_id < 0:
        raise RuntimeError(
            f"Subhalo {subhalo_id_baryonic} has no bijective match in {SIM_DARK} "
            f"(matching value is {dark_id}). Pick a different / more massive candidate."
        )
    print(f"-> TNG50-1 subhalo {subhalo_id_baryonic} matches "
          f"{SIM_DARK} subhalo {dark_id} at snapshot {snap}.")
    return dark_id


# ----------------------------------------------------------------------
# STEP 3 -- download the merger tree for ONLY that one dark-matter halo
# ----------------------------------------------------------------------
def download_merger_tree(dark_subhalo_id, snap=SNAP_Z0, full_tree=True):
    print(f"Downloading merger tree for {SIM_DARK} subhalo {dark_subhalo_id}...")

    subhalo_url = f"{SIM_DARK}/snapshots/{snap}/subhalos/{dark_subhalo_id}/"

    # Main progenitor branch only (mass growth history along the main line)
    mpb_path = os.path.join(OUT_DIR, f"sublink_mpb_{dark_subhalo_id}.hdf5")
    api_download(subhalo_url + "sublink/mpb.hdf5", mpb_path)

    if full_tree:
        # Full subtree, including all merging branches (larger file)
        full_path = os.path.join(OUT_DIR, f"sublink_full_{dark_subhalo_id}.hdf5")
        api_download(subhalo_url + "sublink/full.hdf5", full_path)
        return mpb_path, full_path

    return mpb_path, None


# ----------------------------------------------------------------------
# STEP 3b -- save a local snapshot -> redshift lookup table (used later by
# the plotting script, so it can show redshift on the x-axis without
# needing network access itself)
# ----------------------------------------------------------------------
def save_snapshot_redshifts(sim=SIM_DARK, out_name="snapshot_redshifts.json"):
    snaps = api_get(f"{sim}/snapshots/")
    mapping = {str(s["number"]): s["redshift"] for s in snaps}
    path = os.path.join(OUT_DIR, out_name)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  saved snapshot->redshift lookup to {path}")
    return path


# ----------------------------------------------------------------------
# STEP 4 -- (optional) quick sanity check / plot of the mass growth history
# ----------------------------------------------------------------------
def summarize_mpb(mpb_path):
    with h5py.File(mpb_path, "r") as f:
        snaps = f["SnapNum"][()]
        mass = f["Mass"][()] * 1e10 / LITTLE_H  # to Msun
    print(f"\nMain branch of the merger tree ({mpb_path}):")
    print(f"  {len(snaps)} snapshots, from snap {snaps.min()} to {snaps.max()}")
    print(f"  NOTE: masses below are the matched DARK-MATTER-ONLY halo's total mass "
          f"(dominated by DM), NOT the baryonic galaxy's stellar mass -- these are "
          f"expected to be much larger (typically ~50-100x) than M_star.")
    print(f"  z=0 DM halo mass: {mass[0]:.3e} Msun")
    print(f"  earliest traced DM halo mass: {mass[-1]:.3e} Msun")


# ----------------------------------------------------------------------
# STEP 5 -- extract, per snapshot, the positions/velocities/masses of ALL
# progenitors relative to the main-branch (central) progenitor
# ----------------------------------------------------------------------
def load_full_tree_fields(full_tree_path):
    """Load the fields needed for the dynamics from the full SubLink tree file."""
    fields = [
        "SubhaloID", "SnapNum", "DescendantID",
        "FirstProgenitorID", "NextProgenitorID",
        "SubhaloPos", "SubhaloVel", "SubhaloMassType", "Group_M_Crit200",
    ]
    with h5py.File(full_tree_path, "r") as f:
        data = {k: f[k][()] for k in fields}
    return data


def relative_progenitor_kinematics(full_tree_path, box_size_ckpc_h, h=0.6774):
    """
    Walk the full SubLink subtree and, for every snapshot, return the
    positions/velocities/masses of every progenitor in that snapshot's
    branch, expressed RELATIVE to the main-branch (central) progenitor
    at that same snapshot.

    Returns: dict {snap_num: list of dicts with keys
                   'pos_rel_kpc' (3,), 'vel_rel_kms' (3,), 'mass_msun',
                   'is_central' (bool), 'subhalo_id'}
    """
    data = load_full_tree_fields(full_tree_path)
    n = len(data["SubhaloID"])
    id_to_row = {sid: i for i, sid in enumerate(data["SubhaloID"])}

    # index the main-branch (central) progenitor at each snapshot by
    # walking FirstProgenitorID from the root (row 0 in a SubLink subtree
    # file is always the root of the subtree, e.g. the z=0 subhalo).
    central_row_at_snap = {}
    row = 0
    while row != -1 and row is not None:
        snap = int(data["SnapNum"][row])
        central_row_at_snap[snap] = row
        fp_id = data["FirstProgenitorID"][row]
        row = id_to_row.get(fp_id, -1) if fp_id != -1 else -1

    def wrap(delta, box):
        # periodic boundary wrap into [-box/2, box/2]
        return (delta + box / 2.0) % box - box / 2.0

    out = {}
    for i in range(n):
        snap = int(data["SnapNum"][i])
        if snap not in central_row_at_snap:
            continue  # no central identified at this snapshot (shouldn't normally happen)
        c = central_row_at_snap[snap]

        pos_rel = wrap(data["SubhaloPos"][i] - data["SubhaloPos"][c], box_size_ckpc_h) / h  # -> kpc
        vel_rel = data["SubhaloVel"][i] - data["SubhaloVel"][c]  # km/s, no conversion needed
        mass_msun = data["SubhaloMassType"][i].sum() * 1e10 / h  # total mass, all types -> Msun

        out.setdefault(snap, []).append({
            "subhalo_id": int(data["SubhaloID"][i]),
            "pos_rel_kpc": pos_rel,
            "vel_rel_kms": vel_rel,
            "mass_msun": float(mass_msun),
            "is_central": (i == c),
        })

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find the N most dispersion-dominated ellipticals in TNG50-1, match "
                    "them to TNG50-1-Dark, and download their merger trees."
    )
    parser.add_argument(
        "--mass-min", type=float, default=1e11,
        help="Minimum total stellar mass in Msun (default: 1e11)",
    )
    parser.add_argument(
        "--mass-max", type=float, default=None,
        help="Maximum total stellar mass in Msun (default: no upper bound)",
    )
    parser.add_argument(
        "--cutout-dir", default=CUTOUT_DIR,
        help=f"Directory for the stellar particle cutouts (default: {CUTOUT_DIR})",
    )
    parser.add_argument(
        "--n-select", type=int, default=5,
        help="Number of best (lowest kappa_rot) candidates to download merger trees for "
             "(default: 5)",
    )
    args = parser.parse_args()
    if args.n_select < 1:
        raise SystemExit("--n-select must be at least 1.")
    CUTOUT_DIR = args.cutout_dir  # module-level global, read by download_star_cutout
    print(f"Stellar cutouts will be saved in: {os.path.abspath(CUTOUT_DIR)}")

    if args.mass_max is not None and args.mass_max <= args.mass_min:
        raise SystemExit(f"--mass-max ({args.mass_max:.3e}) must be greater than "
                          f"--mass-min ({args.mass_min:.3e}).")

    ranked = find_prototypical_elliptical(
        mass_min_msun=args.mass_min, mass_max_msun=args.mass_max,
    )

    # Walk down the kappa_rot ranking and keep the best N candidates that have
    # a counterpart in TNG50-1-Dark (a candidate without a bijective match is
    # skipped and the next-ranked one is used instead).
    selected = []  # list of dicts, one per downloaded galaxy
    for sub_id, mass_stars, sfr, morph in ranked:
        if len(selected) == args.n_select:
            break
        try:
            dark_id = match_to_dark(sub_id)
        except RuntimeError as e:
            print(f"  skipping subhalo {sub_id}: {e}")
            continue
        selected.append({"subhalo_id": int(sub_id), "dark_subhalo_id": int(dark_id),
                         "mass_stars_msun": float(mass_stars), **morph})

    if len(selected) < args.n_select:
        print(f"  WARNING: only {len(selected)} of the requested {args.n_select} "
              f"candidates could be matched to {SIM_DARK}.")
    n_rot = sum(s["kappa_rot"] >= KAPPA_ROT_ELLIPTICAL for s in selected)
    if n_rot:
        print(f"  WARNING: {n_rot} of the selected galaxies have kappa_rot >= "
              f"{KAPPA_ROT_ELLIPTICAL}, i.e. they are not clearly ellipticals. "
              f"Consider widening the mass range or lowering --n-select.")

    save_snapshot_redshifts()

    # BoxSize for TNG50 is 35000 ckpc/h (35 Mpc/h); fetch it from the API
    # instead of hardcoding, in case you point this at a different box.
    box_size = api_get(SIM_DARK)["boxsize"]  # ckpc/h

    for rank, s in enumerate(selected, start=1):
        print(f"\n=== Galaxy {rank}/{len(selected)}: TNG50-1 subhalo {s['subhalo_id']} "
              f"-> {SIM_DARK} subhalo {s['dark_subhalo_id']} "
              f"(kappa_rot = {s['kappa_rot']:.3f}) ===")
        mpb_path, full_path = download_merger_tree(s["dark_subhalo_id"], full_tree=True)
        s["mpb_path"], s["full_tree_path"] = mpb_path, full_path
        summarize_mpb(mpb_path)

        kinematics = relative_progenitor_kinematics(full_path, box_size_ckpc_h=box_size)
        n_prog = sum(len(v) for v in kinematics.values())
        print(f"  full tree: {n_prog} subhalo entries over {len(kinematics)} snapshots")

    # record of which galaxies were selected and where their trees are
    sel_path = os.path.join(OUT_DIR, "selected_ellipticals.json")
    with open(sel_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved the selected galaxies to {sel_path}")

    print("Done. Files are in:", os.path.abspath(OUT_DIR))