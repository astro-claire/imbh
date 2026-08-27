"""
Find a prototypical elliptical galaxy in TNG50-1 (full physics), match it to its
dark-matter-only counterpart halo in TNG50-1-Dark, and download the merger tree
for that ONE halo only (no full-catalog / full-tree downloads).

Requires: requests, h5py, numpy
    pip install requests h5py numpy

You need a free TNG API key: register at https://www.tng-project.org/users/register/
then find your key at https://www.tng-project.org/users/profile/
"""

import os
import json
import requests
import h5py
import numpy as np

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
SNAP_Z0 = 99  # z=0
LITTLE_H = 0.6774  # TNG's H0/100

OUT_DIR = "tng_download"
os.makedirs(OUT_DIR, exist_ok=True)


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
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"  downloaded {out_path}")
    return out_path


# ----------------------------------------------------------------------
# STEP 1 -- find a prototypical elliptical in TNG50-1 at z=0
# ----------------------------------------------------------------------
# "Prototypical elliptical" here = massive, quenched (low sSFR), and
# kinematically hot / dispersion-supported (using the stellar circularity
# supplementary catalog field 'CircAbove07Frac', the fraction of stellar mass
# on near-circular orbits -- LOW values mean spheroidal/dispersion-dominated).
#
# Adjust these thresholds to taste.
def find_prototypical_elliptical(mass_min_msun=1e11, mass_max_msun=None):
    """
    Search TNG50-1 at z=0 for a prototypical elliptical with total stellar
    mass in [mass_min_msun, mass_max_msun] (mass_max_msun=None means no
    upper bound).
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

    # Refine using stellar circularity (morphology) for each candidate, since
    # this field isn't queryable directly through the simple search endpoint.
    # NOTE: the lightweight search-results endpoint only returns id/mass_log_msun/url,
    # not mass_stars -- so we fetch the per-subhalo detail endpoint for every
    # candidate anyway, and read the (reliable) stellar mass from there.
    best = None
    chosen_mass_msun = {}
    for cand in candidates:
        sub_id = cand["id"]
        sub = api_get(f"{SIM_BARYONIC}/snapshots/{SNAP_Z0}/subhalos/{sub_id}/")
        mass_stars_msun = sub["mass_stars"] * 1e10 / LITTLE_H
        chosen_mass_msun[sub_id] = mass_stars_msun

        supp = sub.get("supplementary_data", {}) or {}
        circ = supp.get("stellar_circs", {}) or {}
        circ_frac = circ.get("CircAbove07Frac")  # fraction of stars on circular (disky) orbits

        print(f"  subhalo {sub_id}: M_star={mass_stars_msun:.3e} Msun, "
              f"SFR={sub.get('sfr', float('nan')):.3f}, "
              f"CircAbove07Frac={'n/a' if circ_frac is None else f'{circ_frac:.3f}'}")

        if circ_frac is None:
            continue
        # low circ_frac => spheroidal / elliptical-like
        if best is None or circ_frac < best[1]:
            best = (sub_id, circ_frac)

    if best is None:
        print("  circularity catalog not resolvable via API for these candidates; "
              "falling back to the single most massive quenched central.")
        chosen_id = candidates[0]["id"]
    else:
        chosen_id = best[0]

    print(f"-> Selected TNG50-1 subhalo ID {chosen_id} "
          f"(M_star = {chosen_mass_msun[chosen_id]:.3e} Msun) as the prototypical elliptical.")
    return chosen_id


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
        description="Find a prototypical elliptical in TNG50-1, match it to TNG50-1-Dark, "
                    "and download its merger tree."
    )
    parser.add_argument(
        "--mass-min", type=float, default=1e11,
        help="Minimum total stellar mass in Msun (default: 1e11)",
    )
    parser.add_argument(
        "--mass-max", type=float, default=None,
        help="Maximum total stellar mass in Msun (default: no upper bound)",
    )
    args = parser.parse_args()

    if args.mass_max is not None and args.mass_max <= args.mass_min:
        raise SystemExit(f"--mass-max ({args.mass_max:.3e}) must be greater than "
                          f"--mass-min ({args.mass_min:.3e}).")

    elliptical_id = find_prototypical_elliptical(
        mass_min_msun=args.mass_min, mass_max_msun=args.mass_max,
    )
    dark_id = match_to_dark(elliptical_id)
    mpb_path, full_path = download_merger_tree(dark_id, full_tree=True)
    summarize_mpb(mpb_path)
    save_snapshot_redshifts()

    # BoxSize for TNG50 is 35000 ckpc/h (35 Mpc/h); fetch it from the API
    # instead of hardcoding, in case you point this at a different box.
    sim_meta = api_get(SIM_DARK)
    box_size = sim_meta["boxsize"]  # ckpc/h

    kinematics = relative_progenitor_kinematics(full_path, box_size_ckpc_h=box_size)
    example_snap = max(kinematics.keys())
    print(f"\nProgenitors at snapshot {example_snap} (relative to central):")
    for entry in kinematics[example_snap]:
        tag = "CENTRAL" if entry["is_central"] else "progenitor"
        print(f"  [{tag}] id={entry['subhalo_id']}, "
              f"mass={entry['mass_msun']:.3e} Msun, "
              f"pos_rel={entry['pos_rel_kpc']} kpc, "
              f"vel_rel={entry['vel_rel_kms']} km/s")

    print("\nDone. Files are in:", os.path.abspath(OUT_DIR))