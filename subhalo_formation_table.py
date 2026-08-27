"""
For a downloaded full SubLink merger tree (tng_download/sublink_full_<ID>.hdf5),
extract one row per distinct subhalo branch, giving:
  - the DM mass and radius at the highest-redshift (earliest) snapshot that
    branch existed at (i.e. its formation), and
  - delta_t: the cosmic time between that formation snapshot and the
    snapshot where the branch merges into another halo (or, if it never
    merges -- i.e. it's the main branch that survives to z=0 -- the time
    to the present day).

A "branch" here means a single trackable object: the chain of rows
connected by being each other's FirstProgenitorID, from its earliest
appearance (a "leaf" with no progenitor) forward until it either (a) stops
being the main/most-massive progenitor of its descendant (i.e. it merges
into something else -- the descendant continues as a *different* branch),
or (b) reaches the root of the tree (SnapNum=99 / z=0, i.e. it survived).

Requires: h5py, numpy, astropy
    pip install h5py numpy astropy

Usage:
    python subhalo_formation_table.py <subhalo_id> [--tree-dir tng_download] [--out subhalos.csv]

Example:
    python subhalo_formation_table.py 467548
"""

import argparse
import csv
import json
import os
import sys

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# TNG cosmological parameters (Planck-like, matches the simulation's own values)
LITTLE_H = 0.6774
COSMO = FlatLambdaCDM(H0=100 * LITTLE_H, Om0=0.3089)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_tree(path):
    """Load the tree-structure fields plus the physical fields we need."""
    wanted = [
        "SubhaloID", "DescendantID", "FirstProgenitorID", "SnapNum", "SubfindID",
        "SubhaloMass", "SubhaloHalfmassRad",
        "Group_M_Crit200", "Group_R_Crit200",
    ]
    with h5py.File(path, "r") as f:
        available = set(f.keys())
        missing = [k for k in wanted if k not in available]
        data = {k: f[k][()] for k in wanted if k in available}
    if missing:
        print(f"  note: fields not present in this tree file (will be filled with NaN): {missing}")
        n = len(data["SubhaloID"])
        for k in missing:
            data[k] = np.full(n, np.nan)
    return data


def load_snapshot_redshifts(tree_dir):
    path = os.path.join(tree_dir, "snapshot_redshifts.json")
    if not os.path.exists(path):
        sys.exit(
            f"Could not find {path}. This is created by tng50_elliptical_dark_tree.py "
            f"(save_snapshot_redshifts()) -- re-run that script, or copy the file here."
        )
    with open(path) as f:
        mapping = json.load(f)
    return {int(k): float(v) for k, v in mapping.items()}


# ----------------------------------------------------------------------
# Branch enumeration
# ----------------------------------------------------------------------
def enumerate_branches(data):
    """
    Returns a list of dicts, one per distinct subhalo branch, with the row
    index of its formation (earliest/leaf) point and the row index of its
    final point (where it merges into another branch, or the root if it
    survives to z=0).
    """
    n = len(data["SubhaloID"])
    id_to_row = {int(sid): i for i, sid in enumerate(data["SubhaloID"])}

    # A leaf is a row with no progenitor -- i.e. the formation point of a branch.
    leaves = [i for i in range(n) if int(data["FirstProgenitorID"][i]) == -1]

    branches = []
    for leaf in leaves:
        row = leaf
        merged = False
        merged_into_id = None
        while True:
            desc_id = int(data["DescendantID"][row])
            if desc_id == -1:
                # Reached the root -- this branch survives to the final snapshot.
                break
            desc_row = id_to_row.get(desc_id)
            if desc_row is None:
                # Descendant not present in this (sub)tree -- treat as branch end.
                break
            if int(data["FirstProgenitorID"][desc_row]) == int(data["SubhaloID"][row]):
                # Still the main/most-massive progenitor of its descendant --
                # same physical object, continue walking forward.
                row = desc_row
                continue
            else:
                # This row is a secondary progenitor of its descendant: it
                # merges into that (different, more massive) branch here.
                merged = True
                merged_into_id = int(data["SubhaloID"][desc_row])
                break

        branches.append({
            "formation_row": leaf,
            "final_row": row,
            "merged": merged,
            "merged_into_subhalo_id": merged_into_id,
        })

    return branches


# ----------------------------------------------------------------------
# Physical quantities
# ----------------------------------------------------------------------
def comoving_to_physical_kpc(radius_ckpc_h, redshift):
    """Convert comoving ckpc/h (TNG's SubhaloHalfmassRad/Group_R_Crit200 units) to physical kpc."""
    return radius_ckpc_h / LITTLE_H / (1.0 + redshift)


def build_output_rows(data, branches, snap_to_z):
    age_cache = {}

    def age_gyr(snap):
        if snap not in age_cache:
            z = snap_to_z[snap]
            age_cache[snap] = COSMO.age(z).value  # Gyr
        return age_cache[snap]

    rows = []
    for branch_idx, br in enumerate(branches):
        i = br["formation_row"]
        f = br["final_row"]

        snap_form = int(data["SnapNum"][i])
        snap_merge = int(data["SnapNum"][f])
        z_form = snap_to_z[snap_form]
        z_merge = snap_to_z[snap_merge]

        dm_mass_msun = data["SubhaloMass"][i] * 1e10 / LITTLE_H
        halfmass_rad_kpc = comoving_to_physical_kpc(data["SubhaloHalfmassRad"][i], z_form)
        group_m200_msun = data["Group_M_Crit200"][i] * 1e10 / LITTLE_H
        group_r200_kpc = comoving_to_physical_kpc(data["Group_R_Crit200"][i], z_form)

        delta_t_gyr = age_gyr(snap_merge) - age_gyr(snap_form)

        rows.append({
            "branch_id": branch_idx,
            "formation_subhalo_id": int(data["SubhaloID"][i]),
            "formation_subfind_id": int(data["SubfindID"][i]),
            "formation_snap": snap_form,
            "formation_redshift": z_form,
            "dm_mass_msun": dm_mass_msun,               # SubhaloMass: self-bound mass (all DM, in the Dark run)
            "halfmass_rad_kpc": halfmass_rad_kpc,        # SubhaloHalfmassRad, physical kpc
            "group_m_crit200_msun": group_m200_msun,     # host FOF group's virial mass at formation
            "group_r_crit200_kpc": group_r200_kpc,       # host FOF group's virial radius at formation, physical kpc
            "merge_subhalo_id": int(data["SubhaloID"][f]),
            "merge_snap": snap_merge,
            "merge_redshift": z_merge,
            "merged_into_subhalo_id": br["merged_into_subhalo_id"],
            "survived_to_z0": not br["merged"],
            "delta_t_gyr": delta_t_gyr,
        })

    return rows


def write_csv(rows, out_path):
    if not rows:
        raise RuntimeError("No branches found -- nothing to write.")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} subhalo branches to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "subhalo_id", type=int,
        help="The subhalo ID used in the tree filename (e.g. 467548 for sublink_full_467548.hdf5)",
    )
    parser.add_argument("--tree-dir", default="tng_download", help="Directory containing sublink_full_<ID>.hdf5")
    parser.add_argument("--out", default=None, help="Output CSV path (default: subhalo_formation_<ID>.csv)")
    args = parser.parse_args()

    tree_path = os.path.join(args.tree_dir, f"sublink_full_{args.subhalo_id}.hdf5")
    if not os.path.exists(tree_path):
        sys.exit(f"Could not find {tree_path}. Make sure the FULL tree (not just the mpb) has been downloaded.")

    out_path = args.out or f"subhalo_formation_{args.subhalo_id}.csv"

    print(f"Loading tree from {tree_path}...")
    data = load_tree(tree_path)
    snap_to_z = load_snapshot_redshifts(args.tree_dir)

    print("Enumerating subhalo branches...")
    branches = enumerate_branches(data)
    print(f"  found {len(branches)} distinct subhalo branches "
          f"({sum(1 for b in branches if b['merged'])} merge in, "
          f"{sum(1 for b in branches if not b['merged'])} survive to z=0 -- "
          f"normally just the 1 main branch of the tree).")

    rows = build_output_rows(data, branches, snap_to_z)
    write_csv(rows, out_path)