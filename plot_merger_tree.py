"""
Plot the SubLink merger tree stored in tng_download/sublink_full_<ID>.hdf5
(as produced by tng50_elliptical_dark_tree.py).

Usage:
    python plot_merger_tree.py <subhalo_id> [--tree-dir tng_download] [--out tree.png]

Example:
    python plot_merger_tree.py 467548
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LITTLE_H = 0.6774  # TNG's H0/100; used to convert masses to Msun


def load_tree(path):
    """Load only the fields needed for the tree structure and node styling."""
    fields = [
        "SubhaloID", "DescendantID", "FirstProgenitorID", "NextProgenitorID",
        "SnapNum", "SubhaloMassType",
    ]
    with h5py.File(path, "r") as f:
        missing = [k for k in fields if k not in f]
        if missing:
            raise KeyError(
                f"{path} is missing expected field(s) {missing}. "
                f"Available fields: {list(f.keys())}"
            )
        data = {k: f[k][()] for k in fields}
    return data


def build_tree_structure(data):
    """
    Map each row to its children (progenitor rows whose descendant is this row),
    and find the root (the row with DescendantID == -1, i.e. the z=0 object this
    tree was extracted for).
    """
    n = len(data["SubhaloID"])
    id_to_row = {int(sid): i for i, sid in enumerate(data["SubhaloID"])}
    children = defaultdict(list)
    root = None
    for i in range(n):
        desc_id = int(data["DescendantID"][i])
        if desc_id == -1:
            root = i
        else:
            desc_row = id_to_row.get(desc_id)
            if desc_row is not None:
                children[desc_row].append(i)
    if root is None:
        raise RuntimeError("Could not find root node (DescendantID == -1) in this tree.")
    return root, children, id_to_row


def total_mass(data, row):
    return float(np.sum(data["SubhaloMassType"][row]))


def assign_y_positions(data, root, children):
    """
    Simple dendrogram-style layout: leaves get sequential integer y-positions
    (via postorder traversal), internal nodes sit at the mean y of their
    children. Children are visited in order of descending mass so the most
    massive (main) branch stays visually grouped and crossings are reduced.
    """
    y_pos = {}
    leaf_counter = [0]

    def recurse(row):
        kids = children.get(row, [])
        if not kids:
            y_pos[row] = leaf_counter[0]
            leaf_counter[0] += 1
            return
        kids_sorted = sorted(kids, key=lambda r: total_mass(data, r), reverse=True)
        for k in kids_sorted:
            recurse(k)
        y_pos[row] = float(np.mean([y_pos[k] for k in kids_sorted]))

    recurse(root)
    return y_pos


def main_branch_rows(data, id_to_row, root):
    """Rows along the main progenitor branch (FirstProgenitorID chain), from root back in time."""
    rows = [root]
    row = root
    while True:
        fp_id = int(data["FirstProgenitorID"][row])
        if fp_id == -1:
            break
        row = id_to_row[fp_id]
        rows.append(row)
    return rows


def load_snapshot_redshifts(tree_dir):
    """
    Load the snapshot->redshift lookup table saved by the download script
    (tng_download/snapshot_redshifts.json). Returns a dict {snap_num: redshift},
    or None if the file isn't present.
    """
    path = os.path.join(tree_dir, "snapshot_redshifts.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        mapping = json.load(f)
    return {int(k): float(v) for k, v in mapping.items()}


def plot_tree(data, out_path, title=None, snap_to_z=None):
    root, children, id_to_row = build_tree_structure(data)
    y_pos = assign_y_positions(data, root, children)
    main_rows = set(main_branch_rows(data, id_to_row, root))

    n = len(data["SubhaloID"])
    masses_msun = np.array([total_mass(data, i) for i in range(n)]) * 1e10 / LITTLE_H
    log_mass = np.log10(np.clip(masses_msun, 1.0, None))  # avoid log(0)

    use_redshift = snap_to_z is not None
    if use_redshift:
        missing = sorted(set(int(s) for s in data["SnapNum"]) - set(snap_to_z.keys()))
        if missing:
            print(f"  warning: no redshift found for snapshot(s) {missing}; falling back to snapshot number for those.")
        x_all = np.array([
            snap_to_z.get(int(s), float(s)) for s in data["SnapNum"]
        ])
    else:
        print("  no snapshot_redshifts.json found next to the tree file -- plotting vs snapshot "
              "number instead. Re-run the download script (it now saves this lookup table) to get "
              "a redshift axis.")
        x_all = data["SnapNum"].astype(float)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw edges: progenitor -> descendant
    for i in range(n):
        desc_id = int(data["DescendantID"][i])
        if desc_id == -1:
            continue
        j = id_to_row.get(desc_id)
        if j is None:
            continue
        on_main = (i in main_rows) and (j in main_rows)
        ax.plot(
            [x_all[i], x_all[j]],
            [y_pos[i], y_pos[j]],
            color="crimson" if on_main else "0.75",
            linewidth=1.8 if on_main else 0.6,
            zorder=3 if on_main else 1,
        )

    # Draw nodes, sized/colored by total (all-particle-type) mass
    mass_range = max(log_mass.max() - log_mass.min(), 1e-6)
    sizes = 12 + 60 * (log_mass - log_mass.min()) / mass_range
    y = np.array([y_pos[i] for i in range(n)])
    sc = ax.scatter(
        x_all, y, c=log_mass, s=sizes, cmap="viridis",
        zorder=4, edgecolor="k", linewidth=0.1,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label(r"$\log_{10}(M_\mathrm{total} \, / \, M_\odot)$")

    if use_redshift:
        ax.set_xlabel("Redshift")
        ax.invert_xaxis()  # so z=0 (now) is on the right, high-z (past) on the left
    else:
        ax.set_xlabel("Snapshot Number  (higher = later / closer to $z=0$)")
    ax.set_ylabel("Branch (arbitrary vertical spacing)")
    ax.set_yticks([])
    ax.set_title(title or "SubLink merger tree")
    y_span = max(y.max() - y.min(), 1.0)
    ax.set_ylim(y.min() - 0.08 * y_span, y.max() + 0.08 * y_span)

    legend_elems = [
        Line2D([0], [0], color="crimson", lw=1.8, label="Main branch"),
        Line2D([0], [0], color="0.75", lw=0.6, label="Merging progenitors"),
    ]
    # Placed below the axes (not "upper left") so it never overlaps tree
    # nodes/edges, which can legitimately extend into any corner of the plot.
    ax.legend(
        handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=2, frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}  ({n} nodes total, {len(main_rows)} on the main branch)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "subhalo_id", type=int,
        help="The subhalo ID used in the tree filename (e.g. 467548 for sublink_full_467548.hdf5)",
    )
    parser.add_argument("--tree-dir", default="tng_download", help="Directory containing sublink_full_<ID>.hdf5")
    parser.add_argument("--out", default=None, help="Output image path (default: merger_tree_<ID>.png)")
    args = parser.parse_args()

    tree_path = os.path.join(args.tree_dir, f"sublink_full_{args.subhalo_id}.hdf5")
    if not os.path.exists(tree_path):
        sys.exit(
            f"Could not find {tree_path}.\n"
            f"Make sure you've downloaded the FULL tree (not just the mpb) for this subhalo, "
            f"and that --tree-dir points at the right folder."
        )

    out_path = args.out or f"merger_tree_{args.subhalo_id}.png"
    data = load_tree(tree_path)
    snap_to_z = load_snapshot_redshifts(args.tree_dir)
    plot_tree(data, out_path, title=f"Merger tree for subhalo {args.subhalo_id}", snap_to_z=snap_to_z)