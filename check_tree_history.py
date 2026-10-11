"""
Check each SubLink tree for missing assembly history: snapshots where the
main branch is a SATELLITE whose FoF group's central is NOT in the tree.
Uses the snapshot-catalog indices (GroupFirstSub vs SubfindID at the same
snapshot); the tree's FirstSubhaloInFOFGroupID is not reliable in these
downloads (offset by the tree length at some snapshots).
The group's dominant halo and all its progenitors are then absent, so their
clusters are never seeded. Prints, per tree, the snapshot ranges affected and
the main branch's mass relative to its group's M200c there.

    python check_tree_history.py tng_download/sublink_full_*.hdf5
"""
import sys
import h5py
import numpy as np

LITTLE_H = 0.6774
for path in sys.argv[1:]:
    with h5py.File(path, "r") as f:
        snap, ids, fp = f["SnapNum"][()], f["SubhaloID"][()], f["FirstProgenitorID"][()]
        gfs, sfid = f["GroupFirstSub"][()], f["SubfindID"][()]
        m = f["SubhaloMass"][()] * 1e10 / LITTLE_H
        gm = f["Group_M_Crit200"][()] * 1e10 / LITTLE_H
    # (snapshot, SubfindID) of every tree member, to ask "is this group's central in the tree?"
    in_tree = set(zip(snap.tolist(), sfid.tolist()))
    row = {int(i): k for k, i in enumerate(ids)}
    k, mb = 0, []
    while True:
        mb.append(k)
        if fp[k] == -1:
            break
        k = row[int(fp[k])]
    halo = path.split("_")[-1].replace(".hdf5", "")
    bad = [(int(snap[k]), m[k], gm[k]) for k in mb
           if gfs[k] != sfid[k] and (int(snap[k]), int(gfs[k])) not in in_tree]
    if not bad:
        print(f"{halo:>8s}: OK -- main branch is central (or its central is in the tree) at every snapshot")
        continue
    snaps = sorted(s for s, _, _ in bad)
    ranges, start = [], snaps[0]
    for a, b in zip(snaps, snaps[1:] + [None]):
        if b != a + 1:
            ranges.append(f"{start}-{a}" if a != start else f"{a}")
            if b is not None:
                start = b
    worst = min(bad, key=lambda x: x[1] / x[2] if x[2] > 0 else np.inf)
    print(f"{halo:>8s}: main branch is a satellite of a central OUTSIDE the tree at {len(bad)} snapshots "
          f"({', '.join(ranges)}); worst: snap {worst[0]}, M_sub {worst[1]:.2e} vs group M200c {worst[2]:.2e}")
