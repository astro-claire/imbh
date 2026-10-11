"""
Merger-tree checks for the TNG50-1-Dark SubLink trees used by imbh.py.

check_tree(path) returns a dict with a pass/fail verdict and diagnostics:

REJECTING problems (the galaxy's assembly history in the tree is unusable):
  * root_is_central False -- the matched dark subhalo is a SATELLITE at z=0
    (e.g. its FoF group was joined to a much larger one in the dark run), so
    the "galaxy" being modeled sits inside another halo.
  * max_unexplained_frac > max_unexplained_frac -- somewhere along the main
    branch the descendant's mass exceeds ALL the mass in the tree at the
    previous one or two snapshots (two, to allow SubLink's skipped links) by
    more than this fraction of the z=0 subhalo mass. Real mergers and smooth
    accretion are accounted for by mass already in the tree; a large
    unexplained jump means the dominant progenitor's history is missing --
    a broken main-branch link (e.g. 1044309: 1e7 -> 1e11 Msun between
    snapshots 85 and 86) or a central/satellite identity swap (e.g. 467548:
    a 2.8e12 satellite inherits a 1.3e13 group at snapshot 92).

REPORTED only (handled by the pipeline or physical):
  * main_branch_gaps -- snapshots the main branch skips; filled with an
    interpolated stand-in by MergerTreeNavigator (fill_main_branch_gaps).
  * satellite_snaps -- snapshots where the main branch is a satellite of a
    central that is NOT in the tree; usually a fly-through of a larger group
    (the galaxy's own history is complete).
"""
import h5py
import numpy as np

LITTLE_H = 0.6774
MAX_UNEXPLAINED_FRAC = 0.1
FIELDS = ["SnapNum", "SubhaloID", "FirstProgenitorID", "SubhaloMass", "Group_M_Crit200",
          "GroupFirstSub", "SubfindID"]


def _ranges(snaps):
    """[3,4,5,9] -> '3-5, 9'"""
    if not snaps:
        return ""
    out, start, prev = [], snaps[0], snaps[0]
    for s in list(snaps[1:]) + [None]:
        if s != prev + 1 if s is not None else True:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = s
        prev = s if s is not None else prev
    return ", ".join(out)


def check_tree(path, max_unexplained_frac=MAX_UNEXPLAINED_FRAC):
    with h5py.File(path, "r") as f:
        d = {k: f[k][()] for k in FIELDS}
    snap, ids, fp = d["SnapNum"].astype(int), d["SubhaloID"], d["FirstProgenitorID"]
    m = d["SubhaloMass"] * 1e10 / LITTLE_H
    gm = d["Group_M_Crit200"] * 1e10 / LITTLE_H
    row = {int(i): k for k, i in enumerate(ids)}

    # main branch from the root (row 0), then in increasing snapshot order
    k, mb = 0, []
    while True:
        mb.append(k)
        if fp[k] == -1:
            break
        k = row[int(fp[k])]
    mb = mb[::-1]
    m_root = float(m[0])
    root_is_central = bool(d["GroupFirstSub"][0] == d["SubfindID"][0])

    # total mass in the tree at each snapshot (everything there ends up in the root)
    total = np.bincount(snap, weights=m, minlength=snap.max() + 1)
    worst = (0.0, None, None, 0.0, 0.0)          # (frac, snap_a, snap_b, M_b, available)
    for a, b in zip(mb[:-1], mb[1:]):
        sa, sb = int(snap[a]), int(snap[b])
        avail = max(total[s] for s in (sb - 1, sb - 2) if s >= sa)
        frac = (m[b] - avail) / m_root
        if frac > worst[0]:
            worst = (float(frac), sa, sb, float(m[b]), float(avail))

    mb_snaps = [int(snap[k]) for k in mb]
    gaps = sorted(set(range(mb_snaps[0], mb_snaps[-1] + 1)) - set(mb_snaps))
    in_tree = set(zip(snap.tolist(), d["SubfindID"].tolist()))
    sat = [(int(snap[k]), float(m[k]), float(gm[k])) for k in mb
           if d["GroupFirstSub"][k] != d["SubfindID"][k]
           and (int(snap[k]), int(d["GroupFirstSub"][k])) not in in_tree]

    reasons = []
    if not root_is_central:
        reasons.append(f"dark counterpart is a satellite at z=0 (its group M200c = {gm[0]:.2e} Msun)")
    if worst[0] > max_unexplained_frac:
        reasons.append(f"missing history: main branch gains {worst[3]:.2e} Msun at snap {worst[1]}->{worst[2]} "
                       f"with only {worst[4]:.2e} Msun in the tree before ({worst[0]:.0%} of the z=0 mass "
                       f"unaccounted for)")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "root_mass_msun": m_root,
        "root_is_central": root_is_central,
        "max_unexplained_frac": worst[0],
        "max_unexplained_snaps": [worst[1], worst[2]],
        "main_branch_first_snap": mb_snaps[0],
        "main_branch_gaps": gaps,
        "satellite_snaps": [s for s, _, _ in sat],
        "n_rows": int(len(snap)),
    }


def describe(name, chk):
    """Multi-line human-readable summary of check_tree's result."""
    lines = [f"{name}: {'PASS' if chk['ok'] else 'FAIL'} "
             f"(z=0 mass {chk['root_mass_msun']:.2e} Msun, {chk['n_rows']} rows, main branch from snap "
             f"{chk['main_branch_first_snap']}; largest unexplained gain {chk['max_unexplained_frac']:.1%})"]
    lines += [f"    REJECT: {r}" for r in chk["reasons"]]
    if chk["main_branch_gaps"]:
        lines.append(f"    note: main branch skips snapshot(s) {chk['main_branch_gaps']} "
                     f"(filled by MergerTreeNavigator)")
    if chk["satellite_snaps"]:
        swap = (not chk["ok"] and chk["max_unexplained_snaps"][0] in chk["satellite_snaps"])
        lines.append(f"    note: satellite of a group outside the tree at snapshots "
                     f"{_ranges(chk['satellite_snaps'])} "
                     + ("(ends in the identity swap flagged above)" if swap
                        else "(fly-through of a larger group; own history complete)"))
    return "\n".join(lines)
