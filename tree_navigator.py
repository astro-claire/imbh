"""
Direct navigator over a raw SubLink FULL-TREE HDF5 file (the same
sublink_full_<ID>.hdf5 produced by tng50_elliptical_dark_tree.py), used to
follow a single object (e.g. a star cluster's current "host" subhalo)
forward through the tree across merger events.

Why this needs the RAW tree and not the subhalo_formation_table.py summary
CSV: when host A merges into host B, we need B's mass/radius AT THE EXACT
SNAPSHOT of the merge -- not at B's own (possibly much earlier) formation
time, which is all the summary CSV records. The raw tree has a row for B
at every snapshot it existed, so we can look up its state at precisely the
snapshot we need, and likewise get A's position/velocity relative to B at
that same moment.
"""

import h5py
import json
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

LITTLE_H = 0.6774

# Cosmology for the TNG merger tree specifically (NOT the same as whatever
# cosmology the `timescales`/high-res-cluster-simulation pipeline uses for
# the IMBH growth model itself -- those are two different simulations with
# their own cosmologies; don't conflate them).
TNG_COSMO = FlatLambdaCDM(H0=100 * LITTLE_H, Om0=0.3089)


HOST_MASS_MODELS = ("own", "group")


class MergerTreeNavigator:
    def __init__(self, tree_path, snapshot_redshift_path, box_size_ckpc_h, cosmo=TNG_COSMO,
                 host_mass_model="own", fill_main_branch_gaps=True, repair_main_branch_glitches=False,
                 glitch_mass_ratio=5.0):
        """
        Parameters:
            tree_path (str): path to sublink_full_<ID>.hdf5
            snapshot_redshift_path (str): path to snapshot_redshifts.json
                (produced by tng50_elliptical_dark_tree.py's save_snapshot_redshifts())
            box_size_ckpc_h (float): simulation box size, comoving ckpc/h
                (same convention as elsewhere in this pipeline)
            cosmo: astropy cosmology to use for age(z) lookups
            host_mass_model (str): what host_properties() returns for a
                SATELLITE subhalo (centrals get their FoF group's
                Group_M_Crit200/Group_R_Crit200 either way):
                  'own' (default): the satellite's OWN bound mass
                      (SubhaloMass) and an R200c-equivalent radius derived
                      from it at that snapshot's redshift. The FoF group it
                      sits in is supplied separately, as a background
                      potential (see satellite_group_background).
                  'group': the legacy behavior -- the FoF group's
                      Group_M_Crit200/Group_R_Crit200, i.e. the mass of the
                      (much larger) halo the satellite is orbiting inside.
            fill_main_branch_gaps (bool): SubLink sometimes links the main
                branch across a snapshot (e.g. snap 3 -> 5), leaving no
                main-branch subhalo at the skipped snapshot. With True
                (default), each skipped snapshot gets a VIRTUAL main-branch
                row whose position/velocity are interpolated in cosmic time
                between the neighbouring main-branch subhalos (mass in log;
                R200c from the interpolated mass), so main_branch_id_at_snap
                never returns None inside the main branch's lifetime. Virtual
                rows are never linked into the tree (no DescendantID /
                FirstProgenitorID of a real row points at them), so no cluster
                is ever handed to one -- they only serve as the main-branch
                REFERENCE (background potential, separations, R200c history).
                False reproduces the old behavior, where a skipped snapshot
                returned None and every host there was treated as if it
                were the main branch itself.
        """
        if host_mass_model not in HOST_MASS_MODELS:
            raise ValueError(f"host_mass_model must be one of {HOST_MASS_MODELS}, got {host_mass_model!r}")
        self.host_mass_model = host_mass_model
        fields = [
            "SubhaloID", "DescendantID", "FirstProgenitorID", "SnapNum",
            "SubhaloPos", "SubhaloVel", "Group_M_Crit200", "Group_R_Crit200",
            # central/satellite distinction, satellites' own masses, and the
            # position of each subhalo's FoF group center
            "SubhaloMass", "SubfindID", "GroupFirstSub", "GroupPos",
        ]
        with h5py.File(tree_path, "r") as f:
            missing = [k for k in fields if k not in f]
            if missing:
                raise KeyError(f"{tree_path} is missing expected field(s): {missing}")
            self.data = {k: f[k][()] for k in fields}

        self.id_to_row = {int(sid): i for i, sid in enumerate(self.data["SubhaloID"])}
        self.box_size_ckpc_h = box_size_ckpc_h
        self.cosmo = cosmo

        with open(snapshot_redshift_path) as f:
            mapping = json.load(f)
        self._snap_to_redshift = {int(k): float(v) for k, v in mapping.items()}
        self._atime_cache = {}
        self._age_cache = {}

        self.root_id = self._find_root()
        self._main_branch_by_snap = self._build_main_branch_lookup()
        self.virtual_ids = set()
        # Main-branch GLITCHES: snapshots where SubLink's main branch briefly follows a much
        # smaller object (typically a satellite during a major merger: e.g. 1e11 -> 3e8 -> 1e11
        # Msun). With repair_main_branch_glitches, such a snapshot is dropped from the main
        # branch (and then filled like a skipped snapshot), the main branch steps straight over
        # it (step_forward), and any branch merging INTO the glitch row is passed on to the row's
        # descendant instead. Off by default (imbh.py turns it on with --frame-model fixed).
        self._skip_link = {}
        self.glitch_ids = set()
        self.main_branch_glitches = []
        if repair_main_branch_glitches:
            self._repair_main_branch_glitches(self._main_branch_by_snap, glitch_mass_ratio)
            if self.main_branch_glitches:
                print(f"  note: main branch briefly follows a much smaller object at snapshot(s) "
                      f"{self.main_branch_glitches}; stepped over (repair_main_branch_glitches)")
        self.main_branch_gaps = self._main_branch_gaps(self._main_branch_by_snap)
        if fill_main_branch_gaps and self.main_branch_gaps:
            self._fill_main_branch_gaps(self._main_branch_by_snap)
            print(f"  note: main branch skips snapshot(s) {self.main_branch_gaps}; filled with "
                  f"interpolated stand-in main progenitors (see fill_main_branch_gaps)")

    def _find_root(self):
        root_rows = np.where(self.data["DescendantID"] == -1)[0]
        if len(root_rows) == 0:
            raise RuntimeError("No root found in this tree (no row with DescendantID == -1).")
        if len(root_rows) > 1:
            print(f"  warning: {len(root_rows)} rows have DescendantID==-1 in this tree; "
                  f"using the first as the root. This tree may not be a clean single-target subtree.")
        return int(self.data["SubhaloID"][root_rows[0]])

    def _build_main_branch_lookup(self):
        """
        Walk backward from the root via FirstProgenitorID, building a
        {snap: subhalo_id} lookup for the ROOT's OWN main-branch progenitor
        at every snapshot. Used to supply a large-scale background
        potential (see main_branch_id_at_snap) representing the larger
        structure a small proto-halo is embedded within, even before it
        has formally merged into anything.
        """
        lookup = {}
        current = self.root_id
        while current is not None:
            row = self._row(current)
            snap = int(self.data["SnapNum"][row])
            lookup[snap] = current
            fp_id = int(self.data["FirstProgenitorID"][row])
            current = fp_id if fp_id != -1 else None
        return lookup

    def _repair_main_branch_glitches(self, lookup, ratio):
        """One- or two-snapshot dips of the main branch's host mass by > ratio on both sides."""
        snaps = sorted(lookup)
        mass = {s: self.host_properties(lookup[s])[0] for s in snaps}
        drop = set()
        for k in range(1, len(snaps) - 1):
            for w in (1, 2):                      # dip lasting w snapshots
                if k + w >= len(snaps):
                    continue
                inner = snaps[k:k + w]
                before, after = snaps[k - 1], snaps[k + w]
                m_in = max(mass[s] for s in inner)
                if m_in > 0 and mass[before] > ratio * m_in and mass[after] > ratio * m_in:
                    drop.update(inner)
        if not drop:
            return
        for s in sorted(drop):
            self.glitch_ids.add(int(lookup.pop(s)))
        self.main_branch_glitches = sorted(drop)
        # main branch: step straight from the last good row before each run of glitches to the next
        good = sorted(lookup)
        for a, b in zip(good[:-1], good[1:]):
            if any(a < g < b for g in drop):
                self._skip_link[int(lookup[a])] = int(lookup[b])

    @staticmethod
    def _main_branch_gaps(lookup):
        """Snapshots inside the main branch's lifetime with no main-branch subhalo."""
        snaps = sorted(lookup)
        return sorted(set(range(snaps[0], snaps[-1] + 1)) - set(snaps)) if snaps else []

    def _fill_main_branch_gaps(self, lookup):
        """
        Add a virtual main-branch row at every snapshot the main branch skips
        (see __init__'s fill_main_branch_gaps), and register it in `lookup`.
        Interpolation is linear in cosmic time between the bracketing real
        main-branch subhalos: comoving position (with the periodic wrap),
        peculiar velocity, and log mass (SubhaloMass and Group_M_Crit200);
        Group_R_Crit200 is recomputed from the interpolated M200c at the
        virtual snapshot. The row is its own FoF central (SubfindID ==
        GroupFirstSub == -1 - its index among the virtual rows, never equal
        to a real SUBFIND index) with GroupPos = SubhaloPos.
        """
        box = self.box_size_ckpc_h
        new = {k: [] for k in self.data}
        next_id = int(np.max(self.data["SubhaloID"])) + 1
        real = dict(lookup)   # bracket every gap with REAL rows (consecutive gaps: not the stand-ins just made)
        for gap in self._main_branch_gaps(lookup):
            s0 = max(s for s in real if s < gap)
            s1 = min(s for s in real if s > gap)
            i0, i1 = self._row(real[s0]), self._row(real[s1])
            t0, t1, t = (self.age_at_snap(x).value for x in (s0, s1, gap))
            w = (t - t0) / (t1 - t0)

            d = self.data["SubhaloPos"][i1] - self.data["SubhaloPos"][i0]
            d = (d + box / 2) % box - box / 2
            pos = (self.data["SubhaloPos"][i0] + w * d) % box
            vel = (1 - w) * self.data["SubhaloVel"][i0] + w * self.data["SubhaloVel"][i1]

            def log_interp(field):
                a, b = float(self.data[field][i0]), float(self.data[field][i1])
                return a ** (1 - w) * b ** w if a > 0 and b > 0 else (1 - w) * a + w * b

            m200 = log_interp("Group_M_Crit200")                       # 1e10 Msun/h
            r200_kpc = self._r200c_from_mass_kpc(m200 * 1e10 / LITTLE_H, gap) if m200 > 0 else 0.0
            sentinel = -1 - len(new["SubhaloID"])
            row = {
                "SubhaloID": next_id, "DescendantID": -1, "FirstProgenitorID": -1, "SnapNum": gap,
                "SubhaloPos": pos, "SubhaloVel": vel, "GroupPos": pos,
                "Group_M_Crit200": m200, "Group_R_Crit200": r200_kpc * LITTLE_H / self._atime(gap),
                "SubhaloMass": log_interp("SubhaloMass"),
                "SubfindID": sentinel, "GroupFirstSub": sentinel,
            }
            for k in new:
                new[k].append(row[k])
            lookup[gap] = next_id
            self.virtual_ids.add(next_id)
            next_id += 1
        n0 = len(self.data["SubhaloID"])
        for k, vals in new.items():
            arr = np.asarray(vals, dtype=self.data[k].dtype)
            self.data[k] = np.concatenate([self.data[k], arr.reshape((len(vals),) + self.data[k].shape[1:])])
        for j, sid in enumerate(new["SubhaloID"]):
            self.id_to_row[int(sid)] = n0 + j

    def is_virtual(self, subhalo_id):
        """True for a stand-in main-branch row added at a skipped snapshot."""
        return int(subhalo_id) in self.virtual_ids

    def main_branch_id_at_snap(self, snap):
        """
        The root's own main-branch progenitor's SubhaloID at the given
        snapshot, or None if the main branch doesn't extend back that far
        (e.g. the snapshot predates the root's own tracked history).
        """
        return self._main_branch_by_snap.get(snap)

    def main_branch_growth_history(self):
        """
        The root's OWN growth history: (ages_gyr, r200_kpc) as plain numpy
        arrays, one entry per snapshot the main branch (see
        main_branch_id_at_snap) is tracked at, sorted by increasing cosmic
        age -- i.e. how big (R_200, physical kpc) whatever this tree's
        root eventually becomes actually WAS at each point in its own
        history. This is the exact same reference
        trace_cluster_to_snapshot's background potential uses; useful on
        its own too, e.g. as a "does this cluster's separation ever drop
        inside its eventual host's own virial radius" reference curve when
        plotting a cluster's orbit against cosmic time.
        """
        snaps = sorted(self._main_branch_by_snap.keys())
        ages_gyr = np.array([self.age_at_snap(s).to(u.Gyr).value for s in snaps])
        r200_kpc = np.array([self.host_properties(self._main_branch_by_snap[s])[1] for s in snaps])
        return ages_gyr, r200_kpc

    # ------------------------------------------------------------------
    def _row(self, subhalo_id):
        try:
            return self.id_to_row[int(subhalo_id)]
        except KeyError:
            raise KeyError(f"SubhaloID {subhalo_id} not found in this tree.")

    def _atime(self, snap):
        if snap not in self._atime_cache:
            z = self._snap_to_redshift[snap]
            self._atime_cache[snap] = 1.0 / (1.0 + z)
        return self._atime_cache[snap]

    def age_at_snap(self, snap):
        """Cosmic age at the given snapshot, as an astropy Quantity (Gyr)."""
        if snap not in self._age_cache:
            z = self._snap_to_redshift[snap]
            self._age_cache[snap] = self.cosmo.age(z)
        return self._age_cache[snap]

    def hubble_rate_at_snap(self, snap):
        """H(z) at the given snapshot, as an astropy Quantity (1/Gyr)."""
        if not hasattr(self, '_hubble_cache'):
            self._hubble_cache = {}
        if snap not in self._hubble_cache:
            z = self._snap_to_redshift[snap]
            self._hubble_cache[snap] = self.cosmo.H(z).to(1 / u.Gyr)
        return self._hubble_cache[snap]

    def snap_of(self, subhalo_id):
        return int(self.data["SnapNum"][self._row(subhalo_id)])

    # ------------------------------------------------------------------
    def is_central(self, subhalo_id):
        """
        True if this subhalo is its FoF group's central (first/most massive
        subhalo). Uses SubfindID == GroupFirstSub -- both are snapshot-local
        SUBFIND indices on the same row. (FirstSubhaloInFOFGroupID is NOT
        used: in these trees it doesn't reliably point back to the central
        at late times.)
        """
        i = self._row(subhalo_id)
        return int(self.data["SubfindID"][i]) == int(self.data["GroupFirstSub"][i])

    def _r200c_from_mass_kpc(self, mass_msun, snap):
        """Physical radius (kpc) enclosing mean density 200 rho_crit(z) for mass_msun."""
        if not hasattr(self, '_rho_crit_cache'):
            self._rho_crit_cache = {}
        if snap not in self._rho_crit_cache:
            z = self._snap_to_redshift[snap]
            self._rho_crit_cache[snap] = self.cosmo.critical_density(z).to(u.Msun / u.kpc**3).value
        return (3.0 * mass_msun / (4.0 * np.pi * 200.0 * self._rho_crit_cache[snap])) ** (1.0 / 3.0)

    def group_properties(self, subhalo_id):
        """
        (mass_msun, radius_kpc, snap) of the FoF GROUP this subhalo sits in
        (Group_M_Crit200/Group_R_Crit200), using this row's own snapshot's
        scale factor for the comoving -> physical conversion.
        """
        i = self._row(subhalo_id)
        snap = int(self.data["SnapNum"][i])
        a = self._atime(snap)
        mass_msun = float(self.data["Group_M_Crit200"][i]) * 1e10 / LITTLE_H
        radius_kpc = float(self.data["Group_R_Crit200"][i]) * a / LITTLE_H
        return mass_msun, radius_kpc, snap

    def host_properties(self, subhalo_id):
        """
        Returns (mass_msun, radius_kpc, snap): the mass/radius of the halo
        this subhalo itself IS, used as the local NFW host in the orbit
        integration and as the host passed to the cluster samplers.

        Centrals: the FoF group's Group_M_Crit200/Group_R_Crit200 (the
        halo's virial mass). Satellites: depends on host_mass_model (see
        __init__) -- 'own' gives the satellite's own SubhaloMass with an
        R200c-equivalent radius; 'group' gives the group's values.
        """
        if self.host_mass_model == "group" or self.is_central(subhalo_id):
            return self.group_properties(subhalo_id)
        i = self._row(subhalo_id)
        snap = int(self.data["SnapNum"][i])
        mass_msun = float(self.data["SubhaloMass"][i]) * 1e10 / LITTLE_H
        if not mass_msun > 0:
            return 0.0, 0.0, snap
        return mass_msun, self._r200c_from_mass_kpc(mass_msun, snap), snap

    def satellite_group_background(self, subhalo_id, main_branch_id=None):
        """
        For a SATELLITE subhalo (with host_mass_model='own'), the FoF group
        it is orbiting inside, as a background potential:
            (group_mass_msun, group_radius_kpc, offset)
        where offset (astropy Quantity 3-vector, physical kpc) is the group
        center's position (GroupPos -- the potential minimum, which
        coincides with the central's SubhaloPos) RELATIVE TO this subhalo,
        i.e. in the same frame as a cluster's position relative to it.

        Returns None if the subhalo is a central, if host_mass_model is
        'group' (the group is already the local host then), or if the
        group's central is main_branch_id itself (the main-branch
        background already represents that group -- adding it again would
        double count).
        """
        if self.host_mass_model == "group" or self.is_central(subhalo_id):
            return None
        i = self._row(subhalo_id)
        if main_branch_id is not None:
            j = self._row(main_branch_id)
            same_snap = int(self.data["SnapNum"][j]) == int(self.data["SnapNum"][i])
            if same_snap and int(self.data["SubfindID"][j]) == int(self.data["GroupFirstSub"][i]):
                return None
            if same_snap and self.is_virtual(main_branch_id):
                # a stand-in main progenitor has no SUBFIND index to compare:
                # treat this satellite's group as the main branch's own group
                # if the group center lies within half its R200c of the stand-in
                d = self.data["GroupPos"][i] - self.data["SubhaloPos"][j]
                d = (d + self.box_size_ckpc_h / 2) % self.box_size_ckpc_h - self.box_size_ckpc_h / 2
                if np.linalg.norm(d) < 0.5 * float(self.data["Group_R_Crit200"][j]):
                    return None
        mass_msun, radius_kpc, snap = self.group_properties(subhalo_id)
        delta = self.data["GroupPos"][i] - self.data["SubhaloPos"][i]
        delta = (delta + self.box_size_ckpc_h / 2) % self.box_size_ckpc_h - self.box_size_ckpc_h / 2
        offset_kpc = delta * self._atime(snap) / LITTLE_H
        return mass_msun, radius_kpc, offset_kpc * u.kpc

    def _rows_at_snap(self, snap):
        """Row indices of every tree row (real or virtual) at this snapshot (cached)."""
        if not hasattr(self, "_rows_by_snap") or self._rows_by_snap_n != len(self.data["SnapNum"]):
            snaps = np.asarray(self.data["SnapNum"])
            order = np.argsort(snaps, kind="stable")
            uniq, start = np.unique(snaps[order], return_index=True)
            bounds = list(start[1:]) + [len(order)]
            self._rows_by_snap = {int(sv): order[a:b] for sv, a, b in zip(uniq, start, bounds)}
            self._rows_by_snap_n = len(snaps)
        return self._rows_by_snap.get(int(snap), np.array([], dtype=int))

    def nearby_subhalos(self, subhalo_id, rel_pos_kpc, max_group_r200=2.0):
        """
        Candidate new hosts for a cluster at rel_pos_kpc (physical kpc, 3-vector RELATIVE TO
        subhalo_id): SubhaloIDs of the real tree rows at subhalo_id's snapshot (excluding
        subhalo_id itself, virtual stand-ins and main-branch glitch rows) whose centre lies
        within max_group_r200 x their FoF group's R200c of that point. A cheap pre-filter --
        callers check each candidate's own radius/boundedness.
        """
        i = self._row(subhalo_id)
        snap = int(self.data["SnapNum"][i])
        rows = self._rows_at_snap(snap)
        if len(rows) == 0:
            return []
        a = self._atime(snap)
        point = self.data["SubhaloPos"][i] + np.asarray(rel_pos_kpc, dtype=float) * LITTLE_H / a
        d = self.data["SubhaloPos"][rows] - point
        d = (d + self.box_size_ckpc_h / 2) % self.box_size_ckpc_h - self.box_size_ckpc_h / 2
        dist = np.linalg.norm(d, axis=1)
        keep = dist < max_group_r200 * np.asarray(self.data["Group_R_Crit200"][rows], dtype=float)
        out = []
        for sid in self.data["SubhaloID"][rows[keep]]:
            sid = int(sid)
            if sid != int(subhalo_id) and sid not in self.virtual_ids and sid not in self.glitch_ids:
                out.append(sid)
        return out

    def step_forward(self, subhalo_id):
        """
        Advance ONE step forward via DescendantID (regardless of whether it
        stays the primary progenitor there or not).

        Returns (next_id, is_primary_continuation):
            next_id: the descendant's SubhaloID, or None if DescendantID==-1
                (i.e. subhalo_id is genuinely the end of the tracked tree --
                typically the z=0 root, or wherever this tree's tracking stops).
            is_primary_continuation: True if subhalo_id remains next_id's
                FirstProgenitorID (i.e. this is the SAME object simply
                measured at its next tracked snapshot, no formal merger);
                False if a real merger happened at this step.
        """
        sid = int(subhalo_id)
        if sid in self._skip_link:                  # main branch stepping over a glitch
            return self._skip_link[sid], True
        row = self._row(subhalo_id)
        desc_id = int(self.data["DescendantID"][row])
        if desc_id == -1:
            return None, False
        desc_row = self.id_to_row.get(desc_id)
        if desc_row is None:
            return None, False
        is_primary = int(self.data["FirstProgenitorID"][desc_row]) == int(self.data["SubhaloID"][row])
        hops = 0
        while desc_id in self.glitch_ids and hops < 5:   # never hand a cluster to a glitch row
            nxt = int(self.data["DescendantID"][self._row(desc_id)])
            if nxt == -1 or nxt not in self.id_to_row:
                break
            desc_id, is_primary, hops = nxt, False, hops + 1
        return desc_id, is_primary

    def find_next_merge(self, subhalo_id):
        """
        Walk forward via DescendantID from subhalo_id, staying on the SAME
        branch (i.e. while subhalo_id remains the FirstProgenitorID of its
        descendant), until it either:
          - stops being the primary progenitor of its descendant (merges)
            -> returns (last_own_id, descendant_id)
          - reaches the root (DescendantID == -1), or its descendant isn't
            in this tree -> returns (last_own_id, None)
        """
        row = self._row(subhalo_id)
        while True:
            desc_id = int(self.data["DescendantID"][row])
            if desc_id == -1:
                return int(self.data["SubhaloID"][row]), None
            desc_row = self.id_to_row.get(desc_id)
            if desc_row is None:
                return int(self.data["SubhaloID"][row]), None
            if int(self.data["FirstProgenitorID"][desc_row]) == int(self.data["SubhaloID"][row]):
                row = desc_row
                continue
            return int(self.data["SubhaloID"][row]), desc_id

    def relative_state(self, old_id, new_id):
        """
        Position/velocity of old_id RELATIVE TO new_id (physical units),
        typically called right at a merge event (new_id = old_id's direct
        descendant). Each row's own scale factor is used to convert its own
        position/velocity to physical units before differencing (correct
        even in the rare case where old_id/new_id are at adjacent, not
        identical, snapshots -- e.g. after a brief resolution gap); the
        periodic position wrap uses the fixed COMOVING box size, which is
        the same regardless of which snapshot's scale factor applies.
        """
        i_old, i_new = self._row(old_id), self._row(new_id)
        snap_old = int(self.data["SnapNum"][i_old])
        snap_new = int(self.data["SnapNum"][i_new])
        a_old = self._atime(snap_old)
        a_new = self._atime(snap_new)

        pos_old_comoving = self.data["SubhaloPos"][i_old]
        pos_new_comoving = self.data["SubhaloPos"][i_new]
        delta_comoving = pos_old_comoving - pos_new_comoving
        delta_comoving = (
            (delta_comoving + self.box_size_ckpc_h / 2) % self.box_size_ckpc_h
            - self.box_size_ckpc_h / 2
        )
        # convert using old's scale factor for definiteness (old_id is the
        # row we're actively tracking at the moment of the merge)
        rel_pos_kpc = delta_comoving * a_old / LITTLE_H

        # SubhaloVel is ALREADY the physical peculiar velocity in km/s ("No
        # unit conversion is needed" -- TNG data specifications). Only
        # GroupVel (km/s/a) needs the 1/a factor. An earlier version divided
        # SubhaloVel by a here too, inflating every relative velocity (and
        # hence every reframe kick in trace_cluster_to_snapshot) by (1+z).
        vel_old_phys = self.data["SubhaloVel"][i_old]
        vel_new_phys = self.data["SubhaloVel"][i_new]
        rel_vel_kms = vel_old_phys - vel_new_phys

        return rel_pos_kpc * u.kpc, rel_vel_kms * u.km / u.s

    def _same_snap_progenitor(self, new_id, snap_old):
        """new_id's own main progenitor AT snap_old (walking FirstProgenitorID), or the main-branch
        row there if new_id is the main branch; None if neither has a row at exactly snap_old."""
        ref = int(new_id)
        while ref in self.id_to_row and self.snap_of(ref) > snap_old:
            fp = int(self.data["FirstProgenitorID"][self._row(ref)])
            if fp == -1 or fp not in self.id_to_row:
                break
            ref = fp
        if ref in self.id_to_row and self.snap_of(ref) == snap_old:
            return ref
        if self._main_branch_by_snap.get(self.snap_of(new_id)) == int(new_id):
            return self._main_branch_by_snap.get(snap_old)
        return None

    def relative_state_ballistic(self, old_id, new_id, concentration=4.0, hubble=True):
        """
        Like relative_state_matched, but old_id is carried to new_id's time on a BALLISTIC ORBIT
        in new_id's potential instead of a straight line. Starts from old_id's state relative to
        new_id's own progenitor at old_id's snapshot (an exact same-snapshot difference) and
        integrates that relative orbit to new_id's snapshot in an NFW whose M200/R200 go
        log-linearly in time from the progenitor's to new_id's. The straight line x_old + v_old*dt
        badly overshoots when the old host is caught near pericentre (hundreds of km/s, tens of
        kpc per snapshot): it keeps the pericentre speed out to large radii and can leave the
        old host -- and every cluster handed over with it -- unbound. Falls back to
        relative_state_matched when there is no same-snapshot progenitor (or it is old_id itself,
        i.e. a primary continuation).
        """
        snap_old, snap_new = self.snap_of(old_id), self.snap_of(new_id)
        if snap_old == snap_new:
            return self.relative_state(old_id, new_id)
        ref = self._same_snap_progenitor(new_id, snap_old)
        if ref is None or int(ref) == int(old_id):
            return self.relative_state_matched(old_id, new_id)
        m0, r0, _ = self.host_properties(ref)
        m1, r1, _ = self.host_properties(new_id)
        if not (m0 > 0 and r0 > 0 and m1 > 0 and r1 > 0):
            return self.relative_state_matched(old_id, new_id)
        from scipy.integrate import solve_ivp
        from dynamical_friction import NFWHost, _G
        kms_to_kpcgyr = (1 * u.km / u.s).to(u.kpc / u.Gyr).value
        dpos, dvel = self.relative_state(old_id, ref)
        # SubhaloVel is a PECULIAR velocity: add the Hubble flow for the physical relative orbit
        # (and take it off again at the end) when hubble=True
        h0 = self.hubble_rate_at_snap(snap_old).to(1 / u.Gyr).value if hubble else 0.0
        h1 = self.hubble_rate_at_snap(snap_new).to(1 / u.Gyr).value if hubble else 0.0
        x0 = dpos.to(u.kpc).value
        y0 = np.concatenate([x0, dvel.to(u.km / u.s).value * kms_to_kpcgyr + h0 * x0])
        dt = float((self.age_at_snap(snap_new) - self.age_at_snap(snap_old)).to(u.Gyr).value)
        r_soft = 0.01 * min(r0, r1)

        def rhs(t, y):
            w = t / dt
            host = NFWHost(m0 ** (1 - w) * m1 ** w * u.Msun, r0 ** (1 - w) * r1 ** w * u.kpc,
                           concentration=concentration)
            r = np.linalg.norm(y[:3])
            rs = max(r, r_soft)
            acc = -_G * host.mass_enclosed(rs) / rs ** 3 * y[:3] if r > 0 else np.zeros(3)
            return np.concatenate([y[3:], acc])

        sol = solve_ivp(rhs, (0.0, dt), y0, method="RK45", rtol=1e-6, atol=1e-6)
        if not sol.success:
            return self.relative_state_matched(old_id, new_id)
        y1 = sol.y[:, -1]
        return y1[:3] * u.kpc, ((y1[3:] - h1 * y1[:3]) / kms_to_kpcgyr) * u.km / u.s

    def relative_state_matched(self, old_id, new_id):
        """
        Like relative_state, but TIME-MATCHED when old_id and new_id sit at different
        snapshots (a merger: old_id at snap n, its descendant new_id at snap n+1):
        old_id's position is first carried forward to new_id's time with its own
        velocity, x_old + v_old*dt, so the result is old_id's position relative to
        new_id at the SAME moment. Use this to move a cluster that moved WITH its
        old host into the new host's frame (imbh.py frame_model='fixed');
        relative_state alone treats the cluster as if it had stayed put while its
        old host moved. Identical to relative_state for same-snapshot calls.
        """
        rel_pos, rel_vel = self.relative_state(old_id, new_id)
        snap_old, snap_new = self.snap_of(old_id), self.snap_of(new_id)
        if snap_old == snap_new:
            return rel_pos, rel_vel
        dt = self.age_at_snap(snap_new) - self.age_at_snap(snap_old)
        vel_old = np.asarray(self.data["SubhaloVel"][self._row(old_id)], dtype=float) * u.km / u.s
        return (rel_pos + (vel_old * dt).to(u.kpc)), rel_vel