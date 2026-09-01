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


class MergerTreeNavigator:
    def __init__(self, tree_path, snapshot_redshift_path, box_size_ckpc_h, cosmo=TNG_COSMO):
        """
        Parameters:
            tree_path (str): path to sublink_full_<ID>.hdf5
            snapshot_redshift_path (str): path to snapshot_redshifts.json
                (produced by tng50_elliptical_dark_tree.py's save_snapshot_redshifts())
            box_size_ckpc_h (float): simulation box size, comoving ckpc/h
                (same convention as elsewhere in this pipeline)
            cosmo: astropy cosmology to use for age(z) lookups
        """
        fields = [
            "SubhaloID", "DescendantID", "FirstProgenitorID", "SnapNum",
            "SubhaloPos", "SubhaloVel", "Group_M_Crit200", "Group_R_Crit200",
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

    def main_branch_id_at_snap(self, snap):
        """
        The root's own main-branch progenitor's SubhaloID at the given
        snapshot, or None if the main branch doesn't extend back that far
        (e.g. the snapshot predates the root's own tracked history).
        """
        return self._main_branch_by_snap.get(snap)

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
    def host_properties(self, subhalo_id):
        """
        Returns (mass_msun, radius_kpc, snap) for this row, using ITS OWN
        snapshot's scale factor for the comoving -> physical conversion.
        """
        i = self._row(subhalo_id)
        snap = int(self.data["SnapNum"][i])
        a = self._atime(snap)
        mass_msun = float(self.data["Group_M_Crit200"][i]) * 1e10 / LITTLE_H
        radius_kpc = float(self.data["Group_R_Crit200"][i]) * a / LITTLE_H
        return mass_msun, radius_kpc, snap

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
        row = self._row(subhalo_id)
        desc_id = int(self.data["DescendantID"][row])
        if desc_id == -1:
            return None, False
        desc_row = self.id_to_row.get(desc_id)
        if desc_row is None:
            return None, False
        is_primary = int(self.data["FirstProgenitorID"][desc_row]) == int(self.data["SubhaloID"][row])
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

        vel_old_phys = self.data["SubhaloVel"][i_old] / a_old
        vel_new_phys = self.data["SubhaloVel"][i_new] / a_new
        rel_vel_kms = vel_old_phys - vel_new_phys

        return rel_pos_kpc * u.kpc, rel_vel_kms * u.km / u.s