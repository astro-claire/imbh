"""
Calibrated replacement for the draw_clusters() placeholder.

Given a pickled clusters_near_halos_exclusive_*.dat catalog (real star
clusters matched to their host DM halos, with mass/half-mass-radius/
separation/velocity for each), builds a nonparametric sampler that draws
realistic (mass, radius, separation, velocity) tuples for a NEW host halo
of arbitrary (subhalo_mass, subhalo_radius), by kernel-weighted bootstrap
resampling of the real cluster population -- weighted toward halos similar
in mass to the query host, in log-mass space.

Separation and velocity are resampled in dimensionless, self-similar units
(R/R_vir and v/v_circ(host)) and rescaled to the query host's own R_vir/
v_circ. Cluster mass and half-mass radius are resampled as-is.

PERFORMANCE: this version is optimized for two things that matter at your
catalog's scale (~10^5-10^6 halos/clusters):
  1. Construction is vectorized (no per-halo astropy Quantity calls in a
     Python loop -- v_circ is computed once for all halos at once).
  2. draw_clusters() only computes kernel weights over a WINDOW of the
     calibration data within a few bandwidths of the query mass (using a
     presorted array + np.searchsorted), rather than the full catalog
     every call. Since the Gaussian kernel weight is ~0 outside a few
     bandwidths anyway, this changes nothing about the statistics, but
     turns an O(N_total) per-call cost into roughly O(log N + window size).

Requires: numpy, astropy
    pip install numpy astropy
"""

import pickle
import warnings

import numpy as np
import astropy.units as u
from astropy.constants import G

G_KPC_MSUN_KMS = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value  # for fast plain-float v_circ calc


class ClusterPopulationSampler:
    def __init__(self, pickle_path=None, cluster_mass_key='stellarMass_msun',
                 cluster_hmradius_key='hmradii_kpc', bandwidth_dex=0.3,
                 window_sigma=6.0, max_window_points=3000, min_effective_weight=5.0,
                 rng=None, _skip_build=False):
        """
        Parameters:
            pickle_path (str): path to the clusters_near_halos_exclusive_*.dat pickle
            cluster_mass_key (str): per-cluster mass field, Msun
            cluster_hmradius_key (str): per-cluster half-mass radius field, kpc
            bandwidth_dex (float): Gaussian kernel bandwidth in dex (log10 Msun)
            window_sigma (float): only halos/clusters within window_sigma *
                bandwidth_dex of the query (in log-mass) are even considered --
                the Gaussian weight beyond this is negligible, so this is a
                performance cutoff, not a change to the statistics (as long
                as window_sigma isn't set too small; 6 is generous).
            max_window_points (int): HARD CAP on how many calibration points
                (halos, or clusters) are ever considered per draw_clusters()
                call, regardless of how many fall within window_sigma. If the
                mass-based window would include more than this, it's shrunk to
                the nearest max_window_points in log-mass around the query.
                This bounds draw_clusters()'s per-call cost independent of your
                catalog's total size -- for a bootstrap draw of a handful of
                clusters, a few thousand representative neighbors is already
                more than enough; this trades a small, well-justified
                approximation for a large, predictable speedup.
            min_effective_weight (float): warn if the (windowed) total kernel
                weight for a query falls below this -- signals the query is
                poorly supported by the calibration data (likely extrapolating).
            rng (np.random.Generator or None): defaults to np.random.default_rng().
            _skip_build (bool): internal use (for load()) -- skip __init__'s
                normal pickle-loading path.
        """
        self.cluster_mass_key = cluster_mass_key
        self.cluster_hmradius_key = cluster_hmradius_key
        self.bandwidth_dex = bandwidth_dex
        self.window_sigma = window_sigma
        self.max_window_points = max_window_points
        self.min_effective_weight = min_effective_weight
        self.rng = rng if rng is not None else np.random.default_rng()

        if _skip_build:
            return

        with open(pickle_path, 'rb') as f:
            matched = pickle.load(f)

        self._build_calibration_arrays(matched, cluster_mass_key, cluster_hmradius_key)

    # ------------------------------------------------------------------
    # Construction (vectorized)
    # ------------------------------------------------------------------
    def _build_calibration_arrays(self, matched, cluster_mass_key, cluster_hmradius_key):
        halo_mass_msun = np.array([e['halo_mass_msun'] for e in matched], dtype=float)
        halo_radius_kpc = np.array([e['halo_radius'] for e in matched], dtype=float)
        halo_n_clusters = np.array([e['n_clusters'] for e in matched], dtype=int)

        valid_halo = (halo_mass_msun > 0)
        halo_mass_msun = halo_mass_msun[valid_halo]
        halo_radius_kpc = halo_radius_kpc[valid_halo]
        halo_n_clusters = halo_n_clusters[valid_halo]
        matched_valid = [e for e, v in zip(matched, valid_halo) if v]

        self.halo_log_mass = np.log10(halo_mass_msun)
        self.halo_n_clusters = halo_n_clusters
        # presort for windowed lookup in draw_clusters
        order = np.argsort(self.halo_log_mass)
        self.halo_log_mass = self.halo_log_mass[order]
        self.halo_n_clusters = self.halo_n_clusters[order]

        # vectorized v_circ for every halo that actually has clusters (single
        # astropy Quantity call over the whole array, not one per halo)
        has_clusters = halo_n_clusters > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            v_circ_all = np.sqrt(G_KPC_MSUN_KMS * halo_mass_msun / halo_radius_kpc)  # km/s, plain floats

        cl_log_host_mass_parts = []
        cl_mass_parts = []
        cl_hmradius_parts = []
        cl_dist_over_rvir_parts = []
        cl_vel_over_vcirc_parts = []

        missing_mass_key = missing_hmradius_key = missing_vel_key = 0
        skipped_zero_rvir = 0

        for entry, hm, rad, n, vcirc, ok in zip(
            matched_valid, halo_mass_msun, halo_radius_kpc, halo_n_clusters, v_circ_all, has_clusters
        ):
            if not ok:
                continue
            if rad <= 0:
                skipped_zero_rvir += n
                continue
            if cluster_mass_key not in entry:
                missing_mass_key += n
                continue
            if cluster_hmradius_key not in entry:
                missing_hmradius_key += n
                continue
            if 'cluster_rel_vel_mag' not in entry:
                missing_vel_key += n
                continue

            cl_log_host_mass_parts.append(np.full(n, np.log10(hm)))
            cl_mass_parts.append(np.asarray(entry[cluster_mass_key]))
            cl_hmradius_parts.append(np.asarray(entry[cluster_hmradius_key]))
            cl_dist_over_rvir_parts.append(np.asarray(entry['cluster_distance']) / rad)
            cl_vel_over_vcirc_parts.append(np.asarray(entry['cluster_rel_vel_mag']) / vcirc)

        if missing_mass_key or missing_hmradius_key or missing_vel_key:
            print(f"  note: skipped some clusters missing required fields -- "
                  f"mass:{missing_mass_key}, hmradius:{missing_hmradius_key}, vel:{missing_vel_key}")
        if skipped_zero_rvir:
            print(f"  note: skipped {skipped_zero_rvir} cluster(s) belonging to halos with halo_radius<=0")

        if len(cl_log_host_mass_parts) == 0:
            raise RuntimeError(
                "No usable clusters found in the calibration catalog (check that the "
                "pickle has cluster_distance, cluster_rel_vel_mag, and the mass/hmradius keys)."
            )

        cl_log_host_mass = np.concatenate(cl_log_host_mass_parts)
        cl_mass = np.concatenate(cl_mass_parts)
        cl_hmradius = np.concatenate(cl_hmradius_parts)
        cl_dist_over_rvir = np.concatenate(cl_dist_over_rvir_parts)
        cl_vel_over_vcirc = np.concatenate(cl_vel_over_vcirc_parts)

        # presort by host log-mass for windowed lookup
        cl_order = np.argsort(cl_log_host_mass)
        self.cl_log_host_mass = cl_log_host_mass[cl_order]
        self.cl_mass = cl_mass[cl_order]
        self.cl_hmradius = cl_hmradius[cl_order]
        self.cl_dist_over_rvir = cl_dist_over_rvir[cl_order]
        self.cl_vel_over_vcirc = cl_vel_over_vcirc[cl_order]

        print(f"Calibration catalog: {len(self.halo_log_mass)} halos "
              f"(log10 M range [{self.halo_log_mass.min():.2f}, {self.halo_log_mass.max():.2f}]), "
              f"{len(self.cl_mass)} clusters total.")

    # ------------------------------------------------------------------
    # Fast windowed lookup helpers
    # ------------------------------------------------------------------
    def _windowed_weights(self, sorted_log_mass, log_target):
        """
        Returns (i0, i1, weights) where weights are the (unnormalized)
        Gaussian kernel weights for sorted_log_mass[i0:i1] -- the slice of
        the presorted array within window_sigma*bandwidth_dex of log_target,
        further capped to at most max_window_points (shrunk to the nearest
        max_window_points around the query if the mass-window would be larger).
        """
        half_window = self.window_sigma * self.bandwidth_dex
        i0, i1 = np.searchsorted(sorted_log_mass, [log_target - half_window, log_target + half_window])

        if i1 - i0 > self.max_window_points:
            center = np.searchsorted(sorted_log_mass, log_target)
            half_cap = self.max_window_points // 2
            i0 = max(i0, center - half_cap)
            i1 = min(i1, center + half_cap)

        if i1 <= i0:
            return i0, i1, np.array([])
        window = sorted_log_mass[i0:i1]
        weights = np.exp(-0.5 * ((window - log_target) / self.bandwidth_dex) ** 2)
        return i0, i1, weights

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_clusters(self, subhalo_mass, subhalo_radius):
        """
        Drop-in replacement for the draw_clusters() placeholder. Takes plain
        floats (Msun, kpc) and returns the same dict shape as before.
        """
        log_target = np.log10(subhalo_mass)

        h_i0, h_i1, h_weights = self._windowed_weights(self.halo_log_mass, log_target)
        total_weight = h_weights.sum()

        if total_weight < self.min_effective_weight:
            warnings.warn(
                f"Query host mass {subhalo_mass:.3e} Msun (log10={log_target:.2f}) has very low "
                f"kernel-weighted support in the calibration data (effective weight={total_weight:.2f} "
                f"vs threshold {self.min_effective_weight}). This draw is likely extrapolating beyond "
                f"what the calibration catalog resolves -- results may not be trustworthy.",
                stacklevel=2,
            )

        if total_weight == 0:
            # fall back to the single nearest halo in log-mass (still gives
            # *something*, but the warning above already fired)
            nearest = np.searchsorted(self.halo_log_mass, log_target)
            nearest = np.clip(nearest, 0, len(self.halo_log_mass) - 1)
            n_draw = int(self.halo_n_clusters[nearest])
        else:
            h_weights = h_weights / total_weight
            chosen_local = self.rng.choice(h_i1 - h_i0, p=h_weights)
            n_draw = int(self.halo_n_clusters[h_i0 + chosen_local])

        empty = {
            'cluster_mass': np.array([]) * u.Msun,
            'cluster_radius': np.array([]) * u.pc,
            'cluster_sep': np.array([]) * u.kpc,
            'cluster_vel': np.array([]) * u.km / u.s,
        }
        if n_draw == 0:
            return empty

        c_i0, c_i1, c_weights = self._windowed_weights(self.cl_log_host_mass, log_target)
        if c_weights.sum() == 0:
            nearest = np.searchsorted(self.cl_log_host_mass, log_target)
            c_i0, c_i1 = max(nearest - 1, 0), min(nearest + 1, len(self.cl_log_host_mass))
            c_weights = np.ones(c_i1 - c_i0)
        c_weights = c_weights / c_weights.sum()

        local_idx = self.rng.choice(c_i1 - c_i0, size=n_draw, replace=True, p=c_weights)
        idx = c_i0 + local_idx

        v_circ_target = np.sqrt(G_KPC_MSUN_KMS * subhalo_mass / subhalo_radius)  # km/s, plain float

        cluster_mass = self.cl_mass[idx] * u.Msun
        cluster_radius = (self.cl_hmradius[idx] * u.kpc).to(u.pc)
        cluster_sep = self.cl_dist_over_rvir[idx] * subhalo_radius * u.kpc
        cluster_vel = self.cl_vel_over_vcirc[idx] * v_circ_target * u.km / u.s

        return {
            'cluster_mass': cluster_mass,
            'cluster_radius': cluster_radius,
            'cluster_sep': cluster_sep,
            'cluster_vel': cluster_vel,
        }

    # ------------------------------------------------------------------
    # Save / load 
    # ------------------------------------------------------------------
    def __getstate__(self):
        # exclude the RNG from the pickled state -- np.random.Generator IS
        # picklable, but you almost always want a fresh/independently-seeded
        # RNG each time you load a cached sampler rather than replaying
        # whatever internal state it happened to be in when saved.
        state = self.__dict__.copy()
        state.pop('rng', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = np.random.default_rng()

    def save(self, path):
        """
        Save this (already-built) sampler to disk. Loading it back with
        ClusterPopulationSampler.load(path) skips re-reading the raw
        clusters_near_halos pickle and re-deriving the calibration arrays
        entirely -- only worth it if that construction step is your
        bottleneck (see the standalone build_cluster_sampler.py script).
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved sampler to {path}")

    @classmethod
    def load(cls, path, rng=None):
        """Load a sampler previously saved with .save(). See save()'s docstring."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        if rng is not None:
            obj.rng = rng
        return obj


def draw_clusters(subhalo_mass, subhalo_radius, sampler=None, pickle_path=None):
    """
    Functional wrapper matching the original draw_clusters(subhalo_mass,
    subhalo_radius) signature exactly, for a minimal-diff drop-in swap in
    your existing pipeline script.

    Pass either a pre-built `sampler` (recommended -- build/load it ONCE
    outside your per-subhalo loop), or `pickle_path` to a RAW clusters
    catalog to build one lazily on first call (cached on the function
    itself). If you've pre-built and saved a sampler with build_cluster_sampler.py,
    just load it once yourself and pass it in as `sampler` instead.
    """
    if sampler is None:
        if pickle_path is None:
            raise ValueError("Pass either sampler= or pickle_path=.")
        if not hasattr(draw_clusters, "_cached_sampler") or draw_clusters._cached_pickle_path != pickle_path:
            draw_clusters._cached_sampler = ClusterPopulationSampler(pickle_path)
            draw_clusters._cached_pickle_path = pickle_path
        sampler = draw_clusters._cached_sampler
    return sampler.draw_clusters(subhalo_mass, subhalo_radius)