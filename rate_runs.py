"""
Shared helpers for plot_model_ladder.py and plot_tde_rates_seeds.py.

A *run* is one submit_halos.pl / submit_analysis.pl configuration, given by
the same run-selection flags (see run_naming.add_run_args), e.g.
"--mode observational --n-relation bf20_seed --resolution-cuts on --loss-cone full".
Its analysis.py outputs live in <data-dir>/analysis<analysis suffix>/ (or in
<data-dir> itself for an untagged default run) and its imbh.py output in
<data-dir>/cluster_output_<ID>_<snap><model suffix>.dat.

Unlike run_naming.analysis_dir, Run.analysis_dir never falls back to
<data-dir>: a run whose folder doesn't exist is reported missing instead of
silently reading another run's files.
"""
import argparse
import os
import shlex
import warnings

import numpy as np
import pandas as pd

from run_naming import add_run_args, run_suffix, analysis_suffix
from plot_tde_rates_smoothed import smooth, age_to_redshift, DEFAULT_MASS_BINS  # noqa: F401 (re-exported)
from distance_split import main_branch_r200, split_rates

SELECTIONS = ("total", "within", "beyond")
SELECTION_LABELS = {"total": "All TDEs", "within": r"Within ${f}\,R_{{200c}}$",
                    "beyond": r"Beyond ${f}\,R_{{200c}}$"}
IMBH_TDE_MIN_MSUN = 500.0   # analysis.py computes TDEs only above this IMBH mass


def selection_label(selection, factor=3.0):
    return SELECTION_LABELS[selection].replace("{f}", f"{factor:g}")


def parse_run_flags(flag_string):
    p = argparse.ArgumentParser(add_help=False)
    add_run_args(p)
    return p.parse_args(shlex.split(flag_string))


class Run:
    """One model configuration (label + run-selection flags)."""

    def __init__(self, label, flags, data_dir, snap=99):
        self.label = label
        self.flags = " ".join(flags.split())
        self.args = parse_run_flags(self.flags)
        self.model_suffix = run_suffix(self.args)
        self.analysis_suffix = analysis_suffix(self.args)
        self.data_dir = data_dir
        self.snap = snap
        self.analysis_dir = (data_dir if not self.analysis_suffix
                             else os.path.join(data_dir, f"analysis{self.analysis_suffix}"))

    @property
    def exists(self):
        return os.path.isdir(self.analysis_dir)

    def cluster_output_path(self, halo_id):
        return os.path.join(self.data_dir, f"cluster_output_{halo_id}_{self.snap}{self.model_suffix}.dat")

    def cluster_stats(self, halo_id):
        """Counts from imbh.py's output for one halo, or None if the file is missing."""
        path = self.cluster_output_path(halo_id)
        if not os.path.exists(path):
            return None
        df = pd.read_pickle(path) if path.endswith(".dat") else pd.read_csv(path)
        status = df["status"].astype(str)
        m_bh = pd.to_numeric(df["IMBH_mass_msun"], errors="coerce")
        failed_trace = status == "trace_failed"
        imbh = m_bh >= IMBH_TDE_MIN_MSUN
        return {
            "n_clusters": int(len(df)),
            "n_trace_failed": int(failed_trace.sum()),
            "n_imbh_model_failed": int((m_bh.isna() & ~failed_trace).sum()),
            "n_imbh_ge500": int(imbh.sum()),
            "n_escaped_z0": int((status == "escaped").sum()),
            "median_cluster_mass": float(np.nanmedian(df["cluster_mass_msun"])) if len(df) else np.nan,
            "median_r_h_pc": float(np.nanmedian(df["cluster_radius_pc"])) if len(df) else np.nan,
            "median_m_imbh_ge500": float(np.nanmedian(m_bh[imbh])) if imbh.any() else np.nan,
        }


class RateLoader:
    """
    Per-halo TDE rate (Msun/yr = events/yr for 1 Msun stars) on the analysis
    time grid, for selection 'total' (tde_rates_<ID>_alpha<a>.csv) or
    'within'/'beyond' r200_factor * R200c of the main progenitor at the time of
    each event (tde_radial_<ID>_alpha<a>.npz + the main-branch R200c history,
    see distance_split.py).
    """

    def __init__(self, alpha, selection="total", r200_factor=3.0, tree_dir="tng_download",
                 snap_redshift_path="tng_download/snapshot_redshifts.json", r200_cache_dir=None):
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        self.alpha = alpha
        self.selection = selection
        self.r200_factor = r200_factor
        self.tree_dir = tree_dir
        self.snap_redshift_path = snap_redshift_path
        self.r200_cache_dir = r200_cache_dir
        self._r200 = {}

    def _r200_history(self, halo_id):
        if halo_id not in self._r200:
            self._r200[halo_id] = main_branch_r200(halo_id, self.tree_dir, self.snap_redshift_path,
                                                   self.r200_cache_dir)
        return self._r200[halo_id]

    def load(self, analysis_dir, halo_id):
        """(time_yr, rate) or None if this halo's file is missing."""
        if self.selection == "total":
            path = os.path.join(analysis_dir, f"tde_rates_{halo_id}_alpha{self.alpha}.csv")
            if not os.path.exists(path):
                return None
            df = pd.read_csv(path, usecols=["time", "tde_rate_array_msunyr"])
            return df["time"].to_numpy(float), df["tde_rate_array_msunyr"].to_numpy(float)
        path = os.path.join(analysis_dir, f"tde_radial_{halo_id}_alpha{self.alpha}.npz")
        if not os.path.exists(path):
            return None
        z_snap, r200_snap = self._r200_history(halo_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t, near, far, _ = split_rates(path, age_to_redshift, z_snap, r200_snap, self.r200_factor)
        return t, (near if self.selection == "within" else far)


def on_grid(t_ref, t, rate):
    """Put rate onto the reference time grid (no-op when the grids already match)."""
    if len(t) == len(t_ref) and np.allclose(t, t_ref):
        return rate
    return np.interp(t_ref, t, rate, left=0.0, right=0.0)


def band(curves, kind, axis=0):
    """Lower/upper edges of a spread across `axis`: minmax, 16-84 percentile or mean +/- std."""
    if kind == "minmax":
        return curves.min(axis=axis), curves.max(axis=axis)
    if kind == "percentile":
        return tuple(np.percentile(curves, [16, 84], axis=axis))
    m, s = curves.mean(axis=axis), curves.std(axis=axis)
    return np.clip(m - s, 0, None), m + s


def plot_step(dt_yr, smooth_myr):
    """Downsampling stride for plotting smoothed curves (~20 points per smoothing width)."""
    return max(1, int(smooth_myr * 1e6 / dt_yr / 20))
