import argparse
import os
import signal
import sys

# Machine-specific paths (e.g. laptop vs. a shared computing cluster) live
# in imbh_config.py, kept OUT of version control / not synced between
# machines -- see imbh_config.py.example for setup instructions. Look
# next to THIS script rather than relying on the current working
# directory, so it resolves correctly regardless of where imbh.py is
# invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
try:
    from imbh_config import TIMESCALES_SRC_PATH, CLUSTER_SAMPLER_PATH
except ImportError:
    sys.exit(
        "Missing imbh_config.py (expected next to imbh.py, in "
        f"{_SCRIPT_DIR}). This is a small, per-machine config file (kept "
        "out of version control) defining TIMESCALES_SRC_PATH and "
        "CLUSTER_SAMPLER_PATH for THIS machine. Copy imbh_config.py.example "
        "to imbh_config.py and fill in the two paths to get started."
    )
sys.path.append(TIMESCALES_SRC_PATH)

import numpy as np 
import astropy.units as u 
import h5py 
import matplotlib.pyplot as plt 
import pandas as pd
from astropy.constants import G
from cluster_population_sampler import ClusterPopulationSampler

 
# dynamical friction orbit integration (separate module, kept alongside this script)
from dynamical_friction import NFWHost, integrate_orbit
# raw merger-tree navigator, for following a cluster's host across mergers
from tree_navigator import MergerTreeNavigator
#timescales stuff
from timescales import TimescaleEnsemble
from timescales.data import build_single_system_grid
from timescales.analysis.modelv2 import create_dynamical_model_integral
from astropy.cosmology import FlatLambdaCDM
#FIXME hard coded -- cosmology for the IMBH/high-res-cluster-simulation
# model itself (NOT the TNG merger tree -- see tree_navigator.TNG_COSMO
# for that; these are two different simulations, don't conflate them)
cosmo = FlatLambdaCDM(71,0.27,Ob0=0.044, Tcmb0=2.726 *u.K)
alpha = 1.2
# concentration assumed for the host NFW profile. TODO: this is a flat
# placeholder -- at z>=12 you likely want something lower (high-z halos
# are less concentrated), possibly per-halo from a mass-concentration-
# redshift relation rather than one fixed number for every subhalo.
HOST_CONCENTRATION = 4.0
# conversion factor for turning NFWHost.potential()'s internal
# (kpc/Gyr)^2 units into the more readable (km/s)^2, for the
# energy-based boundedness diagnostic in trace_cluster_to_snapshot's
# verbose output. Computed via astropy rather than importing
# dynamical_friction's private _KMS_TO_KPCGYR constant directly.
_KPCGYR_TO_KMS = (1 * u.kpc / u.Gyr).to(u.km / u.s).value
# safety cap on how many snapshot-to-snapshot steps a single cluster can go
# through before we give up (should never be hit in practice; guards
# against any tree pathology causing an infinite loop)
MAX_STEPS = 500

# Time resolution for the optional per-cluster radius-vs-time tracks (see
# --save-radius-tracks / trace_cluster_to_snapshot's record_dt). Matches
# analysis.py's own TDE-rate time resolution (set_up_timebins's default
# 1e5 yr) so the two line up directly without any re-binning.
TRACK_TIME_RESOLUTION = 1e5 * u.yr
# Minimum IMBH_mass_msun for a cluster's radius track to actually be saved
# to disk. Matches analysis.py's own hardcoded "small BHs, placeholder"
# cutoff (timescale_analysis's `if IMBH_mass_msun < 5e2`) -- clusters below
# this never get a TDE rate computed downstream anyway, so there's no
# reason to save (and pay the disk cost for) their track.
TRACK_MIN_IMBH_MASS_MSUN = 500.0
# Default subdirectory (created next to the input CSV) that per-cluster
# radius tracks get saved into.
TRACK_OUTPUT_DIRNAME = "radius_tracks"

# Minimum number of sub-steps used to integrate a leg whenever a background
# potential is active at either end of it (see trace_cluster_to_snapshot),
# REGARDLESS of the leg's own duration relative to max_leg_duration. This
# lets the background's mass/radius/offset be linearly interpolated across
# the leg (same treatment the LOCAL host's mass/radius already gets),
# rather than held fixed for the whole leg and then discretely reset at the
# next leg boundary -- which otherwise produces an artificial instantaneous
# jump in the background-relative distance (and a smaller, less visible one
# in the background's contribution to the force) exactly at every leg
# transition, even though the underlying orbit is evolving continuously.
# Tradeoff: legs that would otherwise take a single fast integrate_orbit
# call now take BG_MIN_SUBSTEPS shorter ones instead, whenever a background
# is present -- which is the common case for most of a cluster's early
# history, before its own branch merges onto the main branch. Increase for
# smoother background evolution at the cost of runtime; 1 disables this
# forced subdivision entirely (falls back to the old fixed-per-leg behavior).
BG_MIN_SUBSTEPS = 4

# Wall-clock timeout for a single cluster's orbit trace (trace_cluster_to_snapshot).
# This is a defensive safeguard, not a fix for any specific known cause --
# some parameter combinations (e.g. a bound orbit with an unusually short
# period relative to a long un-subdivided span, or other stiff/expensive
# regimes we haven't fully characterized) can make a single trace far
# slower than the typical case. Rather than let one pathological cluster
# block the entire batch, log it and move on.
ORBIT_TRACE_TIMEOUT_S = 60
TIMESCALES_CALL_TIMEOUT_S = 60


class TimescalesTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise TimescalesTimeout()


def run_with_timeout(func, timeout_s, *args, **kwargs):
    """
    Run func(*args, **kwargs) with a wall-clock timeout (Unix only -- uses
    SIGALRM, which is fine on macOS/Linux but won't work on Windows).
    Returns (result, None) on success, or (None, error_message) on timeout
    or any other exception, so the caller can log-and-continue rather than
    have the whole batch die or hang on one bad case.
    """
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_s)
    try:
        result = func(*args, **kwargs)
        return result, None
    except TimescalesTimeout:
        return None, f"timed out after {timeout_s}s"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


cluster_sampler = ClusterPopulationSampler.load(CLUSTER_SAMPLER_PATH)


#--------- Load post-processed illustris merger tree 
def load_merger_tree_idx(df,cutoff_z = 7):
    goodidx = np.where((df['delta_t_gyr']>0) & (df['formation_redshift']>=cutoff_z)& (df["group_m_crit200_msun"] < 1e9))[0]
    print("There are "+str(len(goodidx))+" subhalos with nonzero merger time and formation time above z cutoff.")
    return goodidx


# -------- Attach AREPO clusters to the primordial halos
def draw_clusters(subhalo_mass, subhalo_radius):
    return cluster_sampler.draw_clusters(subhalo_mass, subhalo_radius)


def _compute_specific_energy_kms2(r_vec, v_vec, mass_msun, radius_kpc, concentration,
                                   bg_host=None, bg_offset=None):
    """
    Total specific orbital energy (KE + potential), in (km/s)^2, for the
    given position/velocity relative to a host built fresh here from
    mass_msun/radius_kpc, optionally including a background potential
    (bg_host, centered at bg_offset relative to the local host's own
    center). Used both for the verbose per-step boundedness diagnostic
    and for the final_energy_kms2/final_bound fields in
    trace_cluster_to_snapshot's returned result.

    Returns (e_total_kms2, ke_kms2, phi_local_kms2, phi_bg_kms2) -- the
    total (E<0 means bound) plus its three components, so callers that
    want the full breakdown (e.g. the verbose diagnostic) don't need to
    duplicate the calculation, while callers that just want the bottom
    line (e.g. the final result dict) can take e_total_kms2 alone.
    """
    host = NFWHost(mass_msun * u.Msun, radius_kpc * u.kpc, concentration=concentration)
    r_actual_kpc = np.linalg.norm(r_vec.to(u.kpc).value)
    v_actual_kms = np.linalg.norm(v_vec.to(u.km / u.s).value)
    phi_local_kms2 = host.potential(r_actual_kpc) * _KPCGYR_TO_KMS ** 2
    if bg_host is not None:
        r_bg_kpc = np.linalg.norm((r_vec - bg_offset).to(u.kpc).value)
        phi_bg_kms2 = bg_host.potential(r_bg_kpc) * _KPCGYR_TO_KMS ** 2
    else:
        phi_bg_kms2 = 0.0
    ke_kms2 = 0.5 * v_actual_kms ** 2
    e_total_kms2 = ke_kms2 + phi_local_kms2 + phi_bg_kms2
    return e_total_kms2, ke_kms2, phi_local_kms2, phi_bg_kms2


# -------- Follow a single cluster across host mergers, to a target snapshot
def _interpolate_bg(bg_mass_start_msun, bg_radius_start_kpc, bg_offset_start,
                     bg_mass_end_msun, bg_radius_end_kpc, bg_offset_end,
                     frac, concentration):
    """
    Returns (bg_host_k, bg_offset_k) for a sub-step a fraction `frac` of the
    way through a leg, given the background's state at the leg's start and
    end (any of which may be None -- see trace_cluster_to_snapshot).

    Four cases:
      - both None: no background this whole leg -- returns (None, None).
      - both given: ordinary case, a genuinely separate background structure
        the whole leg -- mass/radius/offset are all linearly interpolated
        between their start and end values, so frac=0 exactly reproduces
        the start state and frac=1 exactly reproduces the end state (which,
        by construction, is what the NEXT leg will independently compute as
        ITS OWN start state -- see trace_cluster_to_snapshot's docstring --
        so consecutive legs join up continuously with no jump).
      - start only (fading out): the current host is ABOUT to become the
        main branch itself by this leg's end (or main-branch tracking runs
        out). Rather than holding the background at its full, un-faded
        value all leg and then discontinuously dropping it to nothing at
        the next leg, fade BOTH its mass and its offset to zero together as
        frac->1. Scaling mass to zero (not just moving the offset) is what
        keeps this safe: since the background's mass_enclosed scales with
        bg_mass_k, its force contribution vanishes smoothly regardless of
        where its (now irrelevant) offset ends up -- avoiding any risk of
        double-counting the local host's own gravity as the two structures
        become indistinguishable.
      - end only (fading in): the symmetric, rarer case (see
        trace_cluster_to_snapshot) -- same treatment, mirrored.
    """
    have_start = bg_offset_start is not None
    have_end = bg_offset_end is not None

    if have_start and have_end:
        bg_mass_k = bg_mass_start_msun + (bg_mass_end_msun - bg_mass_start_msun) * frac
        bg_radius_k = bg_radius_start_kpc + (bg_radius_end_kpc - bg_radius_start_kpc) * frac
        bg_offset_k = bg_offset_start + (bg_offset_end - bg_offset_start) * frac
    elif have_start:
        bg_mass_k = bg_mass_start_msun * (1 - frac)
        bg_radius_k = bg_radius_start_kpc
        bg_offset_k = bg_offset_start * (1 - frac)
    elif have_end:
        bg_mass_k = bg_mass_end_msun * frac
        bg_radius_k = bg_radius_end_kpc
        bg_offset_k = bg_offset_end * frac
    else:
        return None, None

    bg_host_k = NFWHost(bg_mass_k * u.Msun, bg_radius_k * u.kpc, concentration=concentration)
    return bg_host_k, bg_offset_k


def _integrate_leg_with_subdivision(cluster_mass, r_vec, v_vec, mass_start, radius_start,
                                     mass_end, radius_end, rel_pos_total, rel_vel_total,
                                     leg_duration, concentration,
                                     bg_mass_start_msun, bg_radius_start_kpc, bg_offset_start,
                                     bg_mass_end_msun, bg_radius_end_kpc, bg_offset_end,
                                     max_leg_duration, hubble_start, hubble_end,
                                     escape_frac=np.inf, record_dt=None, min_bg_substeps=1,
                                     allow_start_inside_stop=False):
    """
    Subdivide a leg into shorter sub-steps rather than treating a
    potentially Gyr-long gap as a single static host/single solve_ivp call
    (which can be extremely slow to integrate in one shot for a bound,
    oscillating orbit spanning many Gyr -- resolving many orbital periods
    inside one adaptive-step call is expensive regardless of the ODE's
    stiffness properties). Linearly interpolates the LOCAL host's
    mass/radius AND the Hubble rate H(z) between their known values at the
    start and end of the leg, and applies the total position/velocity
    reframe (rel_pos_total, rel_vel_total -- the host's own displacement
    over the WHOLE leg) proportionally at each sub-step, rather than all
    at once at the end. The BACKGROUND potential is ALSO linearly
    interpolated across the leg's sub-steps (see _interpolate_bg) -- unlike
    the local host, whose duration-based subdivision this reuses, the
    background is forced into at least min_bg_substeps sub-steps regardless
    of the leg's own duration (see BG_MIN_SUBSTEPS), since otherwise a
    background that's active but has n_sub==1 would still be held fixed at
    a single value for the whole leg.

    allow_start_inside_stop (see integrate_orbit) is passed through
    UNCHANGED to every sub-step's integrate_orbit call, not just the
    first -- so a cluster that's still inside r_stop after one sub-step
    (its dynamics didn't carry it back out within just that one, possibly
    short, sub_dt) keeps getting the exemption for the REST of this leg too,
    rather than spuriously "merging" at the next sub-step boundary purely
    because it hadn't moved out yet.

    Also used for the FINAL (no further host) leg by passing
    mass_end=mass_start, radius_end=radius_start, hubble_end=hubble_start,
    and rel_pos_total/rel_vel_total as zero vectors -- this degenerates to
    simply repeatedly integrating within the SAME unchanging host, which
    is exactly what's needed to cap any single solve_ivp call's duration
    even when there's no "next" host to interpolate toward.

    Returns (status, elapsed, r_vec, v_vec, n_pericenters,
    min_roche_radius_kpc, time_at_min_roche, trajectory) matching
    integrate_orbit's signature, with elapsed being the TOTAL time actually
    used (less than leg_duration if merged/escaped partway through a
    sub-step), n_pericenters the TOTAL pericenter count summed across every
    sub-step, min_roche_radius_kpc the smallest tidal radius found at ANY
    pericenter across every sub-step (None if none occurred),
    time_at_min_roche the time elapsed SINCE THE START OF THIS LEG (i.e.
    this whole _integrate_leg_with_subdivision call, NOT just the one
    sub-step it occurred in) at which that pericenter happened, and
    trajectory the (t_grid, r_grid) pair (see integrate_orbit) with t_grid
    made LOCAL TO THIS WHOLE LEG (not just one sub-step) by shifting each
    sub-step's own t_grid by the cumulative elapsed time of every prior
    sub-step -- None if record_dt was not given.
    """
    n_sub = max(1, int(np.ceil((leg_duration / max_leg_duration).to(u.dimensionless_unscaled).value)))
    bg_active = (bg_offset_start is not None) or (bg_offset_end is not None)
    if bg_active:
        n_sub = max(n_sub, min_bg_substeps)
    sub_dt = leg_duration / n_sub
    total_elapsed = 0 * u.Gyr
    total_pericenters = 0
    min_roche_radius_kpc = None
    time_at_min_roche = None

    track_t_parts = [] if record_dt is not None else None
    track_r_parts = [] if record_dt is not None else None

    for k in range(n_sub):
        frac = k / n_sub
        mass_k = mass_start + (mass_end - mass_start) * frac
        radius_k = radius_start + (radius_end - radius_start) * frac
        hubble_k = hubble_start + (hubble_end - hubble_start) * frac
        host_k = NFWHost(mass_k * u.Msun, radius_k * u.kpc, concentration=concentration)
        bg_host_k, bg_offset_k = _interpolate_bg(
            bg_mass_start_msun, bg_radius_start_kpc, bg_offset_start,
            bg_mass_end_msun, bg_radius_end_kpc, bg_offset_end,
            frac, concentration,
        )

        # only the LAST sub-step of the LAST leg should ever use a real
        # (non-infinite) escape check -- earlier sub-steps always pass
        # escape_frac=np.inf regardless of what the caller asked for, same
        # rule as intermediate legs generally.
        this_escape_frac = escape_frac if k == n_sub - 1 else np.inf

        status, elapsed, r_vec, v_vec, n_peri, roche_this, t_roche_this, traj_sub = integrate_orbit(
            cluster_mass, r_vec, v_vec, host_k, t_max=sub_dt, escape_frac=this_escape_frac,
            background_host=bg_host_k, background_offset=bg_offset_k, hubble_rate=hubble_k,
            record_dt=record_dt, allow_start_inside_stop=allow_start_inside_stop,
        )
        if traj_sub is not None:
            t_sub, r_sub = traj_sub
            # t_sub is local to just this sub-step -- shift by the elapsed
            # time of every PRIOR sub-step (total_elapsed's value from

            # BEFORE this sub-step's own `elapsed` is added below) to make
            # it local to the whole leg instead.
            track_t_parts.append(t_sub + total_elapsed.to(u.Gyr).value)
            track_r_parts.append(r_sub)
        total_pericenters += n_peri
        # NOTE: uses total_elapsed's value from BEFORE this sub-step's
        # `elapsed` is added below, since t_roche_this is local to just
        # this one sub-step -- adding the PRIOR sub-steps' cumulative time
        # converts it to "since the start of this whole leg".
        if roche_this is not None and (min_roche_radius_kpc is None or roche_this < min_roche_radius_kpc):
            min_roche_radius_kpc = roche_this
            time_at_min_roche = total_elapsed + t_roche_this
        total_elapsed += elapsed

        if status in ("merged", "escaped"):
            traj = (np.concatenate(track_t_parts), np.concatenate(track_r_parts)) if record_dt is not None else None
            return status, total_elapsed, r_vec, v_vec, total_pericenters, min_roche_radius_kpc, time_at_min_roche, traj

        r_vec = r_vec + rel_pos_total / n_sub
        v_vec = v_vec + rel_vel_total / n_sub

    traj = (np.concatenate(track_t_parts), np.concatenate(track_r_parts)) if record_dt is not None else None
    return "ongoing", total_elapsed, r_vec, v_vec, total_pericenters, min_roche_radius_kpc, time_at_min_roche, traj


def trace_cluster_to_snapshot(cluster_mass, r0_vec, v0_vec, start_subhalo_id, target_age,
                               navigator, concentration=HOST_CONCENTRATION, max_steps=MAX_STEPS,
                               final_escape_frac=3.0, max_leg_duration=1.0 * u.Gyr,
                               min_host_mass_msun=1e8, verbose=False, record_dt=None,
                               allow_formation_inside_r_stop=True):
    """
    Follow one cluster's dynamical-friction evolution, advancing ONE
    SNAPSHOT AT A TIME along its current host's tree branch, until it
    EITHER inspirals to a host's center, escapes the true final host's
    potential (only checked at the actual end of the tracked tree -- see
    below), or the requested target_age (cosmic age at the output
    snapshot) is reached -- whichever comes first.

    IMPORTANT DESIGN NOTE: earlier versions of this function only
    re-evaluated the host's mass/radius at discrete MERGER events (i.e.
    whenever a branch stopped being the primary progenitor of its
    descendant). That is wrong: a branch can remain the primary
    progenitor of its own growing lineage for a very long time (many Gyr,
    dozens of snapshots) without any formal merger, growing substantially
    in mass/R_vir the whole while. Treating that entire span as "one leg"
    with a single FIXED profile from wherever the cluster happened to land
    used a badly stale (usually far too small) potential for the rest of
    cosmic time, which trivially over-triggers "escaped". This version
    instead refreshes the host's mass/radius (and correctly reframes the
    cluster's position/velocity via the tree's actual relative kinematics)
    at EVERY tracked snapshot the current branch passes through, whether
    or not that particular step happens to be a genuine merger -- so the
    "fixed profile" approximation is only ever stale by one snapshot's
    worth of cosmic time, not potentially the rest of the Hubble time.

    Escape is only ever checked on the TRUE final step of the trace (where
    the tree has no further descendant at all -- typically the z=0 root,
    or wherever this tree's own tracking stops); every other step disables
    escape-checking (escape_frac effectively infinite), since "escaping"
    an intermediate snapshot's potential doesn't mean leaving the larger,
    still-growing structure the cluster is actually embedded in.

    Each leg ALSO includes a background potential (gravity only, no DF)
    from the root's own main-branch progenitor at that same snapshot --
    representing the larger, still-assembling structure the local host is
    itself embedded within. Without this, a cluster only ever feels the
    single small local host's own (often weak, especially at early times)
    potential, and can drift arbitrarily far via ordinary ballistic motion
    over long snapshot gaps -- since we deliberately don't terminate on
    "escape" from an intermediate host (see above), nothing else would
    confine it. This is skipped when the current host is already ON the
    main branch (to avoid double-counting the same structure).

    Each leg ALSO applies HUBBLE DRAG: in an expanding universe, a peculiar
    velocity decays as v~1/a purely from cosmic expansion, with no force
    needed (dv/dt = -H(t)*v for the relative velocity). This matters a lot
    at high redshift, where H(z) is large -- without it, a large peculiar
    velocity correctly recorded in the tree at high-z (real halos genuinely
    have large physical peculiar velocities that early, simply because the
    universe was so much more compact then) never decays the way it
    physically should as the universe expands, and just persists,
    essentially undamped, for the rest of cosmic time.

    Parameters:
        cluster_mass: astropy Quantity (mass)
        r0_vec, v0_vec: astropy Quantity 3-vectors, relative to the
            STARTING host (start_subhalo_id)
        start_subhalo_id: tree-local SubhaloID to start from (typically a
            branch's formation_subhalo_id from the subhalo_formation_table.py CSV)
        target_age: astropy Quantity (Gyr) -- cosmic age at the requested
            output snapshot (from navigator.age_at_snap(output_snap))
        navigator: a MergerTreeNavigator over the raw tree
        concentration: NFW concentration assumed for every host
        final_escape_frac: escape_frac used ONLY on the true final step
        max_leg_duration: astropy Quantity (Gyr) -- legs longer than this
            get subdivided into shorter sub-steps with interpolated host
            mass/radius, rather than treating a potentially Gyr-long gap
            as a single static host the whole way through (see
            _integrate_leg_with_subdivision)
        record_dt: astropy Quantity (time) or None (default). If given, also
            build a full time series of the cluster's separation from the
            CENTRAL/ROOT galaxy (the tree's own z=0 root's main-branch
            position at each snapshot -- i.e. the same reference the
            background-potential term already uses) across the ENTIRE
            trace, sampled every record_dt (see integrate_orbit's
            record_dt and TRACK_TIME_RESOLUTION below). Adds 'track_time_gyr'
            and 'track_radius_kpc' to the returned dict. None (the default)
            skips this at essentially zero extra cost.
        allow_formation_inside_r_stop: bool (default True). Some drawn
            initial separations (from the cluster population sampler) land
            inside r_stop_frac*R_vir already at formation -- i.e. the
            cluster never actually needed to dynamically-friction its way
            to the center. When True (the new default), such a cluster is
            NOT instantly credited with a zero-duration "merger"; it's
            allowed to integrate and evolve normally from there for its
            very first leg (see integrate_orbit's allow_start_inside_stop).
            r_stop is otherwise completely unaffected -- ordinary clusters
            that dynamically friction their way down to r_stop, whether on
            the first leg or any later one, still merge normally the
            instant that happens. Set False to restore the old behavior
            (any cluster starting inside r_stop is an instant t=0 merger).

    Returns a dict:
        status: 'inspiraled' | 'escaped' | 'outskirts'
        total_time_gyr: cumulative cosmic time actually elapsed since the
            cluster's formation, up to whichever of the above stopped it --
            this is what you feed to add_time_evolution as timescale_override
        final_subhalo_id: whichever host the cluster is currently
            associated with when the trace stopped
        final_r_vec, final_v_vec: final position/velocity relative to that host
        final_r_over_rvir: final separation in units of that host's R_vir
            (0 for 'inspiraled', inf for 'escaped')
        n_hops: how many GENUINE mergers it went through (excludes
            same-branch snapshot-to-snapshot continuations)
        n_steps: total number of snapshot-to-snapshot steps taken
        n_pericenters: total number of pericenter passages (radial
            velocity going from infalling to outfalling, relative to
            whichever host is current at the time -- see
            dynamical_friction._make_pericenter_event) detected across the
            ENTIRE trace, summed over every leg/sub-step
        final_energy_kms2: total specific orbital energy (KE + potential,
            including the background term if one was active) AT THE FINAL
            state, in (km/s)^2 -- see _compute_specific_energy_kms2. This
            is a more rigorous boundedness check than final_r_over_rvir
            alone, which can't distinguish a wide-but-bound eccentric
            orbit from genuine unconfined drift.
        final_bound: bool, final_energy_kms2 < 0
        min_roche_radius_kpc: the smallest Jacobi tidal radius (see
            NFWHost.tidal_radius) found at ANY pericenter passage across
            the ENTIRE trace, computed as a cheap post-processing step on
            solve_ivp's already-recorded pericenter-event states (not
            re-evaluated at every integration step). None if the cluster
            never had a pericenter (e.g. it merged/escaped before ever
            completing one).
        time_of_min_roche_gyr: cosmic time elapsed SINCE THE CLUSTER'S
            FORMATION (same convention as total_time_gyr) at which the
            min_roche_radius_kpc pericenter occurred. None under the same
            conditions as min_roche_radius_kpc.
        track_time_gyr, track_radius_kpc: only present if record_dt was
            given. Parallel np.ndarrays: time elapsed SINCE THE CLUSTER'S
            FORMATION (same convention as total_time_gyr -- so e.g. these
            can be directly compared against IMBH_final_formation_time_gyr)
            and the cluster's separation (kpc) from the central/root galaxy
            at that time. Sampled every record_dt, EXCEPT wherever a leg's
            background potential wasn't active (see trace_cluster_to_snapshot's
            docstring on bg_host/bg_offset) -- namely legs where the current
            host itself IS the root's main-branch progenitor, or where the
            main branch isn't tracked back that far -- in which case the
            separation recorded is from the CURRENT host's own center
            instead (the best available proxy for "the center" at that
            point in the trace).
    """
    current_id = start_subhalo_id
    r_vec, v_vec = r0_vec, v0_vec
    _, _, start_snap = navigator.host_properties(start_subhalo_id)
    formation_age = navigator.age_at_snap(start_snap)
    current_age = formation_age
    n_hops = 0
    n_pericenters = 0
    min_roche_radius_kpc = None
    time_of_min_roche_gyr = None
    track_time_parts = [] if record_dt is not None else None
    track_radius_parts = [] if record_dt is not None else None

    def make_result(status, host_id, r, v, r_over_rvir, steps):
        final_energy_kms2, _, _, _ = _compute_specific_energy_kms2(
            r, v, mass_msun, radius_kpc, concentration, bg_host, bg_offset,
        )
        result = {
            'status': status,
            'total_time_gyr': (current_age - formation_age).to(u.Gyr).value,
            'final_subhalo_id': host_id,
            'final_r_vec': r,
            'final_v_vec': v,
            'final_r_over_rvir': r_over_rvir,
            'n_hops': n_hops,
            'n_steps': steps,
            'n_pericenters': n_pericenters,
            'final_energy_kms2': final_energy_kms2,
            'final_bound': bool(final_energy_kms2 < 0),
            'min_roche_radius_kpc': min_roche_radius_kpc,
            'time_of_min_roche_gyr': time_of_min_roche_gyr,
        }
        if record_dt is not None:
            if track_time_parts:
                result['track_time_gyr'] = np.concatenate(track_time_parts)
                result['track_radius_kpc'] = np.concatenate(track_radius_parts)
            else:
                result['track_time_gyr'] = np.array([])
                result['track_radius_kpc'] = np.array([])
        return result

    for step in range(max_steps):
        mass_msun, radius_kpc, snap = navigator.host_properties(current_id)
        hubble_start = navigator.hubble_rate_at_snap(snap)

        # BACKGROUND potential: the larger, still-assembling structure this
        # host is itself embedded within, represented by the root's own
        # main-branch progenitor at this SAME snapshot -- supplies the
        # large-scale confining gravity a single small local host is
        # missing (see dynamical_friction.integrate_orbit's docstring).
        # Skipped if the main branch doesn't reach back this far, or if
        # we're already ON the main branch (would double-count the same
        # structure). Computed HERE, early, rather than further down where
        # it USED to live -- make_result (below) closes over bg_host/
        # bg_offset to compute a final energy diagnostic, and needs them
        # correctly set even on the very first loop iteration, before any
        # of the later per-leg logic runs.
        #
        # bg_mass_msun/bg_radius_kpc/bg_offset here are this leg's START
        # values; bg_host is built from them for use by the non-subdivided
        # path and the energy diagnostic below. The matching END-of-leg
        # values (bg_*_end) are computed further down once next_id/next_snap
        # are known, and together the two let the leg smoothly interpolate
        # the background across itself instead of holding it fixed for the
        # whole leg -- see _interpolate_bg / BG_MIN_SUBSTEPS.
        bg_host, bg_offset = None, None
        bg_mass_msun = bg_radius_kpc = None
        main_branch_id = navigator.main_branch_id_at_snap(snap)
        if main_branch_id is not None and main_branch_id != current_id:
            bg_mass_msun, bg_radius_kpc, _ = navigator.host_properties(main_branch_id)
            rel_pos, _ = navigator.relative_state(current_id, main_branch_id)
            bg_offset = -rel_pos  # root's position relative to current host
            bg_host = NFWHost(bg_mass_msun * u.Msun, bg_radius_kpc * u.kpc, concentration=concentration)

        if verbose:
            r_mag = np.linalg.norm(r_vec.to(u.kpc).value)
            v_mag = np.linalg.norm(v_vec.to(u.km / u.s).value)
            print(f"[step {step}] BEFORE: current_id={current_id}, snap={snap}, "
                  f"host_mass={mass_msun:.3e} Msun, host_R_vir={radius_kpc:.3f} kpc, "
                  f"|r|={r_mag:.4f} kpc ({r_mag/radius_kpc if radius_kpc>0 else float('nan'):.3f} R_vir), "
                  f"|v|={v_mag:.2f} km/s", flush=True)

        remaining = target_age - current_age
        if remaining <= 0 * u.Gyr:
            r_over_rvir = (float(np.linalg.norm(r_vec.to(u.kpc).value)) / radius_kpc
                           if radius_kpc > 0 else np.nan)
            return make_result('outskirts', current_id, r_vec, v_vec, r_over_rvir, step)

        next_id, is_primary = navigator.step_forward(current_id)
        if next_id is not None:
            next_mass_msun, next_radius_kpc, next_snap = navigator.host_properties(next_id)
            hubble_end = navigator.hubble_rate_at_snap(next_snap)
            delta_t_to_next = navigator.age_at_snap(next_snap) - current_age
        else:
            hubble_end = hubble_start
            delta_t_to_next = None

        # END-of-leg background state (see the START-of-leg computation
        # above): what the background would be if evaluated at next_id/
        # next_snap -- i.e. EXACTLY what the FOLLOWING leg will independently
        # compute as ITS OWN start-of-leg background, via the identical
        # main_branch_id_at_snap/relative_state calls. Interpolating THIS
        # leg's background from bg_offset (start) to bg_offset_end here
        # therefore joins up continuously with the next leg's own start
        # value -- no separate bookkeeping needed to "hand off" a value
        # across the loop boundary. No next host (true final step) or no
        # next-id background: hold the start value fixed, matching how
        # hubble_end/next_mass_msun degenerate to their start-of-leg
        # counterparts in that same case.
        bg_mass_end_msun = bg_radius_end_kpc = bg_offset_end = None
        if next_id is not None:
            main_branch_id_end = navigator.main_branch_id_at_snap(next_snap)
            if main_branch_id_end is not None and main_branch_id_end != next_id:
                bg_mass_end_msun, bg_radius_end_kpc, _ = navigator.host_properties(main_branch_id_end)
                rel_pos_end, _ = navigator.relative_state(next_id, main_branch_id_end)
                bg_offset_end = -rel_pos_end
        else:
            bg_mass_end_msun, bg_radius_end_kpc, bg_offset_end = bg_mass_msun, bg_radius_kpc, bg_offset

        if delta_t_to_next is not None and delta_t_to_next <= remaining:
            t_leg_max, capped_by = delta_t_to_next, 'step'
        else:
            t_leg_max, capped_by = remaining, 'output'

        # Only the TRUE final step (no further tracked descendant at all)
        # uses a real escape threshold -- see docstring.
        is_final_step = next_id is None
        escape_frac = final_escape_frac if is_final_step else np.inf

        # Subdivide ANY leg (whether it completes to next_id, or is capped
        # by running out of target_age within the CURRENT host) that spans
        # longer than max_leg_duration -- rather than treating a
        # potentially Gyr-long gap as a single static host/single solve_ivp
        # call. For 'step'-capped legs this interpolates mass/radius/H
        # toward next_id and applies the reframe proportionally; for
        # 'output'-capped legs (including the TRUE final step) there's no
        # "next" host, so it degenerates to repeatedly integrating within
        # the SAME unchanging host (zero reframe each sub-step) -- this
        # still matters because resolving a bound, oscillating orbit over
        # many Gyr in one uninterrupted solve_ivp call can be extremely
        # slow regardless of the leg's stiffness properties.
        #
        # ALSO subdivide (regardless of duration) whenever a background is
        # active at either end of the leg -- see BG_MIN_SUBSTEPS -- so its
        # mass/radius/offset get interpolated across the leg rather than
        # held fixed and then discretely reset at the next boundary.
        bg_active = (bg_offset is not None) or (bg_offset_end is not None)
        subdivided = (t_leg_max > max_leg_duration) or bg_active
        rel_pos_full = rel_vel_full = None
        if next_id is not None:
            rel_pos_full, rel_vel_full = navigator.relative_state(current_id, next_id)

        if verbose:
            if bg_host is not None:
                bg_str = f"bg_mass={bg_mass_msun:.3e} Msun, bg_offset={np.linalg.norm(bg_offset.to(u.kpc).value):.3f} kpc"
            elif main_branch_id is None:
                bg_str = "none (main branch not tracked back this far)"
            elif main_branch_id == current_id:
                bg_str = "none (current host IS the main branch here)"
            else:
                bg_str = "none"
            rp_str = (f"|rel_pos|={np.linalg.norm(rel_pos_full.to(u.kpc).value):.4f} kpc, "
                      f"|rel_vel|={np.linalg.norm(rel_vel_full.to(u.km/u.s).value):.2f} km/s"
                      if rel_pos_full is not None else "n/a (final step)")
            print(f"          leg: capped_by={capped_by}, t_leg_max={t_leg_max:.4f}, "
                  f"is_primary={is_primary}, subdivided={subdivided}, background: {bg_str}, "
                  f"H(z)={hubble_start.to(1/u.Gyr).value:.4f} /Gyr, reframe: {rp_str}", flush=True)

            # ENERGY-BASED BOUNDEDNESS DIAGNOSTIC: E = KE + Phi_local (+ Phi_bg
            # if a background potential is active), evaluated at the CURRENT
            # r_vec/v_vec (i.e. state at the START of this leg, before this
            # step's integration). E<0 means genuinely gravitationally bound
            # (to the local host, plus the background if present) RIGHT NOW,
            # independent of where r sits relative to R_vir -- unlike r/Rvir,
            # this doesn't get confused by a wide but bound eccentric orbit.
            e_total_kms2, ke_kms2, phi_local_kms2, phi_bg_kms2 = _compute_specific_energy_kms2(
                r_vec, v_vec, mass_msun, radius_kpc, concentration, bg_host, bg_offset,
            )
            bound_str = "BOUND" if e_total_kms2 < 0 else "UNBOUND"
            print(f"          energy: KE={ke_kms2:.2f}, Phi_local={phi_local_kms2:.2f}, "
                  f"Phi_bg={phi_bg_kms2:.2f}, E_total={e_total_kms2:.2f} (km/s)^2 -> {bound_str}",
                  flush=True)

        age_at_leg_start = current_age  # for converting a LOCAL pericenter time (relative to
                                         # the start of THIS leg) into an absolute one, below
        # Only the cluster's very FIRST leg (step==0) ever gets the
        # allow_start_inside_stop exemption -- see
        # allow_formation_inside_r_stop's docstring above. Harmless to pass
        # for every step==0 sub-step/call regardless of whether r_vec
        # actually starts inside r_stop (integrate_orbit only acts on it
        # when relevant), and False from step==1 onward restores ordinary
        # r_stop behavior for any genuine later merger.
        allow_start_inside_stop = allow_formation_inside_r_stop and (step == 0)
        leg_traj = None
        if t_leg_max <= 0 * u.Gyr:
            # Degenerate leg (e.g. delta_t_to_next<=0 from a tree quirk) --
            # skip integration entirely, same short-circuit the original
            # non-subdivided path used. Checked here, BEFORE the subdivided
            # branch, since background-driven forced subdivision (bg_active
            # above) would otherwise route this into
            # _integrate_leg_with_subdivision with a zero/negative
            # sub_dt -- solve_ivp isn't guaranteed to handle that cleanly.
            # Force subdivided=False here (even if bg_active made it True
            # above) so the reframe logic below -- which assumes "subdivided
            # means the reframe was already applied incrementally inside
            # _integrate_leg_with_subdivision" -- still applies the full
            # manual reframe for this leg, since no incremental version ran.
            status, elapsed, n_peri, roche_this, t_roche_this = "ongoing", 0 * u.Gyr, 0, None, None
            subdivided = False
        elif subdivided:
            if capped_by == 'step':
                status, elapsed, r_vec, v_vec, n_peri, roche_this, t_roche_this, leg_traj = _integrate_leg_with_subdivision(
                    cluster_mass, r_vec, v_vec, mass_msun, radius_kpc, next_mass_msun, next_radius_kpc,
                    rel_pos_full, rel_vel_full, t_leg_max, concentration,
                    bg_mass_msun, bg_radius_kpc, bg_offset, bg_mass_end_msun, bg_radius_end_kpc, bg_offset_end,
                    max_leg_duration, hubble_start, hubble_end, record_dt=record_dt,
                    min_bg_substeps=BG_MIN_SUBSTEPS, allow_start_inside_stop=allow_start_inside_stop,
                )
            else:
                # 'output'-capped (including the true final step): no next
                # host to interpolate toward -- degenerate to repeatedly
                # integrating within the SAME unchanging host, zero reframe
                # for the LOCAL host (the background, by contrast, may still
                # be actively interpolating if bg_offset != bg_offset_end --
                # e.g. a background that's fading out during this final leg).
                zero_pos = np.zeros(3) * u.kpc
                zero_vel = np.zeros(3) * u.km / u.s
                status, elapsed, r_vec, v_vec, n_peri, roche_this, t_roche_this, leg_traj = _integrate_leg_with_subdivision(
                    cluster_mass, r_vec, v_vec, mass_msun, radius_kpc, mass_msun, radius_kpc,
                    zero_pos, zero_vel, t_leg_max, concentration,
                    bg_mass_msun, bg_radius_kpc, bg_offset, bg_mass_end_msun, bg_radius_end_kpc, bg_offset_end,
                    max_leg_duration, hubble_start, hubble_start, escape_frac=escape_frac, record_dt=record_dt,
                    min_bg_substeps=BG_MIN_SUBSTEPS, allow_start_inside_stop=allow_start_inside_stop,
                )
        else:
            # only reached when NOT subdivided, i.e. t_leg_max <= max_leg_duration
            # AND no background active at either end of this leg (bg_active
            # is False) -- a single un-subdivided integrate_orbit call is
            # safe here precisely because there's no background to smooth.
            host = NFWHost(mass_msun * u.Msun, radius_kpc * u.kpc, concentration=concentration)
            status, elapsed, r_vec, v_vec, n_peri, roche_this, t_roche_this, leg_traj = integrate_orbit(
                cluster_mass, r_vec, v_vec, host, t_max=t_leg_max, escape_frac=escape_frac,
                background_host=bg_host, background_offset=bg_offset, hubble_rate=hubble_start,
                record_dt=record_dt, allow_start_inside_stop=allow_start_inside_stop,
            )
        current_age = current_age + elapsed
        n_pericenters += n_peri
        if roche_this is not None and (min_roche_radius_kpc is None or roche_this < min_roche_radius_kpc):
            min_roche_radius_kpc = roche_this
            time_of_min_roche_gyr = (age_at_leg_start + t_roche_this - formation_age).to(u.Gyr).value
        if leg_traj is not None:
            # leg_traj's times are LOCAL to this leg (t=0 at age_at_leg_start)
            # -- shift to "time since cluster formation", matching
            # total_time_gyr's convention, so this lines up directly against
            # IMBH_final_formation_time_gyr downstream (see analysis.py).
            t_leg, r_leg = leg_traj
            track_time_parts.append(t_leg + (age_at_leg_start - formation_age).to(u.Gyr).value)
            track_radius_parts.append(r_leg)

        if verbose:
            r_mag_post_integrate = np.linalg.norm(r_vec.to(u.kpc).value)
            v_mag_post_integrate = np.linalg.norm(v_vec.to(u.km / u.s).value)
            print(f"          AFTER integrate: status={status}, elapsed={elapsed:.4f}, "
                  f"|r| (before reframe)={r_mag_post_integrate:.4f} kpc, "
                  f"|v| (before reframe)={v_mag_post_integrate:.2f} km/s", flush=True)

        if status == "merged":
            return make_result('inspiraled', current_id, r_vec, v_vec, 0.0, step)
        if status == "escaped":
            # only reachable when is_final_step is True, given escape_frac=inf otherwise
            return make_result('escaped', current_id, r_vec, v_vec, np.inf, step)

        # status == "ongoing"
        if capped_by == 'step':
            if not subdivided:
                # subdivision (if used) already applied the reframe incrementally
                r_vec = r_vec + rel_pos_full
                v_vec = v_vec + rel_vel_full
            if verbose:
                r_mag_post_reframe = np.linalg.norm(r_vec.to(u.kpc).value)
                v_mag_post_reframe = np.linalg.norm(v_vec.to(u.km / u.s).value)
                print(f"          AFTER reframe: |r|={r_mag_post_reframe:.4f} kpc, "
                      f"|v|={v_mag_post_reframe:.2f} km/s", flush=True)
            if not is_primary:
                n_hops += 1
            current_id = next_id
            current_age = navigator.age_at_snap(navigator.snap_of(next_id))
            continue
        else:
            r_over_rvir = (float(np.linalg.norm(r_vec.to(u.kpc).value)) / radius_kpc
                           if radius_kpc > 0 else np.nan)
            return make_result('outskirts', current_id, r_vec, v_vec, r_over_rvir, step)

    r_over_rvir = (float(np.linalg.norm(r_vec.to(u.kpc).value)) / radius_kpc
                   if radius_kpc > 0 else np.nan)
    return make_result('outskirts', current_id, r_vec, v_vec, r_over_rvir, max_steps)


#-------- time evolution for the IMBH analytic model
# 
def create_timescale_model(mass, radius, Nsampling = 5):
    # build_single_system_grid expects `radius` to be r0 of a power-law
    # density profile (rho ~ r^-alpha), with `mass` being the mass
    # ENCLOSED WITHIN r0 -- but the cluster sampler gives us the HALF-MASS
    # radius r_1/2 (paired with the cluster's FULL mass). For a pure
    # power law, M(r) ~ r^(3-alpha), so M(r1)/M(r2) = (r1/r2)^(3-alpha)
    # for any two radii on the same profile. Setting M(r_1/2)=mass/2 and
    # M(r0)=mass (since `mass` is what we're passing in as the enclosed
    # mass at r0) gives (r_1/2 / r0)^(3-alpha) = 1/2, i.e.
    # r0 = r_1/2 * 2^(1/(3-alpha)). Requires alpha < 3 (needed for the
    # power-law mass integral to converge at r->0 in the first place).
    radius = radius * 2 ** (1. / (3 - alpha))
    grid = build_single_system_grid(mass,radius)
    model_nobh = TimescaleEnsemble(grid, 
                        verbose = False,
                        densityModel="power-law",
                        Nsampling = Nsampling,
                        timescales_kwargs={'cosmology':cosmo},
                        profile_kwargs={"alpha":alpha})
    return model_nobh

def add_time_evolution(delta_t, model):
    output = create_dynamical_model_integral(model,verbose = False, timescale_override = delta_t, merger_override = True)
    output['rho0_msun_pc3'] = model.profiles[0].rho0.value
    output['r0_pc'] = model.profiles[0].r0.value
    return output


def _run_timescale_model(mass, radius, delta_t):
    """Bundles create_timescale_model + add_time_evolution into one call for run_with_timeout()."""
    model = create_timescale_model(mass, radius)
    return add_time_evolution(delta_t, model)


def save_radius_track(track_time_gyr, track_radius_kpc, halo_idx, cluster_idx, track_dir):
    """
    Save one cluster's separation-from-central-galaxy time series (see
    trace_cluster_to_snapshot's record_dt/track_time_gyr/track_radius_kpc)
    to its own small CSV, named so it can be uniquely matched back to its
    row in the main per-cluster output table (see save_cluster_output's
    'radius_track_path' column). Returns the path written.

    One file per qualifying cluster (rather than one shared file) keeps
    each file small/simple and avoids re-writing a giant combined table
    every time a single cluster's trace changes -- with clusters at this
    resolution numbering in the thousands-to-millions across a full run,
    consider a more compact format (e.g. one .npz/.hdf5 per HALO rather
    than per cluster) if the sheer file COUNT becomes a filesystem problem.
    """
    path = os.path.join(track_dir, f"radius_track_h{halo_idx}_c{cluster_idx}.csv")
    track_df = pd.DataFrame({
        'time_since_formation_gyr': track_time_gyr,
        'separation_from_host_kpc': track_radius_kpc,
    })
    track_df.to_csv(path, index=False)
    return path


# ------- Iteration over all the subhalos
def iterate_subhalos(df, goodidx, navigator, target_age, debug_trace=False,
                      record_dt=None, track_dir=None, track_min_mass=TRACK_MIN_IMBH_MASS_MSUN,
                      allow_formation_inside_r_stop=True):
    """
    Parameters (new ones only -- see the rest of the module for the others):
        record_dt: astropy Quantity (time) or None (default). Passed straight
            through to trace_cluster_to_snapshot -- if given, every cluster's
            orbit trace also builds a full separation-from-central-galaxy
            time series (see that function's docstring). This alone does
            NOT save anything to disk; it just makes the series available
            in-memory so the mass check below can decide whether to keep it.
        track_dir: directory to save qualifying clusters' radius tracks
            into (one CSV per cluster). Required if record_dt is given;
            ignored otherwise. Created by the caller (main()), not here.
        track_min_mass: only clusters with IMBH_mass_msun >= this get their
            track actually written to disk (see TRACK_MIN_IMBH_MASS_MSUN).
        allow_formation_inside_r_stop: passed straight through to
            trace_cluster_to_snapshot -- see its docstring.
    """
    clusters = []
    failures = []  # (halo_idx, cluster_idx, total_time_gyr, status, error_message) -- orbit trace failures only
    #testing mode-just do the first few
    for idx in goodidx:
        cluster_props = draw_clusters(df['group_m_crit200_msun'][idx], df['group_r_crit200_kpc'][idx])
        print("Generated " + str(len(cluster_props['cluster_mass'])) + " clusters for this halo.")

        start_subhalo_id = int(df['formation_subhalo_id'][idx])

        cluster_props['status'] = []
        cluster_props['total_time_gyr'] = []
        cluster_props['final_subhalo_id'] = []
        cluster_props['initial_subhalo_id'] = []
        cluster_props['final_r_over_rvir'] = []
        cluster_props['n_hops'] = []
        cluster_props['n_steps'] = []
        cluster_props['n_pericenters'] = []
        cluster_props['final_energy_kms2'] = []
        cluster_props['final_bound'] = []
        cluster_props['min_roche_radius_kpc'] = []
        cluster_props['time_of_min_roche_gyr'] = []
        cluster_props['IMBH_mass'] = []
        cluster_props['IMBH_final_formation_time']=[]
        cluster_props['which_final_formation_time']=[]
        cluster_props['initial_subhalo_mass_msun']=[]
        cluster_props['initial_subhalo_formation_redshift']=[]
        cluster_props['rho0_msun_pc3'] =[]
        cluster_props['r0_pc'] =[]
        # path to this cluster's saved separation-from-central-galaxy time
        # series (see TRACK_MIN_IMBH_MASS_MSUN), or None if it wasn't
        # saved (below-threshold IMBH mass, or --save-radius-tracks not
        # requested at all, or the orbit trace itself failed).
        cluster_props['radius_track_path'] = []
        for clusteridx in range(len(cluster_props['cluster_mass'])):
            m_cl = cluster_props['cluster_mass'][clusteridx]
            r0_vec = cluster_props['cluster_sep'][clusteridx]
            v0_vec = cluster_props['cluster_vel'][clusteridx]

            if debug_trace:
                # TEMPORARY diagnostic: print identifying info + exact initial
                # conditions BEFORE the call, so if this specific cluster hangs
                # or crashes, we know exactly what to reproduce standalone
                # without waiting for the run to finish (or timeout) first.
                print(f"    [DEBUG] starting trace: halo_idx={idx}, cluster_idx={clusteridx}, "
                      f"start_subhalo_id={start_subhalo_id}", flush=True)
                print(f"    [DEBUG]   m_cl={m_cl.to(u.Msun).value:.6e} Msun, "
                      f"r0_vec={r0_vec.to(u.kpc).value} kpc, "
                      f"v0_vec={v0_vec.to(u.km/u.s).value} km/s", flush=True)

            trace, trace_err = run_with_timeout(
                trace_cluster_to_snapshot, ORBIT_TRACE_TIMEOUT_S,
                m_cl, r0_vec, v0_vec, start_subhalo_id, target_age, navigator,
                verbose=debug_trace, record_dt=record_dt,
                allow_formation_inside_r_stop=allow_formation_inside_r_stop,
            )
            if trace_err is not None:
                print(f"    WARNING: orbit trace failed/timed out ({trace_err}) -- "
                      f"logging and skipping this cluster.")
                failures.append((idx, clusteridx, np.nan, "trace_failed", trace_err))
                cluster_props['status'].append("trace_failed")
                cluster_props['total_time_gyr'].append(np.nan)
                cluster_props['final_subhalo_id'].append(None)
                cluster_props['initial_subhalo_id'].append(idx)
                cluster_props['initial_subhalo_mass_msun'].append(df['group_m_crit200_msun'][idx])
                cluster_props['initial_subhalo_formation_redshift'].append(df['formation_redshift'][idx])
                cluster_props['final_r_over_rvir'].append(np.nan)
                cluster_props['n_hops'].append(None)
                cluster_props['n_steps'].append(None)
                cluster_props['n_pericenters'].append(None)
                cluster_props['final_energy_kms2'].append(np.nan)
                cluster_props['final_bound'].append(None)
                cluster_props['min_roche_radius_kpc'].append(None)
                cluster_props['time_of_min_roche_gyr'].append(None)
                cluster_props['IMBH_mass'].append(np.nan)
                cluster_props['IMBH_final_formation_time'].append(np.nan)
                cluster_props['which_final_formation_time'].append(None)
                cluster_props['r0_pc'].append(None)
                cluster_props['rho0_msun_pc3'].append(None)
                cluster_props['radius_track_path'].append(None)

                continue

            print(f"    status={trace['status']}, total_time={trace['total_time_gyr']:.4f} Gyr, "
                  f"n_hops={trace['n_hops']}, n_steps={trace['n_steps']}, "
                  f"n_pericenters={trace['n_pericenters']}, final_bound={trace['final_bound']}, "
                  f"min_roche_radius_kpc={trace['min_roche_radius_kpc']}, "
                  f"final_host={trace['final_subhalo_id']}, r/Rvir={trace['final_r_over_rvir']:.4f}")

            cluster_props['status'].append(trace['status'])
            cluster_props['total_time_gyr'].append(trace['total_time_gyr'])
            cluster_props['final_subhalo_id'].append(trace['final_subhalo_id'])
            cluster_props['initial_subhalo_id'].append(idx)
            cluster_props['initial_subhalo_mass_msun'].append(df['group_m_crit200_msun'][idx])
            cluster_props['initial_subhalo_formation_redshift'].append(df['formation_redshift'][idx])

            cluster_props['final_r_over_rvir'].append(trace['final_r_over_rvir'])
            cluster_props['n_hops'].append(trace['n_hops'])
            cluster_props['n_steps'].append(trace['n_steps'])
            cluster_props['n_pericenters'].append(trace['n_pericenters'])
            cluster_props['final_energy_kms2'].append(trace['final_energy_kms2'])
            cluster_props['final_bound'].append(trace['final_bound'])
            cluster_props['min_roche_radius_kpc'].append(trace['min_roche_radius_kpc'])
            cluster_props['time_of_min_roche_gyr'].append(trace['time_of_min_roche_gyr'])


            out, err = run_with_timeout(
                _run_timescale_model, TIMESCALES_CALL_TIMEOUT_S,
                cluster_props['cluster_mass'][clusteridx],
                cluster_props['cluster_radius'][clusteridx],
                trace['total_time_gyr'] * u.Gyr,
            )
            if err is not None:
                print(f"    WARNING: timescales call failed/timed out ({err}) -- "
                      f"using NaN for this cluster's IMBH_mass.")
                cluster_props['IMBH_mass'].append(np.nan)
                cluster_props['IMBH_final_formation_time'].append(np.nan)
                cluster_props['which_final_formation_time'].append(None)
                cluster_props['radius_track_path'].append(None)
                continue
            imbh_mass_out = out['M_VMS'][0]
            cluster_props['IMBH_mass'].append(imbh_mass_out)
            cluster_props['IMBH_final_formation_time'].append(out['minimum_disruption_time'][0])
            cluster_props['which_final_formation_time'].append(out['which_disruption_time'][0])
            cluster_props['r0_pc'].append(out['r0_pc'])
            cluster_props['rho0_msun_pc3'].append(out['rho0_msun_pc3'])

            # `imbh_mass_out` may be a plain float OR an astropy Quantity
            # (with units of Msun) depending on the `timescales` package's
            # own version/behavior -- normalize to a plain float here for
            # the threshold comparison below, regardless of which it is.
            imbh_mass_msun = (imbh_mass_out.to(u.Msun).value if hasattr(imbh_mass_out, 'to')
                               else float(imbh_mass_out))

            track_path = None
            if record_dt is not None and not np.isnan(imbh_mass_msun) and imbh_mass_msun >= track_min_mass:
                track_path = save_radius_track(
                    trace['track_time_gyr'], trace['track_radius_kpc'], idx, clusteridx, track_dir,
                )
            cluster_props['radius_track_path'].append(track_path)
        clusters.append(cluster_props)

    if failures:
        print(f"\n{len(failures)} cluster(s) had a failed/timed-out orbit trace:")
        for halo_idx, cl_idx, ttime, status, err in failures:
            print(f"  halo_idx={halo_idx}, cluster={cl_idx}, total_time={ttime:.4f} Gyr, "
                  f"status={status}, error={err}")

    return clusters


def save_cluster_output(output_clusters, path, file_format="pickle"):
    """
    Flatten output_clusters (a list of per-halo dicts, each holding
    per-cluster lists/arrays -- the return value of iterate_subhalos) into
    a single table with ONE ROW PER CLUSTER (across every halo processed),
    and save it.

    REQUIRES the length-mismatch fix in iterate_subhalos (every key
    appended exactly once per cluster, on every code path) -- otherwise
    the lists for a given halo won't all be the same length and this will
    either raise or (worse) silently misalign columns.

    Column naming:
        - Scalar astropy Quantity fields get a unit suffix baked into the
          column name (e.g. 'cluster_mass_msun') and are stored as plain
          floats -- makes the saved table self-describing and usable
          without needing astropy to read it back.
        - Vector (N,3) fields (cluster_sep, cluster_vel -- the cluster's
          initial position/velocity relative to its FORMATION host) are
          expanded into three separate component columns (e.g.
          'cluster_sep_x_kpc', '..._y_kpc', '..._z_kpc') rather than kept
          as a single array-valued column -- far more usable downstream in
          pandas/numpy, and required at all for a plain-text CSV export.
        - Fields that can be missing on a failed cluster (final_subhalo_id,
          n_hops, n_steps, n_pericenters, min_roche_radius_kpc,
          time_of_min_roche_gyr, which_final_formation_time) are stored as
          NaN (numeric fields) or None (which_final_formation_time, since
          its type isn't fixed here) rather than raising or misaligning
          rows. min_roche_radius_kpc/time_of_min_roche_gyr are ALSO NaN on
          an otherwise-successful trace that never had a pericenter at
          all (e.g. merged/escaped before completing one) -- NaN there
          doesn't necessarily mean the trace itself failed.
        - IMBH_mass and IMBH_final_formation_time are stored as-is,
          assumed to already be plain floats (matching how the rest of
          this script handles them, with no .to()/.value conversion
          anywhere) -- if the `timescales` package actually returns
          astropy Quantities for these, adjust this function accordingly.
        - radius_track_path is None for every cluster unless
          --save-radius-tracks was requested AND that specific cluster's
          IMBH_mass_msun cleared track_min_mass (see iterate_subhalos) --
          stored here as an empty string rather than NaN, since it's a
          path/string field, not numeric.

    Parameters:
        output_clusters: list of per-halo dicts, as returned by iterate_subhalos.
        path: output file path.
        file_format: "pickle" (default -- saves a pandas DataFrame via
            pickle, matching this pipeline's existing .dat convention for
            clusters_near_halos_exclusive_*.dat, and preserves dtypes
            exactly) or "csv" (plain text, more portable/human-readable,
            but a missing which_final_formation_time becomes an empty
            field and dtypes get re-inferred on reload).

    Returns the assembled pandas DataFrame (so you can inspect/use it
    directly without re-loading the saved file).
    """
    rows = []
    for cluster_props in output_clusters:
        n = len(cluster_props['cluster_mass'])
        for i in range(n):
            final_id = cluster_props['final_subhalo_id'][i]
            n_hops = cluster_props['n_hops'][i]
            n_steps = cluster_props['n_steps'][i]
            n_pericenters = cluster_props['n_pericenters'][i]
            final_bound = cluster_props['final_bound'][i]
            min_roche_radius_kpc = cluster_props['min_roche_radius_kpc'][i]
            time_of_min_roche_gyr = cluster_props['time_of_min_roche_gyr'][i]
            rows.append({
                'cluster_mass_msun': cluster_props['cluster_mass'][i].to(u.Msun).value,
                'cluster_radius_pc': cluster_props['cluster_radius'][i].to(u.pc).value,
                'cluster_sep_x_kpc': cluster_props['cluster_sep'][i, 0].to(u.kpc).value,
                'cluster_sep_y_kpc': cluster_props['cluster_sep'][i, 1].to(u.kpc).value,
                'cluster_sep_z_kpc': cluster_props['cluster_sep'][i, 2].to(u.kpc).value,
                'cluster_vel_x_kms': cluster_props['cluster_vel'][i, 0].to(u.km / u.s).value,
                'cluster_vel_y_kms': cluster_props['cluster_vel'][i, 1].to(u.km / u.s).value,
                'cluster_vel_z_kms': cluster_props['cluster_vel'][i, 2].to(u.km / u.s).value,
                'status': cluster_props['status'][i],
                'total_time_gyr': cluster_props['total_time_gyr'][i],
                'initial_subhalo_id': cluster_props['initial_subhalo_id'][i],
                'initial_subhalo_mass_msun': cluster_props['initial_subhalo_mass_msun'][i],
                'initial_subhalo_formation_redshift': cluster_props['initial_subhalo_formation_redshift'][i],
                'final_subhalo_id': final_id if final_id is not None else np.nan,
                'final_r_over_rvir': cluster_props['final_r_over_rvir'][i],
                'n_hops': n_hops if n_hops is not None else np.nan,
                'n_steps': n_steps if n_steps is not None else np.nan,
                'n_pericenters': n_pericenters if n_pericenters is not None else np.nan,
                'final_energy_kms2': cluster_props['final_energy_kms2'][i],
                'final_bound': final_bound if final_bound is not None else np.nan,
                'min_roche_radius_kpc': min_roche_radius_kpc if min_roche_radius_kpc is not None else np.nan,
                'time_of_min_roche_gyr': time_of_min_roche_gyr if time_of_min_roche_gyr is not None else np.nan,
                'IMBH_mass_msun': cluster_props['IMBH_mass'][i].to('Msun').value,
                'IMBH_final_formation_time_gyr': cluster_props['IMBH_final_formation_time'][i].to('Gyr').value,
                'which_final_formation_time': cluster_props['which_final_formation_time'][i],
                'rho0_msun_pc3': cluster_props['rho0_msun_pc3'][i],
                'r0_pc': cluster_props['r0_pc'][i],
                'radius_track_path': cluster_props['radius_track_path'][i] or '',
            })

    df = pd.DataFrame(rows)

    if file_format == "pickle":
        df.to_pickle(path)
    elif file_format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown file_format {file_format!r}; use 'pickle' or 'csv'")

    n_halos = len(output_clusters)
    print(f"Saved {len(df)} clusters ({n_halos} halos) to {path} ({file_format})")
    return df


def summarize_status(clusters):
    """Counts of IMBHs by status (inspiraled/escaped/outskirts) across all clusters drawn."""
    counts = {}
    total = 0
    for cluster_props in clusters:
        for status in cluster_props['status']:
            counts[status] = counts.get(status, 0) + 1
            total += 1
    print(f"\n=== IMBH status summary ({total} clusters total) ===")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}  ({100*n/total:.1f}%)")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the subhalo_formation_<ID>.csv file")
    parser.add_argument("--tree-path", default=None,
                         help="Path to the raw sublink_full_<ID>.hdf5 tree "
                              "(default: guessed from csv_path's <ID>, in tng_download/)")
    parser.add_argument("--snap-redshift-path", default="tng_download/snapshot_redshifts.json",
                         help="Path to snapshot_redshifts.json")
    parser.add_argument("--box-size", type=float, default=35000.0,
                         help="Simulation box size, comoving ckpc/h (default: 35000, TNG50's box)")
    parser.add_argument("--output-snap", type=int, required=True,
                         help="Illustris/TNG snapshot number to report IMBH status at")
    parser.add_argument("--debug-trace", action="store_true",
                         help="TEMPORARY diagnostic: print each cluster's exact starting "
                              "parameters before its trace begins, and the full per-step "
                              "trace_cluster_to_snapshot verbose output, in real time -- so "
                              "if a specific cluster hangs or crashes, you see exactly which "
                              "one and its exact state without waiting for the run (or its "
                              "timeout) to finish. Very noisy; meant to be removed/disabled "
                              "once the issue being chased is resolved.")
    parser.add_argument("--save-path", default=None,
                         help="Where to save the per-cluster output table (default: "
                              "'cluster_output_<tree_id>_snap<output_snap>.dat' next to "
                              "the input CSV)")
    parser.add_argument("--save-format", choices=["pickle", "csv"], default="pickle",
                         help="Format for --save-path (default: pickle, matching this "
                              "pipeline's existing .dat convention; use csv for a "
                              "plain-text/portable table instead)")
    parser.add_argument("--save-radius-tracks", action="store_true",
                         help="Also save, for every cluster whose IMBH_mass_msun ends up "
                              f">= --track-min-mass, a full time series of its separation "
                              "from the central/root galaxy (one CSV per qualifying cluster, "
                              f"time resolution {TRACK_TIME_RESOLUTION.to(u.yr).value:.0e} yr -- "
                              "see TRACK_TIME_RESOLUTION). Downstream, analysis.py uses these "
                              "to report the TDE rate as a function of galactocentric radius, "
                              "not just cosmic time. Off by default: this both slows the run "
                              "(finely-sampled dense ODE output for every cluster, not just "
                              "the ones that end up qualifying) and uses noticeably more disk "
                              "space (one file per qualifying cluster, each up to ~1e5 rows).")
    parser.add_argument("--track-min-mass", type=float, default=TRACK_MIN_IMBH_MASS_MSUN,
                         help="Minimum IMBH_mass_msun for --save-radius-tracks to actually "
                              f"write a cluster's track to disk (default: {TRACK_MIN_IMBH_MASS_MSUN:.0f} "
                              "-- matches analysis.py's own small-BH cutoff, below which no TDE "
                              "rate gets computed for that cluster anyway).")
    parser.add_argument("--track-dir", default=None,
                         help=f"Directory to save --save-radius-tracks output into (default: "
                              f"'{TRACK_OUTPUT_DIRNAME}' next to the input CSV). Pass this same "
                              "path to analysis.py's --track-dir.")
    parser.add_argument("--no-continue-formed-inside-rstop", action="store_true",
                         help="Restore the OLD behavior for clusters whose drawn initial separation "
                              "already lands inside r_stop_frac*R_vir at formation: instantly credit "
                              "them with a zero-duration 'inspiraled' merger (total_time_gyr=0), rather "
                              "than letting them integrate/evolve normally from there (the new default "
                              "-- see trace_cluster_to_snapshot's allow_formation_inside_r_stop). "
                              "r_stop itself is unaffected either way for clusters that dynamically "
                              "friction their way down to it during the trace.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    required = {"delta_t_gyr", "halfmass_rad_kpc", "dm_mass_msun", "formation_subhalo_id"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing expected column(s): {missing}")
    print(f"Loaded {len(df)} subhalo branches from {args.csv_path}")

    tree_path = args.tree_path
    if tree_path is None:
        base = os.path.basename(args.csv_path)
        # subhalo_formation_<ID>.csv -> tng_download/sublink_full_<ID>.hdf5
        tree_id = base.replace("subhalo_formation_", "").replace(".csv", "")
        tree_path = os.path.join("tng_download", f"sublink_full_{tree_id}.hdf5")
        print(f"--tree-path not given, guessing: {tree_path}")

    navigator = MergerTreeNavigator(tree_path, args.snap_redshift_path, box_size_ckpc_h=args.box_size)
    target_age = navigator.age_at_snap(args.output_snap)
    print(f"Output snapshot {args.output_snap} -> cosmic age {target_age:.4f}")

    goodidx = load_merger_tree_idx(df)

    record_dt = TRACK_TIME_RESOLUTION if args.save_radius_tracks else None
    track_dir = args.track_dir
    if track_dir is None:
        track_dir = os.path.join(os.path.dirname(args.csv_path) or ".", TRACK_OUTPUT_DIRNAME)
    if args.save_radius_tracks:
        os.makedirs(track_dir, exist_ok=True)
        print(f"--save-radius-tracks enabled: tracks (IMBH_mass_msun >= {args.track_min_mass:.0f}) "
              f"will be saved under {track_dir}")

    allow_formation_inside_r_stop = not args.no_continue_formed_inside_rstop
    if not allow_formation_inside_r_stop:
        print("--no-continue-formed-inside-rstop set: clusters formed inside r_stop will be treated "
              "as instant t=0 mergers (old behavior).")

    output_clusters = iterate_subhalos(df, goodidx, navigator, target_age, debug_trace=args.debug_trace,
                                        record_dt=record_dt, track_dir=track_dir,
                                        track_min_mass=args.track_min_mass,
                                        allow_formation_inside_r_stop=allow_formation_inside_r_stop)
    summarize_status(output_clusters)

    save_path = args.save_path
    if save_path is None:
        base = os.path.basename(args.csv_path)
        tree_id = base.replace("subhalo_formation_", "").replace(".csv", "")
        ext = "dat" if args.save_format == "pickle" else "csv"
        save_path = os.path.join(os.path.dirname(args.csv_path) or ".",
                                  f"cluster_output_{tree_id}_snap{args.output_snap}.{ext}")
        print(f"--save-path not given, using: {save_path}")
    save_cluster_output(output_clusters, save_path, file_format=args.save_format)


if __name__ == "__main__":
    main()