import argparse
import os
import signal
import sys
sys.path.append("/Users/clairewilliams/Research/Calculations/model-timescales/src")

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


cluster_sampler = ClusterPopulationSampler.load("/Users/clairewilliams/Research/IMBH/cluster_sampler.pkl")


#--------- Load post-processed illustris merger tree 
def load_merger_tree_idx(df):
    goodidx = np.where(df['delta_t_gyr']>0)[0]
    print("There are "+str(len(goodidx))+" subhalos with nonzero merger time.")
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
def _integrate_leg_with_subdivision(cluster_mass, r_vec, v_vec, mass_start, radius_start,
                                     mass_end, radius_end, rel_pos_total, rel_vel_total,
                                     leg_duration, concentration, bg_host, bg_offset,
                                     max_leg_duration, hubble_start, hubble_end,
                                     escape_frac=np.inf):
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
    at once at the end. The background potential is held fixed at its
    start-of-leg value throughout (a secondary structure changing more
    slowly is a reasonable simplification here).

    Also used for the FINAL (no further host) leg by passing
    mass_end=mass_start, radius_end=radius_start, hubble_end=hubble_start,
    and rel_pos_total/rel_vel_total as zero vectors -- this degenerates to
    simply repeatedly integrating within the SAME unchanging host, which
    is exactly what's needed to cap any single solve_ivp call's duration
    even when there's no "next" host to interpolate toward.

    Returns (status, elapsed, r_vec, v_vec, n_pericenters,
    min_roche_radius_kpc) matching integrate_orbit's signature, with
    elapsed being the TOTAL time actually used (less than leg_duration if
    merged/escaped partway through a sub-step), n_pericenters the TOTAL
    pericenter count summed across every sub-step, and
    min_roche_radius_kpc the smallest tidal radius found at ANY pericenter
    across every sub-step (None if none occurred).
    """
    n_sub = max(1, int(np.ceil((leg_duration / max_leg_duration).to(u.dimensionless_unscaled).value)))
    sub_dt = leg_duration / n_sub
    total_elapsed = 0 * u.Gyr
    total_pericenters = 0
    min_roche_radius_kpc = None

    for k in range(n_sub):
        frac = k / n_sub
        mass_k = mass_start + (mass_end - mass_start) * frac
        radius_k = radius_start + (radius_end - radius_start) * frac
        hubble_k = hubble_start + (hubble_end - hubble_start) * frac
        host_k = NFWHost(mass_k * u.Msun, radius_k * u.kpc, concentration=concentration)

        # only the LAST sub-step of the LAST leg should ever use a real
        # (non-infinite) escape check -- earlier sub-steps always pass
        # escape_frac=np.inf regardless of what the caller asked for, same
        # rule as intermediate legs generally.
        this_escape_frac = escape_frac if k == n_sub - 1 else np.inf

        status, elapsed, r_vec, v_vec, n_peri, roche_this = integrate_orbit(
            cluster_mass, r_vec, v_vec, host_k, t_max=sub_dt, escape_frac=this_escape_frac,
            background_host=bg_host, background_offset=bg_offset, hubble_rate=hubble_k,
        )
        total_elapsed += elapsed
        total_pericenters += n_peri
        if roche_this is not None and (min_roche_radius_kpc is None or roche_this < min_roche_radius_kpc):
            min_roche_radius_kpc = roche_this

        if status in ("merged", "escaped"):
            return status, total_elapsed, r_vec, v_vec, total_pericenters, min_roche_radius_kpc

        r_vec = r_vec + rel_pos_total / n_sub
        v_vec = v_vec + rel_vel_total / n_sub

    return "ongoing", total_elapsed, r_vec, v_vec, total_pericenters, min_roche_radius_kpc


def trace_cluster_to_snapshot(cluster_mass, r0_vec, v0_vec, start_subhalo_id, target_age,
                               navigator, concentration=HOST_CONCENTRATION, max_steps=MAX_STEPS,
                               final_escape_frac=3.0, max_leg_duration=1.0 * u.Gyr,
                               min_host_mass_msun=1e8, verbose=False):
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
    """
    current_id = start_subhalo_id
    r_vec, v_vec = r0_vec, v0_vec
    _, _, start_snap = navigator.host_properties(start_subhalo_id)
    formation_age = navigator.age_at_snap(start_snap)
    current_age = formation_age
    n_hops = 0
    n_pericenters = 0
    min_roche_radius_kpc = None

    def make_result(status, host_id, r, v, r_over_rvir, steps):
        final_energy_kms2, _, _, _ = _compute_specific_energy_kms2(
            r, v, mass_msun, radius_kpc, concentration, bg_host, bg_offset,
        )
        return {
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
        }

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
        bg_host, bg_offset = None, None
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
        subdivided = t_leg_max > max_leg_duration
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

        if subdivided:
            if capped_by == 'step':
                status, elapsed, r_vec, v_vec, n_peri, roche_this = _integrate_leg_with_subdivision(
                    cluster_mass, r_vec, v_vec, mass_msun, radius_kpc, next_mass_msun, next_radius_kpc,
                    rel_pos_full, rel_vel_full, t_leg_max, concentration, bg_host, bg_offset, max_leg_duration,
                    hubble_start, hubble_end,
                )
            else:
                # 'output'-capped (including the true final step): no next
                # host to interpolate toward -- degenerate to repeatedly
                # integrating within the SAME unchanging host, zero reframe.
                zero_pos = np.zeros(3) * u.kpc
                zero_vel = np.zeros(3) * u.km / u.s
                status, elapsed, r_vec, v_vec, n_peri, roche_this = _integrate_leg_with_subdivision(
                    cluster_mass, r_vec, v_vec, mass_msun, radius_kpc, mass_msun, radius_kpc,
                    zero_pos, zero_vel, t_leg_max, concentration, bg_host, bg_offset, max_leg_duration,
                    hubble_start, hubble_start, escape_frac=escape_frac,
                )
        else:
            host = NFWHost(mass_msun * u.Msun, radius_kpc * u.kpc, concentration=concentration)
            if t_leg_max > 0 * u.Gyr:
                status, elapsed, r_vec, v_vec, n_peri, roche_this = integrate_orbit(
                    cluster_mass, r_vec, v_vec, host, t_max=t_leg_max, escape_frac=escape_frac,
                    background_host=bg_host, background_offset=bg_offset, hubble_rate=hubble_start,
                )
            else:
                status, elapsed, n_peri, roche_this = "ongoing", 0 * u.Gyr, 0, None
        current_age = current_age + elapsed
        n_pericenters += n_peri
        if roche_this is not None and (min_roche_radius_kpc is None or roche_this < min_roche_radius_kpc):
            min_roche_radius_kpc = roche_this

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
    return output


def _run_timescale_model(mass, radius, delta_t):
    """Bundles create_timescale_model + add_time_evolution into one call for run_with_timeout()."""
    model = create_timescale_model(mass, radius)
    return add_time_evolution(delta_t, model)


# ------- Iteration over all the subhalos
def iterate_subhalos(df, goodidx, navigator, target_age, debug_trace=False):
    clusters = []
    failures = []  # (halo_idx, cluster_idx, total_time_gyr, status, error_message) -- orbit trace failures only
    #testing mode-just do the first few
    for idx in goodidx[1000:1100]:
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
        cluster_props['IMBH_mass'] = []
        cluster_props['IMBH_final_formation_time']=[]
        cluster_props['which_final_formation_time']=[]
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
                verbose=debug_trace,
            )
            if trace_err is not None:
                print(f"    WARNING: orbit trace failed/timed out ({trace_err}) -- "
                      f"logging and skipping this cluster.")
                failures.append((idx, clusteridx, np.nan, "trace_failed", trace_err))
                cluster_props['status'].append("trace_failed")
                cluster_props['total_time_gyr'].append(np.nan)
                cluster_props['final_subhalo_id'].append(None)
                cluster_props['initial_subhalo_id'].append(idx)
                cluster_props['final_r_over_rvir'].append(np.nan)
                cluster_props['n_hops'].append(None)
                cluster_props['n_steps'].append(None)
                cluster_props['n_pericenters'].append(None)
                cluster_props['final_energy_kms2'].append(np.nan)
                cluster_props['final_bound'].append(None)
                cluster_props['min_roche_radius_kpc'].append(None)
                cluster_props['IMBH_mass'].append(np.nan)
                cluster_props['IMBH_final_formation_time'].append(np.nan)
                cluster_props['which_final_formation_time'].append(None)
                continue

            print(f"    status={trace['status']}, total_time={trace['total_time_gyr']:.4f} Gyr, "
                  f"n_hops={trace['n_hops']}, n_steps={trace['n_steps']}, "
                  f"n_pericenters={trace['n_pericenters']}, final_bound={trace['final_bound']}, "
                  f"final_host={trace['final_subhalo_id']}, r/Rvir={trace['final_r_over_rvir']:.4f}")

            cluster_props['status'].append(trace['status'])
            cluster_props['total_time_gyr'].append(trace['total_time_gyr'])
            cluster_props['final_subhalo_id'].append(trace['final_subhalo_id'])
            cluster_props['initial_subhalo_id'].append(idx)
            cluster_props['final_r_over_rvir'].append(trace['final_r_over_rvir'])
            cluster_props['n_hops'].append(trace['n_hops'])
            cluster_props['n_steps'].append(trace['n_steps'])
            cluster_props['n_pericenters'].append(trace['n_pericenters'])
            cluster_props['final_energy_kms2'].append(trace['final_energy_kms2'])
            cluster_props['final_bound'].append(trace['final_bound'])
            cluster_props['min_roche_radius_kpc'].append(trace['min_roche_radius_kpc'])

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
                continue
            cluster_props['IMBH_mass'].append(out['M_VMS'][0])
            cluster_props['IMBH_final_formation_time'].append(out['minimum_disruption_time'][0])
            cluster_props['which_final_formation_time'].append(out['which_disruption_time'][0])

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
          n_hops, n_steps, n_pericenters, which_final_formation_time) are
          stored as NaN (numeric fields) or None (which_final_formation_time,
          since its type isn't fixed here) rather than raising or
          misaligning rows.
        - IMBH_mass and IMBH_final_formation_time are stored as-is,
          assumed to already be plain floats (matching how the rest of
          this script handles them, with no .to()/.value conversion
          anywhere) -- if the `timescales` package actually returns
          astropy Quantities for these, adjust this function accordingly.

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
                'final_subhalo_id': final_id if final_id is not None else np.nan,
                'final_r_over_rvir': cluster_props['final_r_over_rvir'][i],
                'n_hops': n_hops if n_hops is not None else np.nan,
                'n_steps': n_steps if n_steps is not None else np.nan,
                'n_pericenters': n_pericenters if n_pericenters is not None else np.nan,
                'final_energy_kms2': cluster_props['final_energy_kms2'][i],
                'final_bound': final_bound if final_bound is not None else np.nan,
                'IMBH_mass': cluster_props['IMBH_mass'][i],
                'IMBH_final_formation_time': cluster_props['IMBH_final_formation_time'][i],
                'which_final_formation_time': cluster_props['which_final_formation_time'][i],
                'min_roche_radius_kpc': min_roche_radius_kpc if min_roche_radius_kpc is not None else np.nan,
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
    output_clusters = iterate_subhalos(df, goodidx, navigator, target_age, debug_trace=args.debug_trace)
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