import argparse
import numpy as np 
import astropy.units as u 
import h5py 
import matplotlib.pyplot as plt 
import pandas as pd
from astropy.constants import G
from pathlib import Path
import re
import sys
import os
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

from timescales.physics.stars import stellar_radius_approximation
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
#FIXME hard coded -- cosmology for the IMBH/high-res-cluster-simulation
# model itself (NOT the TNG merger tree -- see tree_navigator.TNG_COSMO
# for that; these are two different simulations, don't conflate them)
cosmo = FlatLambdaCDM(71,0.27,Ob0=0.044, Tcmb0=2.726 *u.K)

# Default log-spaced grid (kpc) for the TDE-rate-weighted histogram of each
# contributing cluster's separation from the central galaxy (see
# run_clusters' radial_hist / plot_tde_radial_smoothed.py). Separations
# outside the grid are clipped into the first/last bin (and counted, see
# run_clusters' radial_clipped_msun).
RADIAL_R_MIN_KPC = 1e-2
RADIAL_R_MAX_KPC = 1e3
RADIAL_N_BINS = 50

# Accounting for the IMBH's own mass (see timescale_analysis):
#  * SUBTRACT_IMBH_MASS: the IMBH formed out of the cluster's stars, so the
#    stellar power-law profile is rescaled to hold M_cluster - M_IMBH
#    (rho0 -> rho0 * (1 - M_IMBH/M_cluster)) before any TDE quantity is computed.
#  * TRH_DELAY: TDEs only start this many cluster HALF-MASS relaxation times
#    (Spitzer 1987, see half_mass_relaxation_time_gyr) after the IMBH forms;
#    the time available for TDEs shrinks by the same amount. 0 = no delay.
# Both can be changed from the command line (--no-subtract-imbh, --trh-delay).
SUBTRACT_IMBH_MASS = True
TRH_DELAY = 1.0
# Coulomb logarithm ln(gamma N) for t_rh; gamma matches the 0.2 used in
# calc_relaxation_time / calc_cutoff_radius below, for consistency.
TRH_COULOMB_GAMMA = 0.2

def load_data_file(file_path):
    """
    Checks the file extension and loads a .csv or .dat file into a Pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    file_extension = path.suffix.lower()
    
    if file_extension == '.csv':
        print(f"Processing CSV file: {path.name}")
        return pd.read_csv(path)
        
    elif file_extension == '.dat':
        print(f"Processing DAT file: {path.name}")
        return pd.read_pickle(path)
        # return pd.read_csv(path, sep=None, engine='python')
        
    else:
        raise ValueError(f"Unsupported file type '{file_extension}'. Only .csv and .dat are allowed.")

def set_up_timebins(resolution):
    """Resolution is the length of the timestep in years"""
    age = cosmo.age(0)
    nsteps = int((age.to('yr')/resolution).value)
    tscale_array_yr = np.linspace(1e6,int(age.to('yr').value)+1,nsteps)
    tde_rate_array_msunyr = np.zeros(nsteps)*u.Msun/u.yr
    mass_trelax_array_msun = np.zeros(nsteps)* u.Msun
    return tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun

def half_mass_relaxation_time_gyr(m_star_msun, r_h_pc, m_mean_msun=1.0, gamma=TRH_COULOMB_GAMMA):
    """
    Spitzer (1987) half-mass relaxation time, in Gyr:

        t_rh = 0.138 N^(1/2) r_h^(3/2) / ( m^(1/2) G^(1/2) ln(gamma N) ),   N = M / m

    for a cluster of stellar mass m_star_msun, 3D half-mass radius r_h_pc and
    mean stellar mass m_mean_msun (1 Msun, matching Mstar elsewhere in this file).
    """
    n_stars = m_star_msun / m_mean_msun
    coulomb_log = np.log(gamma * n_stars)
    if not (n_stars > 0 and r_h_pc > 0 and coulomb_log > 0):
        return np.nan
    t = (0.138 * np.sqrt(n_stars) * (r_h_pc * u.pc) ** 1.5
         / (np.sqrt(m_mean_msun * u.Msun) * np.sqrt(G) * coulomb_log))
    return t.to(u.Gyr).value


def timescale_analysis(df, idx, alpha, subtract_imbh=None, trh_delay=None):
    """
    Returns (mass_lost_bins, t_relax_bin, t_cutoff_gyr, t_delay_gyr), or four
    Nones if this cluster produces no TDEs.

    subtract_imbh / trh_delay default to SUBTRACT_IMBH_MASS / TRH_DELAY (see
    their comment above). With the delay, TDEs start t_delay_gyr after the
    IMBH forms and t_cutoff_gyr is the time available AFTER that delay.
    """
    subtract_imbh = SUBTRACT_IMBH_MASS if subtract_imbh is None else subtract_imbh
    trh_delay = TRH_DELAY if trh_delay is None else trh_delay
    stellar_radius = stellar_radius_approximation(1* u.Msun)
    limiting_tscale_gyr = []
    roche = False # start out with no roche effects
    if df['IMBH_mass_msun'][idx]<5e2:
        #placeholder--need to decide what to do with the small BHs. 
        return None,None,None,None
    elif df['IMBH_final_formation_time_gyr'][idx] >= df['total_time_gyr'][idx]:
        #placeholder - if the IMBH hasn't formed yet, don't do anything
        return None,None,None,None
    else: 
        # ---- the stars that formed the IMBH are no longer in the cluster ----
        m_imbh = float(df['IMBH_mass_msun'][idx])
        m_cluster = float(df['cluster_mass_msun'][idx])
        if subtract_imbh:
            f_star = 1.0 - m_imbh / m_cluster
            if f_star <= 0:
                return None,None,None,None  # IMBH as massive as the whole cluster: no stars left
        else:
            f_star = 1.0
        # stellar profile normalisation actually used for every TDE quantity below
        rho0_star = df['rho0_msun_pc3'][idx] * f_star

        # ---- TDEs start one (or trh_delay) half-mass relaxation time(s) after the IMBH forms ----
        t_delay_gyr = 0.0
        if trh_delay > 0:
            t_delay_gyr = trh_delay * half_mass_relaxation_time_gyr(m_cluster * f_star,
                                                                    df['cluster_radius_pc'][idx])
            if not np.isfinite(t_delay_gyr):
                return None,None,None,None
        t_start_gyr = df['IMBH_final_formation_time_gyr'][idx] + t_delay_gyr

        if df['min_roche_radius_kpc'][idx]< df['cluster_radius_pc'][idx] /1e3:
            #If roche lobe goes within cluster, need to consider roche effects (the outer radius is small)
            roche = True
            limiting_tscale_gyr.append(df['time_of_min_roche_gyr'][idx]-t_start_gyr)
        limiting_tscale_gyr.append(df['total_time_gyr'][idx]-t_start_gyr)
        t_cutoff_gyr = min(limiting_tscale_gyr)
        # print(f"The cutoff time is {t_cutoff_gyr}")
        if not t_cutoff_gyr > 0:
            # trace ended (inspiral / output time) or the Roche-limited time
            # passed before the delay was over: no TDEs from this cluster
            return None,None,None,None

        r_sphere_of_influence = sphere_of_influence(df['IMBH_mass_msun'][idx], df['r0_pc'][idx],
                                                     rho0_star, alpha)

        r = calc_cutoff_radius(t_cutoff_gyr, df['IMBH_mass_msun'][idx],
                                df['r0_pc'][idx], rho0_star, alpha,
                                df['cluster_radius_pc'][idx])
        # print(r)
        # r_orbital_inner = calc_orbital_relaxation_radius(df['IMBH_mass_msun'][idx],df['r0_pc'][idx],
                                                    #  rho0_star, alpha,df['cluster_radius_pc'][idx] )
        sigma_r, rho_r, M_r = calc_structural_props(r.value, df['IMBH_mass_msun'][idx],
                                                     rho0_star,
                                                     df['r0_pc'][idx], alpha)
        if roche:
            outer_radius_pc = min(r_sphere_of_influence.value, df['r0_pc'][idx],df['min_roche_radius_kpc'][idx] * 1e3) 
        else: 
            outer_radius_pc = min(r_sphere_of_influence.value, df['r0_pc'][idx])
        r_orbital_inner = stellar_radius.to('pc')
        r_orbital_mstar = calc_min_enclosed_radius(2*u.Msun,df['r0_pc'][idx],rho0_star,alpha  )
        r_orbital_inner = max(r_orbital_inner, r_orbital_mstar)

        Nbins = 6

        if alpha > 1.5:
            """If alpha is greater than 1.5, the relaxation time decreases with radius
                so the condition t < t_relax is satisfied for r < r_crit;
                we're gonna go from the cutoff radius to the outermost radius"""
            outer_radius_pc= min(outer_radius_pc, r.value)
            if r_orbital_inner.value< outer_radius_pc:
                #split the radii into 6 bins
                radii_bins = np.logspace(np.log10(r_orbital_inner.value),np.log10(outer_radius_pc),Nbins)
                radii_bins = np.linspace(r_orbital_inner.value,outer_radius_pc,Nbins)
                structural_props_bins = [calc_structural_props(radii_bins[i], df['IMBH_mass_msun'][idx],
                                                     rho0_star,
                                                     df['r0_pc'][idx], alpha) for i in range(Nbins)]
                mass_bins = [structural_props_bins[i][2] for i in range(Nbins)]
                # mass_bins = [calc_structural_props(radii_bins[i], df['IMBH_mass_msun'][idx],
                #                                      rho0_star,
                #                                      df['r0_pc'][idx], alpha)[2] for i in range(Nbins) ]
                sigma_bins = [structural_props_bins[i][0] for i in range(Nbins)]
                rho_bins = [structural_props_bins[i][1] for i in range(Nbins)]
                mass_lost_bins = [mass_bins[i+1]-mass_bins[i] for i in range(Nbins-1)]
                # _,_, M_r_inner = calc_structural_props(r_orbital_inner.value, df['IMBH_mass_msun'][idx],
                #                                      rho0_star,
                #                                      df['r0_pc'][idx], alpha)
                sigma_r_outer,rho_r_outer, M_r_outer =calc_structural_props(outer_radius_pc, df['IMBH_mass_msun'][idx],
                                                     rho0_star,
                                                     df['r0_pc'][idx], alpha) 
                # M_star_removed_msun = M_r_outer - M_r_inner
                M_star_removed_msun= sum(mass_lost_bins)
                #Using the outer t_relax (could take bin middle but leave that as #TODO)
                t_relax_bin = [calc_relaxation_time(sigma_bins[i],rho_bins[i], mass_bins[i]) for i in range(1,Nbins) ]
            else:
                M_star_removed_msun = 0. 
                mass_lost_bins = [0*u.Msun for i in range(Nbins-1)]
        elif alpha < 1.5:
            """If alpha is below 1.5, the relaxation time increases with radius
                so the condition t < t_relax is satisfied for r > r_crit;
                we go from the inner radius to the cutoff radius"""
            r_orbital_inner = max(r_orbital_inner.value,r.value)

            if r_orbital_inner< outer_radius_pc:
                radii_bins = np.linspace(r_orbital_inner,outer_radius_pc,Nbins)
                structural_props_bins = [calc_structural_props(radii_bins[i], df['IMBH_mass_msun'][idx],
                                                     rho0_star,
                                                     df['r0_pc'][idx], alpha) for i in range(Nbins)]
                mass_bins = [structural_props_bins[i][2] for i in range(Nbins)]
                # mass_bins = [calc_structural_props(radii_bins[i], df['IMBH_mass_msun'][idx],
                #                                      rho0_star,
                #                                      df['r0_pc'][idx], alpha)[2] for i in range(Nbins) ]
                sigma_bins = [structural_props_bins[i][0] for i in range(Nbins)]
                rho_bins = [structural_props_bins[i][1] for i in range(Nbins)]
                mass_lost_bins = [mass_bins[i+1]-mass_bins[i] for i in range(Nbins-1)]
                # _,_, M_r_inner = calc_structural_props(r_orbital_inner.value, df['IMBH_mass_msun'][idx],
                #                                      rho0_star,
                #                                      df['r0_pc'][idx], alpha)
                sigma_r_outer,rho_r_outer, M_r_outer =calc_structural_props(outer_radius_pc, df['IMBH_mass_msun'][idx],
                                                     rho0_star,
                                                     df['r0_pc'][idx], alpha) 
                # M_star_removed_msun = M_r_outer - M_r_inner
                M_star_removed_msun= sum(mass_lost_bins)
                #Using the outer t_relax (could take bin middle but leave that as #TODO)
                t_relax_bin = [calc_relaxation_time(sigma_bins[i],rho_bins[i], mass_bins[i]) for i in range(1,Nbins) ]
            else:
                M_star_removed_msun = 0. 
                mass_lost_bins = [0*u.Msun for i in range(Nbins-1)]
        else:
            # alpha == 1.5: t_relax is radius-independent; calc_cutoff_radius
            # already raises above, so this branch is unreachable in practice,
            # but kept here as an explicit placeholder for future handling
            # (e.g. compare t_cutoff directly against the constant t_relax).
            raise NotImplementedError("alpha == 1.5 case not yet implemented")



        cluster_relaxation_time_gyr = calc_relaxation_time(sigma_r_outer,rho_r_outer, M_r_outer)
        tde_rate = M_star_removed_msun/cluster_relaxation_time_gyr / u.Gyr

        return np.array(mass_lost_bins*u.Msun),np.array(t_relax_bin), t_cutoff_gyr, t_delay_gyr

def sphere_of_influence(imbh_mass,r0,rho0, alpha):
    """
    Set M_BH = M(r) = 4 pi rho0 r^3-alpha / ((3-alpha)r0^-alpha)
    r^(3-alpha) = MBH * (3-alpha)r0^-alpha/ (4 pi rho0)
    """
    imbh_mass = imbh_mass * u.Msun
    rho0 = rho0 * u.Msun / (u.pc)**3
    r0 = r0 * u.pc
    return ((imbh_mass * (3-alpha) * r0**(-alpha)/ (4 * np.pi * rho0))**(1/(3-alpha))).to('pc')

def calc_structural_props(r,imbh_mass, rho0, r0, alpha):
    imbh_mass = imbh_mass * u.Msun
    r = r * u.pc
    rho0 = rho0 * u.Msun / (u.pc)**3
    r0 = r0 * u.pc
    sigma_r = np.sqrt( G * imbh_mass/(1+alpha)/r).to(u.km/ u.s)
    rho_r = (rho0 * (r/r0)**(-alpha)).to(u.Msun/ u.pc**3)
    M_r = 4 * np.pi * rho0 * r**(3-alpha) / (3-alpha)/ r0**(-alpha)
    return sigma_r, rho_r, M_r

def calc_relaxation_time(sigma_r, rho_r, M_r, Mstar = 1*u.Msun):
    coulomb_log = np.log(M_r /Mstar * 0.2)
    return (0.34 * sigma_r**3 / G**2 /rho_r / Mstar/ coulomb_log).to('Gyr').value

def calc_min_enclosed_radius(Mstar, r0, rho0, alpha):
    """
    Radius that encloses exactly Mstar of mass under the power-law
    density profile, i.e. solving M(r) = Mstar for r where

        M(r) = 4*pi*rho0*r^(3-alpha) / ((3-alpha)*r0^(-alpha))

    This is the same functional form as sphere_of_influence, with
    Mstar in place of imbh_mass:

        r_min = ( Mstar*(3-alpha)*r0^(-alpha) / (4*pi*rho0) )^(1/(3-alpha))

    Physically, this is the smallest radius for which treating the
    enclosed mass as a smooth continuum is meaningful at all -- inside
    r_min, the power-law profile implies less than one star's worth of
    mass, which isn't physically sensible for a cusp built out of
    discrete stars of mass Mstar. Useful as an absolute floor on any
    inner cutoff radius (e.g. compare against calc_orbital_relaxation_radius).

    Requires alpha < 3 for M(r) to be finite/increasing as expected;
    raises for alpha >= 3.
    """
    if alpha >= 3:
        raise ValueError(
            "alpha >= 3 makes the enclosed mass profile diverge or "
            "decrease with r (exponent 1/(3-alpha) undefined/negative); "
            "this case needs separate handling."
        )

    Mstar = Mstar * u.Msun if not isinstance(Mstar, u.Quantity) else Mstar
    rho0 = rho0 * u.Msun / (u.pc)**3
    r0 = r0 * u.pc

    return ((Mstar * (3-alpha) * r0**(-alpha) / (4 * np.pi * rho0))**(1/(3-alpha))).to('pc')

def calc_cutoff_radius(t_cutoff_gyr, imbh_mass, r0, rho0, alpha, rtotal, Mstar=1*u.Msun):
    """
    Solve t_cutoff = C * r^(alpha - 3/2) for r.
    Valid for alpha != 1.5 (see guard below); direction of the
    resulting inequality (r > r_crit vs r < r_crit) is handled
    by the caller, not by this function.
    """
    if np.isclose(alpha, 1.5):
        raise ValueError(
            "alpha == 1.5 makes the relaxation time radius-independent "
            "(exponent 3 - 2*alpha = 0); this case needs separate handling."
        )

    rtotal = rtotal * u.pc
    imbh_mass = imbh_mass * u.Msun
    rho0 = rho0 * u.Msun / (u.pc)**3
    r0 = r0 * u.pc
    t_cutoff_gyr = t_cutoff_gyr * u.Gyr
    M_r = 4 * np.pi * rho0 * rtotal**(3-alpha) / (3-alpha) / r0**(-alpha)
    coulomb_log = np.log(M_r / Mstar * 0.2)
    numerator = 0.34 * imbh_mass**(3./2.)
    denominator = t_cutoff_gyr * G**(0.5) * (1+alpha)**(3./2.) * rho0 * Mstar * coulomb_log * r0**alpha
    exponent = 2. / (3. - 2.*alpha)
    return ((numerator / denominator)**exponent).to('pc')


def calc_orbital_relaxation_radius(imbh_mass, r0, rho0, alpha, rtotal, Mstar=1*u.Msun):
    """
    Radius where the Keplerian orbital timescale t_orb(r) equals the
    two-body relaxation timescale t_relax(r):

        t_orb(r)    = 2*pi*sqrt(r^3 / (G*M_BH))
        t_relax(r)  = 0.34*sigma(r)^3 / (G^2*rho(r)*Mstar*ln(Lambda))

    Setting these equal and solving for r (G cancels, and the result
    is independent of any external time t):

        r_crit = ( 0.34*M_BH^2 /
                   (2*pi*(1+alpha)^(3/2)*rho0*r0^alpha*Mstar*ln(Lambda)) )
                 ^ (1/(3-alpha))

    Inside r_crit, orbital dynamics are faster than two-body relaxation
    can act, so treating mass loss as relaxation-driven breaks down;
    use this as the inner cutoff radius for the relaxation calculation.

    As in calc_cutoff_radius, the Coulomb logarithm is approximated
    using M(<rtotal) rather than M(<r_crit), since r_crit is what
    we're solving for.
    """
    if np.isclose(alpha, 3.0):
        raise ValueError(
            "alpha == 3 makes the enclosed mass profile diverge "
            "(exponent 1/(3-alpha) undefined); this case needs separate handling."
        )

    rtotal = rtotal * u.pc
    imbh_mass = imbh_mass * u.Msun
    rho0 = rho0 * u.Msun / (u.pc)**3
    r0 = r0 * u.pc

    M_r = 4 * np.pi * rho0 * rtotal**(3-alpha) / (3-alpha) / r0**(-alpha)
    coulomb_log = np.log(M_r / Mstar * 0.2)

    numerator = 0.34 * imbh_mass**2
    denominator = 2 * np.pi * (1+alpha)**(3./2.) * rho0 * r0**alpha * Mstar * coulomb_log
    exponent = 1. / (3. - alpha)

    return ((numerator / denominator)**exponent).to('pc')

def _load_radius_track(track_path, track_dir, cache):
    """
    Load one cluster's separation-from-central-galaxy time series, as saved
    by imbh.py's --save-radius-tracks (see imbh.py's save_radius_track /
    TRACK_MIN_IMBH_MASS_MSUN). Returns (time_since_formation_gyr,
    separation_kpc) as plain numpy arrays -- the SAME "time since the
    cluster's own formation" convention used by total_time_gyr /
    IMBH_final_formation_time_gyr elsewhere in this pipeline -- or
    (None, None) if the file can't be found/read (logged once; this is a
    diagnostic extra, not something that should ever take down the whole
    TDE-rate calculation).

    `cache` is a plain dict the caller keeps across calls (keyed by the
    literal path string), but ONLY ever holds the (None, None) "missing"
    sentinel -- purely to avoid printing the same missing-file warning
    twice. A SUCCESSFUL load is deliberately never cached: each track file
    belongs to exactly one cluster row (there's no repeat lookup to serve),
    so caching it would only mean permanently retaining that cluster's full
    array (up to ~1e5 rows) in memory for the rest of the run for no
    benefit. With potentially thousands of distinct tracks across a full
    run, that adds up to a genuine, unbounded memory cost -- caching every
    successful read here was a real bug (retaining ~2 MB/track x thousands
    of clusters, all for the run's remaining lifetime), not a hypothetical
    one; caching only the tiny "missing" sentinel keeps the warning-dedup
    benefit without it.
    """
    if track_path in cache:
        return cache[track_path]

    path = Path(track_path)
    if track_dir is not None and not path.is_absolute():
        # allow passing either the exact path imbh.py wrote (already
        # relative/absolute and directly usable) or just re-pointing at
        # a --track-dir the tracks were moved to since -- match on filename.
        candidate = Path(track_dir) / path.name
        if candidate.exists():
            path = candidate

    if not path.exists():
        print(f"    WARNING: radius track '{track_path}' not found -- skipping the "
              f"radius-vs-time diagnostic for this cluster (its TDE rate is unaffected).")
        cache[track_path] = (None, None)
        return cache[track_path]

    track_df = pd.read_csv(path)
    return (track_df['time_since_formation_gyr'].to_numpy(),
            track_df['separation_from_host_kpc'].to_numpy())



def _burst_bins(t_start_yr, t_end_yr, t0_yr, dt_yr, n_bins):
    """
    Time bins covered by a constant-rate burst over [t_start_yr, t_end_yr).

    Bin i of the grid is centred on t0_yr + i*dt_yr and spans +/- dt_yr/2.
    Returns (idx, weights, t_mid_yr): the bin indices the burst overlaps (inside
    the grid), the fraction of each bin's width it covers (0-1), and the
    midpoint time of the covered part of each bin. A burst of rate R therefore
    adds R*weights to the binned rate, and deposits exactly R*(t_end - t_start)
    of mass (minus any part falling off the grid).
    """
    if not t_end_yr > t_start_yr:
        return np.array([], dtype=int), np.array([]), np.array([])
    i0 = int(np.floor((t_start_yr - t0_yr) / dt_yr + 0.5))
    i1 = int(np.floor((t_end_yr - t0_yr) / dt_yr + 0.5))
    i0, i1 = max(i0, 0), min(i1, n_bins - 1)
    if i1 < i0:
        return np.array([], dtype=int), np.array([]), np.array([])
    idx = np.arange(i0, i1 + 1)
    lo = t0_yr + (idx - 0.5) * dt_yr
    hi = lo + dt_yr
    a = np.maximum(lo, t_start_yr)
    b = np.minimum(hi, t_end_yr)
    covered = np.clip(b - a, 0.0, None)
    keep = covered > 0
    return idx[keep], covered[keep] / dt_yr, 0.5 * (a + b)[keep]


def run_clusters(df, alpha, tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun,
                  resolution = 1e5*u.yr, track_dir=None, radial_edges_kpc=None,
                  subtract_imbh=None, trh_delay=None):
    """
    Returns (tde_rate_array_msunyr, mass_trelax_array_msun, mean_radius_kpc,
    n_clusters_with_tde, contributing_rows, radial):

        radial: dict with the TDE rate resolved by separation from the
            central galaxy, on the same time grid as tde_rate_array_msunyr:
              'edges_kpc'   (n_r+1,) log-spaced bin edges (radial_edges_kpc,
                            default RADIAL_R_MIN/MAX_KPC, RADIAL_N_BINS)
              'hist_msunyr' (n_t, n_r) TDE rate [Msun/yr] in each time bin
                            from clusters whose separation fell in each
                            radial bin (float32). Every cluster contributes
                            to exactly ONE radial bin per time bin -- the
                            cluster's own size is << its separation.
              'inspiraled_msunyr' (n_t,) rate from clusters whose burst
                            continues AFTER their orbit trace ended in an
                            inspiral (status == 'inspiraled'): they sit at
                            the center, so they're kept out of hist rather
                            than pinned at the trace's last (stop) radius.
              'no_track_msunyr' (n_t,) rate from contributing clusters with
                            no radius track (so hist + inspiraled + no_track
                            == tde_rate_array_msunyr).
              'clipped_msun' mass (rate x dt) that fell outside edges_kpc
                            and was clipped into the first/last bin.
              'hist_escaped_msunyr', 'no_track_escaped_msunyr',
              'clipped_escaped_msun': the same three quantities restricted
                            to clusters with status == 'escaped' (already
                            included in the totals above -- subtract them to
                            exclude escaped clusters; see plot_tde_*_smoothed.py
                            --escaped). Escaped clusters never inspiral, so
                            they have no 'inspiraled' share.
              'rate_escaped_msunyr' (n_t,) the escaped clusters' share of
                            tde_rate_array_msunyr.

        mean_radius_kpc: at each time bin, the AVERAGE separation from the
            central/root galaxy across every cluster that has a TDE burst
            going off at that same time (i.e. the same time bins
            tde_rate_array_msunyr is nonzero because of) -- NaN wherever no
            cluster is contributing a TDE at that time. Requires each
            contributing cluster's df row to have a valid
            'radius_track_path' (see imbh.py's --save-radius-tracks);
            clusters without one simply don't contribute to this array
            (their TDE rate is still counted in tde_rate_array_msunyr as
            before).
        n_clusters_with_tde: how many clusters' tracks contributed to
            mean_radius_kpc at each time bin -- lets you judge how well
            supported (or not) each time bin's average is.
        contributing_rows: a list of dicts, one per cluster that actually
            contributed at least one mass-loss bin to tde_rate_array_msunyr
            (i.e. timescale_analysis returned a non-None t_cutoff_gyr for
            it) -- an ACTIVELY TDE-generating cluster, as opposed to merely
            having IMBH_mass_msun>0 (many of those still get excluded
            inside timescale_analysis, e.g. by its own 5e2 Msun floor).
            Each dict has 'row_index' (this cluster's position in df),
            'radius_track_path' (straight from df, may be NaN/empty if it
            has none), 'IMBH_mass_msun', 'tde_delay_gyr', 'tde_mass_emitted_msun'
            (mass actually released inside the TDE time window, i.e. what this
            cluster adds to the time-integrated rate), and 'total_mass_lost_msun' (summed
            across every mass-loss bin this cluster contributed) -- enough
            to filter or cross-reference downstream (see
            plot_radius_tracks.py's --tde-contributors-csv) without
            re-running timescale_analysis a second time.
    """
    resolution = resolution.to('yr').value
    n_bins = len(tscale_array_yr)
    # actual spacing of the time grid (set_up_timebins uses linspace, so this is
    # ~resolution but not exactly); bin i covers tscale[i] +/- grid_dt_yr/2
    grid_dt_yr = float(tscale_array_yr[1] - tscale_array_yr[0])
    radius_sum_kpc = np.zeros(n_bins)
    radius_count = np.zeros(n_bins, dtype=int)
    track_cache = {}
    has_track_col = 'radius_track_path' in df.columns
    contributing_rows = []

    if radial_edges_kpc is None:
        radial_edges_kpc = np.logspace(np.log10(RADIAL_R_MIN_KPC), np.log10(RADIAL_R_MAX_KPC),
                                       RADIAL_N_BINS + 1)
    radial_edges_kpc = np.asarray(radial_edges_kpc, dtype=float)
    n_rbins = len(radial_edges_kpc) - 1
    radial_hist = np.zeros((n_bins, n_rbins), dtype=np.float32)
    radial_inspiraled = np.zeros(n_bins)
    radial_no_track = np.zeros(n_bins)
    radial_clipped_msun = 0.0
    has_status_col = 'status' in df.columns
    # escaped clusters' share of every output above (status == 'escaped': beyond
    # 3 R200 of the z=0 host at the last snapshot, see imbh.py), so the plotting
    # scripts can exclude them or show them alone
    rate_escaped = np.zeros(n_bins)
    radial_hist_escaped = np.zeros((n_bins, n_rbins), dtype=np.float32)
    radial_no_track_escaped = np.zeros(n_bins)
    radial_clipped_escaped_msun = 0.0
    if not has_status_col:
        print("NOTE: no 'status' column -- escaped clusters can't be identified; their "
              "share of the outputs will be zero.")
    # Per-cluster formation redshift drawn by imbh.py (--formation-time-draw);
    # older outputs only have the snapshot-quantized subhalo value.
    if 'cluster_formation_redshift' in df.columns:
        formation_z_col = 'cluster_formation_redshift'
    else:
        formation_z_col = 'initial_subhalo_formation_redshift'
        print("NOTE: no cluster_formation_redshift column (older imbh.py output) -- using the "
              "snapshot-quantized initial_subhalo_formation_redshift instead.")

    for clusteridx in range(len(df)):
    # for clusteridx in range(2):
        if df['IMBH_mass_msun'][clusteridx]>0:
            halo_formation_time = cosmo.age(df[formation_z_col][clusteridx])
            mass_lost_bins,t_relax_bin, t_cutoff_gyr, t_delay_gyr = timescale_analysis(
                df, clusteridx, alpha, subtract_imbh=subtract_imbh, trh_delay=trh_delay)
            if t_cutoff_gyr is not None: 
                # TDEs start t_delay_gyr (trh_delay half-mass relaxation times)
                # after the IMBH forms -- see timescale_analysis
                start_time = (halo_formation_time
                              + (df['IMBH_final_formation_time_gyr'][clusteridx] + t_delay_gyr) * u.Gyr)
                # ---- radius-vs-time lookup for this cluster, if it has one ----
                track_t_gyr = track_r_kpc = None
                if has_track_col:
                    track_path = df['radius_track_path'][clusteridx]
                    if isinstance(track_path, str) and track_path:
                        track_t_gyr, track_r_kpc = _load_radius_track(track_path, track_dir, track_cache)

                is_escaped = has_status_col and df['status'][clusteridx] == 'escaped'

                contributing_rows.append({
                    'row_index': clusteridx,
                    'status': df['status'][clusteridx] if has_status_col else None,
                    'radius_track_path': df['radius_track_path'][clusteridx] if has_track_col else None,
                    'IMBH_mass_msun': df['IMBH_mass_msun'][clusteridx],
                    'total_mass_lost_msun': float(np.sum(mass_lost_bins)),
                    'tde_delay_gyr': float(t_delay_gyr),
                    # mass actually emitted as TDEs inside the time window (each
                    # mass-loss bin is cut off at t_cutoff_gyr); filled below
                    'tde_mass_emitted_msun': 0.0,
                })
                emitted_msun = 0.0

                for binidx in range(len(mass_lost_bins)):
                    # Each mass-loss bin releases its mass at a constant rate
                    # mass/t_relax from t_start until min(t_cutoff, t_relax) later.
                    # Spread it over the time grid by the exact fraction of each
                    # bin it covers (see _burst_bins), so the deposited mass is
                    # exactly rate x duration -- no extra end bin, and bursts
                    # shorter than one bin are no longer rounded up to a full bin.
                    t_start_yr = start_time.to(u.yr).value
                    duration_gyr = min(t_cutoff_gyr, t_relax_bin[binidx])
                    t_end_yr = t_start_yr + duration_gyr * 1e9
                    rate = mass_lost_bins[binidx] / t_relax_bin[binidx] * u.Msun / u.Gyr
                    mass = mass_lost_bins[binidx]* u.Msun
                    rate_msunyr = float(mass_lost_bins[binidx] / t_relax_bin[binidx]) / 1e9
                    # mass_trelax diagnostic: unchanged convention (bin after the burst's end)
                    k_end = int(round((t_end_yr - tscale_array_yr[0]) / grid_dt_yr))
                    if 0 <= k_end + 1 < n_bins:
                        mass_trelax_array_msun[k_end+1] += mass

                    tbins, wts, t_mid_yr = _burst_bins(t_start_yr, t_end_yr, tscale_array_yr[0],
                                                       grid_dt_yr, n_bins)
                    if len(tbins) == 0:
                        continue
                    emitted_msun += rate_msunyr * grid_dt_yr * float(np.sum(wts))
                    tde_rate_array_msunyr[tbins] += rate * wts
                    if is_escaped:
                        rate_escaped[tbins] += rate_msunyr * wts

                    if track_t_gyr is None:
                        radial_no_track[tbins] += rate_msunyr * wts
                        if is_escaped:
                            radial_no_track_escaped[tbins] += rate_msunyr * wts
                        continue

                    # cosmic time (Gyr) of the covered part of every bin in THIS
                    # burst -> time since the CLUSTER's own formation (the track's
                    # own convention), then linearly interpolated against the
                    # saved orbit trace (clamped at the ends by np.interp, which is
                    # fine: the track spans the cluster's whole trace, so a burst
                    # time falling outside it just means sub-bin rounding at a
                    # boundary).
                    t_since_formation_gyr = t_mid_yr / 1e9 - halo_formation_time.to(u.Gyr).value
                    r_interp_kpc = np.interp(t_since_formation_gyr, track_t_gyr, track_r_kpc)
                    radius_sum_kpc[tbins] += r_interp_kpc
                    radius_count[tbins] += 1

                    # ---- rate-weighted separation histogram ----
                    # Past the end of an inspiraled cluster's trace, np.interp
                    # would hold the stop radius; the cluster is at the center.
                    in_track = np.ones(len(tbins), dtype=bool)
                    if (has_status_col and df['status'][clusteridx] == 'inspiraled'
                            and len(track_t_gyr)):
                        in_track = t_since_formation_gyr <= track_t_gyr[-1]
                        radial_inspiraled[tbins[~in_track]] += rate_msunyr * wts[~in_track]
                    r_in = r_interp_kpc[in_track]
                    w_in = wts[in_track]
                    ridx = np.searchsorted(radial_edges_kpc, r_in, side='right') - 1
                    out = (ridx < 0) | (ridx >= n_rbins)
                    clipped = rate_msunyr * grid_dt_yr * float(np.sum(w_in[out]))
                    radial_clipped_msun += clipped
                    ridx = np.clip(ridx, 0, n_rbins - 1)
                    np.add.at(radial_hist, (tbins[in_track], ridx), rate_msunyr * w_in)
                    if is_escaped:
                        radial_clipped_escaped_msun += clipped
                        np.add.at(radial_hist_escaped, (tbins[in_track], ridx), rate_msunyr * w_in)
                contributing_rows[-1]['tde_mass_emitted_msun'] = emitted_msun

    with np.errstate(invalid='ignore'):
        mean_radius_kpc = np.where(radius_count > 0, radius_sum_kpc / np.maximum(radius_count, 1), np.nan)

    radial = {
        'edges_kpc': radial_edges_kpc,
        'hist_msunyr': radial_hist,
        'inspiraled_msunyr': radial_inspiraled,
        'no_track_msunyr': radial_no_track,
        'clipped_msun': radial_clipped_msun,
        'hist_escaped_msunyr': radial_hist_escaped,
        'no_track_escaped_msunyr': radial_no_track_escaped,
        'clipped_escaped_msun': radial_clipped_escaped_msun,
        'rate_escaped_msunyr': rate_escaped,
    }
    return (tde_rate_array_msunyr, mass_trelax_array_msun, mean_radius_kpc, radius_count,
            contributing_rows, radial)



def infer_subhalo_id(input_path):
    """
    Pull the subhalo/branch ID out of an imbh.py output filename of the
    form cluster_output_<subhaloid>_<snap>.{csv,dat} (e.g.
    cluster_output_467548_99.dat or cluster_output_467548_snap99.csv).
    Returns the ID as a string, or None if the name doesn't match.
    """
    m = re.match(r"cluster_output_(\d+)_", Path(input_path).name)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="Path to the cluster_output_<branchid>_<snapshot>.csv file")
    parser.add_argument("alpha", help="power law index of cluster profiles")
    parser.add_argument("--track-dir", default=None,
                         help="Directory containing the per-cluster radius_track_*.csv files "
                              "saved by imbh.py's --save-radius-tracks (only needed if the "
                              "'radius_track_path' column's own paths no longer resolve as-is, "
                              "e.g. the files were moved -- matched by filename in that "
                              "directory). If input_path has no radius_track_path column at "
                              "all, this is ignored and the radius-vs-time diagnostic is simply "
                              "skipped (the TDE rate itself is unaffected either way).")
    parser.add_argument("--subhalo-id", default=None,
                         help="Subhalo ID to tag the output files with. If omitted, it is "
                              "read from input_path's name (cluster_output_<subhaloid>_<snap>.*).")
    parser.add_argument("--r-min-kpc", type=float, default=RADIAL_R_MIN_KPC,
                         help="Inner edge of the log-spaced separation grid for the radial TDE "
                              f"histogram (default: {RADIAL_R_MIN_KPC}).")
    parser.add_argument("--r-max-kpc", type=float, default=RADIAL_R_MAX_KPC,
                         help=f"Outer edge of that grid (default: {RADIAL_R_MAX_KPC}).")
    parser.add_argument("--n-rbins", type=int, default=RADIAL_N_BINS,
                         help=f"Number of log-spaced separation bins (default: {RADIAL_N_BINS}). "
                              "Use a fine grid here; plot_tde_radial_smoothed.py can rebin/smooth.")
    parser.add_argument("--output-dir", default=".",
                         help="Directory to write the tde_rates/tde_contributors CSVs to "
                              "(default: current directory).")
    parser.add_argument("--no-subtract-imbh", action="store_true",
                         help="Don't remove the IMBH's mass from the cluster's stellar profile "
                              "(default: the profile is rescaled to hold M_cluster - M_IMBH).")
    parser.add_argument("--trh-delay", type=float, default=TRH_DELAY,
                         help="Delay the start of TDEs by this many cluster half-mass relaxation "
                              f"times after the IMBH forms (default: {TRH_DELAY:g}; 0 = no delay).")
    args = parser.parse_args()

    subhalo_id = args.subhalo_id or infer_subhalo_id(args.input_path)
    if subhalo_id is None:
        parser.error(f"could not infer the subhalo ID from '{args.input_path}'; "
                     "pass --subhalo-id explicitly.")
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{subhalo_id}_alpha{float(args.alpha)}"
    print(f"Subhalo {subhalo_id}, alpha = {float(args.alpha)}")
    print(f"IMBH mass subtracted from the stellar profile: {not args.no_subtract_imbh}; "
          f"TDE onset delayed by {args.trh_delay:g} half-mass relaxation time(s)")
    df = load_data_file(args.input_path)
    print(df.columns)
    tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun = set_up_timebins(1e5*u.yr)
    radial_edges_kpc = np.logspace(np.log10(args.r_min_kpc), np.log10(args.r_max_kpc), args.n_rbins + 1)
    (tde_rate_array_msunyr, mass_trelax_array_msun, mean_radius_kpc, n_clusters_with_tde,
     contributing_rows, radial) = run_clusters(
        df, float(args.alpha), tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun,
        resolution=1e5*u.yr, track_dir=args.track_dir, radial_edges_kpc=radial_edges_kpc,
        subtract_imbh=not args.no_subtract_imbh, trh_delay=args.trh_delay,
    )
    output_df = pd.DataFrame({
        'time': tscale_array_yr[:-2],
        'tde_rate_array_msunyr': tde_rate_array_msunyr[:-2],
        'mass_trelax_array_msun': mass_trelax_array_msun[:-2],
        'mean_radius_kpc': mean_radius_kpc[:-2],
        'n_clusters_with_tde': n_clusters_with_tde[:-2],
        # escaped clusters' share of tde_rate_array_msunyr (already included in it)
        'tde_rate_escaped_msunyr': radial['rate_escaped_msunyr'][:-2],
    })

    rates_path = os.path.join(args.output_dir, f'tde_rates_{tag}.csv')
    output_df.to_csv(rates_path, index=False)
    print(f"TDE rates written to {rates_path}")

    # Rate-weighted histogram of separation from the central galaxy, on the
    # same time grid (see run_clusters' `radial`). Mostly zeros, so a
    # compressed .npz is small; read it with plot_tde_radial_smoothed.py.
    radial_path = os.path.join(args.output_dir, f'tde_radial_{tag}.npz')
    np.savez_compressed(
        radial_path,
        time_yr=tscale_array_yr[:-2],
        edges_kpc=radial['edges_kpc'],
        hist_msunyr=radial['hist_msunyr'][:-2],
        inspiraled_msunyr=radial['inspiraled_msunyr'][:-2],
        no_track_msunyr=radial['no_track_msunyr'][:-2],
        clipped_msun=radial['clipped_msun'],
        hist_escaped_msunyr=radial['hist_escaped_msunyr'][:-2],
        no_track_escaped_msunyr=radial['no_track_escaped_msunyr'][:-2],
        clipped_escaped_msun=radial['clipped_escaped_msun'],
        subhalo_id=subhalo_id, alpha=float(args.alpha),
    )
    dt_yr = float(tscale_array_yr[1] - tscale_array_yr[0])  # actual grid spacing (~1e5 yr)
    tot = float(np.sum(tde_rate_array_msunyr.to(u.Msun / u.yr).value)) * dt_yr
    parts = {k: float(np.sum(radial[k])) * dt_yr for k in ('hist_msunyr', 'inspiraled_msunyr', 'no_track_msunyr')}
    print(f"Radial TDE histogram written to {radial_path}: of {tot:.3g} Msun lost, "
          f"{parts['hist_msunyr']:.3g} placed by separation, {parts['inspiraled_msunyr']:.3g} after "
          f"inspiral (at center), {parts['no_track_msunyr']:.3g} with no radius track; "
          f"{radial['clipped_msun']:.3g} clipped into the edge bins of "
          f"[{args.r_min_kpc:g}, {args.r_max_kpc:g}] kpc.")
    esc = float(np.sum(radial['rate_escaped_msunyr'])) * dt_yr
    print(f"Of that, {esc:.3g} Msun ({esc / tot:.1%}) came from escaped clusters." if tot > 0
          else "No mass lost.")

    # Sidecar file: which clusters actually contributed at least one
    # mass-loss bin to the TDE rate above (see run_clusters' own
    # docstring) -- lets downstream tools (e.g. plot_radius_tracks.py's
    # --tde-contributors-csv) restrict to actively TDE-generating clusters
    # without re-running timescale_analysis themselves.
    contributors_path = os.path.join(args.output_dir, f'tde_contributors_{tag}.csv')
    pd.DataFrame(contributing_rows).to_csv(contributors_path, index=False)
    if contributing_rows and args.trh_delay > 0:
        delays = np.array([r['tde_delay_gyr'] for r in contributing_rows]) * 1e3
        print(f"TDE onset delay (t_rh) for contributing clusters: median {np.median(delays):.3g} Myr, "
              f"16-84%: {np.percentile(delays, 16):.3g}-{np.percentile(delays, 84):.3g} Myr")
    print(f"{len(contributing_rows)} of {len(df)} clusters actively contributed to the TDE rate "
          f"-- see {contributors_path}")

if __name__ == "__main__":
    main()
