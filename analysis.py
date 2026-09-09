import argparse
import numpy as np 
import astropy.units as u 
import h5py 
import matplotlib.pyplot as plt 
import pandas as pd
from astropy.constants import G
from pathlib import Path
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
        return pd.read_csv(path, sep=None, engine='python')
        
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

def timescale_analysis(df, idx, alpha):
    stellar_radius = stellar_radius_approximation(1* u.Msun)
    limiting_tscale_gyr = []
    roche = False # start out with no roche effects
    if df['IMBH_mass_msun'][idx]<5e2:
        #placeholder--need to decide what to do with the small BHs. 
        return None,None,None
    elif df['IMBH_final_formation_time_gyr'][idx] >= df['total_time_gyr'][idx]:
        #placeholder - if the IMBH hasn't formed yet, don't do anything
        return None,None,None
    else: 
        if df['min_roche_radius_kpc'][idx]< df['cluster_radius_pc'][idx] /1e3:
            #If roche lobe goes within cluster, need to consider roche effects (the outer radius is small)
            roche = True
            limiting_tscale_gyr.append(df['time_of_min_roche_gyr'][idx]-df['IMBH_final_formation_time_gyr'][idx])
        limiting_tscale_gyr.append(df['total_time_gyr'][idx]-df['IMBH_final_formation_time_gyr'][idx])
        t_cutoff_gyr = min(limiting_tscale_gyr)
        # print(f"The cutoff time is {t_cutoff_gyr}")

        r_sphere_of_influence = sphere_of_influence(df['IMBH_mass_msun'][idx], df['r0_pc'][idx],
                                                     df['rho0_msun_pc3'][idx], alpha)

        r = calc_cutoff_radius(t_cutoff_gyr, df['IMBH_mass_msun'][idx],
                                df['r0_pc'][idx], df['rho0_msun_pc3'][idx], alpha,
                                df['cluster_radius_pc'][idx])
        # print(r)
        # r_orbital_inner = calc_orbital_relaxation_radius(df['IMBH_mass_msun'][idx],df['r0_pc'][idx],
                                                    #  df['rho0_msun_pc3'][idx], alpha,df['cluster_radius_pc'][idx] )
        sigma_r, rho_r, M_r = calc_structural_props(r.value, df['IMBH_mass_msun'][idx],
                                                     df['rho0_msun_pc3'][idx],
                                                     df['r0_pc'][idx], alpha)
        if roche:
            outer_radius_pc = min(r_sphere_of_influence.value, df['r0_pc'][idx],df['min_roche_radius_kpc'][idx] * 1e3) 
        else: 
            outer_radius_pc = min(r_sphere_of_influence.value, df['r0_pc'][idx])
        r_orbital_inner = stellar_radius.to('pc')
        r_orbital_mstar = calc_min_enclosed_radius(2*u.Msun,df['r0_pc'][idx],df['rho0_msun_pc3'][idx],alpha  )
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
                                                     df['rho0_msun_pc3'][idx],
                                                     df['r0_pc'][idx], alpha) for i in range(Nbins)]
                mass_bins = [structural_props_bins[i][2] for i in range(Nbins)]
                # mass_bins = [calc_structural_props(radii_bins[i], df['IMBH_mass_msun'][idx],
                #                                      df['rho0_msun_pc3'][idx],
                #                                      df['r0_pc'][idx], alpha)[2] for i in range(Nbins) ]
                sigma_bins = [structural_props_bins[i][0] for i in range(Nbins)]
                rho_bins = [structural_props_bins[i][1] for i in range(Nbins)]
                mass_lost_bins = [mass_bins[i+1]-mass_bins[i] for i in range(Nbins-1)]
                # _,_, M_r_inner = calc_structural_props(r_orbital_inner.value, df['IMBH_mass_msun'][idx],
                #                                      df['rho0_msun_pc3'][idx],
                #                                      df['r0_pc'][idx], alpha)
                sigma_r_outer,rho_r_outer, M_r_outer =calc_structural_props(outer_radius_pc, df['IMBH_mass_msun'][idx],
                                                     df['rho0_msun_pc3'][idx],
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
                                                     df['rho0_msun_pc3'][idx],
                                                     df['r0_pc'][idx], alpha) for i in range(Nbins)]
                mass_bins = [structural_props_bins[i][2] for i in range(Nbins)]
                # mass_bins = [calc_structural_props(radii_bins[i], df['IMBH_mass_msun'][idx],
                #                                      df['rho0_msun_pc3'][idx],
                #                                      df['r0_pc'][idx], alpha)[2] for i in range(Nbins) ]
                sigma_bins = [structural_props_bins[i][0] for i in range(Nbins)]
                rho_bins = [structural_props_bins[i][1] for i in range(Nbins)]
                mass_lost_bins = [mass_bins[i+1]-mass_bins[i] for i in range(Nbins-1)]
                # _,_, M_r_inner = calc_structural_props(r_orbital_inner.value, df['IMBH_mass_msun'][idx],
                #                                      df['rho0_msun_pc3'][idx],
                #                                      df['r0_pc'][idx], alpha)
                sigma_r_outer,rho_r_outer, M_r_outer =calc_structural_props(outer_radius_pc, df['IMBH_mass_msun'][idx],
                                                     df['rho0_msun_pc3'][idx],
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

        return np.array(mass_lost_bins*u.Msun),np.array(t_relax_bin), t_cutoff_gyr

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

def run_clusters(df, alpha, tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun, resolution = 1e5*u.yr):
    resolution = resolution.to('yr').value
    for clusteridx in range(len(df)):
    # for clusteridx in range(2):
        if df['IMBH_mass_msun'][clusteridx]>0:
            halo_formation_time = cosmo.age(df['initial_subhalo_formation_redshift'][clusteridx])
            start_time = halo_formation_time+ (df['IMBH_final_formation_time_gyr'][clusteridx]*u.Gyr)
            mass_lost_bins,t_relax_bin, t_cutoff_gyr = timescale_analysis(df, clusteridx, alpha)
            if t_cutoff_gyr is not None: 
                i = int(round((t_cutoff_gyr * 1e9 - tscale_array_yr[0]) / resolution))
                j_start = int(round((start_time.value * 1e9 - tscale_array_yr[0]) / resolution))
                for binidx in range(len(mass_lost_bins)):
                    t_end = min(t_cutoff_gyr, start_time.value + t_relax_bin[binidx])
                    k_end = int(round((t_end * 1e9 - tscale_array_yr[0]) / resolution))
                    rate = mass_lost_bins[binidx] / t_relax_bin[binidx] * u.Msun / u.Gyr
                    mass = mass_lost_bins[binidx]* u.Msun
                    tde_rate_array_msunyr[j_start:k_end+1] += rate
                    mass_trelax_array_msun[k_end+1] += mass
    return tde_rate_array_msunyr, mass_trelax_array_msun



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="Path to the cluster_output_<branchid>_<snapshot>.csv file")
    parser.add_argument("alpha", help="power law index of cluster profiles")
    args = parser.parse_args()
    df = load_data_file(args.input_path)
    print(df.columns)
    tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun = set_up_timebins(1e5*u.yr)
    tde_rate_array_msunyr, mass_trelax_array_msun = run_clusters(df, float(args.alpha), tscale_array_yr, tde_rate_array_msunyr, mass_trelax_array_msun, resolution = 1e5*u.yr)
    output_df = pd.DataFrame({
        'time': tscale_array_yr[:-2],
        'tde_rate_array_msunyr': tde_rate_array_msunyr[:-2],
        'mass_trelax_array_msun': mass_trelax_array_msun[:-2],
    })

    output_df.to_csv(f'tde_rates_alpha{float(args.alpha)}.csv', index=False)

if __name__ == "__main__":
    main()