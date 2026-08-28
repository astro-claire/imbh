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
from dynamical_friction import NFWHost, sink_time
#timescales stuff
from timescales import TimescaleEnsemble
from timescales.data import build_single_system_grid
from timescales.analysis.modelv2 import create_dynamical_model_integral
from astropy.cosmology import FlatLambdaCDM
#FIXME hard coded
cosmo = FlatLambdaCDM(71,0.27,Ob0=0.044, Tcmb0=2.726 *u.K)
alpha = 1.2
# concentration assumed for the host NFW profile. TODO: this is a flat
# placeholder -- at z>=12 you likely want something lower (high-z halos
# are less concentrated), possibly per-halo from a mass-concentration-
# redshift relation rather than one fixed number for every subhalo.
HOST_CONCENTRATION = 4.0





cluster_sampler = ClusterPopulationSampler.load("/Users/clairewilliams/Research/IMBH/cluster_sampler.pkl")


#--------- Load post-processed illustris merger tree 
def load_merger_tree_idx(df):
    goodidx = np.where(df['delta_t_gyr']>0)[0]
    print("There are "+str(len(goodidx))+" subhalos with nonzero merger time.")
    return goodidx


# -------- Attach AREPO clusters to the primordial halos
#          For each cluster, we have delta_r and delta_v away from the central 
def draw_clusters(subhalo_mass, subhalo_radius):
    return cluster_sampler.draw_clusters(subhalo_mass, subhalo_radius)

 
def assign_subhalo_merger_tscale(cluster_props, delta_t, subhalo_mass, subhalo_radius,
                                  concentration=HOST_CONCENTRATION):
    """
    For each cluster, integrate its dynamical friction sink time through
    the host subhalo's (fixed, formation-time) NFW profile, and compare it
    to the host's own merger timescale delta_t:
      - if the cluster sinks to the host center BEFORE the host itself
        merges into its descendant -> outcome "cluster_sinks_first",
        tscale = the dynamical friction sink time.
      - otherwise (the host merges first, or the cluster's sink time
        exceeds delta_t / escapes / doesn't converge) -> outcome
        "host_merger", tscale = delta_t (unchanged from before -- this is
        the FIXED-formation-time model; a later version can hand the
        cluster off to the new host and continue integrating from there).

    cluster_props['cluster_sep'] and ['cluster_vel'] are now (N,3) VECTOR
    Quantities (the direction sampling -- isotropic separation, and a
    velocity direction correlated with separation via a real sampled
    cos_theta -- already happened in ClusterPopulationSampler.draw_clusters()),
    so we integrate the actual orbit directly via sink_time() rather than
    re-inventing a random direction here.
    """
    host = NFWHost(subhalo_mass * u.Msun, subhalo_radius * u.kpc, concentration=concentration)
    delta_t_q = delta_t * u.Gyr
 
    tscales = []
    which_outcome = []
    for clusteridx in range(len(cluster_props['cluster_mass'])):
        m_cl = cluster_props['cluster_mass'][clusteridx]
        r0_vec = cluster_props['cluster_sep'][clusteridx]
        v0_vec = cluster_props['cluster_vel'][clusteridx]
 
        t_df, status = sink_time(
            m_cl, r0_vec, v0_vec, host, t_max=delta_t_q,
        )
 
        if status == "merged" and t_df < delta_t_q:
            tscales.append(t_df.to(u.Gyr).value)
            which_outcome.append("cluster_sinks_first")
        else:
            # host merges first (or the cluster never converges within
            # delta_t, or escapes the host's potential entirely)
            tscales.append(delta_t)
            which_outcome.append("host_merger" if status != "escaped" else "escaped")
 
    return tscales, which_outcome

#-------- time evolution for the IMBH analytic model
# 
def create_timescale_model(mass, radius, Nsampling = 5):
    grid = build_single_system_grid(mass,radius)
    model_nobh = TimescaleEnsemble(grid, 
                        verbose = False,
                        densityModel="power-law",
                        Nsampling = Nsampling,
                        timescales_kwargs={'cosmology':cosmo},
                        profile_kwargs={"alpha":alpha})
    return model_nobh

def add_time_evolution(delta_t, model):
    output = create_dynamical_model_integral(model,verbose = False, timescale_override = delta_t)
    return output


def _run_timescale_model(mass, radius, delta_t):
    """Bundles create_timescale_model + add_time_evolution into one call for run_with_timeout()."""
    model = create_timescale_model(mass, radius)
    return add_time_evolution(delta_t, model)


# ------- Iteration over all the subhalos
def iterate_subhalos(df, goodidx): 
    clusters = []
    failures = []  # (halo_idx, cluster_idx, merger_tscale, which_outcome, error_message)
    #testing mode-just do the first few
    for idx in goodidx[0:10]: 
        cluster_props= draw_clusters(df['group_m_crit200_msun'][idx], df['group_r_crit200_kpc'][idx])
        print("Generated " +str(len(cluster_props['cluster_mass']))+ " clusters for this halo.")
        print(cluster_props['cluster_mass'])
        print(cluster_props['cluster_vel'])
        cluster_props['merger_tscale'], cluster_props['which_outcome'] = assign_subhalo_merger_tscale(cluster_props, df['delta_t_gyr'][idx], df['group_m_crit200_msun'][idx], df['group_r_crit200_kpc'][idx])
        cluster_props['IMBH_mass']= []
        for clusteridx in range(len(cluster_props['cluster_mass'])):
            model = create_timescale_model(cluster_props['cluster_mass'][clusteridx],cluster_props['cluster_radius'][clusteridx])            
            out = add_time_evolution(cluster_props['merger_tscale'][clusteridx] *u.Gyr, model)
            cluster_props['IMBH_mass'].append(out['M_VMS'][0])
        clusters.append(cluster_props)
    return clusters

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the subhalo_formation_<ID>.csv file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    required = {"delta_t_gyr", "halfmass_rad_kpc", "dm_mass_msun"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing expected column(s): {missing}")
    print(f"Loaded {len(df)} subhalo branches from {args.csv_path}")

    goodidx = load_merger_tree_idx(df)
    output_clusters = iterate_subhalos(df, goodidx)
    print(output_clusters[0]['IMBH_mass'])
    print(output_clusters[0]['which_outcome'])
    print(output_clusters[0]['merger_tscale'])


if __name__ == "__main__":
    main()