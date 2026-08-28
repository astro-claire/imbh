"""
One-time preprocessing: build a ClusterPopulationSampler from a raw
clusters_near_halos_exclusive_*.dat catalog and save it to disk, so your
main pipeline script can load it near-instantly instead of re-reading the
raw catalog and rebuilding the calibration arrays on every run.

Usage:
    python build_cluster_sampler.py clusters_near_halos_exclusive_....dat --out cluster_sampler.pkl

Then in your main pipeline script:
    from cluster_population_sampler import ClusterPopulationSampler
    sampler = ClusterPopulationSampler.load("cluster_sampler.pkl")
    # sampler.draw_clusters(subhalo_mass, subhalo_radius) as usual
"""

import argparse

from cluster_population_sampler import ClusterPopulationSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickle_path", help="Path to the RAW clusters_near_halos_exclusive_*.dat catalog")
    parser.add_argument("--out", default="cluster_sampler.pkl", help="Output path for the built sampler")
    parser.add_argument("--bandwidth-dex", type=float, default=0.3,
                         help="Kernel bandwidth in dex (log10 Msun); baked into the saved sampler")
    parser.add_argument("--window-sigma", type=float, default=6.0)
    parser.add_argument("--max-window-points", type=int, default=3000)
    args = parser.parse_args()

    sampler = ClusterPopulationSampler(
        args.pickle_path,
        bandwidth_dex=args.bandwidth_dex,
        window_sigma=args.window_sigma,
        max_window_points=args.max_window_points,
    )
    sampler.save(args.out)


if __name__ == "__main__":
    main()