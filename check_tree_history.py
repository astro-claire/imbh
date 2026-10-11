"""
Run tree_checks.check_tree on SubLink trees and print the verdicts.

    python check_tree_history.py tng_download/sublink_full_*.hdf5 [--max-unexplained-frac 0.1]
"""
import argparse
from tree_checks import check_tree, describe, MAX_UNEXPLAINED_FRAC

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("trees", nargs="+")
p.add_argument("--max-unexplained-frac", type=float, default=MAX_UNEXPLAINED_FRAC)
args = p.parse_args()
for path in args.trees:
    name = path.split("_")[-1].replace(".hdf5", "")
    print(describe(name, check_tree(path, args.max_unexplained_frac)))
