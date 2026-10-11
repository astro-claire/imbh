"""
Check that every galaxy has a complete, consistent imbh.py + analysis.py output
for every configuration the paper uses, and print the commands to fix what isn't.

Configurations (same flags as submit_halos.pl / submit_analysis.pl):
  maximal (--jitter-dex 0.1), its --seeds 1..5 realizations, and the seven
  model-ladder runs of plot_model_ladder.py (incl. the reverse test).
Galaxies: the 15 in plot_tde_rates_smoothed.DEFAULT_MASS_BINS.

For each (configuration, galaxy):
  MODEL     cluster_output_<ID>_<snap><suffix>.dat exists and loads
            (an empty table = a halo that drew no clusters, which is fine)
  TRACKS    every radius_track_path in it exists and is not newer than the .dat
            (a newer track = an interrupted rerun overwrote it)
  ANALYSIS  tde_rates / tde_radial / tde_contributors exist in the run's
            analysis folder and are all newer than the .dat
  FRESH     with --rerun-after: galaxies in --rerun-halos have a .dat and
            analysis written after that time (i.e. with the current code)
Also checks that the r200_history_<ID>.csv used by the within/beyond plots is
newer than --rerun-after for --gap-halos (main-branch gaps, see tree_navigator).

    python check_runs.py --data-dir /u/scratch/c/clairewi/imbh-output \\
        --rerun-after "2026-10-11 09:00"
"""
import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd

from rate_runs import Run, DEFAULT_MASS_BINS
from plot_model_ladder import DEFAULT_LADDER, DEFAULT_REVERSE

RERUN_HALOS = ["697044", "939095", "670354", "802209", "1166615", "1110727", "1099576"]
GAP_HALOS = ["697044", "939095", "670354"]
SLACK_S = 5.0   # mtime tolerance (s) for files written in the same second / NFS clocks


def configurations(n_seeds):
    """(label, submit_halos.pl flags, run-naming flags)."""
    out = [("maximal", "--jitter-dex 0.1", "--jitter-dex 0.1")]
    out += [(f"maximal seed{k}", f"--jitter-dex 0.1 --seeds {k}", f"--jitter-dex 0.1 --label seed{k}")
            for k in range(1, n_seeds + 1)]
    out += [(lbl, fl, fl) for lbl, fl in DEFAULT_LADDER if lbl != "Maximal"]
    out += [DEFAULT_REVERSE[:1] + (DEFAULT_REVERSE[1], DEFAULT_REVERSE[1])]
    return out


def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else None


def check_one(run, halo, alpha, rerun_after, must_be_fresh):
    """List of (stage, message); stage is 'model' or 'analysis'. Empty = OK."""
    problems, notes = [], []
    dat = run.cluster_output_path(halo)
    t_dat = mtime(dat)
    if t_dat is None:
        return [("model", "no cluster_output .dat")], notes
    try:
        df = pd.read_pickle(dat)
    except Exception as e:
        return [("model", f".dat unreadable ({e.__class__.__name__})")], notes
    if len(df) == 0:
        notes.append("0 clusters")
    if must_be_fresh and rerun_after and t_dat < rerun_after:
        problems.append(("model", f".dat older than --rerun-after ({time.ctime(t_dat)})"))

    paths = [p for p in df.get("radius_track_path", pd.Series(dtype=object)) if isinstance(p, str) and p]
    missing = [p for p in paths if not os.path.exists(p)]
    newer = [p for p in paths if os.path.exists(p) and os.path.getmtime(p) > t_dat + SLACK_S]
    if missing:
        problems.append(("model", f"{len(missing)} of {len(paths)} radius tracks missing"))
    if newer:
        problems.append(("model", f"{len(newer)} radius tracks newer than the .dat (interrupted rerun?)"))
    if "status" in df.columns:
        n_fail = int((df["status"].astype(str) == "trace_failed").sum())
        if n_fail:
            notes.append(f"{n_fail} trace_failed")

    for kind, ext in (("tde_rates", "csv"), ("tde_radial", "npz"), ("tde_contributors", "csv")):
        f = os.path.join(run.analysis_dir, f"{kind}_{halo}_alpha{alpha}.{ext}")
        t = mtime(f)
        if t is None:
            problems.append(("analysis", f"no {kind} file"))
        elif t + SLACK_S < t_dat:
            problems.append(("analysis", f"{kind} older than the .dat (analysis ran before the model finished?)"))
    return problems, notes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Base output directory ($OUTDIR).")
    p.add_argument("--loss-cone", default="full")
    p.add_argument("--coulomb-log", default="bh")
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--rerun-after", default=None,
                   help='e.g. "2026-10-11 09:00": --rerun-halos outputs must be newer than this')
    p.add_argument("--rerun-halos", nargs="*", default=RERUN_HALOS)
    p.add_argument("--gap-halos", nargs="*", default=GAP_HALOS)
    args = p.parse_args()

    rerun_after = (datetime.strptime(args.rerun_after, "%Y-%m-%d %H:%M").timestamp()
                   if args.rerun_after else None)
    halos = [h for _, _, _, ids in DEFAULT_MASS_BINS for h in ids]
    lc = f" --loss-cone {args.loss_cone} --coulomb-log {args.coulomb_log}"

    to_model = defaultdict(list)      # submit_halos flags -> halos
    to_analyse = defaultdict(list)    # submit_analysis flags -> halos
    n_ok = n_bad = 0
    for label, submit_flags, naming_flags in configurations(args.n_seeds):
        run = Run(label, naming_flags + lc, args.data_dir, args.snap)
        print(f"\n== {label}   [{os.path.basename(run.analysis_dir)}]")
        for h in halos:
            problems, notes = check_one(run, h, args.alpha, rerun_after, h in args.rerun_halos)
            extra = f"   ({', '.join(notes)})" if notes else ""
            if not problems:
                n_ok += 1
                print(f"   ok      {h:>8s}{extra}")
                continue
            n_bad += 1
            print(f"   RERUN   {h:>8s}: " + "; ".join(f"[{s}] {m}" for s, m in problems) + extra)
            if any(s == "model" for s, _ in problems):
                to_model[submit_flags].append(h)
            to_analyse[submit_flags].append(h)   # a model rerun always needs a new analysis

    if rerun_after:
        print("\n== r200_history caches (within/beyond plots)")
        for h in args.gap_halos:
            f = os.path.join(args.data_dir, f"r200_history_{h}.csv")
            t = mtime(f)
            if t is not None and t < rerun_after:
                print(f"   DELETE  {f}  (made before the main-branch gap fix; the plots rebuild it)")
            else:
                print(f"   ok      {h:>8s}  ({'not cached yet' if t is None else 'fresh'})")

    print(f"\n{n_ok} ok, {n_bad} need rerunning.")
    if to_model:
        print("\n# 1) model runs (wait for these to finish before step 2):")
        for flags, hs in to_model.items():
            print(f'perl submit_halos.pl --halos "{" ".join(hs)}" {flags}')
    if to_analyse:
        print("\n# 2) analysis runs" + (" (after step 1 has finished):" if to_model else ":"))
        for flags, hs in to_analyse.items():
            print(f'perl submit_analysis.pl --halos "{" ".join(hs)}" {flags}{lc}')
    if not (to_model or to_analyse):
        print("Everything is complete and consistent.")


if __name__ == "__main__":
    main()
