"""
Re-run the orbit traces of a few clusters with imbh.py's verbose output on, to
see WHEN and HOW clusters that form near (but not inside) their host's centre
end up leaving it instead of sinking by dynamical friction.

For each selected cluster it rebuilds the exact inputs imbh.py used (mass,
initial position/velocity relative to its formation host, formation time,
formation host) from the cluster table and subhalo_formation_<ID>.csv, calls
imbh.trace_cluster_to_snapshot(..., verbose=True), and parses the per-snapshot
diagnostics into a table:

  step, snap, cosmic age, host id, host mass/R_vir, |r| and |r|/R_vir in the
  CURRENT host's frame, |v|, the specific energy (KE, local and background
  potential, bound/unbound), whether the next step is a genuine merger
  (is_primary=False -> the cluster is handed to a new host), and the leg's
  integration result.

Outputs in --out-dir (default orbit_debug/):
  orbit_<halo>_c<row>.log   imbh.py's raw verbose output
  orbit_<halo>_c<row>.csv   the parsed per-step table
  orbit_<halo>_c<row>.png   |r|/R_vir of the current host and host mass vs cosmic age,
                            hand-offs to a new host marked
  summary.csv               one row per cluster: original vs re-run status, the first
                            step it is outside its host's R_vir and what happened there
                            ('hand-off' = moved to a new host at a merger, 'host shrank'
                            = R_vir dropped by >30%, 'orbit' = it climbed out within one
                            host), and the first step it is energetically unbound

Selection (from the model's cluster_output_<ID>_99<suffix>.dat): IMBH >= --mbh-min
that formed before the trace ended, status in --status (default outskirts,escaped:
clusters that did NOT sink), and starting distance r0/R_host in [--r0-min, --r0-max]
(default 0.01-0.1, i.e. just outside the inspiral radius). --n clusters are drawn at
random per halo (--seed), or give --rows to pick table rows explicitly.

Needs imbh.py's environment (the timescales package) and the merger trees, so run
it on Hoffman2, e.g.:

  python debug_orbits.py --data-dir /u/scratch/c/clairewi/imbh-output --halos 685512 939095 --n 5
  python debug_orbits.py --data-dir ... --model "--jitter-dex 0.1" --halos 685512 --n 5
  python debug_orbits.py --data-dir ... --halos 685512 --rows 12 345 678
  python debug_orbits.py --data-dir ... --frame-model legacy     # reproduce runs made before the fix
"""
import argparse
import contextlib
import io
import os
import re

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rate_runs import Run

CONSERVATIVE = "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12"
NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf)"
RE_STEP = re.compile(rf"\[step (\d+)\] BEFORE: current_id=(-?\d+), snap=(\d+), host_mass=({NUM}) Msun, "
                     rf"host_R_vir=({NUM}) kpc, \|r\|=({NUM}) kpc \(({NUM}) R_vir\), \|v\|=({NUM}) km/s")
RE_LEG = re.compile(rf"leg: capped_by=(\w+), t_leg_max=({NUM}) Gyr, is_primary=(\w+), subdivided=(\w+), "
                    rf"background: (.*?), H\(z\)")
RE_EN = re.compile(rf"energy: KE=({NUM}), Phi_local=({NUM}), Phi_bg=({NUM}), E_total=({NUM}) "
                   rf"\(km/s\)\^2 -> (\w+)")
RE_AFTER = re.compile(rf"AFTER integrate: status=(\w+), elapsed=({NUM}) Gyr, \|r\| \(before reframe\)=({NUM}) kpc")


def parse_log(text):
    rows, cur = [], None
    for line in text.splitlines():
        if (m := RE_STEP.search(line)):
            if cur:
                rows.append(cur)
            cur = dict(step=int(m[1]), host_id=int(m[2]), snap=int(m[3]), host_mass_msun=float(m[4]),
                       host_rvir_kpc=float(m[5]), r_kpc=float(m[6]), r_over_rvir=float(m[7]), v_kms=float(m[8]))
        elif cur is None:
            continue
        elif (m := RE_LEG.search(line)):
            cur.update(capped_by=m[1], t_leg_gyr=float(m[2]), next_is_primary=(m[3] == "True"),
                       background=m[5])
        elif (m := RE_EN.search(line)):
            cur.update(KE=float(m[1]), phi_local=float(m[2]), phi_bg=float(m[3]), E_total=float(m[4]),
                       bound=(m[5] == "BOUND"))
        elif (m := RE_AFTER.search(line)):
            cur.update(leg_status=m[1], leg_elapsed_gyr=float(m[2]), r_end_of_leg_kpc=float(m[3]))
    if cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def how_it_left(steps):
    """First step outside R_vir of the current host, and what happened just before it."""
    out = {"first_outside_step": None, "exit_mechanism": "never left", "first_unbound_step": None}
    if steps.empty:
        return out
    if "bound" in steps:
        ub = steps[steps["bound"] == False]  # noqa: E712
        out["first_unbound_step"] = int(ub["step"].iloc[0]) if len(ub) else None
    outside = steps.index[steps["r_over_rvir"] > 1]
    if not len(outside):
        return out
    k = outside[0]
    out["first_outside_step"] = int(steps.loc[k, "step"])
    if k == steps.index[0]:
        out["exit_mechanism"] = "started outside"
        return out
    prev, now = steps.loc[k - 1], steps.loc[k]
    out["age_at_exit_gyr"] = now.get("age_gyr", np.nan)
    if not prev.get("next_is_primary", True):
        out["exit_mechanism"] = "hand-off"            # merged into a new host; now in its frame
    elif now["host_rvir_kpc"] < 0.7 * prev["host_rvir_kpc"]:
        out["exit_mechanism"] = "host shrank"
    else:
        out["exit_mechanism"] = "orbit"               # climbed out while in the same host
    out["r_over_rvir_before_exit"] = prev["r_over_rvir"]
    out["bound_before_exit"] = prev.get("bound", np.nan)
    return out


def plot_cluster(steps, title, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": 0.05})
    t = steps["age_gyr"]
    a1.plot(t, steps["r_over_rvir"], "-o", ms=3, color="#256abf")
    a1.axhline(1, color="0.5", lw=1)
    a1.axhline(0.01, color="0.5", lw=1, ls=":")
    hop = steps.index[~steps.get("next_is_primary", pd.Series(True, index=steps.index)).astype(bool)]
    for k in hop:
        if k + 1 in steps.index:
            a1.axvline(steps.loc[k + 1, "age_gyr"], color="#eb6834", lw=1, alpha=0.6)
    if "bound" in steps:
        ub = steps[steps["bound"] == False]  # noqa: E712
        a1.plot(ub["age_gyr"], ub["r_over_rvir"], "x", color="#e34948", ms=7, label="unbound")
    a1.set_yscale("log")
    a1.set_ylabel(r"$|r|/R_{\rm vir}$ (current host)")
    a1.set_title(title, fontsize=10)
    a2.plot(t, steps["host_mass_msun"], "-", color="0.3")
    a2.set_yscale("log")
    a2.set_ylabel(r"host mass ($M_\odot$)")
    a2.set_xlabel("cosmic age (Gyr)")
    for k in hop:
        if k + 1 in steps.index:
            a2.axvline(steps.loc[k + 1, "age_gyr"], color="#eb6834", lw=1, alpha=0.6)
    a1.text(0.99, 0.02, "orange: hand-off to a new host at a merger; dotted: inspiral radius",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="Where cluster_output_<ID>_99<suffix>.dat live.")
    p.add_argument("--model", default=CONSERVATIVE, help=f'Run flags (default conservative: "{CONSERVATIVE}").')
    p.add_argument("--halos", nargs="+", default=["685512"])
    p.add_argument("--formation-dir", default=".", help="Where subhalo_formation_<ID>.csv live.")
    p.add_argument("--tree-dir", default="tng_download")
    p.add_argument("--snap-redshift-path", default="tng_download/snapshot_redshifts.json")
    p.add_argument("--host-mass-model", default="own", help="As used for the run (imbh.py default: own).")
    p.add_argument("--snap", type=int, default=99)
    p.add_argument("--n", type=int, default=5, help="Clusters per halo (random).")
    p.add_argument("--rows", type=int, nargs="+", default=None, help="Explicit table rows instead of --n.")
    p.add_argument("--status", default="outskirts,escaped")
    p.add_argument("--r0-min", type=float, default=0.01, help="Min starting distance in R_host.")
    p.add_argument("--r0-max", type=float, default=0.1, help="Max starting distance in R_host.")
    p.add_argument("--mbh-min", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="orbit_debug")
    p.add_argument("--frame-model", choices=["fixed", "legacy"], default="fixed",
                   help="imbh.py --frame-model to re-run with (default fixed; 'legacy' reproduces runs "
                        "made before the Oct-2026 reframe fix).")
    p.add_argument("--inspiral-capture-frac", type=float, default=1.0,
                   help="imbh.py --inspiral-capture-frac to re-run with (default 1).")
    args = p.parse_args()

    import imbh   # needs the timescales package
    from tree_navigator import MergerTreeNavigator

    run = Run("debug", args.model, args.data_dir, args.snap)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = []
    for halo in args.halos:
        df = pd.read_pickle(run.cluster_output_path(halo))
        form = pd.read_csv(os.path.join(args.formation_dir, f"subhalo_formation_{halo}.csv"))
        r0 = np.sqrt(df.cluster_sep_x_kpc ** 2 + df.cluster_sep_y_kpc ** 2 + df.cluster_sep_z_kpc ** 2)
        df["r0_over_Rhost"] = r0 / df.sampler_host_radius_kpc
        if args.rows:
            picks = list(args.rows)
        else:
            ok = ((pd.to_numeric(df.IMBH_mass_msun, errors="coerce") >= args.mbh_min)
                  & (df.IMBH_final_formation_time_gyr < df.total_time_gyr)
                  & df.status.isin(args.status.split(","))
                  & df.r0_over_Rhost.between(args.r0_min, args.r0_max))
            cand = np.flatnonzero(ok.to_numpy())
            print(f"{halo}: {len(cand)} candidate clusters; tracing {min(args.n, len(cand))}")
            picks = list(rng.choice(cand, size=min(args.n, len(cand)), replace=False)) if len(cand) else []

        tree = os.path.join(args.tree_dir, f"sublink_full_{halo}.hdf5")
        nav = MergerTreeNavigator(tree, args.snap_redshift_path, box_size_ckpc_h=35000.0,
                                  host_mass_model=args.host_mass_model)
        target_age = nav.age_at_snap(args.snap)

        for row in picks:
            c = df.iloc[int(row)]
            start_id = int(form["formation_subhalo_id"].iloc[int(c.initial_subhalo_id)])
            m_cl = c.cluster_mass_msun * u.Msun
            r0_vec = np.array([c.cluster_sep_x_kpc, c.cluster_sep_y_kpc, c.cluster_sep_z_kpc]) * u.kpc
            v0_vec = np.array([c.cluster_vel_x_kms, c.cluster_vel_y_kms, c.cluster_vel_z_kms]) * u.km / u.s
            formation_age = c.cluster_formation_age_gyr * u.Gyr
            _, r_host, _ = nav.host_properties(start_id)
            tag = f"orbit_{halo}_c{int(row)}"
            print(f"  {tag}: M_cl={c.cluster_mass_msun:.3g}, r0={np.linalg.norm(r0_vec.value):.4f} kpc "
                  f"({np.linalg.norm(r0_vec.value) / r_host:.3f} R_host), z_form={c.cluster_formation_redshift:.2f}, "
                  f"original status={c.status}")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    trace = imbh.trace_cluster_to_snapshot(m_cl, r0_vec, v0_vec, start_id, target_age, nav,
                                                           verbose=True, formation_age=formation_age,
                                                           frame_model=args.frame_model,
                                                           inspiral_capture_frac=args.inspiral_capture_frac)
                err = None
            except Exception as e:   # keep going; the log shows how far it got
                trace, err = None, repr(e)
            text = buf.getvalue()
            with open(os.path.join(args.out_dir, f"{tag}.log"), "w") as f:
                f.write(text)
            steps = parse_log(text)
            if not steps.empty:
                steps["age_gyr"] = [formation_age.value if i == 0 else nav.age_at_snap(s).to(u.Gyr).value
                                    for i, s in enumerate(steps["snap"])]
                steps.to_csv(os.path.join(args.out_dir, f"{tag}.csv"), index=False)
                plot_cluster(steps, f"{tag}: M_cl={c.cluster_mass_msun:.2g} Msun, "
                                    f"r0={c.r0_over_Rhost:.3f} R_host, original status {c.status}",
                             os.path.join(args.out_dir, f"{tag}.png"))
            info = how_it_left(steps)
            rec = {"halo": halo, "row": int(row), "cluster_mass_msun": c.cluster_mass_msun,
                   "imbh_mass_msun": c.IMBH_mass_msun, "z_form": c.cluster_formation_redshift,
                   "r0_over_Rhost": c.r0_over_Rhost, "host_mass_msun": c.sampler_host_mass_msun,
                   "original_status": c.status, "rerun_status": trace["status"] if trace else f"error {err}",
                   "original_total_time": c.total_time_gyr,
                   "rerun_total_time": trace["total_time_gyr"] if trace else np.nan,
                   "n_steps": len(steps), **info}
            summary.append(rec)
            print(f"    re-run: {rec['rerun_status']}; left host at step {info['first_outside_step']} "
                  f"({info['exit_mechanism']}); first unbound step {info['first_unbound_step']}")

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    if len(s):
        print("\nHow the clusters left their host:")
        print(s["exit_mechanism"].value_counts().to_string())
        mism = (s["original_status"] != s["rerun_status"]).sum()
        if mism:
            print(f"NOTE: {mism} re-run(s) ended with a different status than the original run "
                  f"(different code version or settings?)")
    print(f"\nWrote {args.out_dir}/summary.csv and per-cluster logs, tables and plots.")


if __name__ == "__main__":
    main()
