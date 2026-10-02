"""
Shared run-naming helpers for the plotting scripts.

Rebuilds the output-name suffix that submit_halos.pl / submit_analysis.pl use
for a given run configuration, so the plotting scripts can find that run's
analysis.py outputs and tag their own figures/CSVs with it:

    default mode, no label : suffix ""        -> analysis files in <data-dir>
    otherwise              : suffix "_<...>"  -> analysis files in <data-dir>/analysis<suffix>

    <suffix> = [_obs_<n-relation>_<mass-function>_<radius-relation>_<phase-space>
                _host<host-mass-model>_boost<n-boost>][_<jitter tag>][_<label>]

    <jitter tag> (default mode only) = jit<s> if --mass-jitter-dex == --radius-jitter-dex
                                       = jitM<m>R<r> otherwise; absent if both are 0

KEEP IN SYNC with the naming block in submit_halos.pl and submit_analysis.pl.

Also holds the plot-only --escaped selection (add_escaped_arg), whose suffix
(_noescaped / _escapedonly) is appended to the plotting scripts' output names.
"""
import os
import re

N_RELATIONS = ["bf20", "bf20_seed", "harris17_number", "harris17_mass"]
MASS_FUNCTIONS = ["gclf", "icmf"]
RADIUS_RELATIONS = ["brown_gnedin21", "brown_gnedin21_young", "marks_kroupa12", "const_surface_density"]
PHASE_SPACES = ["analytic", "simulation"]
HOST_MASS_MODELS = ["own", "group"]


def add_run_args(parser):
    """Add the run-selection flags (same names/defaults as submit_halos.pl)."""
    g = parser.add_argument_group(
        "run selection",
        "Which model run to plot -- pass the same flags you gave submit_halos.pl / "
        "submit_analysis.pl. --data-dir is the base output directory ($OUTDIR); the "
        "matching analysis<suffix>/ subdirectory is found automatically.")
    g.add_argument("--mode", choices=["default", "observational"], default="default")
    g.add_argument("--n-relation", choices=N_RELATIONS, default="bf20")
    g.add_argument("--mass-function", choices=MASS_FUNCTIONS, default="gclf")
    g.add_argument("--radius-relation", choices=RADIUS_RELATIONS, default="brown_gnedin21")
    g.add_argument("--phase-space", choices=PHASE_SPACES, default="analytic")
    g.add_argument("--host-mass-model", choices=HOST_MASS_MODELS, default="own")
    g.add_argument("--n-boost", default="1.0",
                   help="As passed to submit_halos.pl (used verbatim in the names, e.g. 1.0).")
    g.add_argument("--jitter-dex", type=float, default=0.0,
                   help="Default mode: smoothed-bootstrap jitter on cluster mass and radius "
                        "[dex], as passed to submit_halos.pl (default 0 = none).")
    g.add_argument("--mass-jitter-dex", type=float, default=None,
                   help="Override --jitter-dex for cluster mass only.")
    g.add_argument("--radius-jitter-dex", type=float, default=None,
                   help="Override --jitter-dex for cluster radius only.")
    g.add_argument("--label", default="")
    return parser


def _jitter_widths(args):
    j = getattr(args, "jitter_dex", 0.0) or 0.0
    m = getattr(args, "mass_jitter_dex", None)
    r = getattr(args, "radius_jitter_dex", None)
    return (j if m is None else m), (j if r is None else r)


def jitter_tag(args):
    """'' / 'jit<s>' / 'jitM<m>R<r>' (default mode only). KEEP IN SYNC with imbh.jitter_tag."""
    if args.mode == "observational":
        return ""
    m, r = _jitter_widths(args)
    if m == 0 and r == 0:
        return ""
    return f"jit{m:g}" if m == r else f"jitM{m:g}R{r:g}"


def run_suffix(args):
    """'' for an unlabelled default-mode run, else '_<obs tag>[_<label>]' / '_<label>'."""
    if args.label and re.search(r"[^\w.\-]", args.label):
        raise SystemExit("--label may only contain letters, digits, '.', '_' or '-'")
    parts = []
    if args.mode == "observational":
        parts.append(f"obs_{args.n_relation}_{args.mass_function}_{args.radius_relation}_"
                     f"{args.phase_space}_host{args.host_mass_model}_boost{args.n_boost}")
    if jitter_tag(args):
        parts.append(jitter_tag(args))
    if args.label:
        parts.append(args.label)
    return "".join(f"_{p}" for p in parts)


def analysis_dir(data_dir, suffix):
    """
    Directory holding this run's analysis.py outputs. For a tagged run that's
    <data_dir>/analysis<suffix>; if that doesn't exist, falls back to
    <data_dir> itself (e.g. --data-dir already points at the analysis folder).
    """
    if not suffix:
        return data_dir
    sub = os.path.join(data_dir, f"analysis{suffix}")
    if os.path.isdir(sub):
        return sub
    print(f"NOTE: {sub} not found; looking for the analysis files in {data_dir} instead")
    return data_dir


ESCAPED_MODES = ["include", "exclude", "only"]


def add_escaped_arg(parser):
    """--escaped include|exclude|only (needs analysis.py outputs that carry the escaped share)."""
    parser.add_argument(
        "--escaped", choices=ESCAPED_MODES, default="include",
        help="Clusters with status 'escaped' (beyond 3 R200 of the z=0 host at the last "
             "snapshot): include them (default), exclude them, or plot only them. "
             "exclude/only need analysis.py outputs written after the escaped split was "
             "added (tde_rate_escaped_msunyr / hist_escaped_msunyr); re-run analysis.py "
             "if they're missing.")
    return parser


def escaped_suffix(args):
    """File-name suffix for the --escaped selection ('' for include)."""
    return {"include": "", "exclude": "_noescaped", "only": "_escapedonly"}[args.escaped]


def escaped_description(args):
    """Title text for the --escaped selection ('' for include)."""
    return {"include": "", "exclude": "escaped clusters excluded",
            "only": "escaped clusters only"}[args.escaped]


def run_description(args):
    """Short human-readable description of the run, for figure titles ('' for default)."""
    bits = []
    if args.mode == "observational":
        bits.append(f"obs: {args.n_relation}, {args.mass_function}, {args.radius_relation}, "
                    f"{args.phase_space}, host {args.host_mass_model}"
                    + (f", boost {args.n_boost}" if args.n_boost not in ("1", "1.0") else ""))
    if jitter_tag(args):
        m, r = _jitter_widths(args)
        bits.append(f"jitter {m:g} dex" if m == r else f"jitter M {m:g} / R {r:g} dex")
    if args.label:
        bits.append(args.label)
    return "; ".join(bits)
