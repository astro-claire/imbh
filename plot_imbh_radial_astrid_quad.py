"""
Four-panel version of plot_imbh_radial_astrid.py (same options, same cached
z = 0 positions): IMBHs per galaxy per dex vs r (left column, kpc) and r/R200
(right column), with our models overplotted on ASTRID's wandering BHs.

  top row     all BHs
  bottom row  only BHs within --max-r200 * R200 of the central (default 3)

Panels touch: shared x within each column, one shared y axis for all four.

Example (as used for the paper figure):
  python plot_imbh_radial_astrid_quad.py --data-dir /u/scratch/c/clairewi/imbh-output \\
      --cache-dir /u/scratch/c/clairewi/imbh-output/z0_positions --max-r200 3 --bin-width 0.3 \\
      --model "Maximal" "--loss-cone full --include-undepleted --jitter-dex 0.1" \\
      --model "Conservative" "--mode observational --n-relation bf20_seed --radius-relation marks_kroupa12 --loss-cone full --include-undepleted" \\
      --output imbh_radial_z0_astrid_quad.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from plot_imbh_radial_astrid import (build_parser, load_inputs, apply_cut, print_summary, compute_edges,
                                     draw_pair, legend_handles, add_legends, log_ticks, FONT_SIZE)


def nice_ylim(values):
    """Log y limits around the data whose edges never sit on a decade (no clashing tick labels)."""
    lo, hi = np.log10(values.min()) - 0.2, np.log10(values.max()) + 0.3
    if abs(lo - round(lo)) < 0.12:
        lo -= 0.15
    if abs(hi - round(hi)) < 0.12:
        hi += 0.15
    return 10 ** lo, 10 ** hi


def main():
    p = build_parser(__doc__, "imbh_radial_z0_astrid_quad.pdf")
    p.set_defaults(max_r200=3.0)
    args = p.parse_args()

    models, astrid = load_inputs(args)
    print("\n== all BHs")
    print_summary(models, astrid)
    print(f"\n== within {args.max_r200:g} R200")
    cut_models, cut_astrid = apply_cut(models, astrid, args.max_r200)
    print_summary(cut_models, cut_astrid)

    lr_edges, lx_edges = compute_edges(models, astrid, args)             # top row sets the x range
    _, lx_cut = compute_edges(cut_models, cut_astrid, args, args.max_r200)

    fig, axes = plt.subplots(2, 2, figsize=(17, 13), sharex="col", sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    v_top = draw_pair(axes[0, 0], axes[0, 1], models, astrid, lr_edges, lx_edges,
                      floor_value=args.noise_floor)
    v_bot = draw_pair(axes[1, 0], axes[1, 1], cut_models, cut_astrid, lr_edges, lx_cut,
                      annotate=False, triangles=False, floor_value=args.noise_floor)

    axes[1, 1].axvline(np.log10(args.max_r200), color="0.4", lw=1.5, ls="--")   # the cut
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(*nice_ylim(np.concatenate([v_top, v_bot])))
    log_ticks(axes[1, 0], lr_edges, margin=0.15)
    log_ticks(axes[1, 1], lx_edges, margin=0.15)
    for ax in axes.flat:
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[1, 0].set_xlabel(r"$r_{\rm BH}$ (kpc)")
    axes[1, 1].set_xlabel(r"$r_{\rm BH}\,/\,R_{200}$")
    fig.supylabel(r"$N_{\rm WBH}$ (gal$^{-1}$ dex$^{-1}$)", x=0.04, fontsize=FONT_SIZE + 4)
    row_labels = ["All BHs", rf"$r_{{\rm BH}}<{args.max_r200:g}\,R_{{200}}$"]
    for r, lab in enumerate(row_labels):
        axes[r, 1].text(1.03, 0.5, lab, transform=axes[r, 1].transAxes, rotation=270,
                        ha="left", va="center", fontsize=FONT_SIZE + 2)

    add_legends(fig, *legend_handles(models, astrid, args, edges=lx_edges), y=0.895)
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    print(f"\nSaved {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
