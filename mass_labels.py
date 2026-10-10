"""
Legend labels for halo-mass bins given on the command line (--bin LABEL IDs ...).

format_mass_label turns plain-text labels such as ">1e11", "gt1e11",
"1e10-1e11" or "<1e9" into matplotlib mathtext with powers of ten and solar
masses -- ">10^11 M_sun", "10^10-10^11 M_sun" -- in the same style as the
DEFAULT_MASS_BINS labels. Shared by plot_tde_rates_smoothed.py,
plot_tde_radial_smoothed.py and plot_tde_rates_quad.py.
"""
import re

_NUM = r"(\d+(?:\.\d+)?)[eE]\+?(-?\d+)"


def _pow10(m):
    """'1e11' -> 10^{11}, '3e10' -> 3\\times10^{10} (mathtext)."""
    mant, exp = m.group(1), int(m.group(2))
    return f"10^{{{exp}}}" if float(mant) == 1 else f"{mant}\\times10^{{{exp}}}"


def format_mass_label(label):
    """
    Turn a --bin label like ">1e11", "gt1e11", "1e10-1e11" or "<1e9" into
    mathtext with powers of ten and solar-mass units, e.g.
    "$>10^{11}\\,M_\\odot$" or "$10^{10}$–$10^{11}\\,M_\\odot$" (same style
    as DEFAULT_MASS_BINS). Labels that already contain '$', or that hold no
    number in 1e11 form, are returned unchanged.
    """
    if "$" in label or not re.search(_NUM, label):
        return label
    s = label.strip()
    s = re.sub(r"^(gt|>)\s*", ">", s, flags=re.I)
    s = re.sub(r"^(lt|<)\s*", "<", s, flags=re.I)
    rng = re.fullmatch(_NUM + r"\s*[-–]\s*" + _NUM, s)
    if rng:
        lo = _pow10(re.match(_NUM, rng.group(0)))
        hi = _pow10(re.search(_NUM + r"$", rng.group(0)))
        return f"${lo}$–${hi}\\,M_\\odot$"
    return "$" + re.sub(_NUM, _pow10, s) + r"\,M_\odot$"
