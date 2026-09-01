"""
Dynamical friction orbit integration for a point-mass "cluster" sinking
through a static NFW dark matter halo (Chandrasekhar 1943 drag force).

This module is deliberately self-contained (no dependency on the
`timescales` package) so it can be imported by any script that needs a
dynamical friction sink time, given a host halo's M200/R200 and a
cluster's mass, separation, and velocity relative to the host center.

This implements the FIXED-profile model: the host's NFW profile (from a
single M200/R200, e.g. at the subhalo's formation snapshot) is treated as
static for the whole integration. It does not account for the host's own
mass growth/stripping over time, or for the cluster losing mass to tides.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
import astropy.units as u
from astropy.constants import G

# Internal unit system for the ODE integration: kpc, Gyr, Msun.
# (astropy Quantities are used at the public API boundary only -- the
# ODE right-hand-side runs on plain floats in these units for speed.)
_G = G.to(u.kpc**3 / (u.Msun * u.Gyr**2)).value      # kpc^3 / (Msun Gyr^2)
_KMS_TO_KPCGYR = (1 * u.km / u.s).to(u.kpc / u.Gyr).value


class NFWHost:
    """A static NFW halo defined by its M200, R200, and concentration."""

    def __init__(self, M200, R200, concentration=10.0):
        self.M200 = M200.to(u.Msun).value            # Msun
        self.R200 = R200.to(u.kpc).value              # kpc
        self.c = float(concentration)
        self.rs = self.R200 / self.c                  # kpc
        mu_c = np.log(1 + self.c) - self.c / (1 + self.c)
        self.rho_s = self.M200 / (4 * np.pi * self.rs**3 * mu_c)   # Msun/kpc^3

    def mass_enclosed(self, r):
        """r in kpc -> enclosed mass in Msun."""
        x = np.maximum(r, 1e-6) / self.rs
        return 4 * np.pi * self.rho_s * self.rs**3 * (np.log(1 + x) - x / (1 + x))

    def density(self, r):
        """r in kpc -> local density in Msun/kpc^3."""
        x = np.maximum(r, 1e-6) / self.rs
        return self.rho_s / (x * (1 + x)**2)

    def circular_velocity(self, r):
        """r in kpc -> circular velocity in kpc/Gyr."""
        return np.sqrt(_G * self.mass_enclosed(r) / np.maximum(r, 1e-6))

    def velocity_dispersion(self, r):
        """
        Approximate 1D velocity dispersion, sigma(r) ~ v_circ(r)/sqrt(2).
        This is the standard order-of-magnitude approximation for an
        NFW-like halo (exact for a singular isothermal sphere). Swap in a
        Jeans-equation solution if you need better accuracy near the center.
        """
        return self.circular_velocity(r) / np.sqrt(2)

    def potential(self, r):
        """
        r in kpc -> Newtonian potential in (kpc/Gyr)^2, i.e. energy per
        unit mass in this module's internal unit system. Standard closed-form
        NFW potential: Phi(r) = -(4 pi G rho_s rs^3 / r) * ln(1 + r/rs).
        """
        r_safe = np.maximum(r, 1e-6)
        x = r_safe / self.rs
        return -4 * np.pi * _G * self.rho_s * self.rs**3 * np.log(1 + x) / r_safe

    def escape_velocity(self, r):
        """r in kpc -> escape velocity in km/s (v_esc = sqrt(2*|Phi(r)|))."""
        v_esc_kpcgyr = np.sqrt(2 * np.abs(self.potential(r)))
        return v_esc_kpcgyr / _KMS_TO_KPCGYR


def _chandrasekhar_xfactor(v, sigma):
    """
    Returns [erf(X) - (2X/sqrt(pi)) exp(-X^2)] / v^3, where X = v/(sqrt(2) sigma),
    using a small-v series expansion to avoid a 0/0 as v -> 0 (physically,
    the drag force smoothly vanishes there, it's not a true singularity).
    """
    X = v / (np.sqrt(2) * sigma)
    if X < 1e-3:
        return (4 / (3 * np.sqrt(np.pi))) / (2 * np.sqrt(2) * sigma**3)
    xterm = erf(X) - (2 * X / np.sqrt(np.pi)) * np.exp(-X**2)
    return xterm / v**3


def _rhs(t, y, m_cluster_msun, host, coulomb_log, r_soft, bg_host, bg_offset, hubble_rate):
    r_vec = y[:3]
    v_vec = y[3:]
    r_actual = np.linalg.norm(r_vec)
    # SOFTENED radius: used for every force/profile evaluation below, to
    # avoid the 1/r^2 gravitational singularity blowing up (and forcing the
    # adaptive integrator to crawl to a near-halt) whenever a near-radial or
    # highly eccentric orbit swings close to the center. r_soft is tied to
    # r_stop (the "merged" threshold) by the caller -- once the true
    # separation would go below that, we're about to declare the cluster
    # merged anyway, so regularizing the force there doesn't change the
    # physics we actually care about.
    r = max(r_actual, r_soft)
    v = max(np.linalg.norm(v_vec), 1e-8)

    Menc = host.mass_enclosed(r)
    rho = host.density(r)
    sigma = host.velocity_dispersion(r)

    # direction uses the TRUE separation (undefined only exactly at the
    # origin, where the softened magnitude below makes the force ~0 anyway)
    r_hat = r_vec / r_actual if r_actual > 1e-12 else np.zeros(3)
    a_grav = -_G * Menc / r**2 * r_hat

    coeff = 4 * np.pi * _G**2 * m_cluster_msun * rho * coulomb_log
    a_df = -coeff * _chandrasekhar_xfactor(v, sigma) * v_vec

    a_total = a_grav + a_df

    # HUBBLE DRAG: in an expanding universe, a peculiar velocity decays as
    # v ~ 1/a purely from cosmic expansion (no force needed) -- this is the
    # standard -H(t)*v term. It's linear in v, so it survives unchanged for
    # a RELATIVE velocity (cluster - host) too. This matters a lot at high
    # redshift, where H(z) is large: without it, a large peculiar velocity
    # correctly imparted at high-z (from the tree's own recorded kinematics)
    # never decays the way it physically should as the universe expands,
    # and just persists for the rest of cosmic time.
    a_total = a_total - hubble_rate * v_vec

    # Optional BACKGROUND potential: gravity only (no dynamical friction --
    # DF is a local-density effect and should be dominated by whatever the
    # cluster is actually embedded in, not a distant structure). This
    # represents the larger, still-assembling structure the local host
    # itself sits within, whose confining gravity we'd otherwise be
    # missing entirely (see trace_cluster_to_snapshot in imbh.py).
    if bg_host is not None:
        r_bg_vec = r_vec - bg_offset
        r_bg_actual = np.linalg.norm(r_bg_vec)
        r_bg = max(r_bg_actual, r_soft)
        Menc_bg = bg_host.mass_enclosed(r_bg)
        r_bg_hat = r_bg_vec / r_bg_actual if r_bg_actual > 1e-12 else np.zeros(3)
        a_total = a_total - _G * Menc_bg / r_bg**2 * r_bg_hat

    return np.concatenate([v_vec, a_total])



def _make_stop_event(r_stop):
    def hit_center(t, y, *args):
        return np.linalg.norm(y[:3]) - r_stop
    hit_center.terminal = True
    hit_center.direction = -1
    return hit_center


def _make_escape_event(r_escape):
    def escaped(t, y, *args):
        return np.linalg.norm(y[:3]) - r_escape
    escaped.terminal = True
    escaped.direction = 1
    return escaped


def _make_pericenter_event():
    """
    Non-terminal event: fires every time the radial velocity (r . v, which
    shares the sign of dr/dt whenever r>0) crosses from negative
    (infalling) to positive (outfalling) -- i.e. every time r(t) passes
    through a LOCAL MINIMUM. direction=1 selects only this crossing
    direction (not apocenter, the opposite crossing), and terminal=False
    means integration continues -- solve_ivp just records every time it
    happens in sol.t_events, so a single call can register multiple
    pericenter passages if the orbital period is short relative to the
    leg's duration.
    """
    def pericenter(t, y, *args):
        return np.dot(y[:3], y[3:])
    pericenter.terminal = False
    pericenter.direction = 1
    return pericenter


def _solve_ivp_robust(rhs_args, t_span, y0, events):
    """
    Try LSODA first (fast, auto-switches between stiff/non-stiff internally
    -- the right choice for most of this problem's parameter space), but
    fall back to RK45 (a simple, robust, non-Jacobian explicit method) if
    LSODA raises anything, or reports failure via sol.success. This exists
    because some parameter combinations in this problem (likely stemming
    from the interaction between the Hubble-drag term's fast timescale at
    high z and the orbital dynamics) have been observed to make LSODA's
    internal stiff-mode solver print "capi_return is NULL / Call-back
    cb_f_in_lsoda__user__routines failed" diagnostics and then stall for a
    long time rather than failing cleanly -- RK45 has not shown this
    failure mode in any testing so far, at the cost of being slower for
    the (common) well-behaved case, which is why it's the fallback rather
    than the default.
    """
    try:
        sol = solve_ivp(_rhs, t_span, y0, args=rhs_args, events=events,
                         method="LSODA", rtol=1e-8, atol=1e-10)
        if sol.success:
            return sol
    except Exception:
        pass
    return solve_ivp(_rhs, t_span, y0, args=rhs_args, events=events,
                      method="RK45", rtol=1e-8, atol=1e-10)


def integrate_orbit(m_cluster, r0_vec, v0_vec, host, coulomb_log=None,
                     t_max=50 * u.Gyr, r_stop_frac=0.01, escape_frac=3.0,
                     background_host=None, background_offset=None,
                     hubble_rate=0.0 / u.Gyr):
    """
    Like sink_time(), but returns the full final STATE (position and
    velocity relative to the host) at whatever time the integration
    actually stops -- whether that's because it merged, escaped, or simply
    ran out of the allotted t_max -- not just the terminal merge time.

    This is what you need to hand off a partially-evolved orbit to a new
    host after a merger (see trace_cluster_to_snapshot in imbh.py), rather
    than only knowing whether/when it eventually merges within a single
    fixed host.

    Parameters:
        background_host (NFWHost or None): an OPTIONAL second, larger-scale
            potential -- e.g. the eventual final structure's own main-branch
            progenitor at this same snapshot -- contributing GRAVITY ONLY
            (no dynamical friction; DF is a local-density effect and should
            be dominated by whichever structure the cluster is actually
            embedded in). Fixed for the whole integration, same
            approximation level as `host` itself.
        background_offset: astropy Quantity 3-vector (kpc) -- the
            background potential's center, relative to `host`'s own center
            (i.e. in the SAME coordinate frame as r0_vec). Required if
            background_host is given.
        hubble_rate: astropy Quantity (1/Gyr) -- H(z) at this leg's cosmic
            time, applied as a -H(t)*v drag on the RELATIVE velocity (see
            _rhs's docstring comment). Defaults to 0 (no expansion damping)
            for backward compatibility / non-cosmological uses of this
            function; pass the real H(z) for cosmological orbit traces,
            especially important at high redshift where H(z) is large.

    Returns
    -------
    status : str
        One of "merged", "escaped", "ongoing" (ran out of t_max without
        merging or escaping -- NOT "never merges", just "not yet, within
        the t_max given").
    elapsed : astropy Quantity (Gyr)
        Time actually elapsed (== t_max if status == "ongoing").
    r_final : astropy Quantity (kpc), shape (3,)
        Position relative to the host at the end of the integration.
    v_final : astropy Quantity (km/s), shape (3,)
        Velocity relative to the host at the end of the integration.
    n_pericenters : int
        Number of pericenter passages (radial velocity going from
        infalling to outfalling, i.e. r(t) passing through a local
        minimum) detected DURING this call -- relative to `host` only,
        not `background_host` (see _make_pericenter_event). A single call
        can register more than one if the orbital period is short
        relative to t_max. Always 0 for the already-merged/already-escaped
        short-circuit returns below, since no integration happens there.
    """
    m_msun = m_cluster.to(u.Msun).value
    r0 = r0_vec.to(u.kpc).value
    v0 = v0_vec.to(u.km / u.s).value * _KMS_TO_KPCGYR
    hubble_rate_gyr = hubble_rate.to(1 / u.Gyr).value

    if coulomb_log is None:
        coulomb_log = np.log(1 + host.M200 / m_msun)

    r_stop = r_stop_frac * host.R200
    r_escape = escape_frac * host.R200
    t_max_gyr = t_max.to(u.Gyr).value

    bg_offset_kpc = background_offset.to(u.kpc).value if background_host is not None else None

    # If we're ALREADY at/inside the merge radius, OR already beyond the
    # escape radius, at t=0 -- short-circuit here rather than relying on
    # solve_ivp's crossing-detection events -- those events only fire on a
    # crossing DURING the integration, and never fire if the starting
    # position already satisfies the condition (no crossing occurs, since
    # there's no sign change to detect). Without this check, an
    # already-unbound cluster silently runs to t_max and gets reported as
    # "ongoing"/never escaped; and an already-merged cluster (r0 ~ 0,
    # sitting essentially exactly at the host's center) gets integrated
    # from a starting point where the RHS function's r_hat direction is
    # genuinely discontinuous (undefined at the exact origin, well-defined
    # a hair away from it) -- this has been observed to make LSODA's
    # stiff-mode Jacobian estimation choke and hang right at t=0 rather
    # than failing cleanly, since it needs to probe the RHS in several
    # directions from the starting point to build a finite-difference
    # Jacobian.
    r0_mag = np.linalg.norm(r0)
    if r0_mag <= r_stop:
        return "merged", 0 * u.Gyr, r0_vec.to(u.kpc), v0_vec.to(u.km / u.s), 0
    if r0_mag > r_escape:
        return "escaped", 0 * u.Gyr, r0_vec.to(u.kpc), v0_vec.to(u.km / u.s), 0

    y0 = np.concatenate([r0, v0])
    events = [_make_stop_event(r_stop), _make_escape_event(r_escape), _make_pericenter_event()]

    sol = _solve_ivp_robust(
        (m_msun, host, coulomb_log, r_stop, background_host, bg_offset_kpc, hubble_rate_gyr),
        (0, t_max_gyr), y0, events,
    )

    y_final = sol.y[:, -1]
    r_final = y_final[:3] * u.kpc
    v_final = (y_final[3:] / _KMS_TO_KPCGYR) * u.km / u.s
    elapsed = sol.t[-1] * u.Gyr
    n_pericenters = int(sol.t_events[2].size)

    if sol.t_events[0].size > 0:
        return "merged", elapsed, r_final, v_final, n_pericenters
    if sol.t_events[1].size > 0:
        return "escaped", elapsed, r_final, v_final, n_pericenters
    return "ongoing", elapsed, r_final, v_final, n_pericenters


def sink_time(m_cluster, r0_vec, v0_vec, host, coulomb_log=None,
              t_max=50 * u.Gyr, r_stop_frac=0.01, escape_frac=3.0):
    """
    Integrate a cluster's orbit under gravity + Chandrasekhar dynamical
    friction through a static NFWHost, starting at position r0_vec and
    velocity v0_vec (both 3-element astropy Quantities, relative to the
    host center), and return the time to sink to the center.

    Returns
    -------
    t_sink : astropy Quantity (Gyr)
        Time to reach r_stop_frac*R200, or np.inf*u.Gyr if it doesn't
        sink within t_max (or unbinds out past escape_frac*R200).
    status : str
        One of "merged", "not_merged_within_tmax", "escaped".
    """
    m_msun = m_cluster.to(u.Msun).value
    r0 = r0_vec.to(u.kpc).value
    v0 = v0_vec.to(u.km / u.s).value * _KMS_TO_KPCGYR   # -> kpc/Gyr

    if coulomb_log is None:
        coulomb_log = np.log(1 + host.M200 / m_msun)

    r_stop = r_stop_frac * host.R200
    r_escape = escape_frac * host.R200
    t_max_gyr = t_max.to(u.Gyr).value

    # Same short-circuit as in integrate_orbit -- see that function's
    # comment for why this is needed (crossing-based events never fire if
    # already inside r_stop or outside r_escape at t=0).
    r0_mag = np.linalg.norm(r0)
    if r0_mag <= r_stop:
        return 0 * u.Gyr, "merged"
    if r0_mag > r_escape:
        return np.inf * u.Gyr, "escaped"

    y0 = np.concatenate([r0, v0])
    events = [_make_stop_event(r_stop), _make_escape_event(r_escape)]

    sol = _solve_ivp_robust(
        (m_msun, host, coulomb_log, r_stop, None, None, 0.0),
        (0, t_max_gyr), y0, events,
    )

    if sol.t_events[0].size > 0:
        return sol.t_events[0][0] * u.Gyr, "merged"
    if sol.t_events[1].size > 0:
        return np.inf * u.Gyr, "escaped"
    return np.inf * u.Gyr, "not_merged_within_tmax"


def sink_time_from_magnitudes(m_cluster, r0_mag, v0_mag, host, rng=None, **kwargs):
    """
    Convenience wrapper for when you only have the *magnitude* of the
    initial separation and velocity (not their orientation) -- e.g. drawn
    from a distribution of separations/speeds rather than actual 3D
    vectors. Draws an isotropically random direction for r0 and an
    independent isotropically random direction for v0.

    Replace this with sink_time(...) directly once you have true 3D
    separation/velocity vectors for each cluster (e.g. matched from your
    high-resolution simulation rather than drawn from a distribution).
    """
    rng = np.random.default_rng() if rng is None else rng

    def random_direction():
        vec = rng.normal(size=3)
        return vec / np.linalg.norm(vec)

    r0_vec = r0_mag * random_direction()
    v0_vec = v0_mag * random_direction()
    return sink_time(m_cluster, r0_vec, v0_vec, host, **kwargs)