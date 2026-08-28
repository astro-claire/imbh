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


def _rhs(t, y, m_cluster_msun, host, coulomb_log, r_soft):
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

    return np.concatenate([v_vec, a_grav + a_df])


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

    y0 = np.concatenate([r0, v0])
    events = [_make_stop_event(r_stop), _make_escape_event(r_escape)]

    sol = solve_ivp(
        _rhs, (0, t_max_gyr), y0, args=(m_msun, host, coulomb_log, r_stop),
        events=events, method="RK45", rtol=1e-8, atol=1e-10,
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