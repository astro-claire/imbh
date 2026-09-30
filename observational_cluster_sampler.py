"""
Observationally-calibrated star-cluster sampler -- an ALTERNATIVE to the
simulation-calibrated cluster_population_sampler.ClusterPopulationSampler.

Instead of bootstrapping clusters from the high-resolution cosmological
simulation's cluster catalog, this draws each host halo's cluster
population from published empirical scaling relations:

  1. HOW MANY clusters  -- the globular cluster number / system mass vs.
     halo mass relation (Burkert & Forbes 2020; Harris, Blakeslee & Harris
     2017). The expected number lambda(M_h) is turned into an integer by a
     Poisson draw, which is what makes the relation usable at all for the
     ~1e7 Msun z>7 halos this pipeline deals with (lambda << 1 there).
  2. HOW MASSIVE        -- either the z=0 GC mass function (log-normal,
     Jordan et al. 2007) or a Schechter-truncated power-law initial cluster
     mass function (dN/dM ~ M^-2 exp(-M/M_c)).
  3. HOW BIG            -- a cluster mass-radius relation (Brown & Gnedin
     2021 for young massive clusters; Marks & Kroupa 2012 for birth radii
     of embedded clusters; or a constant-surface-density relation motivated
     by the z~10 "Cosmic Gems" clusters, Adamo et al. 2024).
  4. WHERE / HOW FAST   -- phase-space initial conditions relative to the
     host. Observations don't constrain these for z>7 proto-halos, so two
     options are offered: 'analytic' (GC-system Sersic profile scaled to
     R200 + isotropic NFW velocity dispersion, bound orbits only), or
     'simulation' (bootstrap r/R200, v/v_circ and the r-v angle from the
     existing ClusterPopulationSampler, so ONLY the number/mass/radius
     assumptions change relative to the default mode).

draw_clusters(subhalo_mass, subhalo_radius) has exactly the same signature
and return format as ClusterPopulationSampler.draw_clusters, so imbh.py can
use either one interchangeably (see imbh.py's --cluster-sampler flag), and
plot_cluster_sampler.py works with it unchanged.

Returned dict (identical to ClusterPopulationSampler):
    'cluster_mass'   : (N,)  Quantity, Msun
    'cluster_radius' : (N,)  Quantity, pc  -- 3D HALF-MASS radius (imbh.py's
                       create_timescale_model converts this to r0 itself)
    'cluster_sep'    : (N,3) Quantity, kpc -- position relative to host center
    'cluster_vel'    : (N,3) Quantity, km/s -- velocity relative to host

CAVEATS worth keeping in mind when interpreting results:
  * The N_GC-M_halo relations are measured at z=0 for SURVIVING GCs in
    halos >~1e10 Msun. Applying them to z>7 progenitors of ~1e7 Msun is an
    extrapolation by ~3 dex in mass. Burkert & Forbes (2020) argue that the
    linearity of the z=0 relation implies the high-z seed halos hosted
    ~1 GC per 5e8 Msun (10x more per unit mass than today, because ~90% of
    z=0 halo mass is smooth accretion that brings no GCs) -- that's the
    'bf20_seed' relation below. 'bf20' applies the z=0 number directly.
  * Surviving != initial. Dissolution/tidal destruction means more (and
    more low-mass) clusters formed than survive; n_boost multiplies the
    expected number to account for this if desired (default 1).
  * TNG's Group_M_Crit200 is used as the halo mass; the observational fits
    use M_vir. At z>7 (Omega_m(z)~1) the two definitions differ by only a
    few percent, well below the relations' own uncertainty.
  * None of these relations has explicit redshift dependence.

Requires: numpy, astropy (and cluster_population_sampler only if
phase_space='simulation').
"""

import argparse
import warnings

import numpy as np
import astropy.units as u
from astropy.constants import G

from dynamical_friction import NFWHost

G_KPC_MSUN_KMS = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value   # for v_circ in km/s
_KPCGYR_TO_KMS = (1 * u.kpc / u.Gyr).to(u.km / u.s).value        # NFWHost returns kpc/Gyr
_trapz = getattr(np, 'trapezoid', None) or np.trapz              # numpy>=2 renamed trapz


# ----------------------------------------------------------------------
# 1. Cluster NUMBER vs halo mass
# ----------------------------------------------------------------------
# kind='per_mass'  : lambda = M_h / halo_mass_per_cluster
# kind='eta_N'     : lambda = eta_N(M_h) * M_h,  log eta_N = a + b log M_h
# kind='eta_M'     : lambda = eta_M * M_h / <m_cluster>  (mass-normalized; the
#                    mean mass comes from whichever mass function is chosen)
# default_scatter_dex: log-normal scatter applied to lambda BEFORE the Poisson
#   draw (mean-preserving). Burkert & Forbes argue the intrinsic scatter is
#   just Poisson (the observed ~constant scatter is M_vir measurement error),
#   so their relations default to 0. Harris et al.'s quoted rms (which
#   includes measurement error) is the default for theirs -- override with
#   n_scatter_dex if you want to turn it off.
N_RELATIONS = {
    'bf20': dict(
        kind='per_mass', halo_mass_per_cluster=5e9, default_scatter_dex=0.0,
        ref="Burkert & Forbes 2020, AJ 159, 56: M_vir = 5e9 Msun x N_GC "
            "(fit over M_vir = 1e10 - 2e15 Msun, z=0 surviving GCs)"),
    'bf20_seed': dict(
        kind='per_mass', halo_mass_per_cluster=5e8, default_scatter_dex=0.0,
        ref="Burkert & Forbes 2020: inferred high-z seed-halo population "
            "hosting ~1 GC per 5e8 Msun of dark matter"),
    'harris17_number': dict(
        kind='eta_N', log_eta_N_a=-8.56, log_eta_N_b=-0.11, default_scatter_dex=0.26,
        ref="Harris, Blakeslee & Harris 2017, ApJ 836, 67: "
            "log eta_N = -8.56 - 0.11 log M_h (rms 0.26 dex)"),
    'harris17_mass': dict(
        kind='eta_M', eta_M=2.9e-5, default_scatter_dex=0.28,
        ref="Harris, Blakeslee & Harris 2017: eta_M = M_GCS/M_h = 2.9e-5 "
            "(rms 0.28 dex); N set by dividing by the mean cluster mass"),
}

# ----------------------------------------------------------------------
# 2. Cluster MASS function
# ----------------------------------------------------------------------
MASS_FUNCTIONS = {
    'gclf': dict(
        kind='lognormal', log10_m_peak=np.log10(2.2e5), sigma_dex=0.5,
        m_min=1e3, m_max=1e8,
        ref="Jordan et al. 2007, ApJS 171, 101: GC mass function turnover "
            "~2.2e5 Msun; width ~0.4-0.55 dex (narrower in dwarf hosts)"),
    'icmf': dict(
        kind='schechter', beta=2.0, m_c=1e6, m_min=1e4, m_max=1e8,
        ref="Initial cluster mass function dN/dM ~ M^-2 exp(-M/M_c) "
            "(e.g. Portegies Zwart et al. 2010; Krumholz et al. 2019). "
            "M_c is poorly constrained at high z -- treat as a free parameter"),
}

# ----------------------------------------------------------------------
# 3. Cluster mass-RADIUS relation
# ----------------------------------------------------------------------
# kind='powerlaw': r = r_ref_pc * (M / m_ref)^slope, log-normal scatter.
# kind='surface_density': R_eff = sqrt(M / (2 pi Sigma_e)) (Sigma_e is the
#   mean surface density within R_eff, which encloses half the mass).
# projected=True means the fit is for the projected half-light radius R_eff;
#   converted to a 3D half-mass radius with r_h = (4/3) R_eff (Spitzer 1987),
#   which is what imbh.py's timescale model expects.
RADIUS_RELATIONS = {
    'brown_gnedin21': dict(
        kind='powerlaw', r_ref_pc=2.548, m_ref=1e4, slope=0.242, scatter_dex=0.25,
        projected=True,
        ref="Brown & Gnedin 2021, MNRAS 508, 5935 (LEGUS, all ages <1 Gyr): "
            "R_eff = 2.548 pc (M/1e4)^0.242, intrinsic scatter 0.25 dex"),
    'brown_gnedin21_young': dict(
        kind='powerlaw', r_ref_pc=2.365, m_ref=1e4, slope=0.180, scatter_dex=0.319,
        projected=True,
        ref="Brown & Gnedin 2021, ages 1-10 Myr: R_eff = 2.365 pc (M/1e4)^0.180, "
            "scatter 0.32 dex"),
    'marks_kroupa12': dict(
        kind='powerlaw', r_ref_pc=0.10, m_ref=1.0, slope=0.13, scatter_dex=0.2,
        projected=False,
        ref="Marks & Kroupa 2012, A&A 543, A8: birth (embedded) 3D half-mass "
            "radius r_h = 0.10(+0.07/-0.04) pc (M/Msun)^0.13; ~0.2 dex scatter "
            "adopted from the normalization uncertainty"),
    'const_surface_density': dict(
        kind='surface_density', sigma_e_msun_pc2=1e5, scatter_dex=0.3,
        projected=True,
        ref="Constant Sigma_e = 1e5 Msun/pc^2, motivated by the z~10.2 Cosmic "
            "Gems clusters (Adamo et al. 2024, Nature 632, 513: M~1e6 Msun, "
            "R_eff~1 pc). NOT a fitted relation -- scatter is a placeholder"),
}

# ----------------------------------------------------------------------
# 4. Spatial distribution of the cluster system
# ----------------------------------------------------------------------
# GC systems follow a de Vaucouleurs (Sersic n=4) projected profile
# (Hudson & Robison 2018, MNRAS 477, 3869), with effective radii of
# roughly 1-5% of R200 for red GC systems and up to ~10% for blue GCs in
# massive hosts (Lim et al. 2024, NGVS XXVII, ApJ 966, 168). Defaults:
# n=4, R_e,GCS = 0.03 R200. Deprojected with the Prugniel & Simien (1997)
# approximation, which can be sampled EXACTLY via a Gamma variate (see
# _sample_sersic_3d_radius).
DEFAULT_SERSIC_N = 4.0
DEFAULT_RE_OVER_R200 = 0.03
DEFAULT_RMAX_OVER_R200 = 1.0


def _random_isotropic_unit_vectors(n, rng):
    vec = rng.normal(size=(n, 3))
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def _random_perpendicular_unit_vectors(r_hat, rng):
    """For each unit vector in r_hat (n,3), a random unit vector perpendicular to it."""
    rand = rng.normal(size=r_hat.shape)
    perp = rand - np.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
    norm = np.linalg.norm(perp, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-8
    while np.any(bad):
        rand2 = rng.normal(size=(int(bad.sum()), 3))
        perp[bad] = rand2 - np.sum(rand2 * r_hat[bad], axis=1, keepdims=True) * r_hat[bad]
        norm[bad] = np.linalg.norm(perp[bad], axis=1, keepdims=True)
        bad = norm[:, 0] < 1e-8
    return perp / norm


def _sample_sersic_3d_radius(n_draw, r_e, sersic_n, r_max, rng, max_iter=100):
    """
    3D radii for tracers whose PROJECTED profile is Sersic(n) with projected
    half-number radius r_e, via the Prugniel-Simien deprojection
        rho(s) ~ s^-p exp(-b s^(1/n)),  s = r/r_e,
        p = 1 - 0.6097/n + 0.05463/n^2   (Lima Neto et al. 1999)
        b = 2n - 1/3 + 0.009876/n        (Prugniel & Simien 1997)
    Substituting x = b s^(1/n) turns the radial number distribution
    r^2 rho(r) dr into x^(n(3-p)-1) e^(-x) dx -- a Gamma(n(3-p)) variate --
    so r = r_e (x/b)^n is an exact draw. Draws beyond r_max are redrawn.
    """
    p = 1.0 - 0.6097 / sersic_n + 0.05463 / sersic_n**2
    b = 2.0 * sersic_n - 1.0 / 3.0 + 0.009876 / sersic_n
    shape = sersic_n * (3.0 - p)

    def draw(k):
        return r_e * (rng.gamma(shape, size=k) / b) ** sersic_n

    r = draw(n_draw)
    for _ in range(max_iter):
        out = r > r_max
        if not np.any(out):
            break
        r[out] = draw(int(out.sum()))
    return np.minimum(r, r_max)


class ObservationalClusterSampler:
    def __init__(self, n_relation='bf20', mass_function='gclf',
                 radius_relation='brown_gnedin21', phase_space='analytic',
                 n_boost=1.0, n_scatter_dex=None,
                 mass_function_kwargs=None, radius_relation_kwargs=None,
                 sersic_n=DEFAULT_SERSIC_N, re_over_r200=DEFAULT_RE_OVER_R200,
                 rmax_over_r200=DEFAULT_RMAX_OVER_R200,
                 host_concentration=4.0, phase_space_sampler=None, rng=None):
        """
        Parameters:
            n_relation (str): key of N_RELATIONS -- sets the expected number
                of clusters per halo.
            mass_function (str): key of MASS_FUNCTIONS.
            radius_relation (str): key of RADIUS_RELATIONS.
            phase_space (str): 'analytic' or 'simulation' (see module docstring).
            n_boost (float): multiplies the expected number of clusters per
                halo, e.g. to convert a SURVIVING-GC relation into an initial
                population (default 1 = relation as published).
            n_scatter_dex (float or None): log-normal scatter on the expected
                number before the Poisson draw; None -> the relation's
                default_scatter_dex.
            mass_function_kwargs / radius_relation_kwargs (dict or None):
                override individual parameters of the chosen mass function /
                radius relation, e.g. {'m_c': 3e6} or {'sigma_e_msun_pc2': 3e4}.
            sersic_n, re_over_r200, rmax_over_r200: cluster-system spatial
                profile (phase_space='analytic' only).
            host_concentration (float): NFW concentration for the velocity
                draw -- pass the SAME value the orbit integration uses
                (imbh.HOST_CONCENTRATION).
            phase_space_sampler (ClusterPopulationSampler or None): required
                if phase_space='simulation'.
            rng (np.random.Generator or None).
        """
        for name, value, table in (('n_relation', n_relation, N_RELATIONS),
                                   ('mass_function', mass_function, MASS_FUNCTIONS),
                                   ('radius_relation', radius_relation, RADIUS_RELATIONS)):
            if value not in table:
                raise ValueError(f"{name} must be one of {sorted(table)}, got {value!r}")
        if phase_space not in ('analytic', 'simulation'):
            raise ValueError(f"phase_space must be 'analytic' or 'simulation', got {phase_space!r}")
        if phase_space == 'simulation' and phase_space_sampler is None:
            raise ValueError("phase_space='simulation' needs phase_space_sampler= "
                             "(a loaded ClusterPopulationSampler)")

        self.n_relation = n_relation
        self.n_params = dict(N_RELATIONS[n_relation])
        self.mass_function = mass_function
        self.mf_params = {**MASS_FUNCTIONS[mass_function], **(mass_function_kwargs or {})}
        self.radius_relation = radius_relation
        self.rr_params = {**RADIUS_RELATIONS[radius_relation], **(radius_relation_kwargs or {})}
        self.phase_space = phase_space
        self.n_boost = float(n_boost)
        self.n_scatter_dex = (self.n_params['default_scatter_dex'] if n_scatter_dex is None
                              else float(n_scatter_dex))
        self.sersic_n = float(sersic_n)
        self.re_over_r200 = float(re_over_r200)
        self.rmax_over_r200 = float(rmax_over_r200)
        self.host_concentration = float(host_concentration)
        self.phase_space_sampler = phase_space_sampler
        self.rng = rng if rng is not None else np.random.default_rng()

        self._build_mass_function_table()

    # ------------------------------------------------------------------
    # Mass function: tabulated inverse CDF on a fine log-mass grid (works
    # for any truncated 1D form, and gives the exact mean mass needed by
    # the eta_M number normalization)
    # ------------------------------------------------------------------
    def _build_mass_function_table(self, n_grid=4001):
        p = self.mf_params
        logm = np.linspace(np.log10(p['m_min']), np.log10(p['m_max']), n_grid)
        m = 10**logm
        if p['kind'] == 'lognormal':
            # dN/dlogM Gaussian in log M
            pdf_logm = np.exp(-0.5 * ((logm - p['log10_m_peak']) / p['sigma_dex'])**2)
        elif p['kind'] == 'schechter':
            # dN/dlogM = M dN/dM ~ M^(1-beta) exp(-M/M_c)
            pdf_logm = m**(1.0 - p['beta']) * np.exp(-m / p['m_c'])
        else:
            raise ValueError(f"unknown mass function kind {p['kind']!r}")
        # trapezoid-integrated CDF in log M
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf_logm[1:] + pdf_logm[:-1]) * np.diff(logm))])
        norm = cdf[-1]
        self._mf_logm = logm
        self._mf_cdf = cdf / norm
        self.mean_cluster_mass = _trapz(pdf_logm * m, logm) / norm

    def _sample_masses(self, n):
        return 10**np.interp(self.rng.random(n), self._mf_cdf, self._mf_logm)

    # ------------------------------------------------------------------
    # Mass-radius relation -> 3D half-mass radius in pc
    # ------------------------------------------------------------------
    def half_mass_radius_pc(self, mass_msun, scatter=True):
        p = self.rr_params
        mass_msun = np.asarray(mass_msun, dtype=float)
        if p['kind'] == 'powerlaw':
            r = p['r_ref_pc'] * (mass_msun / p['m_ref'])**p['slope']
        elif p['kind'] == 'surface_density':
            r = np.sqrt(mass_msun / (2.0 * np.pi * p['sigma_e_msun_pc2']))
        else:
            raise ValueError(f"unknown radius relation kind {p['kind']!r}")
        if scatter and p['scatter_dex'] > 0:
            r = r * 10**(p['scatter_dex'] * self.rng.normal(size=r.shape))
        if p['projected']:
            r = r * 4.0 / 3.0
        return r

    # ------------------------------------------------------------------
    # Number of clusters
    # ------------------------------------------------------------------
    def expected_number(self, halo_mass_msun):
        """Mean number of clusters (before scatter/Poisson) for halo mass(es) in Msun."""
        m = np.asarray(halo_mass_msun, dtype=float)
        p = self.n_params
        with np.errstate(divide='ignore', invalid='ignore'):
            if p['kind'] == 'per_mass':
                lam = m / p['halo_mass_per_cluster']
            elif p['kind'] == 'eta_N':
                lam = 10**(p['log_eta_N_a'] + p['log_eta_N_b'] * np.log10(m)) * m
            elif p['kind'] == 'eta_M':
                lam = p['eta_M'] * m / self.mean_cluster_mass
            else:
                raise ValueError(f"unknown n_relation kind {p['kind']!r}")
        lam = np.where(m > 0, lam, 0.0)
        return self.n_boost * lam

    def _draw_number(self, halo_mass_msun):
        lam = float(self.expected_number(halo_mass_msun))
        if lam <= 0:
            return 0
        if self.n_scatter_dex > 0:
            # mean-preserving log-normal scatter on lambda
            s_ln = self.n_scatter_dex * np.log(10.0)
            lam *= np.exp(s_ln * self.rng.normal() - 0.5 * s_ln**2)
        return int(self.rng.poisson(lam))

    # ------------------------------------------------------------------
    # Phase space
    # ------------------------------------------------------------------
    def _phase_space_analytic(self, n, halo_mass, halo_radius):
        host = NFWHost(halo_mass * u.Msun, halo_radius * u.kpc, concentration=self.host_concentration)
        r_mag = _sample_sersic_3d_radius(n, self.re_over_r200 * halo_radius, self.sersic_n,
                                         self.rmax_over_r200 * halo_radius, self.rng)
        sigma = host.velocity_dispersion(r_mag) * _KPCGYR_TO_KMS      # km/s, 1D
        v_esc = host.escape_velocity(r_mag)                          # km/s
        v = sigma[:, None] * self.rng.normal(size=(n, 3))
        # keep only bound initial velocities: redraw anything at/above v_esc
        for _ in range(100):
            unbound = np.linalg.norm(v, axis=1) >= v_esc
            if not np.any(unbound):
                break
            v[unbound] = sigma[unbound, None] * self.rng.normal(size=(int(unbound.sum()), 3))
        else:
            unbound = np.linalg.norm(v, axis=1) >= v_esc
            v[unbound] *= (0.99 * v_esc[unbound] / np.linalg.norm(v[unbound], axis=1))[:, None]
        r_hat = _random_isotropic_unit_vectors(n, self.rng)
        return r_mag[:, None] * r_hat, v

    def _phase_space_from_simulation(self, n, halo_mass, halo_radius):
        """
        Bootstrap (r/R200, v/v_circ, cos angle) jointly from the simulation-
        calibrated sampler's cluster pool, kernel-weighted toward the query
        halo mass exactly as ClusterPopulationSampler.draw_clusters does.
        """
        s = self.phase_space_sampler
        c_i0, c_i1, w = s._windowed_weights(s.cl_log_host_mass, np.log10(halo_mass))
        if w.sum() == 0:
            nearest = int(np.clip(np.searchsorted(s.cl_log_host_mass, np.log10(halo_mass)),
                                  0, len(s.cl_log_host_mass) - 1))
            c_i0, c_i1 = max(nearest - 1, 0), min(nearest + 1, len(s.cl_log_host_mass))
            w = np.ones(c_i1 - c_i0)
        idx = c_i0 + self.rng.choice(c_i1 - c_i0, size=n, replace=True, p=w / w.sum())
        v_circ = np.sqrt(G_KPC_MSUN_KMS * halo_mass / halo_radius)
        sep_mag = s.cl_dist_over_rvir[idx] * halo_radius
        vel_mag = s.cl_vel_over_vcirc[idx] * v_circ
        cos_theta = s.cl_cos_theta[idx]

        r_hat = _random_isotropic_unit_vectors(n, self.rng)
        v_hat = np.empty((n, 3))
        has_geom = ~np.isnan(cos_theta)
        if np.any(has_geom):
            ct = cos_theta[has_geom][:, None]
            st = np.sqrt(np.clip(1.0 - ct**2, 0.0, 1.0))
            v_hat[has_geom] = ct * r_hat[has_geom] + st * _random_perpendicular_unit_vectors(r_hat[has_geom], self.rng)
        if np.any(~has_geom):
            v_hat[~has_geom] = _random_isotropic_unit_vectors(int((~has_geom).sum()), self.rng)
        return sep_mag[:, None] * r_hat, vel_mag[:, None] * v_hat

    # ------------------------------------------------------------------
    # Public API (same as ClusterPopulationSampler.draw_clusters)
    # ------------------------------------------------------------------
    def draw_clusters(self, subhalo_mass, subhalo_radius):
        """
        subhalo_mass (Msun) and subhalo_radius (kpc) as plain floats --
        imbh.py passes group_m_crit200_msun / group_r_crit200_kpc. Returns
        the same dict of Quantities as ClusterPopulationSampler.draw_clusters.
        Halos with a degenerate (<=0) mass or radius get no clusters.
        """
        subhalo_mass = float(subhalo_mass)
        subhalo_radius = float(subhalo_radius)
        n = self._draw_number(subhalo_mass) if (subhalo_mass > 0 and subhalo_radius > 0) else 0
        if n == 0:
            return {
                'cluster_mass': np.array([]) * u.Msun,
                'cluster_radius': np.array([]) * u.pc,
                'cluster_sep': np.zeros((0, 3)) * u.kpc,
                'cluster_vel': np.zeros((0, 3)) * u.km / u.s,
            }

        mass = self._sample_masses(n)
        radius_pc = self.half_mass_radius_pc(mass)
        if self.phase_space == 'analytic':
            sep, vel = self._phase_space_analytic(n, subhalo_mass, subhalo_radius)
        else:
            sep, vel = self._phase_space_from_simulation(n, subhalo_mass, subhalo_radius)

        return {
            'cluster_mass': mass * u.Msun,
            'cluster_radius': radius_pc * u.pc,
            'cluster_sep': sep * u.kpc,
            'cluster_vel': vel * (u.km / u.s),
        }

    # ------------------------------------------------------------------
    def describe(self):
        """Multi-line summary of every relation/parameter in use (printed by imbh.py)."""
        lines = [
            "ObservationalClusterSampler configuration:",
            f"  number : {self.n_relation}  (n_boost={self.n_boost:g}, "
            f"scatter={self.n_scatter_dex:g} dex, then Poisson)",
            f"           {self.n_params['ref']}",
            f"  mass   : {self.mass_function}  "
            + ", ".join(f"{k}={v:.4g}" for k, v in self.mf_params.items()
                        if k not in ('kind', 'ref')),
            f"           mean cluster mass = {self.mean_cluster_mass:.3e} Msun",
            f"           {self.mf_params['ref']}",
            f"  radius : {self.radius_relation}  "
            + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in self.rr_params.items() if k not in ('kind', 'ref')),
            f"           {self.rr_params['ref']}",
            f"  phase space : {self.phase_space}"
            + (f" (Sersic n={self.sersic_n:g}, R_e={self.re_over_r200:g} R200, "
               f"r_max={self.rmax_over_r200:g} R200, NFW c={self.host_concentration:g}, "
               f"isotropic bound velocities)" if self.phase_space == 'analytic'
               else " (bootstrapped from the simulation-calibrated sampler)"),
        ]
        return "\n".join(lines)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('rng', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = np.random.default_rng()


# ----------------------------------------------------------------------
# Command-line diagnostic: expected cluster counts for a merger-tree table
# ----------------------------------------------------------------------
def main():
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Expected number of clusters (and a sample draw) under each "
                    "observational N-M_halo relation, for the same halo selection "
                    "imbh.py uses (delta_t>0, formation_z>=cutoff, M_crit200<1e9).")
    parser.add_argument("csv_path", help="subhalo_formation_<ID>.csv")
    parser.add_argument("--cutoff-z", type=float, default=7.0)
    parser.add_argument("--mass-function", choices=sorted(MASS_FUNCTIONS), default='gclf')
    parser.add_argument("--radius-relation", choices=sorted(RADIUS_RELATIONS), default='brown_gnedin21')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    sel = df[(df['delta_t_gyr'] > 0) & (df['formation_redshift'] >= args.cutoff_z)
             & (df['group_m_crit200_msun'] < 1e9)]
    m_h = sel['group_m_crit200_msun'].to_numpy()
    r_h = sel['group_r_crit200_kpc'].to_numpy()
    print(f"{len(sel)} halos selected; median M200 = {np.median(m_h):.3e} Msun, "
          f"total M200 = {m_h.sum():.3e} Msun")

    for rel in N_RELATIONS:
        s = ObservationalClusterSampler(n_relation=rel, mass_function=args.mass_function,
                                        radius_relation=args.radius_relation,
                                        rng=np.random.default_rng(args.seed))
        lam = s.expected_number(m_h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            draws = [s.draw_clusters(m, r) for m, r in zip(m_h, r_h)]
        n_drawn = np.array([len(d['cluster_mass']) for d in draws])
        print(f"  {rel:16s} expected N = {lam.sum():9.1f}   one realization: N = {n_drawn.sum():6d} "
              f"in {np.count_nonzero(n_drawn):5d} halos")


if __name__ == "__main__":
    main()
