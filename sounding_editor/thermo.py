"""thermodynamic helpers (numpy, si units: k, pa, kg/kg unless noted)."""
import numpy as np

# constants chosen to match cm1
RD = 287.04
RV = 461.5
CP = 1005.7
G = 9.81
P00 = 100000.0
EPS = RD / RV
KAPPA = RD / CP


def exner(p):
    return (np.asarray(p, dtype=float) / P00) ** KAPPA


def pressure_from_exner(pi):
    return P00 * np.asarray(pi, dtype=float) ** (1.0 / KAPPA)


def virtual_temp(t, qv):
    # works for temperature or potential temperature
    return t * (1.0 + qv / EPS) / (1.0 + qv)


def sat_vapor_pres(t):
    # bolton (1980), pa
    tc = np.asarray(t, dtype=float) - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def sat_mixing_ratio(t, p):
    es = np.minimum(sat_vapor_pres(t), 0.9 * np.asarray(p, dtype=float))
    return EPS * es / (p - es)


def vapor_pres(qv, p):
    return p * qv / (EPS + qv)


def dewpoint(qv, p):
    # inverse of the bolton saturation vapor pressure
    e = np.maximum(vapor_pres(np.maximum(qv, 0.0), p), 1e-6)
    ln = np.log(e / 611.2)
    return 243.5 * ln / (17.67 - ln) + 273.15


def rel_humidity(t, qv, p):
    return vapor_pres(qv, p) / sat_vapor_pres(t)


def theta_e(t, p, qv, t_lcl):
    # bolton (1980) eq. 43 pseudo-equivalent potential temperature
    r = qv * 1000.0
    return (t * (P00 / p) ** (0.2854 * (1.0 - 0.28e-3 * r))
            * np.exp((3.376 / t_lcl - 0.00254) * r * (1.0 + 0.81e-3 * r)))


def lcl_temperature(t, td):
    # bolton (1980) eq. 15
    return 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0


def theta_e_env(t, p, qv):
    td = np.minimum(dewpoint(qv, p), t)
    return theta_e(t, p, qv, lcl_temperature(t, td))


def sat_theta_e(t, p):
    # theta_e of a saturated parcel at (t, p)
    return theta_e(t, p, sat_mixing_ratio(t, p), t)


def temp_on_pseudoadiabat(the, p, n_iter=45):
    """temperature where the saturated theta_e equals `the` at each pressure (bisection)."""
    p = np.asarray(p, dtype=float)
    lo = np.full(p.shape, 120.0)
    hi = np.full(p.shape, 400.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_warm = sat_theta_e(mid, p) > the
        hi = np.where(too_warm, mid, hi)
        lo = np.where(too_warm, lo, mid)
    return 0.5 * (lo + hi)
