"""edit operations on a Sounding. every function returns a new Sounding.

smoothing: `sigma` (m) is the gaussian half-width used to spread an edit to
neighboring levels; sigma <= 0 means the edit touches only the target level
(drags) or stops sharply at the layer edges (layer operations).
"""
import numpy as np

from . import thermo


# ---- shared helpers -----------------------------------------------------------
def weights(z, k, sigma):
    if sigma <= 0:
        w = np.zeros_like(z)
        w[k] = 1.0
        return w
    w = np.exp(-0.5 * ((z - z[k]) / sigma) ** 2)
    w[w < 1e-4] = 0.0
    return w


def layer_mask(z, zbot, ztop):
    return (z >= zbot - 1e-6) & (z <= ztop + 1e-6)


def _taper(z, delta, mask, sigma):
    """extend the edge values of a layer delta outward with a gaussian decay."""
    if sigma <= 0 or not mask.any():
        return delta
    idx = np.where(mask)[0]
    ib, it = idx[0], idx[-1]
    out = delta.copy()
    below = np.arange(len(z)) < ib
    above = np.arange(len(z)) > it
    out[below] = delta[ib] * np.exp(-0.5 * ((z[ib] - z[below]) / sigma) ** 2)
    out[above] = delta[it] * np.exp(-0.5 * ((z[above] - z[it]) / sigma) ** 2)
    return out


def _set_temperature(snd, t_target, mask, n_iter=3):
    """set theta so that temperature equals t_target on mask (pressure is re-derived)."""
    new = snd.copy()
    for _ in range(n_iter):
        prof = new.derived()
        new.theta[mask] = t_target[mask] / prof.pi[mask]
    return new


def _wind_mask(n):
    m = np.ones(n, dtype=bool)
    m[0] = False
    return m


def _mass_weights(snd, mask):
    prof = snd.derived()
    rho = prof.p / (thermo.RD * prof.tv)
    dz = np.gradient(snd.z)
    return (rho * dz)[mask]


# ---- drags --------------------------------------------------------------------
def drag_temperature(base, k, t_new, sigma=0.0, hold="qv"):
    """move temperature at level k to t_new (k); hold='qv' keeps mixing ratio, 'rh' keeps rh."""
    prof = base.derived()
    w = weights(base.z, k, sigma)
    mask = w > 0
    t_target = prof.t + w * (t_new - prof.t[k])
    new = _set_temperature(base, t_target, mask)
    if hold == "rh":
        p2 = new.derived()
        qs = thermo.sat_mixing_ratio(t_target, p2.p)
        new.qv[mask] = (prof.rh * qs)[mask] * 1000.0
    return new


def drag_dewpoint(base, k, td_new, sigma=0.0):
    """move dewpoint at level k to td_new (k); dewpoint is capped at the temperature."""
    prof = base.derived()
    w = weights(base.z, k, sigma)
    mask = w > 0
    td = np.minimum(prof.td + w * (td_new - prof.td[k]), prof.t)
    new = base.copy()
    new.qv[mask] = thermo.sat_mixing_ratio(td[mask], prof.p[mask]) * 1000.0
    return new


def drag_wind(base, k, u_new, v_new, sigma=0.0):
    w = weights(base.z, k, sigma) * _wind_mask(len(base.z))
    new = base.copy()
    new.u = base.u + w * (u_new - base.u[k])
    new.v = base.v + w * (v_new - base.v[k])
    new.u[0] = new.v[0] = np.nan
    return new


def set_level(base, k, field, value):
    """set one value directly (data table edits)."""
    new = base.copy()
    if field == "psfc":
        new.psfc = float(value)
    else:
        getattr(new, field)[k] = float(value)
    return new


# ---- thermodynamic layer operations -------------------------------------------
def _apply_thermo_delta(snd, dth, dqv, mask, sigma):
    new = snd.copy()
    new.theta = snd.theta + _taper(snd.z, dth, mask, sigma)
    new.qv = np.maximum(snd.qv + _taper(snd.z, dqv, mask, sigma), 0.0)
    return new


def set_lapse_rate(snd, zbot, ztop, gamma, sigma=0.0):
    """linear temperature profile with lapse rate gamma (k/km) from the layer base."""
    mask = layer_mask(snd.z, zbot, ztop)
    if not mask.any():
        return snd.copy()
    ib = np.where(mask)[0][0]
    prof = snd.derived()
    t_target = prof.t[ib] - gamma * (snd.z - snd.z[ib]) / 1000.0
    new = snd.copy()
    for _ in range(3):
        trial = _set_temperature(new, t_target, mask, n_iter=1)
        dth = np.where(mask, trial.theta - snd.theta, 0.0)
        new = _apply_thermo_delta(snd, dth, np.zeros_like(dth), mask, sigma)
    return new


def offset_theta(snd, zbot, ztop, dtheta, sigma=0.0):
    mask = layer_mask(snd.z, zbot, ztop)
    dth = np.where(mask, dtheta, 0.0)
    return _apply_thermo_delta(snd, dth, np.zeros_like(dth), mask, sigma)


def well_mixed(snd, zbot, ztop, sigma=0.0):
    """constant (mass-weighted mean) theta and qv in the layer."""
    mask = layer_mask(snd.z, zbot, ztop)
    if mask.sum() < 2:
        return snd.copy()
    w = _mass_weights(snd, mask)
    th_m = np.average(snd.theta[mask], weights=w)
    qv_m = np.average(snd.qv[mask], weights=w)
    dth = np.where(mask, th_m - snd.theta, 0.0)
    dqv = np.where(mask, qv_m - snd.qv, 0.0)
    return _apply_thermo_delta(snd, dth, dqv, mask, sigma)


def scale_qv(snd, zbot, ztop, factor, sigma=0.0):
    mask = layer_mask(snd.z, zbot, ztop)
    dqv = np.where(mask, snd.qv * (factor - 1.0), 0.0)
    return _apply_thermo_delta(snd, np.zeros_like(dqv), dqv, mask, sigma)


def set_rh(snd, zbot, ztop, rh_percent, sigma=0.0):
    mask = layer_mask(snd.z, zbot, ztop)
    prof = snd.derived()
    qs = thermo.sat_mixing_ratio(prof.t, prof.p) * 1000.0
    dqv = np.where(mask, rh_percent / 100.0 * qs - snd.qv, 0.0)
    return _apply_thermo_delta(snd, np.zeros_like(dqv), dqv, mask, sigma)


def cap_saturation(snd, zbot, ztop, sigma=0.0):
    mask = layer_mask(snd.z, zbot, ztop)
    prof = snd.derived()
    qs = thermo.sat_mixing_ratio(prof.t, prof.p) * 1000.0
    new = snd.copy()
    new.qv[mask] = np.minimum(snd.qv[mask], qs[mask])
    return new


def _gauss_smooth(z, y, sigma):
    out = np.empty_like(y)
    for i in range(len(z)):
        w = np.exp(-0.5 * ((z - z[i]) / sigma) ** 2)
        out[i] = np.sum(w * y) / np.sum(w)
    return out


def smooth_thermo(snd, zbot, ztop, width):
    """gaussian smoothing (half-width in m) of theta and qv inside the layer."""
    mask = layer_mask(snd.z, zbot, ztop)
    if width <= 0 or mask.sum() < 3:
        return snd.copy()
    new = snd.copy()
    zz = snd.z[mask]
    new.theta[mask] = _gauss_smooth(zz, snd.theta[mask], width)
    new.qv[mask] = _gauss_smooth(zz, snd.qv[mask], width)
    return new


def remove_superadiabatic(snd, zbot, ztop):
    """make theta non-decreasing in the layer with a mass-weighted isotonic fit.

    within each corrected layer the column-integrated theta is preserved.
    """
    mask = layer_mask(snd.z, zbot, ztop)
    if mask.sum() < 2:
        return snd.copy()
    y = snd.theta[mask].astype(float)
    w = _mass_weights(snd, mask).astype(float)
    # pool adjacent violators
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnt.append(c1 + c2)
    fit = np.repeat(vals, cnt)
    new = snd.copy()
    new.theta[mask] = fit
    return new


# ---- wind layer operations ------------------------------------------------------
def _wind_layer(snd, zbot, ztop):
    mask = layer_mask(snd.z, zbot, ztop) & _wind_mask(len(snd.z))
    return mask


def _apply_wind_delta(snd, du, dv, mask, carry):
    new = snd.copy()
    if carry and mask.any():
        it = np.where(mask)[0][-1]
        above = np.arange(len(snd.z)) > it
        du = du.copy()
        dv = dv.copy()
        du[above] = du[it]
        dv[above] = dv[it]
    new.u = snd.u + du
    new.v = snd.v + dv
    new.u[0] = new.v[0] = np.nan
    return new


def rotate_shear(snd, zbot, ztop, degrees, carry=True):
    """rotate the hodograph in the layer about the wind at the layer base (positive = clockwise)."""
    mask = _wind_layer(snd, zbot, ztop)
    if not mask.any():
        return snd.copy()
    ib = np.where(mask)[0][0]
    a = np.radians(-degrees)
    du0 = snd.u - snd.u[ib]
    dv0 = snd.v - snd.v[ib]
    ur = du0 * np.cos(a) - dv0 * np.sin(a)
    vr = du0 * np.sin(a) + dv0 * np.cos(a)
    du = np.where(mask, ur - du0, 0.0)
    dv = np.where(mask, vr - dv0, 0.0)
    return _apply_wind_delta(snd, du, dv, mask, carry)


def scale_shear(snd, zbot, ztop, factor, carry=True):
    """scale the shear in the layer relative to the wind at the layer base."""
    mask = _wind_layer(snd, zbot, ztop)
    if not mask.any():
        return snd.copy()
    ib = np.where(mask)[0][0]
    du = np.where(mask, (snd.u - snd.u[ib]) * (factor - 1.0), 0.0)
    dv = np.where(mask, (snd.v - snd.v[ib]) * (factor - 1.0), 0.0)
    return _apply_wind_delta(snd, du, dv, mask, carry)


def shift_winds(snd, zbot, ztop, du, dv, carry=False):
    mask = _wind_layer(snd, zbot, ztop)
    dua = np.where(mask, du, 0.0)
    dva = np.where(mask, dv, 0.0)
    return _apply_wind_delta(snd, dua, dva, mask, carry)


def smooth_winds(snd, zbot, ztop, width):
    mask = _wind_layer(snd, zbot, ztop)
    if width <= 0 or mask.sum() < 3:
        return snd.copy()
    new = snd.copy()
    zz = snd.z[mask]
    new.u[mask] = _gauss_smooth(zz, snd.u[mask], width)
    new.v[mask] = _gauss_smooth(zz, snd.v[mask], width)
    return new


# ---- registry used by the gui ---------------------------------------------------
# key: (label, function, [(arg label, default, min, max, step, decimals)], uses_taper, uses_carry)
LAYER_OPS = {
    "lapse": ("Set lapse rate", set_lapse_rate, [("Lapse rate (K/km)", 7.0, -20.0, 12.0, 0.1, 2)], True, False),
    "dtheta": ("Add to θ", offset_theta, [("Δθ (K)", 1.0, -30.0, 30.0, 0.1, 2)], True, False),
    "mix": ("Well-mix θ and qv", well_mixed, [], True, False),
    "scaleq": ("Scale qv", scale_qv, [("Factor", 1.1, 0.0, 5.0, 0.01, 3)], True, False),
    "rh": ("Set RH", set_rh, [("RH (%)", 90.0, 0.0, 100.0, 1.0, 1)], True, False),
    "cap": ("Cap at saturation", cap_saturation, [], False, False),
    "superad": ("Remove superadiabatic layers", remove_superadiabatic, [], False, False),
    "smooth_t": ("Smooth θ and qv", smooth_thermo, [("Half-width (m)", 150.0, 0.0, 5000.0, 10.0, 0)], False, False),
    "rotate": ("Rotate shear", rotate_shear, [("Angle (°, + = clockwise)", 10.0, -180.0, 180.0, 1.0, 1)], False, True),
    "scales": ("Scale shear", scale_shear, [("Factor", 1.2, 0.0, 5.0, 0.01, 3)], False, True),
    "shift": ("Shift winds", shift_winds, [("Δu (m/s)", 0.0, -50.0, 50.0, 0.5, 2),
                                            ("Δv (m/s)", 0.0, -50.0, 50.0, 0.5, 2)], False, True),
    "smooth_w": ("Smooth winds", smooth_winds, [("Half-width (m)", 150.0, 0.0, 5000.0, 10.0, 0)], False, False),
}


def apply_layer_op(snd, key, zbot, ztop, args, sigma=0.0, carry=True):
    label, fn, specs, uses_taper, uses_carry = LAYER_OPS[key]
    kwargs = {}
    if uses_taper:
        kwargs["sigma"] = sigma
    if uses_carry:
        kwargs["carry"] = carry
    return fn(snd, zbot, ztop, *args, **kwargs)
