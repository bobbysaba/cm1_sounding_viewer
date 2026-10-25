"""severe-weather parameters for a cm1 sounding.

conventions (spc / sharppy style):
- cape/cin use the virtual temperature correction and a pseudoadiabatic parcel
- ml parcel: pressure-weighted mean theta and qv of the lowest 100 hpa, lifted from the surface
- mu parcel: highest theta_e in the lowest 300 hpa
- effective inflow layer: thompson et al. (2007), cape >= 100 j/kg and cin >= -250 j/kg
- buoyancy is integrated in height on the sounding's own z levels
"""
from dataclasses import dataclass, field

import numpy as np

from . import thermo
from .thermo import G


@dataclass
class Settings:
    surface_wind: str = "lowest"      # "lowest", "extrapolate", "zero"
    storm_motion: str = "rm"          # "rm", "lm", "mean", "custom"
    custom_u: float = 0.0
    custom_v: float = 0.0
    virtual: bool = True              # virtual temperature correction in cape/cin


@dataclass
class Parcel:
    name: str
    k: int                 # start level index
    z: np.ndarray          # heights from the start level up
    p: np.ndarray
    t: np.ndarray          # parcel temperature (k)
    b: np.ndarray          # buoyancy (m/s2)
    p_lcl: float
    z_lcl: float
    cape: float = 0.0
    cin: float = 0.0
    lfc: float = np.nan
    el: float = np.nan
    cape3: float = 0.0


@dataclass
class Results:
    values: dict = field(default_factory=dict)   # key -> float
    parcels: dict = field(default_factory=dict)  # name -> Parcel
    u: np.ndarray = None                          # winds with the surface filled in
    v: np.ndarray = None
    bunkers_rm: tuple = (np.nan, np.nan)
    bunkers_lm: tuple = (np.nan, np.nan)
    mean_wind: tuple = (np.nan, np.nan)
    storm: tuple = (np.nan, np.nan)
    eff_layer: tuple = (np.nan, np.nan)


# (group, key, label, unit, format)
TABLE = [
    ("Thermodynamics", "sbcape", "SBCAPE", "J/kg", "{:.0f}"),
    ("Thermodynamics", "sbcin", "SBCIN", "J/kg", "{:.0f}"),
    ("Thermodynamics", "sblcl", "SB LCL", "m", "{:.0f}"),
    ("Thermodynamics", "sblfc", "SB LFC", "m", "{:.0f}"),
    ("Thermodynamics", "sbel", "SB EL", "m", "{:.0f}"),
    ("Thermodynamics", "mlcape", "MLCAPE", "J/kg", "{:.0f}"),
    ("Thermodynamics", "mlcin", "MLCIN", "J/kg", "{:.0f}"),
    ("Thermodynamics", "ml3cape", "ML 0–3 km CAPE", "J/kg", "{:.0f}"),
    ("Thermodynamics", "mllcl", "ML LCL", "m", "{:.0f}"),
    ("Thermodynamics", "mllfc", "ML LFC", "m", "{:.0f}"),
    ("Thermodynamics", "mlel", "ML EL", "m", "{:.0f}"),
    ("Thermodynamics", "mucape", "MUCAPE", "J/kg", "{:.0f}"),
    ("Thermodynamics", "mucin", "MUCIN", "J/kg", "{:.0f}"),
    ("Thermodynamics", "muz", "MU parcel height", "m", "{:.0f}"),
    ("Thermodynamics", "lr03", "0–3 km lapse rate", "K/km", "{:.2f}"),
    ("Thermodynamics", "lr75", "700–500 hPa lapse rate", "K/km", "{:.2f}"),
    ("Thermodynamics", "pw", "Precipitable water", "mm", "{:.1f}"),
    ("Thermodynamics", "sfc_t", "Surface T", "°C", "{:.1f}"),
    ("Thermodynamics", "sfc_td", "Surface Td", "°C", "{:.1f}"),
    ("Kinematics", "shr05", "0–500 m bulk shear", "m/s", "{:.1f}"),
    ("Kinematics", "shr1", "0–1 km bulk shear", "m/s", "{:.1f}"),
    ("Kinematics", "shr3", "0–3 km bulk shear", "m/s", "{:.1f}"),
    ("Kinematics", "shr6", "0–6 km bulk shear", "m/s", "{:.1f}"),
    ("Kinematics", "storm_u", "Storm motion u", "m/s", "{:.1f}"),
    ("Kinematics", "storm_v", "Storm motion v", "m/s", "{:.1f}"),
    ("Kinematics", "srh05", "0–500 m SRH", "m²/s²", "{:.0f}"),
    ("Kinematics", "srh1", "0–1 km SRH", "m²/s²", "{:.0f}"),
    ("Kinematics", "srh3", "0–3 km SRH", "m²/s²", "{:.0f}"),
    ("Kinematics", "eff_base", "Eff. inflow base", "m", "{:.0f}"),
    ("Kinematics", "eff_top", "Eff. inflow top", "m", "{:.0f}"),
    ("Kinematics", "esrh", "Effective SRH", "m²/s²", "{:.0f}"),
    ("Kinematics", "ebwd", "Effective bulk shear", "m/s", "{:.1f}"),
    ("Kinematics", "crit_angle", "Critical angle", "°", "{:.0f}"),
    ("Kinematics", "swv05", "0–500 m streamwise vort.", "1/s", "{:.4f}"),
    ("Kinematics", "swfrac05", "0–500 m streamwise frac.", "%", "{:.0f}"),
    ("Composites", "stp_eff", "STP (effective)", "", "{:.2f}"),
    ("Composites", "stp_fix", "STP (fixed)", "", "{:.2f}"),
    ("Composites", "scp", "SCP", "", "{:.2f}"),
    ("Composites", "sherbs3", "SHERBS3", "", "{:.2f}"),
    ("Composites", "sherbe", "SHERBE", "", "{:.2f}"),
]


# ---- parcels ------------------------------------------------------------------
def _lcl(t0, q0, p0):
    """pressure and temperature where a dry-adiabatically lifted parcel saturates."""
    th = t0 / thermo.exner(p0)
    if q0 >= thermo.sat_mixing_ratio(t0, p0):
        return p0, t0
    lo, hi = 0.2 * p0, p0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if thermo.sat_mixing_ratio(th * thermo.exner(mid), mid) > q0:
            hi = mid
        else:
            lo = mid
    p_lcl = 0.5 * (lo + hi)
    return p_lcl, th * thermo.exner(p_lcl)


def _insert_zero_crossings(z, b):
    s = np.sign(b)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return z.copy(), b.copy()
    zc = z[idx] + (z[idx + 1] - z[idx]) * b[idx] / (b[idx] - b[idx + 1])
    return np.insert(z, idx + 1, zc), np.insert(b, idx + 1, 0.0)


def _area(zf, bf, lo, hi, positive):
    if not hi > lo:
        return 0.0
    inside = (zf > lo) & (zf < hi)
    zz = np.concatenate([[lo], zf[inside], [hi]])
    bb = np.interp(zz, zf, bf)
    bb = np.maximum(bb, 0.0) if positive else np.minimum(bb, 0.0)
    return float(np.sum(0.5 * (bb[1:] + bb[:-1]) * np.diff(zz)))


def lift_parcel(prof, k, t0, q0, name="", virtual=True):
    """lift a parcel from level k with temperature t0 (k) and mixing ratio q0 (kg/kg)."""
    p0 = prof.p[k]
    z = prof.z[k:]
    p = prof.p[k:]
    p_lcl, t_lcl = _lcl(t0, q0, p0)
    th = t0 / thermo.exner(p0)
    the = thermo.theta_e(t_lcl, p_lcl, q0, t_lcl)
    dry = p >= p_lcl
    tp = np.empty_like(p)
    qp = np.empty_like(p)
    tp[dry] = th * thermo.exner(p[dry])
    qp[dry] = q0
    if np.any(~dry):
        tp[~dry] = thermo.temp_on_pseudoadiabat(the, p[~dry])
        qp[~dry] = thermo.sat_mixing_ratio(tp[~dry], p[~dry])
    if virtual:
        tenv = prof.tv[k:]
        tpar = thermo.virtual_temp(tp, qp)
    else:
        tenv = prof.t[k:]
        tpar = tp
    b = G * (tpar - tenv) / tenv
    # lcl height from the environment's z(ln p)
    z_lcl = float(np.interp(-np.log(p_lcl), -np.log(prof.p), prof.z))
    par = Parcel(name=name, k=k, z=z, p=p, t=tp, b=b, p_lcl=p_lcl, z_lcl=z_lcl)
    _buoyancy_stats(par)
    return par


def _buoyancy_stats(par):
    zf, bf = _insert_zero_crossings(par.z, par.b)
    # contiguous positive regions
    seg_pos = (bf[:-1] + bf[1:]) > 0
    regions = []
    start = None
    for i, pos in enumerate(seg_pos):
        if pos and start is None:
            start = i
        if not pos and start is not None:
            regions.append((zf[start], zf[i]))
            start = None
    if start is not None:
        regions.append((zf[start], zf[-1]))
    regions = [r for r in regions if r[1] > par.z_lcl]
    if not regions:
        return
    lfc = max(regions[0][0], par.z_lcl)
    el = regions[-1][1]
    par.lfc = float(lfc)
    par.el = float(el)
    par.cape = _area(zf, bf, lfc, el, True)
    par.cin = _area(zf, bf, zf[0], lfc, False)
    par.cape3 = _area(zf, bf, lfc, min(el, 3000.0), True)


def ml_parcel(prof, depth=10000.0, virtual=True):
    psfc = prof.p[0]
    pp = np.linspace(psfc, psfc - depth, 101)
    lnp = -np.log(prof.p)
    th = np.interp(-np.log(pp), lnp, prof.theta)
    qv = np.interp(-np.log(pp), lnp, prof.qv)
    th_m, qv_m = th.mean(), qv.mean()   # uniform in p, so pressure weighted
    return lift_parcel(prof, 0, th_m * prof.pi[0], qv_m, "ML", virtual)


def mu_parcel(prof, depth=30000.0, virtual=True):
    ok = prof.p >= prof.p[0] - depth
    the = thermo.theta_e_env(prof.t, prof.p, prof.qv)
    k = int(np.argmax(np.where(ok, the, -np.inf)))
    return lift_parcel(prof, k, prof.t[k], prof.qv[k], "MU", virtual)


def effective_layer(prof, virtual=True, depth=30000.0):
    base = top = None
    last = None
    for k in range(len(prof.z)):
        if prof.p[k] < prof.p[0] - depth:
            break
        par = lift_parcel(prof, k, prof.t[k], prof.qv[k], virtual=virtual)
        ok = par.cape >= 100.0 and par.cin >= -250.0
        if base is None:
            if ok:
                base = k
        elif not ok:
            top = k - 1
            break
        last = k
    if base is None:
        return np.nan, np.nan
    if top is None:
        top = last
    return float(prof.z[base]), float(prof.z[top])


# ---- kinematics ---------------------------------------------------------------
def filled_winds(snd, mode):
    u = snd.u.copy()
    v = snd.v.copy()
    if mode == "zero":
        u[0] = v[0] = 0.0
    elif mode == "extrapolate" and len(u) > 2:
        f = snd.z[1] / (snd.z[2] - snd.z[1])
        u[0] = u[1] - (u[2] - u[1]) * f
        v[0] = v[1] - (v[2] - v[1]) * f
    else:
        u[0], v[0] = u[1], v[1]
    return u, v


def wind_at(z, u, v, h):
    return float(np.interp(h, z, u)), float(np.interp(h, z, v))


def mean_wind(z, u, v, bot, top, dz=10.0):
    hh = np.arange(bot, top + 0.5 * dz, dz)
    return float(np.interp(hh, z, u).mean()), float(np.interp(hh, z, v).mean())


def bulk_shear(z, u, v, bot, top):
    u1, v1 = wind_at(z, u, v, bot)
    u2, v2 = wind_at(z, u, v, top)
    return float(np.hypot(u2 - u1, v2 - v1))


def bunkers(z, u, v):
    """bunkers et al. (2000), 0-6 km non-pressure-weighted mean, d = 7.5 m/s."""
    mu_, mv_ = mean_wind(z, u, v, 0.0, 6000.0)
    lu, lv = mean_wind(z, u, v, 0.0, 500.0)
    hu, hv = mean_wind(z, u, v, 5500.0, 6000.0)
    su, sv = hu - lu, hv - lv
    mag = np.hypot(su, sv)
    if mag < 1e-6:
        return (mu_, mv_), (mu_, mv_), (mu_, mv_)
    d = 7.5
    rm = (mu_ + d * sv / mag, mv_ - d * su / mag)
    lm = (mu_ - d * sv / mag, mv_ + d * su / mag)
    return rm, lm, (mu_, mv_)


def srh(z, u, v, cu, cv, bot, top):
    inside = (z > bot) & (z < top)
    zz = np.concatenate([[bot], z[inside], [top]])
    sru = np.interp(zz, z, u) - cu
    srv = np.interp(zz, z, v) - cv
    return float(np.sum(sru[1:] * srv[:-1] - sru[:-1] * srv[1:]))


def streamwise(z, u, v, cu, cv, top=500.0, dz=10.0):
    hh = np.arange(0.0, top + 0.5 * dz, dz)
    uu = np.interp(hh, z, u)
    vv = np.interp(hh, z, v)
    dudz = np.diff(uu) / dz
    dvdz = np.diff(vv) / dz
    um = 0.5 * (uu[1:] + uu[:-1]) - cu
    vm = 0.5 * (vv[1:] + vv[:-1]) - cv
    spd = np.maximum(np.hypot(um, vm), 1e-6)
    # horizontal vorticity = k x dV/dz
    wx, wy = -dvdz, dudz
    ws = (wx * um + wy * vm) / spd
    mag = np.hypot(wx, wy)
    frac = ws.mean() / mag.mean() if mag.mean() > 0 else np.nan
    return float(ws.mean()), float(100.0 * frac)


def critical_angle(z, u, v, cu, cv):
    """esterheld and giuliano (2008): angle between 0-500 m shear and surface storm-relative inflow."""
    us, vs = u[0], v[0]
    u5, v5 = wind_at(z, u, v, 500.0)
    sx, sy = u5 - us, v5 - vs
    ix, iy = us - cu, vs - cv
    n = np.hypot(sx, sy) * np.hypot(ix, iy)
    if n < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip((sx * ix + sy * iy) / n, -1.0, 1.0))))


# ---- composites ---------------------------------------------------------------
def _lcl_term(lcl):
    if lcl < 1000.0:
        return 1.0
    if lcl > 2000.0:
        return 0.0
    return (2000.0 - lcl) / 1000.0


def _shear_term_stp(s):
    if s < 12.5:
        return 0.0
    return min(s, 30.0) / 20.0


def _cin_term(cin):
    if cin > -50.0:
        return 1.0
    if cin < -200.0:
        return 0.0
    return (200.0 + cin) / 150.0


def _env_t_at_height(prof, h):
    return float(np.interp(h, prof.z, prof.t))


def _height_at_pressure(prof, p):
    return float(np.interp(-np.log(p), -np.log(prof.p), prof.z))


# ---- main entry ---------------------------------------------------------------
def compute(snd, settings=None):
    s = settings or Settings()
    prof = snd.derived()
    z = prof.z
    u, v = filled_winds(snd, s.surface_wind)
    r = Results(u=u, v=v)
    val = r.values

    sb = lift_parcel(prof, 0, prof.t[0], prof.qv[0], "SB", s.virtual)
    ml = ml_parcel(prof, virtual=s.virtual)
    mu = mu_parcel(prof, virtual=s.virtual)
    r.parcels = {"SB": sb, "ML": ml, "MU": mu}
    for tag, par in (("sb", sb), ("ml", ml), ("mu", mu)):
        val[f"{tag}cape"] = par.cape
        val[f"{tag}cin"] = par.cin
        val[f"{tag}lcl"] = par.z_lcl
        val[f"{tag}lfc"] = par.lfc
        val[f"{tag}el"] = par.el
    val["ml3cape"] = ml.cape3
    val["muz"] = float(z[mu.k])

    val["lr03"] = (_env_t_at_height(prof, 0.0) - _env_t_at_height(prof, 3000.0)) / 3.0
    z7, z5 = _height_at_pressure(prof, 70000.0), _height_at_pressure(prof, 50000.0)
    t7 = float(np.interp(z7, z, prof.t))
    t5 = float(np.interp(z5, z, prof.t))
    val["lr75"] = (t7 - t5) / ((z5 - z7) / 1000.0)
    rho = prof.p / (thermo.RD * prof.tv)
    q = rho * prof.qv
    val["pw"] = float(np.sum(0.5 * (q[1:] + q[:-1]) * np.diff(z)))
    val["sfc_t"] = float(prof.t[0] - 273.15)
    val["sfc_td"] = float(prof.td[0] - 273.15)

    # kinematics
    for key, h in (("shr05", 500.0), ("shr1", 1000.0), ("shr3", 3000.0), ("shr6", 6000.0)):
        val[key] = bulk_shear(z, u, v, 0.0, h)
    rm, lm, mw = bunkers(z, u, v)
    r.bunkers_rm, r.bunkers_lm, r.mean_wind = rm, lm, mw
    cu, cv = {"rm": rm, "lm": lm, "mean": mw}.get(s.storm_motion, (s.custom_u, s.custom_v))
    r.storm = (cu, cv)
    val["storm_u"], val["storm_v"] = cu, cv
    for key, h in (("srh05", 500.0), ("srh1", 1000.0), ("srh3", 3000.0)):
        val[key] = srh(z, u, v, cu, cv, 0.0, h)

    eb, et = effective_layer(prof, s.virtual)
    r.eff_layer = (eb, et)
    val["eff_base"], val["eff_top"] = eb, et
    if np.isfinite(eb):
        val["esrh"] = srh(z, u, v, cu, cv, eb, et)
        if np.isfinite(mu.el):
            val["ebwd"] = bulk_shear(z, u, v, eb, eb + 0.5 * (mu.el - eb))
        else:
            val["ebwd"] = 0.0
    else:
        val["esrh"] = 0.0
        val["ebwd"] = 0.0
    val["crit_angle"] = critical_angle(z, u, v, cu, cv)
    val["swv05"], val["swfrac05"] = streamwise(z, u, v, cu, cv)

    # composites
    stp_eff = ((ml.cape / 1500.0) * _lcl_term(ml.z_lcl) * (val["esrh"] / 150.0)
               * _shear_term_stp(val["ebwd"]) * _cin_term(ml.cin))
    val["stp_eff"] = max(stp_eff, 0.0)
    stp_fix = ((sb.cape / 1500.0) * _lcl_term(sb.z_lcl) * (val["srh1"] / 150.0)
               * _shear_term_stp(val["shr6"]))
    val["stp_fix"] = max(stp_fix, 0.0)
    eb_term = 0.0 if val["ebwd"] < 10.0 else min(val["ebwd"], 20.0) / 20.0
    val["scp"] = (mu.cape / 1000.0) * (val["esrh"] / 50.0) * eb_term
    lapse = (val["lr03"] / 5.2) * (val["lr75"] / 5.6)
    val["sherbs3"] = (val["shr3"] / 26.0) * lapse
    val["sherbe"] = (val["ebwd"] / 27.0) * lapse
    return r
