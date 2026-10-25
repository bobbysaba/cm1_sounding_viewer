"""cm1 input_sounding i/o and hydrostatic base state.

index 0 of every array is the surface line of the file (z = 0, psfc, theta, qv);
the file has no surface wind, so u[0] and v[0] are nan.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import thermo


@dataclass
class Profile:
    """derived quantities on the sounding levels (si units)."""
    z: np.ndarray       # m agl
    p: np.ndarray       # pa
    pi: np.ndarray      # exner function
    theta: np.ndarray   # k
    qv: np.ndarray      # kg/kg
    t: np.ndarray       # k
    td: np.ndarray      # k
    tv: np.ndarray      # k
    rh: np.ndarray      # fraction


@dataclass
class Sounding:
    psfc: float         # surface pressure (hpa)
    z: np.ndarray       # height agl (m), z[0] = 0
    theta: np.ndarray   # potential temperature (k)
    qv: np.ndarray      # water vapor mixing ratio (g/kg)
    u: np.ndarray       # m/s, u[0] = nan
    v: np.ndarray       # m/s, v[0] = nan

    # ---- i/o ----------------------------------------------------------------
    @classmethod
    def read(cls, path):
        rows = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if parts:
                rows.append(parts)
        if len(rows) < 3:
            raise ValueError("file has fewer than 3 non-empty lines")
        try:
            psfc, th0, qv0 = (float(x) for x in rows[0][:3])
            data = np.array([[float(x) for x in r[:5]] for r in rows[1:]])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"could not parse numbers: {exc}") from exc
        if data.ndim != 2 or data.shape[1] != 5:
            raise ValueError("expected 5 columns (z, theta, qv, u, v) after the first line")
        nan = np.array([np.nan])
        return cls(
            psfc=psfc,
            z=np.concatenate([[0.0], data[:, 0]]),
            theta=np.concatenate([[th0], data[:, 1]]),
            qv=np.concatenate([[qv0], data[:, 2]]),
            u=np.concatenate([nan, data[:, 3]]),
            v=np.concatenate([nan, data[:, 4]]),
        )

    def to_text(self):
        lines = [f"{self.psfc:.3f}  {self.theta[0]:.3f}  {self.qv[0]:.3f}"]
        for k in range(1, len(self.z)):
            lines.append(f"{self.z[k]:.3f}  {self.theta[k]:.3f}  {self.qv[k]:.3f}  "
                         f"{self.u[k]:.3f}  {self.v[k]:.3f}")
        return "\n".join(lines) + "\n"

    def write(self, path):
        Path(path).write_text(self.to_text())

    # ---- helpers ------------------------------------------------------------
    def copy(self):
        return Sounding(self.psfc, self.z.copy(), self.theta.copy(), self.qv.copy(),
                        self.u.copy(), self.v.copy())

    def equals(self, other, tol=5e-4):
        # equality at the precision the file is written with
        if other is None or len(self.z) != len(other.z):
            return False
        if abs(self.psfc - other.psfc) > tol:
            return False
        for a, b in ((self.z, other.z), (self.theta, other.theta), (self.qv, other.qv),
                     (self.u[1:], other.u[1:]), (self.v[1:], other.v[1:])):
            if np.any(np.abs(a - b) > tol):
                return False
        return True

    def __len__(self):
        return len(self.z)

    # ---- base state ---------------------------------------------------------
    def derived(self):
        """hydrostatic pressure from psfc, integrating the exner function in theta_v."""
        qv = np.maximum(self.qv, 0.0) / 1000.0
        thv = thermo.virtual_temp(self.theta, qv)
        pi = np.empty_like(self.z)
        pi[0] = thermo.exner(self.psfc * 100.0)
        dz = np.diff(self.z)
        inv = 1.0 / thv
        pi[1:] = pi[0] - thermo.G / thermo.CP * np.cumsum(dz * 0.5 * (inv[1:] + inv[:-1]))
        p = thermo.pressure_from_exner(pi)
        t = self.theta * pi
        td = np.minimum(thermo.dewpoint(qv, p), t)
        return Profile(z=self.z.copy(), p=p, pi=pi, theta=self.theta.copy(), qv=qv, t=t,
                       td=td, tv=thermo.virtual_temp(t, qv), rh=thermo.rel_humidity(t, qv, p))

    # ---- checks -------------------------------------------------------------
    def check(self):
        """return a list of (severity, message, list_of_heights) issues."""
        issues = []
        dz = np.diff(self.z)
        if np.any(dz <= 0):
            bad = self.z[1:][dz <= 0]
            issues.append(("error", "heights not strictly increasing", list(bad)))
        if np.any(self.qv < 0):
            issues.append(("error", "negative qv", list(self.z[self.qv < 0])))
        if not np.all(np.isfinite(np.concatenate([self.theta, self.qv, self.u[1:], self.v[1:]]))):
            issues.append(("error", "non-finite values", []))
        prof = self.derived()
        sup = prof.rh > 1.005
        if np.any(sup):
            issues.append(("warning", "supersaturated", list(self.z[sup])))
        dth = np.diff(self.theta)
        neg = dth < 0
        if np.any(neg):
            # report the bottom of each layer where theta decreases with height
            issues.append(("info", "theta decreases with height (superadiabatic)",
                           list(self.z[:-1][neg])))
        return issues
