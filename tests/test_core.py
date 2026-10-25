"""core tests: run with `pytest` from the project folder."""
from pathlib import Path

import numpy as np
import pytest

from sounding_editor import edits, params
from sounding_editor.sounding import Sounding

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "rm_tor"


@pytest.fixture
def snd():
    return Sounding.read(EXAMPLE)


def test_round_trip_is_byte_identical(snd):
    assert snd.to_text() == EXAMPLE.read_text()


def test_hydrostatic_pressure_is_sane(snd):
    prof = snd.derived()
    assert prof.p[0] == pytest.approx(99025.7, abs=0.1)
    assert np.all(np.diff(prof.p) < 0)
    # dry-ish check: 1 km pressure near 880 hpa for this warm, moist profile
    assert 87000 < np.interp(1000.0, prof.z, prof.p) < 89000


def test_drag_without_smoothing_changes_one_level(snd):
    prof = snd.derived()
    new = edits.drag_temperature(snd, 20, prof.t[20] + 2.0, sigma=0.0)
    changed = np.where(np.abs(new.theta - snd.theta) > 1e-9)[0]
    assert list(changed) == [20]
    assert new.derived().t[20] - prof.t[20] == pytest.approx(2.0, abs=1e-3)


def test_drag_with_smoothing_spreads(snd):
    prof = snd.derived()
    new = edits.drag_temperature(snd, 20, prof.t[20] + 2.0, sigma=300.0)
    assert (np.abs(new.theta - snd.theta) > 1e-9).sum() > 10


def test_wind_drag_never_touches_surface_line(snd):
    new = edits.drag_wind(snd, 1, 50.0, 50.0, sigma=500.0)
    assert np.isnan(new.u[0]) and np.isnan(new.v[0])
    assert "nan" not in new.to_text()


def test_remove_superadiabatic(snd):
    new = edits.remove_superadiabatic(snd, 0.0, snd.z[-1])
    assert np.all(np.diff(new.theta) >= -1e-9)
    assert not any(sev == "info" for sev, *_ in new.check())


def test_rotate_shear_keeps_shear_above_layer(snd):
    new = edits.rotate_shear(snd, 0.0, 1000.0, 30.0, carry=True)
    above = snd.z > 1000.0
    assert np.allclose(np.diff(new.u[above]), np.diff(snd.u[above]))
    assert np.allclose(np.diff(new.v[above]), np.diff(snd.v[above]))


def test_lapse_rate(snd):
    new = edits.set_lapse_rate(snd, 0.0, 2000.0, 9.8)
    prof = new.derived()
    t0, t2 = np.interp([0.0, 2000.0], prof.z, prof.t)
    assert (t0 - t2) / 2.0 == pytest.approx(9.8, abs=0.01)


def test_parameters_reasonable(snd):
    v = params.compute(snd).values
    assert 200 < v["sbcape"] < 500
    assert v["sbcin"] < 0
    assert v["srh1"] > 0
    assert np.isfinite(v["stp_eff"])


def test_against_metpy(snd):
    metpy = pytest.importorskip("metpy.calc")
    from metpy.units import units
    prof = snd.derived()
    res = params.compute(snd)
    p, t, td = prof.p * units.Pa, prof.t * units.K, prof.td * units.K
    # parcel temperatures agree to a few tenths of a kelvin
    pp = metpy.parcel_profile(p, t[0], td[0]).m
    assert np.max(np.abs(pp - res.parcels["SB"].t)) < 0.5
    # srh is a pure kinematic integral and should match closely
    z = prof.z * units.m
    u, v = res.u * units("m/s"), res.v * units("m/s")
    srh = metpy.storm_relative_helicity(z, u, v, 1000 * units.m,
                                        storm_u=res.storm[0] * units("m/s"),
                                        storm_v=res.storm[1] * units("m/s"))[2].m
    assert res.values["srh1"] == pytest.approx(srh, rel=0.01)
