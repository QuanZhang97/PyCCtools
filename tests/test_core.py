import importlib
import json
import numpy as np
import pytest
from obspy import Trace, Stream, UTCDateTime
from obspy.core.event import Event, Origin
from obspy.taup import TauPyModel
from pycctools import phase_pair_curve, travel_time
from pycctools.common import (distance_bearing, regular_edges, distance_bin_index,
                              event_slug, coverage_fraction)
from pycctools.signal import GCC, correlate_pair, linear_lags, stack_data


def test_prem_predictions_and_generic_phases(tmp_path):
    curve = phase_pair_curve([30., 60.], 20., 'P', 'S')
    model = TauPyModel('prem')
    expected = []
    for d in curve.distance_deg:
        p = model.get_travel_times(20., d, phase_list=['P'])[0].time
        s = model.get_travel_times(20., d, phase_list=['S'])[0].time
        expected.append(s - p)
    np.testing.assert_allclose(curve.lag_sec, expected)
    reverse = phase_pair_curve([30., 60.], 20., 'S', 'P')
    np.testing.assert_allclose(reverse.lag_sec, -curve.lag_sec)
    out = curve.save(tmp_path / 'curve.txt')
    metadata = json.loads(out.read_text().splitlines()[0][2:])
    assert metadata['model'] == 'prem' and metadata['phase_a'] == 'P'
    np.testing.assert_allclose(np.loadtxt(out)[:, 1], curve.lag_sec, atol=1e-8)


def test_inner_core_pair_and_shadow_zone():
    curve = phase_pair_curve([150.], phase_a='PKIKP', phase_b='PKJKP')
    assert np.isfinite(curve.lag_sec[0]) and curve.lag_sec[0] > 0
    assert np.isnan(travel_time(0., phase='PKIKP').time_sec)
    assert np.isfinite(travel_time(30., phase='P', model='iasp91').time_sec)


@pytest.mark.parametrize('distance,depth', [(-1, 0), (181, 0), (np.nan, 0), (30, -1)])
def test_bad_coordinates(distance, depth):
    with pytest.raises(ValueError):
        travel_time(distance, depth)


def test_bad_phase_is_not_missing_path():
    with pytest.raises(Exception):
        travel_time(30, phase='not-a-phase')


def test_geometry_bins_and_half_open_coverage():
    distance, bearing = distance_bearing(0, 0, 0, 90)
    assert distance == pytest.approx(90) and bearing == pytest.approx(90)
    assert np.isnan(distance_bearing(0, 0, 0, 180)[1])
    edges = regular_edges(0, 10, 3)
    np.testing.assert_equal(edges, [0, 3, 6, 9, 10])
    assert distance_bin_index(10, edges) == 3
    assert distance_bin_index(11, edges) is None
    trace = Trace(np.ones(10), header={'starttime': UTCDateTime(0), 'delta': 1.})
    assert coverage_fraction(Stream([trace, trace.copy()]), UTCDateTime(0), UTCDateTime(20)) == .5
    assert event_slug(Event(resource_id='smi:local/test/event')).startswith('test_')


def test_correlation_lags_and_buffer_ownership():
    a = Trace(np.array([0., 1., 0., 0., 0., 0., 0., 0.]))
    b = Trace(np.array([0., 0., 0., 1., 0., 0., 0., 0.]))
    lag, corr = correlate_pair(a, b, temp_width=None, spec_width=None, ram_fband=None)
    assert lag[np.argmax(corr)] == -2
    lag, corr = correlate_pair(a, a, temp_width=None, spec_width=None, ram_fband=None)
    assert lag[np.argmax(corr)] == 0 and corr[np.argmax(corr)] == pytest.approx(1)
    np.testing.assert_equal(a.data, [0, 1, 0, 0, 0, 0, 0, 0])
    gcc = GCC(16, None, None, None)
    spec = gcc.pre_proc(a)
    saved = spec.copy()
    gcc.pre_proc(b)
    np.testing.assert_equal(spec, saved)
    lag_ba, corr_ba = correlate_pair(b, a, temp_width=None, spec_width=None, ram_fband=None)
    lag_ab, corr_ab = correlate_pair(a, b, temp_width=None, spec_width=None, ram_fband=None)
    np.testing.assert_array_equal(lag_ab, lag_ba)
    np.testing.assert_allclose(corr_ba, corr_ab[::-1], atol=1e-7)
    assert lag_ba[np.argmax(corr_ba)] == 2


def test_short_zero_trace_whitening_and_stack():
    trace = Trace(np.zeros(16, dtype=np.float32))
    _, corr = correlate_pair(trace, trace, temp_width=128, spec_width=2., ram_fband=None)
    assert np.isfinite(corr).all() and not corr.any()
    assert np.isfinite(stack_data(Stream([trace, trace.copy()]), order=2).data).all()


@pytest.mark.parametrize('stage', ['events', 'stations', 'download', 'correlation', 'plotting'])
def test_import_and_reject_typo(stage, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module('pycctools.' + stage)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match='Unknown settings'):
        module.run(NOT_A_SETTING=True)
    assert list(tmp_path.iterdir()) == []
