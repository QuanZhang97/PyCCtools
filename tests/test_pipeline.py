import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, Stream, Trace, UTCDateTime, read_events
from obspy.core.event import Event, Origin, Magnitude, FocalMechanism, PrincipalAxes, Axis
from pycctools.common import event_slug
from pycctools import correlation, plotting, events
from pycctools.cli import main


def test_local_catalogue_selection(tmp_path):
    origin = Origin(time=UTCDateTime('2010-01-01'), latitude=0., longitude=130.,
                    depth=10000., origin_type='hypocenter')
    event = Event(resource_id='smi:local/fixture/event', origins=[origin],
                  magnitudes=[Magnitude(mag=7.5, magnitude_type='Mw')],
                  focal_mechanisms=[FocalMechanism(principal_axes=PrincipalAxes(
                      t_axis=Axis(plunge=70, azimuth=0, length=1),
                      p_axis=Axis(plunge=10, azimuth=90, length=-1),
                      n_axis=Axis(plunge=10, azimuth=180, length=0)))])
    event.preferred_origin_id = origin.resource_id
    path = tmp_path / 'source.xml'
    Catalog([event]).write(str(path), format='QUAKEML')
    result = events.run(LOCAL_CATALOGUE=path, LOCAL_FORMAT='QUAKEML',
                        START_TIME='2009-01-01', END_TIME='2011-01-01',
                        REQUIRE_PLOTTABLE_STF=False, PREVIEW_COUNT=0,
                        CATALOGUE_DIR=tmp_path / 'catalogue', FIG_DIR=tmp_path / 'fig')
    assert len(read_events(str(result['catalogue_file']))) == 1
    # Repeatability, isolated settings, reproducible run identity.
    again = events.run(LOCAL_CATALOGUE=path, LOCAL_FORMAT='QUAKEML',
                       START_TIME='2009-01-01', END_TIME='2011-01-01',
                       REQUIRE_PLOTTABLE_STF=False, PREVIEW_COUNT=0,
                       CATALOGUE_DIR=tmp_path / 'catalogue', FIG_DIR=tmp_path / 'fig')
    assert again == result


@pytest.mark.parametrize("latitude,longitude", [(0., 0.), (5., 130.)])
def test_waveform_to_stack_to_figures(tmp_path, latitude, longitude):
    origin = Origin(time=UTCDateTime(0), latitude=latitude, longitude=longitude, depth=10000., origin_type='hypocenter')
    event = Event(resource_id='smi:local/synthetic/event', origins=[origin])
    event.preferred_origin_id = origin.resource_id
    cat = tmp_path / 'events.xml'
    Catalog([event]).write(str(cat), format='QUAKEML')
    name = event_slug(event)
    station_dir, waveform_dir = tmp_path / 'stations', tmp_path / 'waveforms'
    (station_dir / name).mkdir(parents=True)
    (waveform_dir / name).mkdir(parents=True)
    pd.DataFrame([dict(station_id='XX.A', latitude=latitude+10., longitude=longitude),
                  dict(station_id='XX.B', latitude=latitude+20., longitude=longitude)]).to_csv(
                      station_dir / name / 'stations.csv', index=False)
    rng = np.random.default_rng(42)
    data = np.sin(2*np.pi*.04*np.arange(3000)/10) + .1*rng.normal(size=3000)
    for station in ['A', 'B']:
        trace_data = data if station == 'A' else np.roll(data, 7)
        trace = Trace(trace_data.astype(np.float32), header=dict(network='XX', station=station,
                      channel='BHZ', starttime=origin.time, sampling_rate=10.))
        Stream([trace]).write(str(waveform_dir / name / f'{station}_XX_--_BHZ.mseed'), format='MSEED')
    # A second vertical channel for A must not produce A->A or duplicate A->B.
    alternate = Trace(data.astype(np.float32), header=dict(network='XX', station='A',
                      channel='HHZ', starttime=origin.time, sampling_rate=10.))
    Stream([alternate]).write(str(waveform_dir / name / 'A_XX_--_HHZ.mseed'), format='MSEED')
    output = tmp_path / 'correlation'
    settings = dict(CATALOGUE_FILE=cat, WAVEFORM_RUN_DIR=waveform_dir, STATION_RUN_DIR=station_dir,
                    START_HOURS=0., END_HOURS=300/3600,
                    TARGET_SAMPLING_RATE_HZ=10., CORRELATION_DIR=output,
                    N_PREPROCESS_WORKERS=2, N_STACK_WORKERS=2)
    correlation.run(**settings)
    files = list(output.glob('*/correlations/*.npz'))
    assert len(files) == 1
    with np.load(files[0]) as stack:
        assert stack['pair_count'].sum() == 2
        assert stack['correlation'].shape == (60, 5999)
        assert np.isfinite(stack['correlation']).all()
        saved = stack['correlation'].copy()
        assert stack['lag_sec'][2999] == 0
        assert not bool(stack['symmetrized'])
        assert stack['center_lat'] == latitude
        assert stack['center_lon'] == longitude
    pair_files = list((output / name / 'pairs').rglob('*.npz'))
    assert len(pair_files) == 2
    directed = {}
    for file in pair_files:
        with np.load(file) as pair:
            directed[(str(pair['station_a']), str(pair['station_b']))] = pair['correlation'].copy()
            assert pair['distance_deg'] == pytest.approx(10.)
            assert pair['epicenter_latitude'] == latitude
            assert pair['epicenter_longitude'] == longitude
    assert set(directed) == {('XX.A', 'XX.B'), ('XX.B', 'XX.A')}
    np.testing.assert_allclose(directed['XX.A', 'XX.B'], directed['XX.B', 'XX.A'][::-1], atol=1e-4)
    assert not np.allclose(directed['XX.A', 'XX.B'], directed['XX.A', 'XX.B'][::-1], atol=1e-6)
    from obspy.signal.filter import bandpass
    expected = bandpass((directed['XX.A', 'XX.B'] + directed['XX.B', 'XX.A']) / 2,
                        freqmin=1/50, freqmax=1/15, df=10., corners=4, zerophase=True)
    with np.load(files[0]) as stack:
        selected_bin = np.flatnonzero(stack['pair_count'])[0]
        np.testing.assert_allclose(stack['correlation'][selected_bin], expected, atol=1e-5)

    correlation.run(**settings)
    with np.load(files[0]) as stack:
        np.testing.assert_allclose(stack['correlation'], saved)
    fig_dir = tmp_path / 'figures'
    plotting.run(CORRELATION_DIR=output, FIG_DIR=fig_dir, SAVE_PDF=False, PLOT_DPI=40,
                 DISPLAY_LAG_RANGE_SEC=(0, 100), FIGURE_SIZE=(4, 4))
    pngs = list(fig_dir.rglob('*.png'))
    assert len(pngs) == 2 and all(p.stat().st_size > 1000 for p in pngs)


def test_cli_other_phases(tmp_path):
    path = tmp_path / 'p-s.txt'
    main(['traveltime', '--phase-a', 'P', '--phase-b', 'S', '--distance-min', '30',
          '--distance-max', '31', '--output', str(path)])
    assert np.loadtxt(path).shape == (2, 2)


def test_station_and_download_services_offline(tmp_path, monkeypatch):
    """Exercise real stage selection/I/O using deterministic FDSN responses."""
    from io import BytesIO
    import requests
    import cartopy.mpl.geoaxes
    from obspy.core.inventory import Inventory, Network, Station, Channel, Site
    from pycctools import stations, download

    origin = Origin(time=UTCDateTime('2010-01-01'), latitude=0., longitude=0.,
                    depth=10000., origin_type='hypocenter')
    event = Event(resource_id='smi:local/offline/event', origins=[origin])
    event.preferred_origin_id = origin.resource_id
    second_origin = Origin(time=origin.time+1000, latitude=0., longitude=90.,
                           depth=10000., origin_type='hypocenter')
    second_event = Event(resource_id='smi:local/second/event', origins=[second_origin])
    second_event.preferred_origin_id = second_origin.resource_id
    cat = tmp_path / 'catalogue.xml'
    Catalog([event, second_event]).write(str(cat), format='QUAKEML')
    inventory = Inventory([Network(code='XX', stations=[Station(
        code=name, latitude=lat, longitude=0., elevation=0., site=Site(name=name),
        channels=[Channel(code='BHZ', location_code='', latitude=lat, longitude=0.,
                          elevation=0., depth=0., azimuth=0., dip=-90., sample_rate=10.,
                          start_date=origin.time-86400, end_date=origin.time+86400)])
        for name, lat in [('A', 10.), ('B', 20.)]])], source='synthetic')
    buffer = BytesIO()
    inventory.write(buffer, format='STATIONXML')
    xml = buffer.getvalue()
    calls = []

    class Response:
        status_code = 200
        url = 'https://example.test/fdsnws'
        def __init__(self, content): self.content = content
        def raise_for_status(self): pass

    def fake_get(self, url, params, **kwargs):
        calls.append(url)
        if '/station/' in url:
            return Response(xml)
        n = int((UTCDateTime(params['endtime'])-UTCDateTime(params['starttime'])) * 10) + 1
        trace = Trace(np.sin(np.arange(n) * .025).astype(np.float32), header=dict(
            network='XX', station=params['station'], channel='BHZ',
            starttime=UTCDateTime(params['starttime']), sampling_rate=10.))
        payload = BytesIO()
        Stream([trace]).write(payload, format='MSEED')
        return Response(payload.getvalue())

    monkeypatch.setattr(requests.Session, 'get', fake_get)
    # No Natural Earth download: numerical map geometry/rendering still runs.
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, 'coastlines', lambda *a, **kw: None)
    monkeypatch.setattr(cartopy.mpl.geoaxes.GeoAxes, 'add_feature', lambda *a, **kw: None)
    station_result = stations.run(
        CATALOGUE_FILE=cat,
        WINDOW_START_OFFSET_S=0., WINDOW_END_OFFSET_S=300.,
        PROVIDERS={'TEST': 'https://example.test/fdsnws/station/1/query'},
        AZIMUTH_INCREMENT_DEG=90., RADIAL_INCREMENT_DEG=90., GREAT_CIRCLE_STEP_DEG=10.,
        OUTPUT_DIR=tmp_path/'stations', CACHE_DIR=tmp_path/'station_cache',
        FIG_DIR=tmp_path/'fig', PREVIEW_EVENT_COUNT=0)
    name = event_slug(event)
    selected = pd.read_csv(station_result['run_dir']/name/'stations.csv')
    assert len(selected) == 2
    other = pd.read_csv(station_result['run_dir']/event_slug(second_event)/'stations.csv')
    assert len(other) == 2
    np.testing.assert_allclose(selected.center_azimuth_deg, selected.event_azimuth_deg)
    np.testing.assert_allclose(other.center_azimuth_deg, other.event_azimuth_deg)
    assert not np.allclose(selected.center_azimuth_deg, other.center_azimuth_deg)
    result = download.run(
        CATALOGUE_FILE=cat, STATION_RUN_DIR=station_result['run_dir'],
        START_HOURS=0., END_HOURS=300/3600,
        AZIMUTH_BIN_WIDTH_DEG=90.,
        PROVIDERS={'TEST': 'https://example.test/fdsnws/dataselect/1/query'},
        WAVEFORM_DIR=tmp_path/'waveforms', CACHE_DIR=tmp_path/'waveform_cache', FIG_DIR=tmp_path/'fig')
    usable = pd.read_csv(result['run_dir']/name/'usable_stations.csv')
    assert len(usable) == 2
    assert len(list((result['run_dir']/name).glob('*.mseed'))) == 2
    groups = pd.read_csv(result['run_dir']/'group_summary.csv')
    assert groups.pair_count.sum() == 4
    assert len(calls) == 5
