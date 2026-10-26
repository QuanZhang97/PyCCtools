"""Coda waveform and station-pair download stage.

No processing occurs at import. Settings use uppercase names.
Each run has independent settings, helpers and FFT worker state.
"""
from ._config import setting, validate_settings

SETTINGS = ('CATALOGUE_FILE', 'STATION_RUN_DIR', 'START_HOURS', 'END_HOURS', 'AZIMUTH_START_DEG', 'AZIMUTH_STOP_DEG', 'AZIMUTH_BIN_WIDTH_DEG', 'CHANNEL_PREFERENCE', 'LOCATION_PREFERENCE', 'DOWNLOAD_ALL_CHANNELS', 'REQUEST_CHUNK_SECONDS', 'REQUEST_TIMEOUT_SECONDS', 'REFRESH_DOWNLOADS', 'CHECK_CHANNEL_EPOCHS', 'MAX_EVENTS', 'MAX_STATIONS_PER_EVENT', 'DISTANCE_HISTOGRAM_BIN_DEG', 'WAVEFORM_DIR', 'FIG_DIR', 'CACHE_DIR', 'PROVIDERS')

def run(**overrides):
    """Execute this stage and return its output paths. See SETTINGS for options."""
    validate_settings(overrides, SETTINGS)
    #!/usr/bin/env python
    # coding: utf-8

    # # 03 · Download 3–10 h vertical coda and analyse station-pair geometry
    #
    # Run after notebooks 01 and 02. This notebook loops through the selected earthquake catalogue, reads each event's station selection, downloads vertical recordings, and analyses every unordered pair of successfully downloaded stations within each opposite-azimuth group.
    #
    # **Files:** `./waveforms/<run_id>/<event>/<STA>_<NET>_<LOC>_<BHZ-or-HHZ>.mseed`. Network and location prevent filename collisions; a blank location is written as `--`. Raw instrument counts are retained. No filtering, resampling, response removal, normalization or correlation is applied yet.
    #
    # **Window:** origin +3 h through origin +10 h, inclusive sample trimming. Requests are split into one-hour pieces and cached. Only complete, finite, single-sample-rate, gap-free recordings enter pair analysis by default. Partial recordings are saved separately and flagged. No gaps are padded or interpolated.
    #
    # **Station scope:** this uses the station list selected in notebook 02. It rechecks selected channel epochs for the 3–10 h window, but cannot recover stations omitted by a prior 0–3 h selection. For a selection specifically tailored to coda, rerun notebook 02 with offsets 10800 and 36000 seconds.
    #
    # **CSV correction:** notebook 02 writes geographic bearings in `stations.csv`; `channels.csv` contains NSLC identifiers, provider and coordinates. This notebook joins them and recalculates the centre-to-station bearing. A channel's instrument azimuth must never be used as its geographic bearing.

    from pathlib import Path
    from io import BytesIO
    from itertools import permutations
    import csv
    import hashlib
    import json
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from obspy import read, read_events, read_inventory, Stream, UTCDateTime

    CATALOGUE_FILE = None
    CATALOGUE_FILE = setting(overrides, "CATALOGUE_FILE", CATALOGUE_FILE)
    STATION_RUN_DIR = None
    STATION_RUN_DIR = setting(overrides, "STATION_RUN_DIR", STATION_RUN_DIR)
    START_HOURS, END_HOURS = 3.0, 10.0
    START_HOURS = setting(overrides, "START_HOURS", START_HOURS)
    END_HOURS = setting(overrides, "END_HOURS", END_HOURS)
    AZIMUTH_START_DEG, AZIMUTH_STOP_DEG = 0.0, 180.0
    AZIMUTH_START_DEG = setting(overrides, "AZIMUTH_START_DEG", AZIMUTH_START_DEG)
    AZIMUTH_STOP_DEG = setting(overrides, "AZIMUTH_STOP_DEG", AZIMUTH_STOP_DEG)
    AZIMUTH_BIN_WIDTH_DEG = 10.0
    AZIMUTH_BIN_WIDTH_DEG = setting(overrides, "AZIMUTH_BIN_WIDTH_DEG", AZIMUTH_BIN_WIDTH_DEG)
    CHANNEL_PREFERENCE = ('BHZ', 'HHZ')
    CHANNEL_PREFERENCE = setting(overrides, "CHANNEL_PREFERENCE", CHANNEL_PREFERENCE)
    LOCATION_PREFERENCE = ('00', '', '10', '01')
    LOCATION_PREFERENCE = setting(overrides, "LOCATION_PREFERENCE", LOCATION_PREFERENCE)
    DOWNLOAD_ALL_CHANNELS = False  # False: first complete channel per NET.STA; try alternatives on failure
    DOWNLOAD_ALL_CHANNELS = setting(overrides, "DOWNLOAD_ALL_CHANNELS", DOWNLOAD_ALL_CHANNELS)
    REQUEST_CHUNK_SECONDS = 3600.0
    REQUEST_CHUNK_SECONDS = setting(overrides, "REQUEST_CHUNK_SECONDS", REQUEST_CHUNK_SECONDS)
    REQUEST_TIMEOUT_SECONDS = 180
    REQUEST_TIMEOUT_SECONDS = setting(overrides, "REQUEST_TIMEOUT_SECONDS", REQUEST_TIMEOUT_SECONDS)
    REFRESH_DOWNLOADS = False
    REFRESH_DOWNLOADS = setting(overrides, "REFRESH_DOWNLOADS", REFRESH_DOWNLOADS)
    CHECK_CHANNEL_EPOCHS = True
    CHECK_CHANNEL_EPOCHS = setting(overrides, "CHECK_CHANNEL_EPOCHS", CHECK_CHANNEL_EPOCHS)
    MAX_EVENTS = None  # Optional small test, e.g. 1. None processes every event.
    MAX_EVENTS = setting(overrides, "MAX_EVENTS", MAX_EVENTS)
    MAX_STATIONS_PER_EVENT = None  # Optional test limit; None retains all selected stations.
    MAX_STATIONS_PER_EVENT = setting(overrides, "MAX_STATIONS_PER_EVENT", MAX_STATIONS_PER_EVENT)
    DISTANCE_HISTOGRAM_BIN_DEG = 5.0
    DISTANCE_HISTOGRAM_BIN_DEG = setting(overrides, "DISTANCE_HISTOGRAM_BIN_DEG", DISTANCE_HISTOGRAM_BIN_DEG)
    WAVEFORM_DIR = Path('./waveforms')
    WAVEFORM_DIR = setting(overrides, "WAVEFORM_DIR", WAVEFORM_DIR)
    FIG_DIR = Path('./fig')
    FIG_DIR = setting(overrides, "FIG_DIR", FIG_DIR)
    CACHE_DIR = Path('./data/coda_cache')
    CACHE_DIR = setting(overrides, "CACHE_DIR", CACHE_DIR)
    PROVIDERS = {
        'EARTHSCOPE': 'https://service.earthscope.org/fdsnws/dataselect/1/query',
        'GEOFON': 'https://geofon.gfz.de/fdsnws/dataselect/1/query',
    }
    PROVIDERS = setting(overrides, "PROVIDERS", PROVIDERS)


    if CATALOGUE_FILE is None:
        raise ValueError("CATALOGUE_FILE must be supplied explicitly.")
    CATALOGUE_FILE = Path(CATALOGUE_FILE)
    if STATION_RUN_DIR is None:
        raise ValueError("STATION_RUN_DIR must be supplied explicitly.")
    STATION_RUN_DIR = Path(STATION_RUN_DIR)

    def single_match(paths, name):
        paths = sorted(paths)
        if len(paths) != 1:
            raise ValueError('Set {} explicitly; candidates: {}'.format(name, paths))
        return paths[0]

    if CATALOGUE_FILE is None:
        CATALOGUE_FILE = single_match(Path('./catalogue').glob('*/selected_events.xml'), 'CATALOGUE_FILE')
    if STATION_RUN_DIR is None:
        STATION_RUN_DIR = single_match([p.parent for p in Path('./stations').glob('*/manifest.json')], 'STATION_RUN_DIR')
    CATALOGUE_FILE, STATION_RUN_DIR = Path(CATALOGUE_FILE), Path(STATION_RUN_DIR)
    cat = read_events(str(CATALOGUE_FILE))
    if not cat: raise ValueError('Empty earthquake catalogue.')
    assert 0 <= START_HOURS < END_HOURS
    assert 0 <= AZIMUTH_START_DEG < AZIMUTH_STOP_DEG <= 180
    assert AZIMUTH_BIN_WIDTH_DEG > 0 and REQUEST_CHUNK_SECONDS > 0
    assert 0 < DISTANCE_HISTOGRAM_BIN_DEG <= 180
    assert MAX_EVENTS is None or MAX_EVENTS > 0
    assert MAX_STATIONS_PER_EVENT is None or MAX_STATIONS_PER_EVENT > 0
    az_edges = np.r_[np.arange(AZIMUTH_START_DEG, AZIMUTH_STOP_DEG, AZIMUTH_BIN_WIDTH_DEG), AZIMUTH_STOP_DEG]
    old_manifest = STATION_RUN_DIR / 'manifest.json'
    catalogue_hash = hashlib.sha256(CATALOGUE_FILE.read_bytes()).hexdigest()
    if old_manifest.exists():
        prior = json.loads(old_manifest.read_text())
        expected = prior.get('configuration', {}).get('catalogue_sha256')
        if expected and expected != catalogue_hash:
            raise ValueError('Catalogue differs from the station-selection catalogue. Use matching inputs.')

    from .common import located_origin as event_origin

    from .common import event_slug as slug

    from .common import vector as vector

    from .common import distance_bearing as distance_bearing

    def pair_bin(azimuth):
        from .common import azimuth_bin
        return azimuth_bin(azimuth, az_edges)

    from .common import axial_angle as axial_angle

    def included_angles(lat1,lon1,lat2,lon2,midpoint):
        a,b = vector(lat1,lon1),vector(lat2,lon2)
        n = np.cross(a,b)
        if np.linalg.norm(n)<1e-10: return np.nan,np.nan
        n /= np.linalg.norm(n)
        c = vector(CENTER_LAT,CENTER_LON)
        p,l = np.deg2rad([CENTER_LAT,CENTER_LON])
        north = np.array([-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)])
        east = np.array([-np.sin(l),np.cos(l),0.0])
        tangent = np.cos(np.deg2rad(midpoint))*north+np.sin(np.deg2rad(midpoint))*east
        reference = np.cross(c,tangent)
        dot = float(np.clip(np.dot(reference,n),-1,1))
        unsigned = float(np.rad2deg(np.arccos(abs(dot))))
        if unsigned < 1e-7: return 0.0,0.0
        cross = np.cross(reference,n)
        handedness = np.dot(c,cross)
        if abs(handedness)<1e-12: return np.nan,unsigned
        oriented = np.rad2deg(np.arctan2(np.sign(handedness)*np.linalg.norm(cross),dot))
        return axial_angle(oriented),unsigned

    events = sorted(cat,key=lambda e:event_origin(e).time)
    assert len(set(slug(e) for e in events)) == len(events), 'Duplicate event IDs.'
    if MAX_EVENTS is not None: events = events[:MAX_EVENTS]
    settings = dict(catalogue_sha256=catalogue_hash,station_run=str(STATION_RUN_DIR.resolve()),
        start_hours=START_HOURS,end_hours=END_HOURS,center="event_epicenter",
        azimuth_edges=az_edges.tolist(),channels=list(CHANNEL_PREFERENCE),locations=list(LOCATION_PREFERENCE),
        all_channels=DOWNLOAD_ALL_CHANNELS,epoch_check=CHECK_CHANNEL_EPOCHS,providers=PROVIDERS,
        max_events=MAX_EVENTS,max_stations=MAX_STATIONS_PER_EVENT,
        station_manifest_sha256=hashlib.sha256(old_manifest.read_bytes()).hexdigest() if old_manifest.exists() else None)
    run_id = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
    run_dir,fig_dir = WAVEFORM_DIR/run_id,FIG_DIR/('coda_pairs_'+run_id)
    for p in (run_dir,fig_dir,CACHE_DIR): p.mkdir(parents=True,exist_ok=True)
    (run_dir/'configuration.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print('Events:',len(events),'| coda hours:',START_HOURS,END_HOURS,'| run:',run_id)


    session = requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,
        status_forcelist=[429,500,502,503,504])))

    def epoch_covers(inventory, row, start, end):
        spans=[]; positions=[]
        for net in inventory:
            if net.code!=row['network']: continue
            for sta in net:
                if sta.code!=row['station']: continue
                for ch in sta:
                    if ch.code!=row['channel'] or ch.location_code!=row['location']: continue
                    nodes=(net,sta,ch)
                    lo=max([start]+[x.start_date for x in nodes if x.start_date is not None])
                    hi=min([end]+[x.end_date for x in nodes if x.end_date is not None])
                    if hi>lo:
                        spans.append((lo,hi));positions.append((float(ch.latitude),float(ch.longitude)))
        cursor=start
        for lo,hi in sorted(spans):
            if lo>cursor+1e-6: return False,'metadata_gap',None
            cursor=max(cursor,hi)
        if cursor<end: return False,'metadata_does_not_cover_3_to_10h',None
        if any(distance_bearing(*positions[0],*pos)[0]>.01 for pos in positions):
            return False,'station_coordinates_change',None
        return True,'covered',positions[0]

    def fetch_piece(row, start, end):
        endpoint=PROVIDERS[row['provider']]
        params=dict(network=row['network'],station=row['station'],location=row['location'] or '--',
                    channel=row['channel'],starttime=str(start),endtime=str(end),nodata=204)
        key=hashlib.sha256(json.dumps([endpoint,params],sort_keys=True).encode()).hexdigest()
        path=CACHE_DIR/(key+'.mseed')
        if path.exists() and not REFRESH_DOWNLOADS:
            try: return read(str(path),format='MSEED')
            except Exception: pass  # Invalid cache is replaced only after a successful request.
        response=session.get(endpoint,params=params,timeout=(20,REQUEST_TIMEOUT_SECONDS))
        if response.status_code==204: return Stream()
        response.raise_for_status()
        stream=read(BytesIO(response.content),format='MSEED')
        temp=path.with_suffix('.tmp')
        temp.write_bytes(response.content);temp.replace(path)
        return stream

    def quality(stream, row, start, end):
        stream=stream.select(network=row['network'],station=row['station'],
                             location=row['location'],channel=row['channel']).copy()
        stream.trim(starttime=start,endtime=end,nearest_sample=False,pad=False)
        stream=Stream([tr for tr in stream if tr.stats.npts>0])
        if not stream: return stream,dict(complete=False,reason='no_data',coverage_fraction=0.0)
        # Compute time coverage from the union of sample-support intervals; overlaps do not double count.
        spans=sorted((max(start,tr.stats.starttime),min(end,tr.stats.endtime+tr.stats.delta)) for tr in stream)
        seconds=0.0;cursor=start
        for lo,hi in spans:
            seconds+=max(0.0,float(hi-max(cursor,lo)));cursor=max(cursor,hi)
        fraction=min(1.0,seconds/float(end-start))
        try: stream.merge(method=-1)
        except Exception as exc:
            return stream,dict(complete=False,reason='incompatible_segments: '+str(exc),coverage_fraction=fraction)
        complete=False;reason='gaps_overlaps_or_rate_changes'
        if len(stream)==1:
            tr=stream[0];dt=tr.stats.delta
            finite=np.isfinite(tr.data).all() and not np.ma.is_masked(tr.data)
            complete=bool(finite and tr.stats.starttime<=start+dt+1e-6 and tr.stats.endtime>=end-dt-1e-6)
            reason='complete' if complete else 'nonfinite_or_incomplete_endpoints'
        return stream,dict(complete=complete,reason=reason,coverage_fraction=fraction)

    def preference(row):
        loc=row['location']
        return (CHANNEL_PREFERENCE.index(row['channel']),
            LOCATION_PREFERENCE.index(loc) if loc in LOCATION_PREFERENCE else len(LOCATION_PREFERENCE),loc,row['provider'])

    def write_table(path,rows,columns):
        pd.DataFrame(rows,columns=columns).to_csv(path,index=False)


    log_columns=['event','station_id','provider','network','station','location','channel','starttime','endtime',
                 'status','reason','coverage_fraction','file','chunk_errors']
    station_columns=['station_id','network','station','location','channel','provider','latitude','longitude',
                     'center_azimuth_deg','old_center_azimuth_deg','pair_bin','file']
    event_status=[]
    usable_by_event={}
    for event in events:
        name=slug(event);origin=event_origin(event)
        CENTER_LAT, CENTER_LON = origin.latitude, origin.longitude
        start,end=origin.time+START_HOURS*3600,origin.time+END_HOURS*3600
        source=STATION_RUN_DIR/name
        destination=run_dir/name;destination.mkdir(parents=True,exist_ok=True)
        log=[];usable={}
        channel_path=source/'channels.csv'
        if not channel_path.exists(): channel_path=source/'channel.csv'
        try:
            table=pd.read_csv(channel_path,dtype=str,keep_default_na=False)
            stations=pd.read_csv(source/'stations.csv',dtype=str,keep_default_na=False)
            required={'provider','network','station','location','channel','latitude','longitude'}
            if not required.issubset(table.columns): raise ValueError('Missing channel fields: '+str(required-set(table.columns)))
            inventory=read_inventory(str(source/'stations.xml')) if CHECK_CHANNEL_EPOCHS else None
        except Exception as exc:
            event_status.append(dict(event=name,status='missing_or_invalid_station_inputs',detail=str(exc),usable_stations=0))
            usable_by_event[name]=[]
            continue
        old_bearings=dict(zip(stations.station_id,stations.get('center_azimuth_deg',['']*len(stations))))
        selected_ids=set(stations.station_id)
        table=table[table.channel.isin(CHANNEL_PREFERENCE)].drop_duplicates(['provider','network','station','location','channel'])
        candidates={}
        for row in table.to_dict('records'):
            sid=row['network']+'.'+row['station']
            if sid in selected_ids: candidates.setdefault(sid,[]).append(row)
        station_ids=sorted(candidates)
        if MAX_STATIONS_PER_EVENT is not None: station_ids=station_ids[:MAX_STATIONS_PER_EVENT]
        print(name,'| stations:',len(station_ids),'|',start,'to',end,flush=True)
        for sid in station_ids:
            for row in sorted(candidates[sid],key=preference):
                entry=dict(event=name,station_id=sid,provider=row['provider'],network=row['network'],station=row['station'],
                    location=row['location'],channel=row['channel'],starttime=str(start),endtime=str(end),
                    status='failed',reason='',coverage_fraction=0.0,file='',chunk_errors='')
                try:
                    lat,lon=float(row['latitude']),float(row['longitude'])
                    if CHECK_CHANNEL_EPOCHS:
                        ok,why,position=epoch_covers(inventory,row,start,end)
                        if not ok: raise ValueError(why+'; rerun station selection with the coda window if needed')
                        lat,lon=position
                    radius,az=distance_bearing(CENTER_LAT,CENTER_LON,lat,lon)
                    group=pair_bin(az)
                    if group is None: raise ValueError('outside_azimuth_bins_or_undefined_bearing')
                    stream=Stream();errors=[];t=start
                    while t<end:
                        stop=min(t+REQUEST_CHUNK_SECONDS,end)
                        try: stream+=fetch_piece(row,t,stop)
                        except Exception as exc: errors.append(str(t)+': '+str(exc))
                        t=stop
                    stream,qc=quality(stream,row,start,end)
                    # Any failed chunk remains an explicit acquisition failure, even if neighbouring records overlap.
                    if errors: qc['complete']=False;qc['reason']='one_or_more_chunk_requests_failed'
                    entry.update(status='complete' if qc['complete'] else 'partial',reason=qc['reason'],
                                 coverage_fraction=qc['coverage_fraction'],chunk_errors='; '.join(errors))
                    if stream:
                        folder=destination if qc['complete'] else destination/'partial'
                        folder.mkdir(parents=True,exist_ok=True)
                        filename='{}_{}_{}_{}.mseed'.format(row['station'],row['network'],row['location'] or '--',row['channel'])
                        path=folder/filename;temp=path.with_suffix('.tmp')
                        stream.write(str(temp),format='MSEED')
                        restored=read(str(temp),format='MSEED')
                        assert sum(tr.stats.npts for tr in restored)==sum(tr.stats.npts for tr in stream)
                        temp.replace(path);entry['file']=str(path)
                    if qc['complete'] and sid not in usable:
                        usable[sid]=dict(station_id=sid,network=row['network'],station=row['station'],location=row['location'],
                            channel=row['channel'],provider=row['provider'],latitude=lat,longitude=lon,
                            center_azimuth_deg=az%360.0,old_center_azimuth_deg=old_bearings.get(sid,''),pair_bin=group,file=entry['file'])
                except Exception as exc:
                    entry['reason']=str(exc)
                log.append(entry)
                write_table(destination/'download_log.csv',log,log_columns)
                if entry['status']=='complete' and not DOWNLOAD_ALL_CHANNELS: break
        usable_by_event[name]=[usable[s] for s in sorted(usable)]
        write_table(destination/'usable_stations.csv',usable_by_event[name],station_columns)
        write_table(destination/'download_log.csv',log,log_columns)
        event_status.append(dict(event=name,status='processed',detail='',usable_stations=len(usable)))
        print('  Complete station recordings:',len(usable),flush=True)
    write_table(run_dir/'event_status.csv',event_status,['event','status','detail','usable_stations'])
    print('Waveforms and logs:',run_dir.resolve())


    pair_columns=['station_a','station_b','file_a','file_b','distance_deg','pair_forward_azimuth_deg',
                  'azimuth_a_deg','azimuth_b_deg','bin_midpoint_deg','signed_plane_angle_deg',
                  'unsigned_plane_angle_deg','local_bearing_offset_deg','opposite_sides']
    group_summary=[]
    distance_edges=np.r_[np.arange(0,180,DISTANCE_HISTOGRAM_BIN_DEG),180.0]
    for event in events:
        name=slug(event);rows=usable_by_event.get(name,[])
        origin = event_origin(event)
        CENTER_LAT, CENTER_LON = origin.latitude, origin.longitude
        event_fig=fig_dir/name;event_fig.mkdir(parents=True,exist_ok=True)
        for group in range(len(az_edges)-1):
            members=sorted([r for r in rows if r['pair_bin']==group],key=lambda r:r['station_id'])
            lo,hi=az_edges[group:group+2];mid=(lo+hi)/2
            tag='az_{:g}_{:g}_and_{:g}_{:g}'.format(lo,hi,lo+180,hi+180)
            pair_file=run_dir/name/(tag+'_pairs.csv')
            count=len(members)*(len(members)-1)
            group_summary.append(dict(event=name,group=group,azimuth_start=lo,azimuth_end=hi,
                                      station_count=len(members),pair_count=count,pair_file=str(pair_file)))
            distances=[];angles=[];bearing_offsets=[]
            with pair_file.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=pair_columns);writer.writeheader()
                for a,b in permutations(members,2):
                    distance,bearing=distance_bearing(a['latitude'],a['longitude'],b['latitude'],b['longitude'])
                    signed,unsigned=included_angles(a['latitude'],a['longitude'],b['latitude'],b['longitude'],mid)
                    writer.writerow(dict(station_a=a['station_id'],station_b=b['station_id'],file_a=a['file'],file_b=b['file'],
                        distance_deg=distance,pair_forward_azimuth_deg=bearing,azimuth_a_deg=a['center_azimuth_deg'],
                        azimuth_b_deg=b['center_azimuth_deg'],bin_midpoint_deg=mid,signed_plane_angle_deg=signed,
                        unsigned_plane_angle_deg=unsigned,local_bearing_offset_deg=axial_angle(bearing-mid),
                        opposite_sides=(a['center_azimuth_deg']>=180)!=(b['center_azimuth_deg']>=180)))
                    distances.append(distance);angles.append(signed);bearing_offsets.append(axial_angle(bearing-mid))
            if count==0: continue
            distances=np.asarray(distances);angles=np.asarray(angles)
            title='{} | {}–{}° + {}–{}° | {} stations, {} pairs'.format(name,lo,hi,lo+180,hi+180,len(members),count)
            fig,ax=plt.subplots(figsize=(10,4))
            ax.hist(distances,bins=distance_edges,color='#246c8e',edgecolor='white')
            ax.set(xlim=(0,180),xlabel='Inter-station distance (°)',ylabel='Number of pairs',title=title)
            ax.grid(axis='y',alpha=.2);fig.tight_layout()
            fig.savefig(str(event_fig/(tag+'_distance.png')),dpi=180);plt.close(fig)
            finite=np.isfinite(angles)
            fig,ax=plt.subplots(figsize=(10,5))
            if np.count_nonzero(finite)>20000:
                image=ax.hexbin(distances[finite],angles[finite],gridsize=(90,60),extent=(0,180,-90,90),mincnt=1,cmap='viridis')
                fig.colorbar(image,ax=ax,label='Pairs per hexagon')
            else:
                ax.scatter(distances[finite],angles[finite],s=12,alpha=.55,color='#246c8e',rasterized=True)
            ax.axhline(0,color='gray',linestyle='--',linewidth=.8)
            ax.set(xlim=(0,180),ylim=(-90,90),xlabel='Inter-station distance (°)',
                   ylabel='Signed great-circle plane angle (°)',title=title)
            ax.text(.01,.02,'Undefined signed angles: {}'.format(int((~finite).sum())),transform=ax.transAxes,fontsize=9)
            ax.grid(alpha=.2);fig.tight_layout()
            fig.savefig(str(event_fig/(tag+'_angle_vs_distance.png')),dpi=180);plt.close(fig)
            offsets=np.asarray(bearing_offsets);valid=np.isfinite(offsets)
            fig,ax=plt.subplots(figsize=(10,5))
            if np.count_nonzero(valid)>20000:
                image=ax.hexbin(distances[valid],offsets[valid],gridsize=(90,60),extent=(0,180,-90,90),mincnt=1,cmap='viridis')
                fig.colorbar(image,ax=ax,label='Pairs per hexagon')
            else:
                ax.scatter(distances[valid],offsets[valid],s=12,alpha=.55,color='#327f69',rasterized=True)
            ax.axhline(0,color='gray',linestyle='--',linewidth=.8)
            ax.set(xlim=(0,180),ylim=(-90,90),xlabel='Inter-station distance (°)',
                   ylabel='Pair forward bearing − bin midpoint, folded (°)',title=title)
            ax.grid(alpha=.2);fig.tight_layout()
            fig.savefig(str(event_fig/(tag+'_bearing_offset_vs_distance.png')),dpi=180);plt.close(fig)
    write_table(run_dir/'group_summary.csv',group_summary,['event','group','azimuth_start','azimuth_end','station_count','pair_count','pair_file'])
    print('Pair tables:',run_dir.resolve())
    print('Figures:',fig_dir.resolve())


    return {"run_dir": run_dir}
