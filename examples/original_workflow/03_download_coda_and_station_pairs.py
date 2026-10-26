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

# In[ ]:


# Install once if needed; use package versions compatible with your Python.
# %pip install obspy numpy pandas matplotlib requests


# In[7]:


from pathlib import Path
from io import BytesIO
from itertools import combinations
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

CATALOGUE_FILE = './catalogue/09eacf4ce01d/selected_events.xml'  # Set to ./catalogue/<run_id>/selected_events.xml
STATION_RUN_DIR = './stations/4777cb801a0d'  # Set to ./stations/<run_id> from notebook 02
START_HOURS, END_HOURS = 3.0, 10.0
CENTER_LON, CENTER_LAT = 130.0, 0.0
AZIMUTH_START_DEG, AZIMUTH_STOP_DEG = 0.0, 180.0
AZIMUTH_BIN_WIDTH_DEG = 10.0
CHANNEL_PREFERENCE = ('BHZ', 'HHZ')
LOCATION_PREFERENCE = ('00', '', '10', '01')
DOWNLOAD_ALL_CHANNELS = False  # False: first complete channel per NET.STA; try alternatives on failure
REQUEST_CHUNK_SECONDS = 3600.0
REQUEST_TIMEOUT_SECONDS = 180
REFRESH_DOWNLOADS = False
CHECK_CHANNEL_EPOCHS = True
MAX_EVENTS = None  # Optional small test, e.g. 1. None processes every event.
MAX_STATIONS_PER_EVENT = None  # Optional test limit; None retains all selected stations.
DISTANCE_HISTOGRAM_BIN_DEG = 5.0
WAVEFORM_DIR = Path('./waveforms')
FIG_DIR = Path('./fig')
CACHE_DIR = Path('./data/coda_cache')
PROVIDERS = {
    'EARTHSCOPE': 'https://service.earthscope.org/fdsnws/dataselect/1/query',
    'GEOFON': 'https://geofon.gfz.de/fdsnws/dataselect/1/query',
}


# ## 1 · Resolve inputs and geometry conventions
# 
# Set both input paths explicitly when you have multiple catalogue or station-selection runs. The catalogue hash is checked against notebook 02's manifest. Azimuth edges are generated in `[0,180]` and each sector is paired with its +180° counterpart. `360°` normalizes to `0°`; no station is counted twice.
# 
# All distances are spherical angular distances (0–180°). Each `NET.STA` contributes once per event, regardless of available BHZ/HHZ alternatives. Every combination within a group is used, including pairs on the same side. Radius bands from notebook 02 do not restrict pairing here.
# 
# The reference great circle passes through the fixed centre in the bin-midpoint direction. Let its unit normal be `n_ref` and a station-pair circle's normal be `n_pair`. The **unsigned included angle** is `acos(abs(n_ref·n_pair))` in 0–90°. A signed version uses the intersection hemisphere facing the fixed centre, then folds orientations into `[-90,90)`. This sign is a declared plotting convention; only its magnitude is independent of the convention. The reference circle is unique even though the bin joins opposite directions.
# 
# Colocated or exactly antipodal station pairs have no unique great-circle plane: their distances remain in the table/histogram but their angle is NaN. Pairs with an indeterminate sign have a valid unsigned angle and a NaN signed angle. A separate CSV column gives the forward-bearing offset from the bin midpoint; it is not the same quantity as the plane-intersection angle.

# In[8]:


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
assert -90 < CENTER_LAT < 90 and -180 <= CENTER_LON <= 180
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

def event_origin(e):
    origins = [o for o in e.origins if o.origin_type == 'hypocenter']
    o = origins[0] if origins else e.preferred_origin()
    if o is None or o.time is None: raise ValueError('No timed origin for ' + str(e.resource_id))
    return o

def slug(e):
    ident = str(e.resource_id)
    parts = ident.rstrip('/').split('/')
    label = parts[-2] if parts[-1] == 'event' and len(parts)>1 else parts[-1]
    return re.sub(r'[^A-Za-z0-9_-]', '_', label) + '_' + hashlib.sha256(ident.encode()).hexdigest()[:8]

def vector(lat, lon):
    p,l = np.deg2rad([lat,lon])
    return np.array([np.cos(p)*np.cos(l), np.cos(p)*np.sin(l), np.sin(p)])

def distance_bearing(lat0, lon0, lat1, lon1):
    a,b = vector(lat0,lon0),vector(lat1,lon1)
    distance = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))
    p,q = np.deg2rad([lat0,lat1]); dl = np.deg2rad(lon1-lon0)
    y = np.sin(dl)*np.cos(q)
    x = np.cos(p)*np.sin(q)-np.sin(p)*np.cos(q)*np.cos(dl)
    bearing = np.rad2deg(np.arctan2(y,x)) % 360.0 if np.hypot(x,y)>1e-12 else np.nan
    return float(distance),float(bearing)

def pair_bin(azimuth):
    if not np.isfinite(azimuth): return None
    axial = (azimuth % 360.0) % 180.0
    if not AZIMUTH_START_DEG <= axial < AZIMUTH_STOP_DEG: return None
    return int(np.searchsorted(az_edges,axial,side='right')-1)

def axial_angle(value):
    return float((value+90.0)%180.0-90.0)

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
    start_hours=START_HOURS,end_hours=END_HOURS,center=[CENTER_LON,CENTER_LAT],
    azimuth_edges=az_edges.tolist(),channels=list(CHANNEL_PREFERENCE),locations=list(LOCATION_PREFERENCE),
    all_channels=DOWNLOAD_ALL_CHANNELS,epoch_check=CHECK_CHANNEL_EPOCHS,providers=PROVIDERS,
    max_events=MAX_EVENTS,max_stations=MAX_STATIONS_PER_EVENT,
    station_manifest_sha256=hashlib.sha256(old_manifest.read_bytes()).hexdigest() if old_manifest.exists() else None)
run_id = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
run_dir,fig_dir = WAVEFORM_DIR/run_id,FIG_DIR/('coda_pairs_'+run_id)
for p in (run_dir,fig_dir,CACHE_DIR): p.mkdir(parents=True,exist_ok=True)
(run_dir/'configuration.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
print('Events:',len(events),'| coda hours:',START_HOURS,END_HOURS,'| run:',run_id)


# In[17]:


# ## 2 · Download helpers, epoch and waveform checks
# 
# The cache is keyed by provider, exact NSLC and time interval. Only valid miniSEED responses enter the cache. Empty responses are retried on a later run; `REFRESH_DOWNLOADS=True` also retries cached responses. HTTP errors are recorded per channel, so a failed channel does not terminate other events. No authentication bypass is attempted.
# 
# Instrument response metadata are not rewritten here, avoiding the earlier StationXML serialization issues. Existing StationXML is read only to check channel/station/network dates for the longer coda window. Epoch coverage does not establish waveform completeness: the downloaded data are checked separately.
# 
# Overlap cleanup is conservative (`merge(method=-1)`): only compatible contiguous or identical-overlap traces are combined. Conflicting overlaps, gaps, rate changes, nonfinite data and incomplete endpoints prevent a channel from entering geometry analysis. A complete alternate channel can still be used. Raw partial traces are retained in `partial/`, with their status in `download_log.csv`.

# In[9]:


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


# ## 3 · Event loop: download vertical coda
# 
# Both `channels.csv` (the original filename) and `channel.csv` are accepted. A station is grouped by its **geographic** bearing from the fixed centre, recomputed from the chosen recording's channel coordinates. The old geographic bearing is retained for comparison. Channel preference is BHZ then HHZ; a failing preferred channel triggers the next available channel/location.
# 
# Each event writes `download_log.csv`, `usable_stations.csv` and miniSEED files. Download logs are saved after every attempt. Cached time chunks let an interrupted run resume. Existing files from previous attempts are not deleted; the current log and usable-station table define which files belong to the current result.

# In[19]:


log_columns=['event','station_id','provider','network','station','location','channel','starttime','endtime',
             'status','reason','coverage_fraction','file','chunk_errors']
station_columns=['station_id','network','station','location','channel','provider','latitude','longitude',
                 'center_azimuth_deg','old_center_azimuth_deg','pair_bin','file']
event_status=[]
usable_by_event={}
events.pop(2)
for event in events:
    name=slug(event);origin=event_origin(event)
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


# ## 4 · Every station pair within each opposite-azimuth group
# 
# For a group with N usable stations, this produces N(N−1)/2 unordered pairs. Pair rows are streamed to CSV to avoid storing all pair metadata in memory. Three figures per populated group are saved to `./fig/coda_pairs_<run_id>/<event>/`:
# 
# 1. Inter-station distance histogram, x-axis **0–180°**.
# 2. Signed great-circle included angle versus distance, axes **0–180°** and **−90–90°**.
# 3. Station-pair forward-bearing offset from the bin midpoint versus distance, with the same axis limits. This implements the direct azimuth-subtraction interpretation as a separately labeled quantity; it uses the forward bearing from alphabetically first station to second and therefore depends on that endpoint convention.
# 
# Small groups use a scatter plot; large groups use a hexbin density plot containing all finite-angle pairs. No pair subsampling is applied. Zero/one-station groups remain in `group_summary.csv` but do not produce empty figures. The pair CSV includes station identifiers, files, both geographic bearings, signed/unsigned plane angles, local forward-bearing offset and same/opposite-side classification.

# In[20]:


pair_columns=['station_a','station_b','file_a','file_b','distance_deg','pair_forward_azimuth_deg',
              'azimuth_a_deg','azimuth_b_deg','bin_midpoint_deg','signed_plane_angle_deg',
              'unsigned_plane_angle_deg','local_bearing_offset_deg','opposite_sides']
group_summary=[]
distance_edges=np.r_[np.arange(0,180,DISTANCE_HISTOGRAM_BIN_DEG),180.0]
for event in events:
    name=slug(event);rows=usable_by_event.get(name,[])
    event_fig=fig_dir/name;event_fig.mkdir(parents=True,exist_ok=True)
    for group in range(len(az_edges)-1):
        members=sorted([r for r in rows if r['pair_bin']==group],key=lambda r:r['station_id'])
        lo,hi=az_edges[group:group+2];mid=(lo+hi)/2
        tag='az_{:g}_{:g}_and_{:g}_{:g}'.format(lo,hi,lo+180,hi+180)
        pair_file=run_dir/name/(tag+'_pairs.csv')
        count=len(members)*(len(members)-1)//2
        group_summary.append(dict(event=name,group=group,azimuth_start=lo,azimuth_end=hi,
                                  station_count=len(members),pair_count=count,pair_file=str(pair_file)))
        distances=[];angles=[];bearing_offsets=[]
        with pair_file.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=pair_columns);writer.writeheader()
            for a,b in combinations(members,2):
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


# ## Interpretation and next step
# 
# These plots describe the geometry of successfully acquired coda recordings. Geometry does not establish usable signal-to-noise, common instrument response, timing accuracy or freedom from other earthquakes inside the 3–10 h window. Those checks belong in the preprocessing/correlation stage.
# 
# The signed-plane angle is an orientation convention, not a directed propagation angle. Pair order is deterministic (`NET.STA` alphabetical); the axial folding makes swapping stations equivalent except for the ±90° boundary, represented as −90°.
# 
# **Service reference:** [EarthScope dataselect documentation](https://service.earthscope.org/fdsnws/dataselect/1/) documents the HTTPS endpoint, NSLC/time requests, and retired request parameters. This notebook uses only basic supported query parameters. Blank locations are sent as `--` and preserved as strings in CSV. MiniSEED files do not embed the full station response; retain notebook 02 metadata and obtain response-level metadata before instrument correction.
