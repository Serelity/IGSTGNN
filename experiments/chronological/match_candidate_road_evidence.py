"""Offline, exploratory crosswalk for three Contra road candidates; never certifies physics.

Input: existing train-only candidate inventory, public station metadata, staged
Caltrans Esri JSON (XYM retained), 2023 ramp workbook and HPMS event tables.
Requires geopandas, shapely, pyproj, openpyxl. No training or network operations.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

os.environ['PROJ_NETWORK'] = 'OFF'
import geopandas as gpd
import openpyxl
import pandas as pd
import pyproj
import shapely
from shapely.geometry import LineString, Point

GUARD_M = 50.0  # Sensitivity buffer, not a statistical confidence interval.
DATE = datetime(2023, 7, 10, tzinfo=timezone.utc).timestamp() * 1000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def select_candidates(rows, count=3):
    """One pair per numbered road, selected without examining new geometry or Y."""
    eligible = [r for r in rows if r['metadata_prefilter_pass'] == 'True'
                and int(r['both_report_supported_train_windows']) > 0]
    eligible.sort(key=lambda r: (-int(r['both_report_supported_train_windows']),
                  -float(r['both_valid_unique_train_x_fraction']),
                  float(r['postmile_difference']), r['road'],
                  r['low_postmile_station_id'], r['high_postmile_station_id']))
    selected, used = [], set()
    for r in eligible:
        road = r['road'].rsplit('-', 1)[0]
        if road not in used:
            selected.append(r)
            used.add(road)
        if len(selected) == count:
            break
    require(len(selected) == count, 'Insufficient distinct supported roads')
    return selected


def project_xym(point, geometry):
    """Project XY in EPSG:3310; interpolate M explicitly, never as a Z coordinate."""
    best = None
    px, py = point.x, point.y
    for pi, path in enumerate(geometry['paths']):
        chain = 0.0
        for vi, (a, b) in enumerate(zip(path, path[1:])):
            require(len(a) == len(b) == 3 and all(
                v is not None and math.isfinite(v) for v in a + b), 'Invalid XYM')
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            t = max(0., min(1., ((px-a[0])*dx+(py-a[1])*dy)/length**2))
            x, y = a[0]+t*dx, a[1]+t*dy
            distance = math.hypot(px-x, py-y)
            record = dict(distance_m=distance, measure=a[2]+t*(b[2]-a[2]),
                          chainage_m=chain+t*length, path_index=pi,
                          vertex_index=vi, x=x, y=y)
            if best is None or distance < best['distance_m']:
                best = record
            chain += length
    require(best is not None, 'Empty projected route')
    return best


def direction_of_ramp(description):
    # FIRST roadway direction: "WBONFR NB WILLOW..." is a WB ramp, not NB.
    found = re.search(r'(?:^|[^A-Z])(NB|SB|EB|WB)', description.upper())
    return found.group(1) if found else None


def interpolate_bracket(anchors, target, key, value):
    """Only interpolate within one prefix/suffix/direction/path, never extrapolate."""
    groups = {}
    for a in anchors:
        groups.setdefault((a['pm_route_id'], a['path_index']), []).append(a)
    fits = []
    for group in groups.values():
        group.sort(key=lambda a: a[key])
        for left, right in zip(group, group[1:]):
            delta = right[key] - left[key]
            if delta <= 0 or not left[key] <= target <= right[key]:
                continue
            if abs(right['pm']-left['pm']) > .111 or abs(
                    right['chainage_m']-left['chainage_m']) > 220:
                continue
            t = (target-left[key])/delta
            fits.append(dict(value=left[value]+t*(right[value]-left[value]),
                             anchor_ids=[left['objectid'], right['objectid']],
                             pm_route_id=left['pm_route_id'],
                             prefix=left['prefix'], suffix=left['suffix'],
                             path_index=left['path_index']))
    # At a shared anchor there can be two equivalent bracketing intervals.
    require(bool(fits), 'No local same-prefix bracket')
    require(len({f['pm_route_id'] for f in fits}) == 1 and
            max(f['value'] for f in fits)-min(f['value'] for f in fits) < 1e-5,
            'Ambiguous postmile bracket')
    return fits[0]


def overlapping_events(rows, route_id, low, high, item):
    out = []
    for r in rows:
        if r['routeid'] != route_id or r['dataitem'] != item:
            continue
        a, b = float(r['beginpoint']), float(r['endpoint'])
        require(b > a, 'Non-positive event interval')
        if min(b, high) > max(a, low):
            out.append(dict(r, clipped_begin=max(a, low), clipped_end=min(b, high)))
    return sorted(out, key=lambda r: float(r['beginpoint']))


def interval_coverage(rows, low, high):
    end, length = low, 0.
    for r in rows:
        a, b = max(low, float(r['beginpoint'])), min(high, float(r['endpoint']))
        if b > max(a, end):
            length += b-max(a, end)
        end = max(end, b)
    return length/(high-low)


def audit_esri(obj, name):
    require(not obj.get('exceededTransferLimit'), name + ' is truncated')
    require(obj['spatialReference']['wkid'] == 3310, 'Expected explicit EPSG:3310')
    geoms, m_count = [], 0
    for f in obj['features']:
        g = f['geometry']
        if 'paths' in g:
            require(obj.get('hasM') is True and obj.get('hasZ') is False,
                    'Expected explicit XYM encoding')
            for path in g['paths']:
                require(all(len(p) == 3 and all(v is not None and math.isfinite(v)
                            for v in p) for p in path), 'Invalid coordinate')
                geoms.append(LineString([p[:2] for p in path]))
                m_count += len(path)
        else:
            geoms.append(Point(g['x'], g['y']))
    series = gpd.GeoSeries(geoms, crs='EPSG:3310')
    report = dict(features=len(obj['features']), xy_geometries=len(series),
                  null=int(series.isna().sum()), empty=int(series.is_empty.sum()),
                  invalid=int((~series.is_valid).sum()), m_vertices=m_count,
                  types=series.geom_type.value_counts().to_dict(), crs=series.crs.to_string())
    require(not (report['null'] or report['empty'] or report['invalid']), 'Geometry audit failed')
    return report


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    geo, supplement = Path(args.geometry_dir), Path(args.supplement_dir)
    input_paths = [Path(args.inventory), Path(args.stations),
        geo/'odometer_routes.json', geo/'nhs_routes.json', geo/'postmile_points.json',
        geo/'hpms_small_context.json', supplement/'hpms_2023_candidate_route_lanes.json',
        supplement/'caltrans_2023_d4_ramp_adt.xlsx']
    rows = list(csv.DictReader(input_paths[0].open(encoding='utf-8-sig')))
    selected = select_candidates(rows)
    write_json(out/'selected_candidates.json', selected)
    all_stations = list(csv.DictReader(input_paths[1].open(encoding='utf-8-sig'), delimiter='\t'))
    stations = {s['station_id']:s for s in all_stations}
    require(len(stations) == len(all_stations), 'Duplicate station IDs')
    odo, nhs, pms = [load(p) for p in input_paths[2:5]]
    audit = {name:audit_esri(obj, name) for name,obj in [('odometer',odo),('nhs',nhs),('postmile',pms)]}
    context = load(input_paths[5])
    lane_rows = load(input_paths[6])['records']
    require(all(r['datayear']=='2023' and r['stateid']=='6' for r in context+lane_rows), 'Wrong HPMS scope')
    workbook = openpyxl.load_workbook(input_paths[7], read_only=True, data_only=True, keep_links=False)
    it = iter(workbook['D4 Ramp ADT 2023'].values)
    header = next(it)
    ramps = [dict(zip(header, r), excel_row=i) for i,r in enumerate(it,2)]
    workbook.close()
    # Public station lat/lon datum is not explicitly documented in the source.
    # WGS84 is an explicit working assumption, not source-level certification.
    tx = pyproj.Transformer.from_crs(4326, 3310, always_xy=True)
    matches, endpoint_rows, ramp_rows = [], [], []
    for candidate in selected:
        pairid = candidate['road']+'_'+candidate['low_postmile_station_id']+'_'+candidate['high_postmile_station_id']
        srows = [stations[candidate[k]] for k in ['low_postmile_station_id','high_postmile_station_id']]
        number = int(srows[0]['Fwy'])
        direction = candidate['direction']+'B'
        align = 'L' if direction in ['WB','SB'] else 'R'
        route_id = f'SHS_{number:03d}._P'
        route_options = [f for f in odo['features'] if f['attributes']['RouteId']==f'{number:03d}._{align}'
            and f['attributes']['LRSFromDate'] <= DATE
            and (f['attributes']['LRSToDate'] is None or f['attributes']['LRSToDate'] > DATE)]
        require(len(route_options)==1, 'Ambiguous time-valid carriageway')
        route = route_options[0]
        anchors = []
        for f in pms['features']:
            a,g=f['attributes'],f['geometry']
            if a['Route']!=number or a['County']!='CC' or a['Direction']!=direction:
                continue
            pp=project_xym(Point(g['x'],g['y']),route['geometry'])
            if pp['distance_m'] > 10:  # Reject anchors off the chosen historical alignment.
                continue
            anchors.append(dict(pp, objectid=a['OBJECTID'],pm_route_id=a['PMRouteID'],
                                pm=a['PM'],prefix=a['PMPrefix'],suffix=a['PMSuffix']))
        ends=[]
        for s in srows:
            require(s['Fwy Name']==candidate['road'] and s['Direction']==candidate['direction'], 'Station identity mismatch')
            point=Point(tx.transform(float(s['Lng']),float(s['Lat'])))
            pos=project_xym(point,route['geometry'])
            require(pos['distance_m'] < 25, 'Station too far from directional geometry')
            independent=LineString([v[:2] for v in route['geometry']['paths'][pos['path_index']]])
            require(abs(independent.distance(point)-pos['distance_m'])<1e-6 and
                    abs(independent.project(point)-pos['chainage_m'])<1e-6,
                    'Independent GEOS projection disagrees')
            pm=interpolate_bracket([a for a in anchors if a['path_index']==pos['path_index']],
                                   pos['chainage_m'],'chainage_m','pm')
            nhs_fits=sorted([(project_xym(point,f['geometry']), f) for f in nhs['features']
                            if f['attributes']['RouteID']==route_id],key=lambda x:x[0]['distance_m'])
            require(nhs_fits and nhs_fits[0][0]['distance_m']<100, 'No nearby NHS route')
            require(len(nhs_fits)==1 or nhs_fits[1][0]['distance_m']-nhs_fits[0][0]['distance_m']>10,
                    'Ambiguous NHS route segment')
            fit,nf=nhs_fits[0]
            require(nf['attributes']['FromARMeasure']-1e-6<=fit['measure']<=nf['attributes']['ToARMeasure']+1e-6,
                    'NHS measure outside attribute limits')
            e=dict(pair_id=pairid,station_id=s['station_id'],name=s['Name'],direction=direction,
                   lat=float(s['Lat']),lon=float(s['Lng']),pems_abs_pm=float(s['Abs PM']),
                   road=number,odometer_objectid=route['attributes']['OBJECTID'],
                   odometer_route_id=route['attributes']['RouteId'],directional_projection=pos,
                   county_postmile=pm,nhs_objectid=nf['attributes']['OBJECTID'],
                   hpms_route_id=route_id,nhs_projection=fit)
            ends.append(e)
            endpoint_rows.append(e)
        require(ends[0]['directional_projection']['path_index']==ends[1]['directional_projection']['path_index'], 'Pair crosses paths')
        require(ends[0]['county_postmile']['pm_route_id']==ends[1]['county_postmile']['pm_route_id'], 'Pair crosses PM reset')
        require(ends[0]['nhs_objectid']==ends[1]['nhs_objectid'], 'Pair spans NHS features')
        low,high=sorted(e['nhs_projection']['measure'] for e in ends)
        lane_hits=overlapping_events(lane_rows,route_id,low,high,'THROUGH_LANES')
        require(abs(interval_coverage(lane_hits,low,high)-1)<1e-8, 'HPMS coverage gap')
        require(abs(sum(r['clipped_end']-r['clipped_begin'] for r in lane_hits)/(high-low)-1)<1e-8,
                'Overlapping HPMS lane events require review')
        items={item:overlapping_events(context,route_id,low,high,item)
               for item in ['FACILITY_TYPE','PEAK_LANES','COUNTER_PEAK_LANES','DIR_THROUGH_LANES']}
        require(items['FACILITY_TYPE'] and all(float(r['valuenumeric'])==2 for r in items['FACILITY_TYPE']), 'Facility semantics need review')
        require(abs(interval_coverage(items['FACILITY_TYPE'],low,high)-1)<1e-8, 'Facility type coverage gap')
        start,end=sorted(e['directional_projection']['chainage_m'] for e in ends)
        travel_ends=ends[::-1] if direction in ['WB','SB'] else ends
        same_direction_ramps=[r for r in ramps if r['CNTY']=='CC' and int(r['RTE'])==number
                              and direction_of_ramp(r['LOCATION_DESC'])==direction]
        mapped_ramps, unmapped_ramps=[],[]
        for r in same_direction_ramps:
            compatible=[a for a in anchors if a['prefix']==(r['PM_PFX'] or '')
                        and a['suffix']==(r['PM_SFX'] or '')
                        and a['path_index']==ends[0]['directional_projection']['path_index']]
            rr=dict(excel_row=r['excel_row'],description=r['LOCATION_DESC'],
                    prefix=r['PM_PFX'] or '',postmile=r['POSTMILE'],suffix=r['PM_SFX'] or '',
                    direction=direction,adt_2023=r['YR_2023'])
            try:
                m=interpolate_bracket(compatible,float(r['POSTMILE']),'pm','chainage_m')
                chain=m['value']
                relation='interior' if start<chain<end else 'outside'
                distance=max(start-chain,chain-end,0.)
                rr.update(chainage_m=chain,anchor_ids=m['anchor_ids'],relation=relation,
                          distance_to_pair_m=distance,within_50m_guard=distance<=GUARD_M)
                mapped_ramps.append(rr)
            except ValueError as error:
                unmapped_ramps.append(dict(rr,reason=str(error)))
        nearby=sorted([r for r in mapped_ramps if r['distance_to_pair_m']<=1609.344],
                      key=lambda r:r['chainage_m'])
        ramp_rows.extend([dict(r,pair_id=pairid) for r in nearby])
        full_meta_near=[dict(station_id=s['station_id'],name=s['Name'],type=s['Type'],abs_pm=s['Abs PM'])
            for s in all_stations if s['Fwy Name']==candidate['road'] and s['Type']!='Mainline'
            and float(candidate['low_postmile'])-1<=float(s['Abs PM'])<=float(candidate['high_postmile'])+1]
        matches.append(dict(pair_id=pairid,road=candidate['road'],route_number=number,
            train_report_supported_windows=int(candidate['both_report_supported_train_windows']),
            train_x_valid_fraction=float(candidate['both_valid_unique_train_x_fraction']),
            upstream_station=travel_ends[0]['station_id'],downstream_station=travel_ends[1]['station_id'],
            directional_geometry_length_m=end-start,
            county_postmile_range=sorted(e['county_postmile']['value'] for e in ends),
            county_postmile_prefix=ends[0]['county_postmile']['prefix'],
            hpms_measure_range=[low,high],through_lanes_both_directions=lane_hits,
            hpms_context=items,hpms_coverage=interval_coverage(lane_hits,low,high),
            lane_values_with_50m_measure_guard=sorted({float(r['valuenumeric']) for r in
                overlapping_events(lane_rows,route_id,low-GUARD_M/1609.344,high+GUARD_M/1609.344,'THROUGH_LANES')}),
            ramp_adt_same_direction_total=len(same_direction_ramps),
            ramp_adt_mapped_total=len(mapped_ramps),ramp_adt_unmapped=unmapped_ramps,
            nearby_ramp_adt=nearby,
            ramp_adt_interior_count=sum(r['relation']=='interior' for r in mapped_ramps),
            ramp_adt_guard_count=sum(r['within_50m_guard'] for r in mapped_ramps),
            full_source_nearby_nonmainline=full_meta_near,
            odometer_validity={k:route['attributes'][k] for k in ['LRSFromDate','LRSToDate']},
            matching_status='EXPLORATORY_SPATIAL_CROSSWALK_COMPLETE',
            closed_boundary_certified=False,directional_lanes_certified=False,
            physical_contract_ready=False))
    summary=dict(status='THREE_ROAD_STATIC_MATCH_COMPLETE_NOT_PHYSICS_CERTIFIED',
        selection_rule='One pair per numbered road: descending train report support, valid fraction, then ascending spacing and identifiers; top 3 roads.',
        inventory_rows=len(rows),prefilter_pass_rows=sum(r['metadata_prefilter_pass']=='True' for r in rows),
        selected_pairs=len(matches),matched_endpoints=len(endpoint_rows),
        versions={k.__name__:k.__version__ for k in [gpd,pd,shapely,pyproj,openpyxl]},
        source_hashes=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in input_paths],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coordinate_policy=dict(station_crs_assumed='EPSG:4326; undocumented source datum',
             output_crs='EPSG:3310',always_xy=True,proj_network=False,geometry_repair=False,
             m_policy='Original XYM retained; explicit per-edge interpolation; XY-only geometry for metric distance.'),
        source_vintages=dict(ramp_adt=2023,hpms_events=2023,nhs_lrs_export='2023-07-18',
             postmile_geometry='TSN extraction 2025-06-08 (not a 2023 snapshot)',
             odometer='Retrieved live service, filter validity on 2023-07-10; edit timestamps are later.'),
        limits=['Road membership is supported; exact 2023 ramp merge/diverge locations remain unverified.',
                'Ramp ADT inventory is not exhaustive and annual ADT is not five-minute ramp flow.',
                'THROUGH_LANES is two-way here; peak/counterpeak does not identify compass direction.',
                'No lane count division by two; no capacity or queue labels derived.',
                'Training report support counts windows, not unique crashes or certified bottlenecks.',
                '50 m guard is a sensitivity check, not source accuracy certification.',
                'No X values, validation targets, or test targets loaded; no training performed.'],
        geometry_audit=audit,matches=matches,endpoints=endpoint_rows)
    summary['independent_checks'] = dict(geos_projection_endpoints_passed=len(endpoint_rows),
        geos_projection_tolerance_m=1e-6,positive_overlap_full_coverage_no_lane_overlap=True)
    write_json(out/'road_match_evidence.json',summary)
    for filename,data in [('endpoint_matches.csv',endpoint_rows),('nearby_ramps.csv',ramp_rows)]:
        flat=[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in data]
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]) if flat else ['pair_id'])
            writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:summary[k] for k in ['status','selected_pairs','matched_endpoints']},indent=2))
    for m in matches:
        print(m['road'],m['upstream_station'],'->',m['downstream_station'],round(m['directional_geometry_length_m'],1),
              m['county_postmile_range'],'ramp interior/guard',m['ramp_adt_interior_count'],m['ramp_adt_guard_count'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['inventory','stations','geometry-dir','supplement-dir','output-dir']:
        parser.add_argument('--'+flag,required=True)
    run(parser.parse_args())
