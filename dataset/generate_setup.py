import json
import time
import os
import pyproj
import matplotlib.pyplot as plt
from requests import HTTPError
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import osmnx
import contextily as cx
from insite_parser import write_city, write_terrain, write_txrx, write_setup
import warnings
warnings.filterwarnings("ignore", message="DeprecationWarning: In a future version")

import argparse

wgs84 = 'wgs84'
albers = 'USA_Contiguous_Albers_Equal_Area_Conic_USGS_version'
rng = np.random.default_rng(seed=42)
HEIGHT_PER_LEVEL=14.2 #TODO this number is probably inaccurate
MAX_HEIGHT=95.

def calc_h(row):
    #TODO check units 
    if 'height' in row.keys() and pd.notna(row.height):
        if type(row.height) == str and row.height.isnumeric():
            h = pd.to_numeric(row.height)
        else:
            h = float(''.join(c for c in row.height if c.isnumeric()))
    elif 'building:levels' in row.keys() and pd.notna(row['building:levels']):
        try:
            h = pd.to_numeric(row['building:levels'])*HEIGHT_PER_LEVEL
        except ValueError:
            print('weird `building:levels` entry')
            h = 1 * HEIGHT_PER_LEVEL
    else:
        h = 1 * HEIGHT_PER_LEVEL
    
    if h == 0.:
        h = 1 * HEIGHT_PER_LEVEL

    return min(h, MAX_HEIGHT)

def construct_scene(args):
    # set the original origin
    lon0, lat0 = args.origin
    origin = shapely.geometry.Point(lon0, lat0)
    df_origin = gpd.GeoDataFrame(data={'name': ['origin']},          geometry=[origin],          crs=wgs84).to_crs(albers)

    # translate the origin
    dx, dy = args.offset
    df_origin = gpd.GeoDataFrame(
        data={'name': ['origin']}, 
        geometry=df_origin.translate(dx, dy),
    )
    lon0, lat0 = df_origin.to_crs(wgs84).iloc[0].geometry.x, df_origin.to_crs(wgs84).iloc[0].geometry.y
    args.lon, args.lat = lon0, lat0


    # create buildings df
    side_length = args.side_length
    rect = [[-side_length, -side_length], [-side_length, side_length], [side_length, side_length], [side_length, -side_length]]
    test_area = shapely.geometry.Polygon(
        np.array(df_origin.loc[0].geometry.xy).T + rect)
    df_test_area = gpd.GeoDataFrame(
        {'name': ['test_area']}, geometry=[test_area], crs=albers)

    # do the same for the terrain
    rect = [[-1.5*side_length, -1.5*side_length], [-1.5*side_length, 1.5*side_length],
            [1.5*side_length, 1.5*side_length], [1.5*side_length, -1.5*side_length]]
    terrain_area = shapely.geometry.Polygon(
        np.array(df_origin.loc[0].geometry.xy).T + rect)
    df_terrain = gpd.GeoDataFrame({'name': ['terrain']}, geometry=[           terrain_area], crs=albers)


    #   # assign heights to buildings
    #   # clip buildings to test area
    df_buildings = osmnx.features_from_point(
        (lat0, lon0), dist=1.1*side_length, tags={'building': True, 'element_type': 'relation'})
    df_buildings = gpd.clip(df_buildings.to_crs(
        albers), test_area)  # clip to test area
    df_buildings['calc_h'] = df_buildings.apply(calc_h, axis=1)  # add height
    df_buildings = df_buildings[df_buildings.type ==         'Polygon']  # remove non polygons

    # create transcievers
    #   # create receive grid
    #   #   # create coord lists
    x0 = df_origin[df_origin.name == 'origin'].geometry.x
    y0 = df_origin[df_origin.name == 'origin'].geometry.y

    spacing_air = args.spacing_air
    N_air = int(2*side_length//spacing_air)  # number of points
    offset = 2*side_length/(2*N_air)  # offset from boundary
    args.offset = offset

    x_air = np.linspace(x0-side_length+offset, x0+side_length-offset, N_air)
    y_air = np.linspace(y0-side_length+offset, y0+side_length-offset, N_air)

    #   #   # create mesh
    X_air, Y_air = np.meshgrid(x_air, y_air)

    #   #   # create df
    h_air = args.h_air
    df_air = gpd.GeoDataFrame(
        {'name': [f'air_{i:04d}' for i in range(X_air.size)]},
        crs=albers,
        geometry=[shapely.geometry.Point((x, y, h_air))
                  for x, y in zip(X_air.flat, Y_air.flat)]
    )
    #   #   # calculate rotation
    geod = pyproj.crs.CRS(wgs84).get_geod()
    sw, se = df_air.iloc[[0, int(2*side_length/spacing_air-1)]].to_crs(wgs84).geometry
    sw.x, se.x, sw.y, se.y
    fwd_az, _, _ = geod.inv(sw.x, sw.y, se.x, se.y)
    rot = (90-fwd_az)
    args.rot = rot

    #   # create tx UE points
    N_ue = args.N_ue
    h_ue = args.h_ue
    spacing_ue = args.spacing_ue

    #   #   # compute valid ground (4 meter outside all buildings)
    valid_ground = (test_area.buffer(-4) - df_buildings.buffer(4).unary_union)
    minx, miny, maxx, maxy = valid_ground.bounds

    #   #   # build up points list
    ue_points = []
    max_attempts = 10e3
    attempts = 0
    while len(ue_points) < N_ue:
        attempts += 1
        pnt = shapely.Point(rng.uniform(minx, maxx),     rng.uniform(miny, maxy), h_ue)
        if valid_ground.contains(pnt) and pnt not in set(ue_points):
            min_d = gpd.GeoSeries(ue_points).distance(pnt).min()
            if min_d < spacing_ue:
                continue
            ue_points.append(pnt)
        if attempts > max_attempts:
            raise Exception("could not place {N_ue} UEs at {spaccing_ue}m spacing in {max_attempts} attempts: I give up.")
            
    #   #   # create df
    df_ue = gpd.GeoDataFrame(
        {'name': [f'ue_{i:04d}' for i in range(len(ue_points))]},
        crs=albers,
        geometry=ue_points
    )

    return {
        'origin': df_origin,
        'test_area': df_test_area,
        'terrain': df_terrain,
        'buildings': df_buildings,
        'air': df_air,
        'ue': df_ue
    }, args


def _generate_setup(args, show=False):
    project_dir = args.project_dir

    # make directories
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
    insite_dir = os.path.join(project_dir, 'insite')
    if not os.path.exists(insite_dir):
        os.makedirs(insite_dir, exist_ok=True)
    df_dir = os.path.join(project_dir, 'dfs')
    if not os.path.exists(df_dir):
        os.makedirs(df_dir, exist_ok=True)

    if os.path.exists(os.path.join(project_dir, 'scene.png')): 
        print('scene already exists')
        return

    dfs, args = construct_scene(args)
    
    # save data and config
    for df_name, df in dfs.items():
            df.to_csv(os.path.join(df_dir, f'{df_name}.csv'))
    with open(os.path.join(project_dir, 'config.json'), 'w', newline='\r\n') as f:
        json.dump(vars(args), f)

    # plot and save scene
    fig, ax = plt.subplots(figsize=(8, 8))

    dfs['origin'].to_crs(wgs84).plot(ax=ax, color='k', label='origin')
    dfs['test_area'].to_crs(wgs84).plot(ax=ax, color='b', alpha=0.35)
    dfs['terrain'].to_crs(wgs84).plot(ax=ax, color='g', alpha=0.2)
    dfs['buildings'].to_crs(wgs84).plot(ax=ax, color='b', alpha=0.5)
    dfs['air'].to_crs(wgs84).plot(ax=ax, color='r', marker='.')
    dfs['ue'].to_crs(wgs84).plot(ax=ax, color='y', marker='o')

    try:
        cx.add_basemap(ax, crs=wgs84)
    except HTTPError:
        print('could not find the basemap')
    plt.title(args.name)
    if show:
        plt.show()
    
    
    fig.savefig(os.path.join(project_dir, 'scene.png'))
    plt.close(fig)
    
    # write files
    #   # write .ter
    terrain = write_terrain(args, dfs['terrain'])
    with open(os.path.join(insite_dir, f'{args.name}.ter'), 'w', newline='\r\n') as f:
        f.writelines('Format type:keyword version: 1.1.0\n')
        f.writelines(terrain)
    
    #   # write .city
    city = write_city(args, dfs['buildings'])
    with open(os.path.join(insite_dir, f'{args.name}.city'), 'w', newline='\r\n') as f:
        f.writelines('Format type:keyword version: 1.1.0\n')
        f.writelines(city)

    #   # write .txrx
    txrx = write_txrx(args, dfs['ue'], dfs['air'])
    with open(os.path.join(insite_dir, f'{args.name}.txrx'), 'w', newline='\r\n') as f:
        f.writelines(txrx)

    #   # write .setup
    setup = write_setup(args, dfs['terrain'])
    with open(os.path.join(insite_dir, f'{args.name}.setup'), 'w', newline='\r\n') as f:
        f.writelines('Format type:keyword version: 1.1.0\n')
        f.writelines(setup)

def make_subgrids(args, show=False):
    delta = [-2.1*args.side_length, 0, 2.1*args.side_length]
    ex = 0
    for dx in delta:
        for dy in delta:
            ex+=1 
            print(ex)
            args.offset = (dx, dy) 
            
            args.project_dir = os.path.join(args.base_dir, f'ha{args.h_air}_hu{args.h_ue}', args.name, f'ex{ex:02d}')
            _generate_setup(args, show=show)

            time.sleep(0.5)

def read_places_file(places_file):
    with open(places_file, 'r') as f:
        lines = f.readlines()
    
    places = {}
    for line in lines[1:]:
        place, lon, lat = line.strip().split(',')
        if '//' in place:continue
        places[place.strip()] = (float(lon), float(lat))
    return places

def generate_setup(args):
    if args.places:
        places = read_places_file(args.places)
    else:
        places = {args.name: args.origin}
    
    for place, coords in places.items():
        args.name=place
        args.origin=coords
    
        if args.do_grid:
            make_subgrids(args, show=args.show_plot)
        else:
            args.project_dir = os.path.join(args.base_dir, f'ha{args.h_air}_hu{args.h_ue}', args.name)
            args.offset = (0, 0)   
            _generate_setup(args, show=args.show_plot)

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Description of your script')
    place_group = parser.add_mutually_exclusive_group(required=True)
    place_group.add_argument('--origin', type=float, nargs=2, default=None, help='Origin of the AUT (area under test) in (long, lat) format')
    place_group.add_argument('--places', type=str, default=None, help='places file to read origins from')
    parser.add_argument('-n', '--name', type=str, default='test', help='Name of the project (default: test)')
    parser.add_argument('--side_length', type=float, default=256., help='1/2 Side length of the AUT (area under test) in meters (default: 256 m)')
    parser.add_argument('--h_air', type=float, default=100., help='Height of the UAV in meters above the ground (default: 100 m)')
    parser.add_argument('--spacing_air', type=float, default=32., help='Spacing between UAV grid points in meters (default: 32 m)')
    parser.add_argument('--N_ue', type=int, default=15, help='Number of UEs in the simulation (default: 15)')
    parser.add_argument('--h_ue', type=float, default=2., help='Height of the UEs in meters above the ground (default: 2 m)')
    parser.add_argument('--spacing_ue', type=float, default=16., help='Minimum spacing for UEs (default 16 m)')
    parser.add_argument('--ptx', type=float, default=30., help='Transmit power in dBm (default: 30 dBm)')
    parser.add_argument('--fc', type=float, default=2.484e9, help='Carrier frequency in Hz (default: 2.484 GHz)')
    parser.add_argument('--max_reflections', type=int, default=4, help='Maximum number of reflections to consider (default: 4)')
    parser.add_argument('--max_diffractions', type=int, default=1, help='Maximum number of diffractions to consider (default: 1)')
    parser.add_argument('--ray_spacing', type=float, default=0.5, help='Spacing between rays in degrees (default: 0.5 deg)')
    parser.add_argument('--base_dir', type=str, default=os.path.join('.', 'data'), help='output data directory')
    parser.add_argument('--do_grid', action=argparse.BooleanOptionalAction, help='whether or not to do a 3x3 grid around the specified origin')
    parser.add_argument('--show_plot', action=argparse.BooleanOptionalAction, help='show the scene figure?')

    args = parser.parse_args()

    generate_setup(args)
    
