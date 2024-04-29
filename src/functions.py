import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
import numpy as np
import random
import datetime as dt
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point

def transform_coords(source_crs, target_crs, x, y):
    '''transforms point coordinates from the source coordinate system to the target'''
    return Transformer.from_crs(source_crs, target_crs, always_xy=True).transform(x, y)

def find_containing_polygon(point: Point, gdf: gpd) -> float:
    '''For a given point finds the polygon containing it
    input: point - type Point; gdf - geopandas dataframe
    output: number of polygon that contains point - type int'''
    for i in range(len(gdf)):
        if gdf.geometry[i].contains(point):
            return gdf.NO[i]
    return None

def calc_demand(gdf_a, df_demo, gdf_c, od_probs):
    '''Determines travel demand that occurs within the specified city area during the morning rush hour
        input:  gdf_a - SUM area [shapefile]
                df_demo - city demographic [csv]
                gdf_c - centroids of city zones [geojson]
                od_probs - ODM with probabilities [excel]
    '''
    for i in range(8, len(gdf_a)):
        df_area = df_demo.copy()
        polygon = gdf_a.geometry[i]
        # area centroids - are they needed???
        # x, y = polygon.centroid.x, polygon.centroid.y
        # x_center, y_center = fncs.transform_coords("EPSG:3857", "EPSG:4326", x, y)
        # area exterior
        xs, ys = polygon.exterior.xy[0], polygon.exterior.xy[1] # lists with polygon exterior coords
        xs, ys = transform_coords("EPSG:3857", "EPSG:4326", xs, ys)
        poly = Polygon(zip(xs, ys))
        
        # check, if SUM area containes addresses points
        df_area['inside_poly'] = df_area.apply(lambda row: poly.contains(Point(row['x'], row['y'])), axis=1)
        # filter addresses belonging to ith SUM area
        df_area = df_area[df_area.inside_poly == True]
        # repeat Rows N times (equal "ogolem / total"): N Rows = N people 
        df_area = df_area.loc[df_area.index.repeat(df_area.total)].reset_index(drop=True)

        # TODO: EDIT !!!
        # assign probabilities to specified zone, then choose destination using probs
        df_area['probs'] = [[]] * len(df_area)
        df_area['desti_zones'] = [[]] * len(df_area)
        for j in range(len(df_area)):
            z = int(df_area.iloc[j].zone_NO) # zone number
            row = od_probs.loc[od_probs['zone_NO'] == z].iloc[0]
            row = row[2:]
            df_area.at[j, 'desti_zones'] = list(row.index)
            df_area.at[j, 'probs'] =list(row)
            dic = row.to_dict()
            df_area.at[j, 'prob_dict'] = [dic]

        # assign destination zone number based on probabilities
        df_area['desti_zone'] = df_area.apply(lambda row: random.choices(row.desti_zones, weights=row.probs, k=1)[0], axis=1)
        # assign centroid coordinates for the destination zone
        df_area['desti_x'] = df_area.apply(lambda row: 
                                 gdf_c[gdf_c.NO == row.desti_zone].geometry.iloc[0].coords.xy[0][0], axis=1)
        df_area['desti_y'] = df_area.apply(lambda row: 
                                 gdf_c[gdf_c.NO == row.desti_zone].geometry.iloc[0].coords.xy[1][0], axis=1)
        # save output files
        # (f'C:\path\to\file\{filename}.csv', index=False)
        df_area.to_csv('output/demand_' + str(i) + '.csv', index=False)

        # generate the request time
        df_area['treq'] = pd.NA
        time_format = '%Y-%m-%d %H:%M:%S'
        time_lb = dt.datetime.strptime('2024-03-28 07:45:00', time_format)
        time_ub = dt.datetime.strptime('2024-03-28 08:15:00', time_format)
        df_area['treq'] = df_area['treq'].apply(lambda _: time_lb + 
                                                  dt.timedelta(seconds=np.random.randint(0, (time_ub - time_lb).seconds)))
        df_area.to_csv('requests/georequests_' + str(i) + '.csv', index=False)


def calc_demand_0(s_areas, zones, demo, odm, sheet, zone_c):
    gdf_areas = gpd.read_file(s_areas)
    gdf_zones = gpd.read_file(zones)
    gdf_centroid = gpd.read_file(zone_c)
    
    od = pd.read_excel(odm, sheet)
    od.rename(columns = {'464 x 464' : 'zone_NO', 'Unnamed: 2': 'sum'}, inplace = True)
    od.drop([0, 1], inplace=True)
    od.drop(od.filter(regex="Unname"), axis=1, inplace=True)
    od.reset_index(drop=True, inplace=True)
    od[0:] = od[:].astype(float)
    od['zone_NO'] = od['zone_NO'].astype(int)
    # filter OD zone number to match city zones
    od_filtered = od.loc[od['zone_NO'].isin(gdf_zones['NO']), ['zone_NO', 'sum'] + gdf_zones['NO'].to_list()]
    # calculate probabilities for each origin point
    od_probs = od_filtered.loc[:, 'sum':].astype('float64').divide(od_filtered.loc[:, 'sum'].astype('float64'), axis=0)
    od_probs.insert(0, 'zone_NO', od_filtered['zone_NO'])

    df = pd.read_csv(demo)
    df = df.rename(columns={"adr_pelny": "address", "ogolem": "total" })
    # assign zone number to each address point
    df['zone_NO'] = df.apply(lambda row: find_containing_polygon(Point(row['x'], row['y']), gdf_zones), axis=1)
    # remove the trips with nan zone
    df.dropna(subset='zone_NO', inplace=True, ignore_index=True)
    
    for i in range(8, len(gdf_areas)):
        df_area = df.copy()
        polygon = gdf_areas.geometry[i]
        # area centroids - are they needed???
        # x, y = polygon.centroid.x, polygon.centroid.y
        # x_center, y_center = fncs.transform_coords("EPSG:3857", "EPSG:4326", x, y)
        # area exterior
        xs, ys = polygon.exterior.xy[0], polygon.exterior.xy[1] # lists with polygon exterior coords
        xs, ys = transform_coords("EPSG:3857", "EPSG:4326", xs, ys)
        poly = Polygon(zip(xs, ys))
        
        # check, if SUM area containes addresses points
        df_area['inside_poly'] = df_area.apply(lambda row: poly.contains(Point(row['x'], row['y'])), axis=1)
        # filter addresses belonging to ith SUM area
        df_area = df_area[df_area.inside_poly == True]
        # repeat Rows N times (equal "ogolem / total"): N Rows = N people 
        df_area = df_area.loc[df_area.index.repeat(df_area.total)].reset_index(drop=True)

        # TODO: EDIT !!!
        # assign probabilities to specified zone, then choose destination using probs
        df_area['probs'] = [[]] * len(df_area)
        df_area['desti_zones'] = [[]] * len(df_area)
        for j in range(len(df_area)):
            z = int(df_area.iloc[j].zone_NO) # zone number
            row = od_probs.loc[od_probs['zone_NO'] == z].iloc[0]
            row = row[2:]
            df_area.at[j, 'desti_zones'] = list(row.index)
            df_area.at[j, 'probs'] =list(row)
            dic = row.to_dict()
            df_area.at[j, 'prob_dict'] = [dic]

        # assign destination zone number based on probabilities
        df_area['desti_zone'] = df_area.apply(lambda row: random.choices(row.desti_zones, weights=row.probs, k=1)[0], axis=1)
        # assign centroid coordinates for the destination zone
        df_area['desti_x'] = df_area.apply(lambda row: 
                                 gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[0][0], axis=1)
        df_area['desti_y'] = df_area.apply(lambda row: 
                                 gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[1][0], axis=1)
        # save output files
        # (f'C:\path\to\file\{filename}.csv', index=False)
        df_area.to_csv('output/demand_' + str(i) + '.csv', index=False)

        # generate the request time
        df_area['treq'] = pd.NA
        time_format = '%Y-%m-%d %H:%M:%S'
        time_lb = dt.datetime.strptime('2024-03-28 07:45:00', time_format)
        time_ub = dt.datetime.strptime('2024-03-28 08:15:00', time_format)
        df_area['treq'] = df_area['treq'].apply(lambda _: time_lb + 
                                                  dt.timedelta(seconds=np.random.randint(0, (time_ub - time_lb).seconds)))
        df_area.to_csv('requests/georequests_' + str(i) + '.csv', index=False)

def PT_utility(requests, params):
    if 'walkDistance' in requests.columns:
        requests = requests
        requests['PT_fare'] = 1 + requests.transitTime * params.avg_speed/1000 * params.ticket_price
        requests['u_PT'] = requests['PT_fare'] + \
                           params.VoT * (params.walk_factor * requests.walkDistance / params.speeds.walk +
                                           params.wait_factor * requests.waitingTime +
                                           params.transfer_penalty * requests.transfers + requests.transitTime)
    return requests