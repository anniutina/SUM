from ExMAS import main
from query_PT import main as qpt_main
import requests
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
import datetime

# To calculate sample size
KRA_17 = 767348     # Krakow population for 2017
KRA_23 = 804200     # Krakow population for 2023
COEF = KRA_23 / KRA_17

def transform_coords(source_crs, target_crs, x, y):
    '''transforms point coordinates from the source coordinate system to the target
       default: "EPSG:3857" to "EPSG:4326"
    '''
    return Transformer.from_crs(source_crs, target_crs, always_xy=True).transform(x, y)

def find_containing_polygon(point: Point, gdf: gpd) -> float:
    '''For a given point finds the polygon containing it
    input: point - type Point; gdf - geopandas dataframe
    output: number of polygon that contains point - type int'''
    for i in range(len(gdf)):
        if gdf.geometry[i].contains(point):
            return gdf.NO[i]
    return None

def define_demand(sum_areas, df_demo, gdf_centroid, od, od_probs, to_csv=False):
    '''Determines travel demand that occurs within the specified city area during the morning rush hour
        input:  sum_areas - SUM areas [shapefile]
                df_demo - city demographic [csv]
                gdf_centroid - centroids of city zones [geojson]
                od - ODM [excel]
                od_probs - ODM with probabilities [excel]
                to_csv - save results csv format
        output: dictionary with dataframes containing requests ex. {'Skotniki': (O, D, T)} 
    '''
    res = {}
    for i in range(len(sum_areas)):
        area = sum_areas.loc[i]
        demo_area = df_demo.copy()
        demo_area['inside_poly'] = demo_area.apply(lambda row: 
                                            area.geometry.contains(Point(row['x'], row['y'])), axis=1)
        demo_area = demo_area[demo_area.inside_poly].reset_index(drop=True)
        # repeat rows N times (equal "total"): N rows = N people
        demo_area = demo_area.loc[demo_area.index.repeat(demo_area.total)]
        # select a sample of origins
        sample_area = demo_area.sample(sample_size(od, demo_area)).reset_index(drop=True)
        sample_area.rename(columns = {'x' : 'origin_x', 'y': 'origin_y'}, inplace = True)
        sample_area['probs'] = [None] * len(sample_area)
        sample_area['desti_zones'] = [None] * len(sample_area)
        for n in range(len(sample_area['zone_NO'])):
            sample_area.at[n, 'desti_zones'] = list(od_probs.zone_NO)
            sample_area.at[n, 'probs'] = list(od_probs.loc[sample_area['zone_NO'][n], 1:])
        sample_area['desti_zone'] = sample_area.apply(lambda row: random.choices(row.desti_zones, 
                                                                        weights=row.probs, k=1)[0], axis=1)
        sample_area['destination_x'] = sample_area.apply(lambda row: 
                                    gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[0][0], axis=1)
        sample_area['destination_y'] = sample_area.apply(lambda row: 
                                    gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[1][0], axis=1)    
        sample_area['treq'] = pd.NA
        time_format = '%Y-%m-%d %H:%M:%S'
        time_lb = dt.datetime.strptime('2024-03-28 07:45:00', time_format)
        time_ub = dt.datetime.strptime('2024-03-28 08:15:00', time_format)
        sample_area['treq'] = sample_area['treq'].apply(lambda _: time_lb + 
                            dt.timedelta(seconds=np.random.randint(0, (time_ub - time_lb).seconds)))
        requests = sample_area[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'treq']]
        if to_csv:
            requests.to_csv('requests/reqs_' + str(sum_areas.name[i]) + '.csv', index=False)
        res[sum_areas.name[i]] = requests
    return res

def sample_size(od, df):
    '''calculate area production:
    how many travellers leave the area during the morning rush hour
    input: od - ODM, df - area demography with assigned zone numbers
    output: int, sample size
    '''
    return round((sum(od[(od['zone_NO'].isin(df['zone_NO'].unique()))]['sum'])) / 2 * COEF)

def PT_utility(requests, params):
    if 'walkDistance' in requests.columns:
        requests = requests
        requests['PT_fare'] = 1 + requests.transitTime * params.avg_speed/1000 * params.ticket_price
        requests['u_PT'] = requests['PT_fare'] + \
                           params.VoT * (params.walk_factor * requests.walkDistance / params.speeds.walk +
                                           params.wait_factor * requests.waitingTime +
                                           params.transfer_penalty * requests.transfers + requests.transitTime)
    return requests

def haversine(loc1, loc2):
    ''' Haversine formula [km]
        coordinates in decimal degrees (y, x), e.g. (19.881557, 50.012738)'''
    # latitude is the y-coordinate, longitude is the x-coordinate
    Earth_radius = 6371  # [km];   R = 3959.87433 [mi]
    lat1, lon1 = np.radians((loc1[0], loc1[1]))
    lat2, lon2 = np.radians((loc2[0], loc2[1]))
    a = np.sin(0.5 * (lat2 - lat1))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1))**2
    return 2 * Earth_radius * np.arcsin(np.sqrt(a))

def run_ExMAS(df, inData, params, hub=None, degree=1):
    '''input: df - dataframe with columns [origin_x, origin_y, destination_x, destination_y, treq]
              inData - loaded graph and parameters
              hub - tuple with hub coordinates
              degree - params.max_degree - max number of travellers
        output: runs ExMAS, calculates utilities and KPI's
    '''
    # inData.requests.columns = [origin, destination, treq, tdep, ttrav, tarr, tdrop]
    params.nP = len(df)  # sample size
    params.max_degree = degree
    inData.requests = df.copy()
    inData.requests['origin'] = inData.requests.apply(lambda row: ox.nearest_nodes(inData.G, row['origin_x'], row['origin_y']), axis=1)
    if hub is not None:
        inData.requests['destination'] = ox.nearest_nodes(inData.G, hub[0], hub[1])
    else:
        inData.requests['destination'] = inData.requests.apply(lambda row: ox.nearest_nodes(inData.G, row['destination_x'], row['destination_y']), axis=1)
    # TODO: - DONT NEED TO CHECK if not reading from csv
    if type(inData.requests['treq'][0]) == str:
        inData.requests['treq'] = inData.requests['treq'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    inData.requests['dist'] = inData.requests.apply(lambda request: inData.skim.loc[request.origin, request.destination], axis=1)
    inData.requests['ttrav'] = inData.requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    inData.requests['tarr'] = [request.treq + request.ttrav for _, request in inData.requests.iterrows()]
    inData.requests['pax_id'] = list(range(len(inData.requests)))
    
    inData = main.main(inData, params)

def run_OTP(df, OTP_API):
    '''using OpenTripPlanner server returns routes for the given request
    returns dataframe with OTP requests, drops requests for access=False
    ATTENTION to dataframe indexes'''
    df_query = df.copy()
    query = df_query.apply(lambda row: 
                    qpt_main.parse_OTP_response(requests.get(OTP_API, 
                                                params=qpt_main.make_query(row.squeeze())).json()), axis=1)
    names = []
    for q in query:
        if q['success']:
            names = list(q.keys())
            break
        
    if len(names) > 0:
        for name in names:
            vals = []
            for i in range(len(query)):
                if query[i]['success']:
                    vals.append(query[i][name])
                else:
                    vals.append(pd.NA)
            df_query[name] = vals

    # add ignore_index=True if indexes must be sorted
    df_query.dropna(inplace=True)
    return df_query