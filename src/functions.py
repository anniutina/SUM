import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import random
import datetime as dt
import requests
from pyproj import Transformer
import geopandas as gpd
import math

from ExMAS import main
from query_PT import main as qpt_main

# To calculate sample size
KRA_17 = 767348     # Krakow population for 2017
KRA_23 = 804200     # Krakow population for 2023
COEF = KRA_23 / KRA_17

def sample_size(od, df_demo, area_pop):
    '''Calculates the number of travellers leaving the area during the rush hour
        as the proportion between zone and area population and their production
        Parameters: od - ODM, df_demo - DataFrame with city population distribution, 
                    area_pop - area population distribution
        Returns: area production
    '''
    a_zns = np.sort(area_pop['zone_NO'].unique()) # area zones
    # zone production, zone population, partial area population
    z_prod, z_pop, p_a_pop = 0, 0, 0  
    p_a_prod = [] # partial area production

    for z_num in a_zns:
        z_prod = od[(od['zone_NO'] == z_num)]['sum'].item()
        z_pop = sum(df_demo[df_demo["zone_NO"] == z_num]["total"])
        p_a_pop = sum(area_pop[area_pop["zone_NO"] == z_num]["total"])
        p_a_prod.append(z_prod * p_a_pop / z_pop)
    return round(sum(p_a_prod) / 2 * COEF, 2)

def reverse_coords(geom):
    '''Reverse the order of lon, lat in a Shapely geometry
        Parameters: geom - Polygon or MultiPolygon object with coords to reverse
        Returns: a new Polygon or MultiPolygon object with reversed coords, lat lon'''
    if type(geom) == Polygon:
        new_exterior = [(y, x) for x, y in geom.exterior.coords]
        new_interior = [[(y, x) for x, y in interior.coords] for interior in geom.interiors]
        return Polygon(new_exterior, new_interior)
    elif type(geom) == MultiPolygon:
        reversed_multi_poly = []
        for poly in geom.geoms:
            reversed_multi_poly.append(reverse_coords(poly))
        return MultiPolygon(reversed_multi_poly)

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

def haversine(loc1, loc2):
    ''' Haversine formula [km]
        coordinates in decimal degrees (y, x), e.g. (19.881557, 50.012738)'''
    # latitude is the y-coordinate, longitude is the x-coordinate
    Earth_radius = 6371  # [km];   R = 3959.87433 [mi]
    lat1, lon1 = np.radians((loc1[0], loc1[1]))
    lat2, lon2 = np.radians((loc2[0], loc2[1]))
    a = np.sin(0.5 * (lat2 - lat1))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1))**2
    return 2 * Earth_radius * np.arcsin(np.sqrt(a))

def define_demand(sum_areas, df_demo, gdf_centroid, od, od_probs, params, to_csv=False):
    '''Determine travel demand that occurs within the specified city area during the morning rush hour
        input:  sum_areas - SUM areas [geopandas dataframe]
                df_demo - city demographic [csv]
                gdf_centroid - centroids of city zones [geojson]
                od - ODM [excel]
                od_probs - ODM with probabilities [excel]
                to_csv - save results csv format
        output: dictionary with dataframes containing requests ex. {'Skotniki': (O, D, T)} 
    '''
    res = {}
    for i in range(len(sum_areas)):
        if isinstance(sum_areas, pd.Series):
            area = sum_areas
        else:    
            area = sum_areas.loc[i]
        area_pop = df_demo.copy()
        # define, if area polygon contains address points (area population)
        area_pop['inside_poly'] = area_pop.apply(lambda row: 
                                            area.geometry.contains(Point(row['x'], row['y'])), axis=1)
        area_pop = area_pop[area_pop.inside_poly].reset_index(drop=True)
        # repeat rows N times (equal "total"): N rows = N people
        area_pop_repeated = area_pop.loc[area_pop.index.repeat(area_pop.total)]
        # select a sample of origins
        area_sample = area_pop_repeated.sample(round(sample_size(od, df_demo, area_pop))).reset_index(drop=True)
        area_sample.rename(columns = {'x' : 'origin_x', 'y': 'origin_y'}, inplace = True)
        area_sample['probs'] = [None] * len(area_sample)
        area_sample['desti_zones'] = [None] * len(area_sample)
        for n in range(len(area_sample['zone_NO'])):
            area_sample.at[n, 'desti_zones'] = list(od_probs.zone_NO)
            area_sample.at[n, 'probs'] = list(od_probs.loc[area_sample['zone_NO'][n], 1:])
        area_sample['desti_zone'] = area_sample.apply(lambda row: random.choices(row.desti_zones, 
                                                                        weights=row.probs, k=1)[0], axis=1)
        area_sample['destination_x'] = area_sample.apply(lambda row: 
                                    gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[0][0], axis=1)
        area_sample['destination_y'] = area_sample.apply(lambda row: 
                                    gdf_centroid[gdf_centroid.NO == row.desti_zone].geometry.iloc[0].coords.xy[1][0], axis=1)    
        area_sample['treq'] = pd.NA
        time_format = '%Y-%m-%d %H:%M:%S'
        time_lb = dt.datetime.strptime('2024-03-28 07:45:00', time_format)
        time_ub = dt.datetime.strptime('2024-03-28 08:15:00', time_format)
        area_sample['treq'] = area_sample['treq'].apply(lambda _: time_lb + 
                            dt.timedelta(seconds=np.random.randint(0, (time_ub - time_lb).seconds)))
        requests = area_sample[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'treq']]
        if isinstance(sum_areas, pd.Series):
            res[sum_areas["name"]] = requests
        else:
            if to_csv:
                requests.to_csv('requests/reqs_' + str(sum_areas.name[i]) + '.csv', index=False)
            res[sum_areas.name[i]] = requests
    return res

def PT_utility(requests, params):
    if 'walkDistance' in requests.columns:
        requests = requests
        requests['PT_fare'] = 1 + requests.transitTime * params.avg_speed/1000 * params.ticket_price
        requests['u_PT'] = requests['PT_fare'] + \
                           params.VoT * (params.walk_factor * requests.walkDistance / params.speeds.walk +
                                           params.wait_factor * requests.waitingTime +
                                           params.transfer_penalty * requests.transfers + requests.transitTime)
    return requests

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

    # ignore_index=True if indexes must be sorted
    df_query.dropna(inplace=True, ignore_index=True)
    return df_query

def run_ExMAS(df, inData, params, hub=None, degree=8):
    '''input: df - dataframe with columns [origin_x, origin_y, destination_x, destination_y, treq]
              inData - dotMap object with loaded graph and parameters
              hub - tuple with hub coordinates
              degree - params.max_degree - max number of travellers
        output: runs ExMAS, calculates utilities and KPI's
    '''
    params.nP = len(df)  # sample size
    params.max_degree = degree
    inData.requests = df.copy()
    if hub is None:
        inData.requests['destination'] = inData.requests.apply(lambda row: ox.nearest_nodes(inData.G, row['destination_x'], row['destination_y']), axis=1)
    inData.requests['ttrav'] = inData.requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    inData.requests['pax_id'] = list(range(len(inData.requests)))
    
    inData = main.main(inData, params)

#  TODO: update function
def simulate(gdf_areas, df_demo, gdf_centroid, od, od_probs, hubs, inData, params, OTP_API, degree=1, N=1, ASC=2.58):
    '''calculate utilities for each traveller of the given area, (single rides, no ExMAS)
    input:  gdf_areas - SUM areas [geopandas dataframe]
            df_demo - city demographic [csv]
            gdf_centroid - centroids of city zones [geojson]
            od - ODMs [excel]
            od_probs - ODM with probabilities [excel]
            N - number of replications
            ASC - alternative specific constant
    output: tuple[0] - dictionary with mean results for N iterations for each area sample
               ex. {'Skotniki': tw_PT_OD, tw_PT_HD, u_PT_OD, u_PT_HD, u_SUM_OD, p_SUM}
            tuple[1] - dictionary with results of the last iteration for sample of each area:
               ex. {'Skotniki': origin_x, origin_y, destination_x, destination_y, treq, u_PT_OD,
                                origin, hub, dist, ttrav, tarr, u, u_SUM_OD, p_SUM}
            '''
    areas_res = {}
    sum_res = {} # resultind dataFrame from the last iteration
    for _, area in gdf_areas.iterrows():
        key = area["name"]
        dfres = pd.DataFrame()
        for repl in range(N):
            # print("area ", key, " iteration # ", repl + 1)
            area_reqs = define_demand(area, df_demo, gdf_centroid, od, od_probs, params)
            df = area_reqs[key].copy() # df with {O, D, Treq} for the area
            hub = hubs[key]

            # Utility for PT OD
            u_pt_od = df.copy()
            u_pt_od = run_OTP(u_pt_od, OTP_API) # define PT routes for each traveller
            PT_utility(u_pt_od, params)  
            
            df = df.loc[u_pt_od.index, :] # select requests with successful OD trips 
            df.reset_index(drop=True, inplace=True)

            # Utility for SUM (NSM OH + PT HD)
            df_sum = df.copy()
            df_sum['u_PT_OD'] = u_pt_od.u_PT

            if degree == 1:
                df_sum['origin'] = df_sum.apply(lambda row: ox.nearest_nodes(inData.G, row['origin_x'], row['origin_y']), axis=1)
                df_sum['hub'] = ox.nearest_nodes(inData.G, hub[0], hub[1])
                df_sum['dist'] = df_sum.apply(lambda request: inData.skim.loc[request.origin, request.hub], axis=1)
                df_sum['ttrav'] = df_sum['dist'].apply(lambda request: request / params.avg_speed)
                df_sum['tarr'] = df_sum.treq + df_sum.apply(lambda df_sum: pd.Timedelta(df_sum.ttrav, 's').floor('s'), axis=1)
                df_sum['u'] = df_sum.apply(lambda request: request['ttrav'] * params.VoT + request['dist'] * params.price / 1000, axis=1)
            else:
                run_ExMAS(df_sum, inData, params, hub, degree)
                df_sum['tarr'] = df_sum.treq + inData.sblts.requests.apply(
                    lambda request: pd.Timedelta(request.ttrav_sh, 's').floor('s'), axis=1)
                df_sum['u'] = inData.sblts.requests.u
                df_sum['u_sh'] = inData.sblts.requests.u_sh

            # Utility for PT HD
            u_pt_hd = df.copy()
            u_pt_hd = u_pt_hd.rename(columns = {'treq': 'treq_origin'})
            u_pt_hd['origin_x'] = hub[0]
            u_pt_hd['origin_y'] = hub[1]
            u_pt_hd['treq'] = pd.to_datetime(df_sum.tarr) + pd.Timedelta(params.transfertime, unit='s') # treq for PT_HD
            u_pt_hd = run_OTP(u_pt_hd, OTP_API)
            PT_utility(u_pt_hd, params)

            u_pt_od = u_pt_od.loc[u_pt_hd.index, :] # drop rows with unsuccessful HD trips 
            u_pt_od.reset_index(drop=True, inplace=True)
            df_sum = df_sum.loc[u_pt_hd.index, :] 
            df_sum.reset_index(drop=True, inplace=True)
            u_pt_hd.reset_index(drop=True, inplace=True)
            df_sum['u_PT_HD'] = u_pt_hd.u_PT
            
            if degree == 1:
                df_sum['u_SUM_OD'] = df_sum.u + u_pt_hd.u_PT + ASC
            else:
                df_sum['u_SUM_OD'] = df_sum.u_sh + u_pt_hd.u_PT + ASC
            df_sum['p_SUM'] = df_sum.apply(lambda row: math.exp(-row.u_SUM_OD) / \
                                           (math.exp(-row.u_SUM_OD) + math.exp(-row.u_PT_OD)), axis=1)
            df_means = pd.DataFrame([[u_pt_od.waitingTime.mean(), u_pt_hd.waitingTime.mean(), u_pt_od.u_PT.mean(),
                                    u_pt_hd.u_PT.mean(), df_sum.u_SUM_OD.mean(), df_sum.p_SUM.mean()]], 
                                    columns=['tw_PT_OD', 'tw_PT_HD', 'u_PT_OD', 'u_PT_HD', 'u_SUM_OD', 'p_SUM'])
            dfres = pd.concat([dfres, df_means], ignore_index=True)
        sum_res[key] = df_sum
        areas_res[key] = dfres
    return (areas_res, sum_res)

def calc_E_Psum(df, ASC=0):
    df['u_SUM_OD'] = df.u + df.u_PT_HD + ASC
    df['p_SUM'] = df.apply(lambda row: math.exp(-row.u_SUM_OD) / \
                        (math.exp(-row.u_SUM_OD) + math.exp(-row.u_PT_OD)), axis=1)
    return df.p_SUM.mean()