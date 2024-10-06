# -*- coding:utf-8 -*-
##############################################################
# Created Date: Tuesday, September 5th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

from __future__ import absolute_import
import itertools
import copy
from multiprocessing import Pool
from multiprocessing import Manager

import pandas as pd
import shapely
import numpy as np
import shapely.geometry
from shapely.strtree import STRtree
from joblib import Parallel, delayed
from tqdm import tqdm
from pyufunc import (calc_distance_on_unit_sphere,
                     calc_distance_on_unit_haversine,
                     cvt_int_to_alpha,
                     func_running_time,
                     find_closest_point)

from grid2demand.utils_lib.net_utils import Zone, Node
from grid2demand.utils_lib.pkg_settings import pkg_settings
from tqdm.contrib.concurrent import process_map
import datetime
import random
import math


# supporting functions
def _get_lng_lat_min_max(node_dict: dict[int, Node]) -> list:
    """Get the boundary of the study area

    Args:
        node_dict (dict[int, Node]): node_dict {node_id: Node}

    Returns:
        list: [min_lng, max_lng, min_lat, max_lat]
    """
    first_key = list(node_dict.keys())[0]

    coord_x_min, coord_x_max = node_dict[first_key].x_coord, node_dict[first_key].x_coord
    coord_y_min, coord_y_max = node_dict[first_key].y_coord, node_dict[first_key].y_coord

    for node_id in node_dict:
        if node_dict[node_id].x_coord < coord_x_min:
            coord_x_min = node_dict[node_id].x_coord
        if node_dict[node_id].x_coord > coord_x_max:
            coord_x_max = node_dict[node_id].x_coord
        if node_dict[node_id].y_coord < coord_y_min:
            coord_y_min = node_dict[node_id].y_coord
        if node_dict[node_id].y_coord > coord_y_max:
            coord_y_max = node_dict[node_id].y_coord

    return [coord_x_min - 0.000001, coord_x_max + 0.000001, coord_y_min - 0.000001, coord_y_max + 0.000001]


def _sync_zones_geometry_with_node(args: tuple) -> tuple:
    """Check if a batch of nodes is within any zone and return updated node and zone information."""
    # node is dictionary here, pool.imap or process_map will not work for dataclass object

    # node_id, node, zone_dict = args
    # node_id, node, zone_dict, zone_geometries, zone_ids = args

    # for zone_name in zone_dict:
    #     # convert str geometry to shapely geometry
    #     if isinstance(node["geometry"], str):
    #         node["geometry"] = shapely.from_wkt(node["geometry"])
    #     if isinstance(zone_dict[zone_name]["geometry"], str):
    #         zone_dict[zone_name]["geometry"] = shapely.from_wkt(zone_dict[zone_name]["geometry"])
    #     # add node_id and zone_id if node inside the zone geometry
    #     if shapely.within(node["geometry"], zone_dict[zone_name]["geometry"]):
    #         node["zone_id"] = zone_dict[zone_name]["id"]
    #         zone_dict[zone_name]["node_id_list"].append(node_id)
    #         return node_id, node, zone_name
    # return node_id, node, None
    # Parse node geometry once, if it is a WKT string
    try:
        nodes_batch, zone_dict, zone_index, zone_ids = args
        results = []

        for node_id, node in nodes_batch:
            node_geom = node["geometry"]

            # Query spatial index for candidate zones using bounding boxes
            candidate_zones = zone_index.query(node_geom)

            for candidate_zone in candidate_zones:
                zone_name = zone_ids[zone_index.geoms.index(candidate_zone)]
                if node_geom.within(candidate_zone):
                    node["zone_id"] = zone_dict[zone_name]["id"]
                    zone_dict[zone_name]["node_id_list"].append(node_id)
                    results.append((node_id, node, zone_name))
                    break  # If node is in one zone, no need to check others

            # If no zone found, add the node with None zone
            if "zone_id" not in node:
                results.append((node_id, node, None))

        return results
    except Exception as e:
        print(f"  : could not mapping {node_id} with zone for {e}")

    # return node_id, node, None


def _sync_zones_geometry_with_poi(args: tuple) -> tuple:

    # poi is dictionary here
    # pool.imap or process_map will not work for dataclass object

    # poi_id, poi, zone_dict = args
    # for zone_name in zone_dict:
    #     # convert str geometry to shapely geometry
    #     if isinstance(poi["geometry"], str):
    #         poi["geometry"] = shapely.from_wkt(poi["geometry"])
    #     if isinstance(zone_dict[zone_name]["geometry"], str):
    #         zone_dict[zone_name]["geometry"] = shapely.from_wkt(zone_dict[zone_name]["geometry"])
    #     # add poi_id and zone_id, if poi inside zone geometry
    #     if shapely.within(poi["geometry"], zone_dict[zone_name]["geometry"]):
    #         poi["zone_id"] = zone_dict[zone_name]["id"]
    #         zone_dict[zone_name]["poi_id_list"].append(poi_id)
    #         return poi_id, poi, zone_name
    # return poi_id, poi, None

    try:
        pois_batch, zone_dict, zone_index, zone_ids = args
        results = []

        for poi_id, poi in pois_batch:
            poi_geom = poi["geometry"]

            # Query spatial index for candidate zones using bounding boxes
            candidate_zones = zone_index.query(poi_geom)

            for candidate_zone in candidate_zones:
                zone_name = zone_ids[zone_index.geoms.index(candidate_zone)]
                if poi_geom.within(candidate_zone):
                    poi["zone_id"] = zone_dict[zone_name]["id"]
                    zone_dict[zone_name]["node_id_list"].append(poi_id)
                    results.append((poi_id, poi, zone_name))
                    break  # If node is in one zone, no need to check others

            # If no zone found, add the node with None zone
            if "zone_id" not in poi:
                results.append((poi_id, poi, None))

        return results
    except Exception as e:
        print(f"  : could not mapping poi with zone for {e}")


def _sync_zones_centroid_with_node(args: tuple) -> tuple:
    # node is dictionary here
    node_id, node, multipoint_zone, zone_point_id = args
    node_point = shapely.geometry.Point(node["x_coord"], node["y_coord"])
    closest_zone_point = find_closest_point(node_point, multipoint_zone)[0]
    zone_id = zone_point_id[closest_zone_point]
    return node_id, zone_id


def _sync_zones_centroid_with_poi(args: tuple) -> tuple:

    # poi is dictionary here
    # pool.imap or process_map will not work for dataclass object

    poi_id, poi, multipoint_zone, zone_point_id = args
    poi_point = shapely.geometry.Point(poi["x_coord"], poi["y_coord"])
    closest_zone_point = find_closest_point(poi_point, multipoint_zone)[0]
    zone_id = zone_point_id[closest_zone_point]
    return poi_id, zone_id

# Main functions


@func_running_time
def net2zone(node_dict: dict[int, Node],
             num_x_blocks: int = 0,
             num_y_blocks: int = 0,
             cell_width: float = 0,
             cell_height: float = 0,
             unit: str = "km",
             verbose: bool = False) -> dict[str, Zone]:
    """convert node_dict to zone_dict by grid.
    The grid can be defined by num_x_blocks and num_y_blocks, or cell_width and cell_height.
    if num_x_blocks and num_y_blocks are specified, the grid will be divided into num_x_blocks * num_y_blocks.
    if cell_width and cell_height are specified, the grid will be divided into cells with cell_width * cell_height.
    Note: num_x_blocks and num_y_blocks have higher priority to cell_width and cell_height.
            if num_x_blocks and num_y_blocks are specified, cell_width and cell_height will be ignored.

    Args:
        node_dict (dict[int, Node]): node_dict {node_id: Node}
        num_x_blocks (int, optional): total number of blocks/grids from x direction. Defaults to 10.
        num_y_blocks (int, optional): total number of blocks/grids from y direction. Defaults to 10.
        cell_width (float, optional): the width for each block/grid . Defaults to 0. unit: km.
        cell_height (float, optional): the height for each block/grid. Defaults to 0. unit: km.
        unit (str, optional): the unit of cell_width and cell_height. Defaults to "km". Options:"meter", "km", "mile".
        use_zone_id (bool, optional): whether to use zone_id in node_dict. Defaults to False.
        verbose (bool, optional): print processing information. Defaults to False.

    Raises
        ValueError: Please provide num_x_blocks and num_y_blocks or cell_width and cell_height

    Returns
        Zone: dictionary, Zone cells with keys are zone names, values are Zone

    Examples:
        >>> zone_dict = net2zone(node_dict, num_x_blocks=10, num_y_blocks=10)
        >>> zone_dict['A1']
        Zone(id=0, name='A1', x_coord=0.05, y_coord=0.95, centroid='POINT (0.05 0.95)', x_min=0.0, x_max=0.1,
        y_min=0.9, y_max=1.0, geometry='POLYGON ((0.05 0.9, 0.1 0.9, 0.1 1, 0.05 1, 0.05 0.9))')

    """

    # # convert node_dict to dataframe
    # df_node = pd.DataFrame(node_dict.values())
    # # get the boundary of the study area
    # coord_x_min, coord_x_max = df_node['x_coord'].min(
    # ) - 0.000001, df_node['x_coord'].max() + 0.000001
    # coord_y_min, coord_y_max = df_node['y_coord'].min(
    # ) - 0.000001, df_node['y_coord'].max() + 0.000001

    # generate zone based on zone_id in node.csv
    # if use_zone_id:
    #     node_dict_zone_id = {}
    #     for node_id in node_dict:
    #         with contextlib.suppress(AttributeError):
    #             if node_dict[node_id]._zone_id != -1:
    #                 node_dict_zone_id[node_id] = node_dict[node_id]
    #     if not node_dict_zone_id:
    #         print("  : No zone_id found in node_dict, will generate zone based on original node_dict")
    #     else:
    #         node_dict = node_dict_zone_id

    coord_x_min, coord_x_max, coord_y_min, coord_y_max = _get_lng_lat_min_max(node_dict)

    # get nodes within the boundary
    # if use_zone_id:
    #     node_dict_within_boundary = {}
    #     for node_id in node_dict:
    #         if node_dict[node_id].x_coord >= coord_x_min and node_dict[node_id].x_coord <= coord_x_max and \
    #                 node_dict[node_id].y_coord >= coord_y_min and node_dict[node_id].y_coord <= coord_y_max:
    #             node_dict_within_boundary[node_id] = node_dict[node_id]
    # else:
    #     node_dict_within_boundary = node_dict

    # Priority: num_x_blocks, number_y_blocks > cell_width, cell_height
    # if num_x_blocks and num_y_blocks are given, use them to partition the study area
    # else if cell_width and cell_height are given, use them to partition the study area
    # else raise error

    if num_x_blocks > 0 and num_y_blocks > 0:
        x_block_width = (coord_x_max - coord_x_min) / num_x_blocks
        y_block_height = (coord_y_max - coord_y_min) / num_y_blocks
    elif cell_width > 0 and cell_height > 0:
        x_dist_km = calc_distance_on_unit_sphere(
            (coord_x_min, coord_y_min), (coord_x_max, coord_y_min), unit=unit)
        y_dist_km = calc_distance_on_unit_sphere(
            (coord_x_min, coord_y_min), (coord_x_min, coord_y_max), unit=unit)

        num_x_blocks = int(np.ceil(x_dist_km / cell_width))
        num_y_blocks = int(np.ceil(y_dist_km / cell_height))

        x_block_width = (coord_x_max - coord_x_min) / num_x_blocks
        y_block_height = (coord_y_max - coord_y_min) / num_y_blocks
    else:
        raise ValueError(
            'Please provide num_x_blocks and num_y_blocks or cell_width and cell_height')

    # partition the study area into zone cells
    x_block_min_lst = [coord_x_min + i *
                       x_block_width for i in range(num_x_blocks)]
    y_block_min_lst = [coord_y_min + i *
                       y_block_height for i in range(num_y_blocks)]

    x_block_minmax_list = list(zip(
        x_block_min_lst[:-1], x_block_min_lst[1:])) + [(x_block_min_lst[-1], coord_x_max)]
    y_block_minmax_list = list(zip(
        y_block_min_lst[:-1], y_block_min_lst[1:])) + [(y_block_min_lst[-1], coord_y_max)]

    def generate_polygon(x_min, x_max, y_min, y_max) -> shapely.geometry.Polygon:
        """Generate polygon from min and max coordinates

        Parameters
            x_min: float, Min x coordinate
            x_max: float, Max x coordinate
            y_min: float, Min y coordinate
            y_max: float, Max y coordinate

        Returns
            polygon: sg.Polygon, Polygon

        """
        return shapely.geometry.Polygon([(x_min, y_min),
                                         (x_max, y_min),
                                         (x_max, y_max),
                                         (x_min, y_max),
                                         (x_min, y_min)])

    zone_dict = {}
    zone_upper_row = []
    zone_lower_row = []
    zone_left_col = []
    zone_right_col = []
    zone_id_flag = 0

    # convert y from min-max to max-min
    y_block_maxmin_list = y_block_minmax_list[::-1]

    # generate zone cells with id and labels
    # id: A1, A2, A3, ...,
    #     B1, B2, B3, ...,
    #     C1, C2, C3, ...
    for j in range(len(y_block_maxmin_list)):
        for i in range(len(x_block_minmax_list)):
            x_min = x_block_minmax_list[i][0]
            x_max = x_block_minmax_list[i][1]
            y_min = y_block_maxmin_list[j][0]
            y_max = y_block_maxmin_list[j][1]

            cell_polygon = generate_polygon(x_min, x_max, y_min, y_max)
            row_alpha = cvt_int_to_alpha(j)
            zone_dict[f"{row_alpha}{i}"] = Zone(
                id=zone_id_flag,
                name=f"{row_alpha}{i}",
                x_coord=cell_polygon.centroid.x,
                y_coord=cell_polygon.centroid.y,
                centroid=cell_polygon.centroid,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                geometry=cell_polygon
            )

            # add boundary zone names to list
            if j == 0:
                zone_upper_row.append(f"{row_alpha}{i}")
            if j == len(y_block_maxmin_list) - 1:
                zone_lower_row.append(f"{row_alpha}{i}")

            if i == 0:
                zone_left_col.append(f"{row_alpha}{i}")
            if i == len(x_block_minmax_list) - 1:
                zone_right_col.append(f"{row_alpha}{i}")

            # update zone id
            zone_id_flag += 1

    # generate outside boundary centroids
    upper_points = [shapely.geometry.Point(zone_dict[zone_name].x_coord,
                                           zone_dict[zone_name].y_coord + y_block_height
                                           ) for zone_name in zone_upper_row]
    lower_points = [shapely.geometry.Point(zone_dict[zone_name].x_coord,
                                           zone_dict[zone_name].y_coord - y_block_height
                                           ) for zone_name in zone_lower_row]
    left_points = [shapely.geometry.Point(zone_dict[zone_name].x_coord - x_block_width,
                                          zone_dict[zone_name].y_coord
                                          ) for zone_name in zone_left_col]
    right_points = [shapely.geometry.Point(zone_dict[zone_name].x_coord + x_block_width,
                                           zone_dict[zone_name].y_coord
                                           ) for zone_name in zone_right_col]
    points_lst = upper_points + lower_points + left_points + right_points
    for i in range(len(points_lst)):
        zone_dict[f"gate{i}"] = Zone(
            id=zone_id_flag,
            name=f"gate{i}",
            x_coord=points_lst[i].x,
            y_coord=points_lst[i].y,
            centroid=points_lst[i],
            geometry=points_lst[i]
        )
        zone_id_flag += 1

    if verbose:
        print(
            f"  : Successfully generated zone dictionary: {len(zone_dict) - 4 * len(zone_upper_row)} Zones generated,")
        print(f"  : plus {4 * len(zone_upper_row)} boundary gates (points)")
    return zone_dict


@func_running_time
def sync_zone_geometry_and_node(zone_dict: dict, node_dict: dict, cpu_cores: int = 1, verbose: bool = False) -> dict:
    """Map nodes to zone cells

    Parameters
        node_dict: dict, Nodes
        zone_dict: dict, zone cells

    Returns
        node_dict and zone_dict: dict, Update Nodes with zone id, update zone cells with node id list

    """
    # Create a pool of worker processes
    if verbose:
        print(f"  : Parallel synchronizing Nodes and Zones using Pool with {cpu_cores} CPUs. Please wait...")

    # create deepcopy of zone_dict and node_dict to avoid modifying the original dict
    zone_cp = copy.deepcopy(zone_dict)
    node_cp = copy.deepcopy(node_dict)

    # Pre-process zone geometries
    for zone_id, zone in zone_dict.items():
        if isinstance(zone["geometry"], str):
            zone["geometry"] = shapely.from_wkt(zone["geometry"])

    # Pre-process node geometries
    for node_id, node in node_dict.items():
        if isinstance(node["geometry"], str):
            node["geometry"] = shapely.from_wkt(node["geometry"])

    # Create a spatial index for zone geometries
    zone_geometries = [zone["geometry"] for zone in zone_dict.values()]
    zone_ids = list(zone_dict.keys())
    zone_index = STRtree(zone_geometries)

    # Prepare arguments for the pool
    # args_list = [(node_id, node, zone_cp) for node_id, node in node_cp.items()]

    # Prepare node batches for multiprocessing
    chunk_size = pkg_settings["data_chunk_size"]

    node_items = list(node_dict.items())
    node_batches = [node_items[i:i + chunk_size] for i in range(0, len(node_items), chunk_size)]

    # Prepare arguments for multiprocessing
    # args_list = [(node_id, node, zone_dict, zone_index, zone_ids) for node_id, node in node_dict.items()]
    args_list = [(batch, zone_dict, zone_index, zone_ids) for batch in node_batches]

    if verbose:
        print(f"  : Parallel sync zone geometry and node using {cpu_cores} cpu cores.")

    with Pool(processes=cpu_cores) as pool:
        results = list(tqdm(pool.imap(_sync_zones_geometry_with_node, args_list), total=len(args_list)))
        pool.close()
        pool.join()
    # results = process_map(_sync_zones_geometry_with_node, args_list, workers=cpu_cores)

    # Gather results
    # Ensure results is a list of lists and not None
    results = [result for result in results if result is not None]
    for batch_results in results:
        for node_id, node, zone_name in batch_results:
            if zone_name is not None:
                zone_cp[zone_name]["node_id_list"].append(node_id)
            node_cp[node_id] = node

    if verbose:
        print("  : Successfully synchronized zone and node geometry")

    return {"zone_dict": zone_cp, "node_dict": node_cp}


def sync_zone_centroid_and_node(zone_dict: dict, node_dict: dict, cpu_cores: int = 1, verbose: bool = False) -> dict:
    """Synchronize zone in centroids and nodes to update zone_id attribute for nodes

    Args:
        zone_dict (dict): Zone cells
        node_dict (dict): Nodes

    Returns:
        dict: the updated zone_dict and node_dict

    """

    # node_point_id = {
    #     shapely.geometry.Point(
    #         node_dict[node_id].x_coord, node_dict[node_id].x_coord
    #     ): node_id
    #     for node_id in node_dict
    # }

    # Deepcopy the dictionary
    zone_cp = copy.deepcopy(zone_dict)
    node_cp = copy.deepcopy(node_dict)

    # Create zone_point_id dictionary
    zone_point_id = {
        shapely.geometry.Point(zone_cp[zone_id]["x_coord"], zone_cp[zone_id]["y_coord"]): zone_id
        for zone_id in zone_cp
    }

    # create multipoint object for zone centroids
    multipoint_zone = shapely.geometry.MultiPoint(
        [shapely.geometry.Point(zone_cp[i]["x_coord"], zone_cp[i]["y_coord"]) for i in zone_cp])

    # Prepare data for multiprocessing
    args = [(node_id, node, multipoint_zone, zone_point_id) for node_id, node in node_cp.items()]

    # cpu_cores = pkg_settings["set_cpu_cores"]

    if verbose:
        print(f"  : Parallel sync zone centroid and node using {cpu_cores} cpu cores.")

    with Pool(cpu_cores) as pool:
        results = list(tqdm(pool.imap(_sync_zones_centroid_with_node, args), total=len(node_cp)))
        pool.close()
        pool.join()

    # Update zone_cp with the results
    for node_id, zone_id in results:
        node_cp[node_id]["zone_id"] = zone_id
        zone_cp[zone_id]["node_id_list"].append(node_id)

    # flag = 0
    # for node_id, node in tqdm(node_cp.items()):
    #     if flag + 1 % 1000 == 0:
    #         print(f"Processing node {flag + 1}/{len(node_cp)}")
    #     node_point = shapely.geometry.Point(node.x_coord, node.y_coord)
    #     closest_zone_point = find_closest_point(node_point, multipoint_zone)[0]
    #     zone_id = zone_point_id[closest_zone_point]
    #     node.zone_id = zone_id
    #     zone_cp[zone_id].node_id_list.append(node_id)

    if verbose:
        print("  : Successfully synchronized zone and node geometry")

    return {"zone_dict": zone_cp, "node_dict": node_cp}


@func_running_time
def sync_zone_geometry_and_poi(zone_dict: dict, poi_dict: dict, cpu_cores: int = 1, verbose: bool = False) -> dict:
    """Synchronize zone cells and POIs to update zone_id attribute for POIs and poi_id_list attribute for zone cells

    Args:
        zone_dict (dict): Zone cells
        poi_dict (dict): POIs

    Returns:
        dict: the updated zone_dict and poi_dict
    """

    # Create a pool of worker processes
    if verbose:
        print(f"  : Parallel synchronizing POIs and Zones using Pool with {cpu_cores} CPUs. Please wait...")

    # create deepcopy of zone_dict and poi_dict to avoid modifying the original dict
    zone_cp = copy.deepcopy(zone_dict)
    poi_cp = copy.deepcopy(poi_dict)

    # Pre-process zone geometries
    for zone_id, zone in zone_dict.items():
        if isinstance(zone["geometry"], str):
            zone["geometry"] = shapely.from_wkt(zone["geometry"])

    # Pre-process node geometries
    for poi_id, poi in poi_dict.items():
        if isinstance(poi["geometry"], str):
            poi["geometry"] = shapely.from_wkt(poi["geometry"])

    # Create a spatial index for zone geometries
    zone_geometries = [zone["geometry"] for zone in zone_dict.values()]
    zone_ids = list(zone_dict.keys())
    zone_index = STRtree(zone_geometries)

    # Prepare arguments for the pool
    # args_list = [(poi_id, poi, zone_cp) for poi_id, poi in poi_cp.items()]

    # Prepare node batches for multiprocessing
    chunk_size = pkg_settings["data_chunk_size"]

    poi_items = list(poi_dict.items())
    poi_batches = [poi_items[i:i + chunk_size] for i in range(0, len(poi_items), chunk_size)]

    # Prepare arguments for multiprocessing
    # args_list = [(node_id, node, zone_dict, zone_index, zone_ids) for node_id, node in node_dict.items()]
    args_list = [(batch, zone_dict, zone_index, zone_ids) for batch in poi_batches]

    with Pool(processes=cpu_cores) as pool:
        # Distribute work to the pool
        results = list(tqdm(pool.imap(_sync_zones_geometry_with_poi, args_list), total=len(args_list)))
        pool.close()
        pool.join()

    # results = process_map(_sync_zones_geometry_with_poi, args_list)

    # Gather results
    # Ensure results is a list of lists and not None
    results = [result for result in results if result is not None]
    for batch_results in results:
        for poi_id, poi, zone_name in batch_results:
            if zone_name is not None:
                zone_cp[zone_name]["poi_id_list"].append(poi_id)
            poi_cp[poi_id] = poi

    if verbose:
        print("  : Successfully synchronized zone and poi geometry")
    return {"zone_dict": zone_cp, "poi_dict": poi_cp}


def sync_zone_centroid_and_poi(zone_dict: dict, poi_dict: dict, cpu_cores: int = 1, verbose: bool = False) -> dict:
    """Synchronize zone in centroids and nodes to update zone_id attribute for nodes

    Args:
        zone_dict (dict): Zone cells
        node_dict (dict): Nodes

    Returns:
        dict: the updated zone_dict and node_dict

    """

    zone_cp = copy.deepcopy(zone_dict)
    poi_cp = copy.deepcopy(poi_dict)

    zone_point_id = {
        shapely.geometry.Point(zone_cp[zone_id]["x_coord"], zone_cp[zone_id]["y_coord"]): zone_id
        for zone_id in zone_cp
    }

    multipoint_zone = shapely.geometry.MultiPoint(
        [shapely.geometry.Point(zone_cp[i]["x_coord"], zone_cp[i]["y_coord"]) for i in zone_cp])

    # Prepare data for multiprocessing
    args = [(poi_id, poi, multipoint_zone, zone_point_id) for poi_id, poi in poi_cp.items()]
    cpu_cores = pkg_settings["set_cpu_cores"]

    with Pool(cpu_cores) as pool:
        results = list(tqdm(pool.imap(_sync_zones_centroid_with_poi, args), total=len(poi_cp)))
        pool.close()
        pool.join()

    # Update zone_cp with the results
    for poi_id, zone_id in results:
        poi_cp[poi_id]["zone_id"] = zone_id
        zone_cp[zone_id]["poi_id_list"].append(poi_id)

    # for poi_id, poi in tqdm(poi_cp.items()):
    #     poi_point = shapely.geometry.Point(poi.x_coord, poi.y_coord)
    #     closest_zone_point = find_closest_point(poi_point, multipoint_zone)[0]
    #     zone_id = zone_point_id[closest_zone_point]
    #     poi.zone_id = zone_id
    #     zone_cp[zone_id].poi_id_list.append(poi_id)

    if verbose:
        print("  : Successfully synchronized zone and poi geometry")

    return {"zone_dict": zone_cp, "poi_dict": poi_cp}


@func_running_time
def calc_zone_od_matrix(zone_dict: dict,
                        *,
                        cpu_cores: int = -1,
                        sel_orig_zone_id: list = [],
                        sel_dest_zone_id: list = [],
                        pct: float = 0.1, verbose: bool = False) -> dict[tuple[str, str], float]:
    """Calculate the zone-to-zone distance matrix in chunks to handle large datasets.

    Args:
        zone_dict (dict): Zone cells
        cpu_cores (int): number of cpu cores
        sel_orig_zone_id (list): selected origin zones for calculation
        sel_dest_zone_od (list): selected destination zones for calculation
        pct: the percentage to randomly select zones from given zone_dict.
        verbose: whether to printout processing messages.

    Returns:
        dict: the zone-to-zone distance matrix
    """
    # Prepare arguments for the pool
    print(f"  : Parallel calculating zone-to-zone distance matrix using Pool with {cpu_cores} CPUs. Please wait...")

    # deepcopy the origin dictionary to avoid potential error.
    zone_inside = copy.deepcopy(zone_dict)
    total_zones = len(zone_inside)

    # define selected_zone_id
    selected_zone_id = []

    # Prepare node batches for multiprocessing
    chunk_size = pkg_settings["data_chunk_size"]

    # not sel_orig_zone_id and sel_dest_zone_id are not specified
    # use pct to randomly select zones from zone_inside
    if not sel_orig_zone_id and not sel_dest_zone_id:

        # Calculate the number of keys to select based on the given percentage
        num_keys_to_select = int(total_zones * pct)

        # Randomly select keys from the dictionary
        selected_keys = random.sample(list(zone_inside.keys()), num_keys_to_select)

        # Create a sub-dictionary with the selected keys
        zone_inside = {key: zone_inside[key] for key in selected_keys}
        print(f"  : Randomly select {num_keys_to_select} zones from {pct * 100} % of {total_zones}")

        selected_zone_id = list(zone_inside.keys())

    # if origin and destination zones are specified
    # origin -> all, plus all -> destination, replace duplicated pairs
    elif sel_orig_zone_id and sel_dest_zone_id:
        selected_zone_id = list(set(sel_orig_zone_id + sel_dest_zone_id))

    # if only origin zones specified, origin -> all pairs
    elif sel_orig_zone_id:
        selected_zone_id = list(set(sel_orig_zone_id))

    # if only destination zones specified, all -> destinations pairs
    else:
        selected_zone_id = list(set(sel_dest_zone_id))

    # convert zone_inside to dataframe with only id, x_coord and y_coord
    selected_data = [(zone['id'], zone['x_coord'], zone['y_coord'])
                     for zone in zone_inside.values()]
    df_zone_coords = pd.DataFrame(selected_data, columns=['id', 'x_coord', 'y_coord'])

    # crate selected df
    df_zone_coords_selected = df_zone_coords[df_zone_coords["id"].isin(selected_zone_id)]

    # calculate total OD combinations from given number of zones
    num_selected_zones = df_zone_coords.shape[0]

    if verbose:
        print("  : Generate OD combinations...")

    # Generate OD combinations using combinations_with_replacement
    if not sel_orig_zone_id and not sel_dest_zone_id:
        combinations = itertools.combinations(df_zone_coords.itertuples(index=False), 2)
        total_combinations = math.comb(num_selected_zones + 1, 2)

    else:
        # Define a function to process a combination
        def process_combinations(chunk):
            unique_pairs = set()
            for row1, row2 in chunk:
                # Use frozenset to eliminate ordering differences
                if row1 != row2:
                    row_pair = frozenset([row1, row2])
                    unique_pairs.add(row_pair)
            return unique_pairs

        # Create combinations between rows of df1 and df2
        combinations = list(itertools.product(df_zone_coords_selected.itertuples(index=False),
                                              df_zone_coords.itertuples(index=False)))

        # Split combinations into chunks for parallel processing
        chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

        # Use joblib to parallelize the processing of combinations
        results = Parallel(n_jobs=cpu_cores)(
            delayed(process_combinations)(chunk) for chunk in chunks)

        # Merge results from all workers into a single set
        combinations = set()
        for result in results:
            combinations.update(result)
        total_combinations = len(combinations)

    selected_combinations = list(tqdm(itertools.islice(combinations, total_combinations),
                                      total=total_combinations, desc="  :Generate OD Combinations"))

    # printout message to inform total number of zones and total number of combinations
    print(f"  : Total zones {num_selected_zones}, will generate OD combinations {total_combinations}.")

    if verbose:
        print("  : Extract OD coordinates from zones...")

    # Extract latitudes and longitudes for OD pairs in parallel
    def extract_coordinates(zone_pair):
        zone1, zone2 = zone_pair
        return (zone1.y_coord, zone1.x_coord, zone2.y_coord, zone2.x_coord, str(zone1.id), str(zone2.id))

    extracted_data = Parallel(n_jobs=cpu_cores)(delayed(extract_coordinates)(
        zone_pair) for zone_pair in tqdm(selected_combinations, desc="  :Extract OD Coordinates from zones"))

    if verbose:
        print("  : Prepare OD longitudes and latitudes from combinations for parallel computing...")

    # Initialize empty numpy arrays and mapping dictionary
    zone_od_dist = {}

    # Extract latitudes, longitudes, and zone pairs
    lat1, lon1, lat2, lon2, zone_o_str, zone_d_str = zip(*extracted_data)

    # Create the OD DataFrame
    df_od = pd.DataFrame({
        'lat1': lat1,
        'lon1': lon1,
        'lat2': lat2,
        'lon2': lon2})

    # Split dataset into batches
    batches = [df_od.iloc[i:i + chunk_size]
               for i in range(0, len(df_od), chunk_size)]

    # Batch processing function
    def process_batch(batch):
        lat1, lon1, lat2, lon2 = batch
        return calc_distance_on_unit_haversine(lat1, lon1, lat2, lon2)

    # Define a function that processes each batch in parallel
    def parallel_process_batches(batches):
        # Use Joblib to process batches in parallel
        results = Parallel(n_jobs=cpu_cores)(delayed(process_batch)(
            (batch['lat1'].values, batch['lon1'].values, batch['lat2'].values, batch['lon2'].values))
            for batch in tqdm(batches, desc="  :Calculate Distance"))
        return np.concatenate(results)

    # calculate the distance
    distances = parallel_process_batches(batches)

    # od paris
    print("  : preparing zone_od_dist...")
    df_od_dist = pd.DataFrame()
    df_od_dist["zone_o"] = zone_o_str
    df_od_dist["zone_d"] = zone_d_str
    df_od_dist["dist"] = distances

    zone_od_dist = df_od_dist.set_index(["zone_o", "zone_d"])["dist"].to_dict()

    if verbose:
        print("  : Successfully calculated zone-to-zone distance matrix")

    return zone_od_dist
