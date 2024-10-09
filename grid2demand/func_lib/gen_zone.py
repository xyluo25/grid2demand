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
from joblib import Parallel, delayed


# supporting functions
def _get_lng_lat_min_max(node_dict: dict[str, "Node"]) -> list:
    """Get the boundary of the study area using faster numpy operations.

    Args:
        node_dict (dict[int, Node]): node_dict {node_id: Node}

    Returns:
        list: [min_lng, max_lng, min_lat, max_lat]
    """

    # Extract x_coords and y_coords from node_dict
    x_coords = np.array([node.x_coord for node in node_dict.values()])
    y_coords = np.array([node.y_coord for node in node_dict.values()])

    # Get min/max values using numpy operations (vectorized)
    coord_x_min = np.min(x_coords)
    coord_x_max = np.max(x_coords)
    coord_y_min = np.min(y_coords)
    coord_y_max = np.max(y_coords)

    # Return the boundary with a small buffer
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


def cvt_node_poi_to_arrays(input_dict: dict) -> tuple:
    """Convert node or poi dictionary to arrays

    Args:
        input_dict (dict): Node or POI dictionary

    Returns:
        tuple: Node or POI arrays
    """
    # Create numpy arrays for node coordinates
    coords = np.array([[node_info["x_coord"], node_info["y_coord"]]
                      for node_info in input_dict.values()])
    ids = np.array([node_info["id"] for node_info in input_dict.values()])
    return (ids, coords)


def cvt_zone_geometry_to_arrays(zone_dict: dict) -> tuple:
    """Convert zone dictionary to arrays

    Args:
        zone_dict (dict): Zone dictionary

    Returns:
        tuple: Zone arrays
    """
    # Create numpy arrays for zone bounding boxes and geometries
    zone_bboxes = []
    zone_polygons = []
    for zone_info in zone_dict.values():
        zone_bboxes.append([zone_info["x_min"], zone_info["x_max"],
                           zone_info["y_min"], zone_info["y_max"]])
        zone_polygon = shapely.from_wkt(zone_info["geometry"])
        zone_polygons.append(zone_polygon)
    return (np.array(zone_bboxes), np.array(zone_polygons))


def cvt_zone_centroid_to_arrays(zone_dict: dict) -> tuple:
    """Convert zone dictionary to arrays

    Args:
        zone_dict (dict): Zone dictionary

    Returns:
        tuple: Zone arrays
    """
    # Create numpy arrays for zone with x_coord and y_coord
    zone_centroids = np.array([[zone_info["x_coord"], zone_info["y_coord"]]
                              for zone_info in zone_dict.values()])
    zone_ids = np.array([zone_info["id"] for zone_info in zone_dict.values()])
    return (zone_ids, zone_centroids)


def points_map_to_zone(zone_bbox: np.array, zone_polygon: np.array, pt_coords: np.array, pt_ids: np.array) -> np.array:
    """Map points to zones

    Args:
        zone_bbox (np.array): the boundary of the zone
        zone_polygon (np.array): the geometry of the zone in WKT format
        pt_coords (np.array): the coordinates of the all points
        pt_ids (np.array): the ids of the all points

    Returns:
        np.array: the ids of the points within the zone
    """

    # Extract bounding box coordinates
    x_min, x_max, y_min, y_max = zone_bbox

    # Vectorized bounding box check
    within_bbox = (pt_coords[:, 0] >= x_min) & (pt_coords[:, 0] <= x_max) & \
                  (pt_coords[:, 1] >= y_min) & (pt_coords[:, 1] <= y_max)

    # Check within the bounding box
    candidate_nodes = pt_coords[within_bbox]
    candidate_ids = pt_ids[within_bbox]

    return [
        candidate_ids[i]
        for i, node in enumerate(candidate_nodes)
        if zone_polygon.contains(shapely.Point(node[0], node[1]))
    ]


def add_zone_to_nodes(node_dict: dict, zone_dict: dict) -> dict:
    """Update node_dict with zone_id

    Args:
        node_dict (dict): the node dictionary
        zone_dict (dict): the zone dictionary

    Returns:
        dict: the updated node dictionary
    """

    sum_missing_nodes = 0
    for zone_id, zone_info in zone_dict.items():
        node_id_list = zone_info["node_id_list"]
        for node_id in node_id_list:
            try:
                # Update zone_id for each node
                node_dict[str(node_id)]["zone_id"] = zone_id
            except KeyError:
                sum_missing_nodes += 1
                # print(f"  : Node {node_id} not found in node_dict")
    if sum_missing_nodes > 0:
        print(f"  : {sum_missing_nodes} Nodes not found in node_dict")
    return node_dict


def add_zone_to_pois(poi_dict: dict, zone_dict: dict) -> dict:
    """Update poi_dict with zone_id

    Args:
        poi_dict (dict): the poi dictionary
        zone_dict (dict): the zone dictionary

    Returns:
        dict: the updated poi dictionary
    """

    sum_missing_poi = 0
    for zone_id, zone_info in zone_dict.items():
        poi_id_list = zone_info["poi_id_list"]
        for poi_id in poi_id_list:
            try:
                # Update zone_id for poi
                poi_dict[str(poi_id)]["zone_id"] = zone_id
            except KeyError:
                sum_missing_poi += 1
                # print(f"  : Poi {poi_id} not found in poi_dict")
    if sum_missing_poi > 0:
        print(f"  : {sum_missing_poi} POIs not found in poi_dict")
    return poi_dict

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

    coord_x_min, coord_x_max, coord_y_min, coord_y_max = _get_lng_lat_min_max(node_dict)

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
    x_block_min_lst = [coord_x_min + i * x_block_width for i in range(num_x_blocks)]
    y_block_min_lst = [coord_y_min + i * y_block_height for i in range(num_y_blocks)]

    x_block_minmax_list = list(zip(x_block_min_lst[:-1],
                                   x_block_min_lst[1:])) + [(x_block_min_lst[-1],
                                                             coord_x_max)]
    y_block_minmax_list = list(zip(y_block_min_lst[:-1],
                                   y_block_min_lst[1:])) + [(y_block_min_lst[-1],
                                                             coord_y_max)]

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
def sync_zone_geometry_and_node(zone_dict: dict, node_dict: dict, cpu_cores: int = -1, verbose: bool = False) -> dict:
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

    # Step 1: Add node_id_list to each zone in zone_dict

    # convert node_dict and zone_dict to arrays
    node_ids, node_coords = cvt_node_poi_to_arrays(node_cp)
    zone_bboxes, zone_polygons = cvt_zone_geometry_to_arrays(zone_cp)

    if verbose:
        print(f"  : Parallel sync zone geometry and node using {cpu_cores} cpu cores.")

    # Parallel processing for zones
    results = Parallel(n_jobs=cpu_cores)(delayed(points_map_to_zone)(
        zone_bboxes[idx], zone_polygons[idx], node_coords, node_ids) for idx in tqdm(range(len(zone_bboxes)), desc="  :Node to Zone"))

    # Collect results back into zone_dict
    for idx, (zone_id, zone_info) in enumerate(zone_cp.items()):
        zone_info["node_id_list"] = results[idx]

    if verbose:
        print("  : Successfully synchronized zone and node geometry")

    # Step 2: Add zone_id back to each node in node_dict
    node_cp = add_zone_to_nodes(node_cp, zone_cp)

    return {"zone_dict": zone_cp, "node_dict": node_cp}


def sync_zone_centroid_and_node(zone_dict: dict, node_dict: dict, cpu_cores: int = -1, verbose: bool = False) -> dict:
    """Synchronize zone in centroids and nodes to update zone_id attribute for nodes

    Args:
        zone_dict (dict): Zone cells
        node_dict (dict): Nodes

    Returns:
        dict: the updated zone_dict and node_dict

    """

    # Deepcopy the dictionary
    zone_cp = copy.deepcopy(zone_dict)
    node_cp = copy.deepcopy(node_dict)

    # convert node_dict and zone_dict to arrays
    node_ids, node_coords = cvt_node_poi_to_arrays(node_cp)
    zone_ids, zone_centroids = cvt_zone_centroid_to_arrays(zone_cp)

    if verbose:
        print(f"  : Parallel sync zone centroid and node using {cpu_cores} cpu cores.")

    # Step 1: mapping nodes with closest zone centroids
    # Function to compute the closest zone for each batch of nodes
    def process_node_batch(node_batch_coords, node_batch_ids, zone_centroids, zone_ids):
        closest_zone_ids = []
        for i in range(len(node_batch_coords)):
            # Calculate distances between the node and all zone centroids
            distances = np.sqrt((zone_centroids[:, 0] - node_batch_coords[i, 0])**2 +
                                (zone_centroids[:, 1] - node_batch_coords[i, 1])**2)
            # Find the index of the closest zone
            closest_zone_idx = np.argmin(distances)
            closest_zone_ids.append(zone_ids[closest_zone_idx])
        return node_batch_ids, np.array(closest_zone_ids)

    # Split nodes into batches and process them in parallel
    node_chunks = np.array_split(np.arange(len(node_coords)), cpu_cores)
    results = Parallel(n_jobs=cpu_cores)(
        delayed(process_node_batch)(
            node_coords[chunk], node_ids[chunk], zone_centroids, zone_ids)
        for chunk in tqdm(node_chunks, desc="  :Node to closest Zone")
    )

    # Create a mapping from node_ids to their index positions
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Update node_zone_mapping using the mapping to get the correct index positions
    node_zone_mapping = np.zeros(len(node_ids), dtype=zone_ids.dtype)
    for node_batch_ids, closest_zone_batch in results:
        node_batch_indices = [node_id_to_index[node_id]
                              for node_id in node_batch_ids]
        node_zone_mapping[node_batch_indices] = closest_zone_batch

    # Step 2: Update node_dict with zone_id, and zone_dict with node_id_list
    missing_nodes = 0
    missing_zones = 0
    for node_id, zone_id in zip(node_ids, node_zone_mapping):
        try:
            node_cp[node_id]["zone_id"] = zone_id
        except KeyError:
            missing_nodes += 1

        try:
            zone_cp[zone_id]["node_id_list"].append(node_id)
        except KeyError:
            missing_zones += 1

    if missing_nodes > 0:
        print(f"  : {missing_nodes} Nodes not found in node_dict")
    if missing_zones > 0:
        print(f"  : {missing_zones} Zones not found in zone_dict")

    return {"zone_dict": zone_cp, "node_dict": node_cp}


@func_running_time
def sync_zone_geometry_and_poi(zone_dict: dict, poi_dict: dict, cpu_cores: int = -1, verbose: bool = False) -> dict:
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

    # Step 1: Add node_id_list to each zone in zone_dict

    # convert node_dict and zone_dict to arrays
    poi_ids, poi_coords = cvt_node_poi_to_arrays(poi_cp)
    zone_bboxes, zone_polygons = cvt_zone_geometry_to_arrays(zone_cp)

    if verbose:
        print(f"  : Parallel sync zone geometry and node using {cpu_cores} cpu cores.")

    # Parallel processing for zones
    results = Parallel(n_jobs=cpu_cores)(delayed(points_map_to_zone)(
        zone_bboxes[idx], zone_polygons[idx], poi_coords, poi_ids) for idx in tqdm(range(len(zone_bboxes)), desc="  :POI to Zone"))

    # Collect results back into zone_dict
    for idx, (zone_id, zone_info) in enumerate(zone_cp.items()):
        zone_info["poi_id_list"] = results[idx]

    if verbose:
        print("  : Successfully synchronized zone and node geometry")

    # Step 2: Add zone_id back to each node in node_dict
    poi_cp = add_zone_to_pois(poi_cp, zone_cp)

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

    # convert node_dict and zone_dict to arrays
    poi_ids, poi_coords = cvt_node_poi_to_arrays(poi_cp)
    zone_ids, zone_centroids = cvt_zone_centroid_to_arrays(zone_cp)

    if verbose:
        print(f"  : Parallel sync zone centroid and poi using {cpu_cores} cpu cores.")

    # Step 1: mapping nodes with closest zone centroids
    # Function to compute the closest zone for each batch of nodes
    def process_node_batch(poi_batch_coords, poi_batch_ids, zone_centroids, zone_ids):
        closest_zone_ids = []
        for i in range(len(poi_batch_coords)):
            # Calculate distances between the node and all zone centroids
            distances = np.sqrt((zone_centroids[:, 0] - poi_batch_coords[i, 0])**2 +
                                (zone_centroids[:, 1] - poi_batch_coords[i, 1])**2)
            # Find the index of the closest zone
            closest_zone_idx = np.argmin(distances)
            closest_zone_ids.append(zone_ids[closest_zone_idx])
        return poi_batch_ids, np.array(closest_zone_ids)

    # Split nodes into batches and process them in parallel
    poi_chunks = np.array_split(np.arange(len(poi_coords)), cpu_cores)
    results = Parallel(n_jobs=cpu_cores)(
        delayed(process_node_batch)(
            poi_coords[chunk], poi_ids[chunk], zone_centroids, zone_ids)
        for chunk in tqdm(poi_chunks, desc="  :POI to closest Zone")
    )

    # Create a mapping from node_ids to their index positions
    poi_id_to_index = {poi_id: idx for idx, poi_id in enumerate(poi_ids)}

    # Update node_zone_mapping using the mapping to get the correct index positions
    poi_zone_mapping = np.zeros(len(poi_ids), dtype=zone_ids.dtype)
    for poi_batch_ids, closest_zone_batch in results:
        poi_batch_indices = [poi_id_to_index[poi_id]
                             for poi_id in poi_batch_ids]
        poi_zone_mapping[poi_batch_indices] = closest_zone_batch

    # Step 2: Update node_dict with zone_id, and zone_dict with node_id_list
    missing_poi = 0
    missing_zones = 0
    for poi_id, zone_id in zip(poi_ids, poi_zone_mapping):
        try:
            poi_cp[poi_id]["zone_id"] = zone_id
        except KeyError:
            missing_poi += 1

        try:
            zone_cp[zone_id]["node_id_list"].append(poi_id)
        except KeyError:
            missing_zones += 1

    if missing_poi > 0:
        print(f"  : {missing_poi} POIs not found in poi_dict")
    if missing_zones > 0:
        print(f"  : {missing_zones} Zones not found in zone_dict")

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
    print(f"  : Parallel calculating zone-to-zone distance matrix using {cpu_cores} CPUs. Please wait...")

    total_zones = len(zone_dict)
    chunk_size = pkg_settings["data_chunk_size"]

    # Use pct to randomly select zones if no origin/destination zones are specified
    if not sel_orig_zone_id and not sel_dest_zone_id:
        num_keys_to_select = int(total_zones * pct)
        selected_zone_ids = random.sample(list(zone_dict.keys()), num_keys_to_select)
        print(f"  : Randomly selected {num_keys_to_select} from {pct * 100} % of {total_zones} zones")
    else:
        selected_zone_ids = set(sel_orig_zone_id + sel_dest_zone_id)

    # Extract selected zones and their coordinates as numpy arrays
    zone_ids = np.array([zone['id'] for zone in zone_dict.values() if zone['id'] in selected_zone_ids])
    zone_coords = np.array(
        [[zone['x_coord'],
          zone['y_coord']] for zone in zone_dict.values() if zone['id'] in selected_zone_ids])

    # Parallel generation of combinations using joblib
    zone_chunks = [zone_ids[i:i + chunk_size] for i in range(0, len(zone_ids), chunk_size)]

    # function to generate combinations for a chunk of zones
    def generate_combinations_chunk(chunk1, chunk2, combination_type="product"):
        if combination_type == "product":
            return list(itertools.product(chunk1, chunk2))
        elif combination_type == "combinations":
            return list(itertools.combinations(chunk1, 2))

    if sel_orig_zone_id or sel_dest_zone_id:
        # Using itertools.product if origin/destination are specified
        combinations = Parallel(n_jobs=cpu_cores)(
            delayed(generate_combinations_chunk)(chunk1, zone_ids, "product")
            for chunk1 in tqdm(zone_chunks, desc="  :Generating OD combinations (Product)"))
    else:
        # Using itertools.combinations if no specific origin/destination
        combinations = Parallel(n_jobs=cpu_cores)(
            delayed(generate_combinations_chunk)(chunk1, None, "combinations")
            for chunk1 in tqdm(zone_chunks, desc="  :Generating OD combinations (Combinations)"))

    # Flatten the list of combinations
    combinations = [item for sublist in combinations for item in sublist]
    total_combinations = len(combinations)

    # Prepare OD coordinates from the combinations
    print(f"  : Preparing OD coordinates for {total_combinations} zone combinations...")

    # function to extract coordinates for a chunk of OD combinations
    def extract_coordinates_chunk(combinations_chunk, zone_ids, zone_coords):
        extracted = []
        for comb in combinations_chunk:
            orig_zone, dest_zone = comb
            orig_idx = np.nonzero(zone_ids == orig_zone)[0][0]
            dest_idx = np.nonzero(zone_ids == dest_zone)[0][0]
            lat1, lon1 = zone_coords[orig_idx]
            lat2, lon2 = zone_coords[dest_idx]
            extracted.append((lat1, lon1, lat2, lon2, str(orig_zone), str(dest_zone)))
        return extracted

    # Extract OD coordinates in chunks for parallel processing
    combination_chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

    print(f"  : Extracting OD coordinates in {len(combination_chunks)} chunks...")

    # Extract coordinates in parallel using chunking
    extracted_data = Parallel(n_jobs=cpu_cores)(
        delayed(extract_coordinates_chunk)(chunk, zone_ids, zone_coords)
        for chunk in tqdm(combination_chunks, desc="  :Extract OD Coordinates"))

    # Flatten extracted data
    extracted_data = [item for sublist in extracted_data for item in sublist]

    # Store extracted data into numpy arrays
    lat1, lon1, lat2, lon2, zone_o_str, zone_d_str = zip(*extracted_data)
    lat1, lon1, lat2, lon2 = np.array(lat1), np.array(lon1), np.array(lat2), np.array(lon2)

    # Calculate distances in chunks
    chunk_size = int(len(lat1) / cpu_cores)

    def process_batch(lon1, lat1, lon2, lat2):
        return calc_distance_on_unit_haversine(lon1, lat1, lon2, lat2)

    # Split into batches for parallel processing
    batches = [(lon1[i:i + chunk_size], lat1[i:i + chunk_size], lon2[i:i + chunk_size], lat2[i:i + chunk_size])
               for i in range(0, len(lat1), chunk_size)]

    # Process batches in parallel
    distances = Parallel(n_jobs=cpu_cores)(
        delayed(process_batch)(batch[0], batch[1], batch[2], batch[3])
        for batch in tqdm(batches, desc="  :Calculating Distances"))

    # Concatenate results from all batches
    distances = np.concatenate(distances)

    # Create the OD distance matrix
    zone_od_dist = {(zone_o, zone_d): dist for zone_o, zone_d, dist in zip(zone_o_str, zone_d_str, distances)}

    print("  : Successfully calculated zone-to-zone distance matrix.")
    return zone_od_dist
