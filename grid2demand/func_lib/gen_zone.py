# -*- coding:utf-8 -*-
##############################################################
# Created Date: Tuesday, September 5th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

from __future__ import absolute_import
import itertools
import pandas as pd
import shapely
import numpy as np

from grid2demand.utils_lib.net_utils import Zone, Node
from grid2demand.utils_lib.utils import calc_distance_on_unit_sphere, int2alpha
from grid2demand.utils_lib.utils import func_running_time


@func_running_time
def net2zone(node_dict: dict[int, Node],
             num_x_blocks: int = 0,
             num_y_blocks: int = 0,
             cell_width: float = 0,
             cell_height: float = 0) -> dict[str, Zone]:
    """Partition the study area into zone cells

    The priority of partition parameters is num_x_blocks, num_y_blocks > cell_width, cell_height.
    If num_x_blocks and num_y_blocks are given, use them to partition the study area.
    Else if cell_width and cell_height are given, use them to partition the study area.
    Else raise error.

    Parameters
        path_node: str, Path to the node.csv file, node.csv is GMNS format file
        num_x_blocks: int, Number of blocks in x direction, default 0
        num_y_blocks: int, Number of blocks in y direction, default 0
        cell_width: float, Width of each cell, unit in km, default 0
        cell_height: float, Height of each cell, unit in km, default 0

    Raises
        ValueError: Please provide num_x_blocks and num_y_blocks or cell_width and cell_height

    Returns
        Zone: dictionary, Zone cells with keys are zone names, values are Zone

    Examples:
        >>> zone_dict = net2zone(node_dict, num_x_blocks=10, num_y_blocks=10)
        >>> zone_dict['A1']
        Zone(id=0, name='A1', centroid_x=0.05, centroid_y=0.95, centroid='POINT (0.05 0.95)', x_min=0.0, x_max=0.1,
        y_min=0.9, y_max=1.0, geometry='POLYGON ((0.05 0.9, 0.1 0.9, 0.1 1, 0.05 1, 0.05 0.9))')

    """

    # convert node_dict to dataframe
    df_node = pd.DataFrame(node_dict.values())

    # get the boundary of the study area
    coord_x_min, coord_x_max = df_node['x_coord'].min(
    ) - 0.1, df_node['x_coord'].max() + 0.1
    coord_y_min, coord_y_max = df_node['y_coord'].min(
    ) - 0.1, df_node['y_coord'].max() + 0.1

    # Priority: num_x_blocks, number_y_blocks > cell_width, cell_height
    # if num_x_blocks and num_y_blocks are given, use them to partition the study area
    # else if cell_width and cell_height are given, use them to partition the study area
    # else raise error

    if num_x_blocks > 0 and num_y_blocks > 0:
        x_block_width = (coord_x_max - coord_x_min) / num_x_blocks
        y_block_height = (coord_y_max - coord_y_min) / num_y_blocks
    elif cell_width > 0 and cell_height > 0:
        x_dist_km = calc_distance_on_unit_sphere(
            (coord_x_min, coord_y_min), (coord_x_max, coord_y_min), unit='km')
        y_dist_km = calc_distance_on_unit_sphere(
            (coord_x_min, coord_y_min), (coord_x_min, coord_y_max), unit='km')

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
        return shapely.geometry.Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)])

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
            row_alpha = int2alpha(j)
            zone_dict[f"{row_alpha}{i}"] = Zone(
                id=zone_id_flag,
                name=f"{row_alpha}{i}",
                centroid_x=cell_polygon.centroid.x,
                centroid_y=cell_polygon.centroid.y,
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
    upper_points = [shapely.geometry.Point(zone_dict[zone_name].centroid_x,
                                           zone_dict[zone_name].centroid_y + y_block_height
                                           ) for zone_name in zone_upper_row]
    lower_points = [shapely.geometry.Point(zone_dict[zone_name].centroid_x,
                                           zone_dict[zone_name].centroid_y - y_block_height
                                           ) for zone_name in zone_lower_row]
    left_points = [shapely.geometry.Point(zone_dict[zone_name].centroid_x - x_block_width,
                                          zone_dict[zone_name].centroid_y
                                          ) for zone_name in zone_left_col]
    right_points = [shapely.geometry.Point(zone_dict[zone_name].centroid_x + x_block_width,
                                           zone_dict[zone_name].centroid_y
                                           ) for zone_name in zone_right_col]
    points_lst = upper_points + lower_points + left_points + right_points
    for i in range(len(points_lst)):
        zone_dict[f"gate{i}"] = Zone(
            id=zone_id_flag,
            name=f"gate{i}",
            centroid_x=points_lst[i].x,
            centroid_y=points_lst[i].y,
            centroid=points_lst[i],
            geometry=points_lst[i]
        )
        zone_id_flag += 1
    return zone_dict


@func_running_time
def sync_zone_and_node_geometry(zone_dict: dict, node_dict: dict) -> dict:
    """Map nodes to zone cells

    Parameters
        node_dict: dict, Nodes
        zone_dict: dict, zone cells

    Returns
        node_dict and zone_dict: dict, Update Nodes with zone id, update zone cells with node id list

    """
    for node_id in node_dict:
        for zone_name in zone_dict:
            if isinstance(node_dict[node_id].geometry, str):
                node_dict[node_id].geometry = shapely.from_wkt(node_dict[node_id].geometry)
            if isinstance(zone_dict[zone_name].geometry, str):
                zone_dict[zone_name].geometry = shapely.from_wkt(zone_dict[zone_name].geometry)

            if shapely.within(node_dict[node_id].geometry, zone_dict[zone_name].geometry):
                node_dict[node_id].zone_id = zone_dict[zone_name].id
                zone_dict[zone_name].node_id_list.append(node_id)
                break

    return {"node_dict": node_dict, "zone_dict": zone_dict}


@func_running_time
def sync_zone_and_poi_geometry(zone_dict: dict, poi_dict: dict) -> dict:
    """Synchronize zone cells and POIs to update zone_id attribute for POIs and poi_id_list attribute for zone cells

    Args:
        zone_dict (dict): Zone cells
        poi_dict (dict): POIs

    Returns:
        dict: the updated zone_dict and poi_dict
    """
    for poi_id in poi_dict:
        for zone_name in zone_dict:
            if isinstance(poi_dict[poi_id].geometry, str):
                poi_dict[poi_id].geometry = shapely.from_wkt(poi_dict[poi_id].geometry)
            if isinstance(zone_dict[zone_name].geometry, str):
                zone_dict[zone_name].geometry = shapely.from_wkt(zone_dict[zone_name].geometry)

            if shapely.within(poi_dict[poi_id].geometry, zone_dict[zone_name].geometry):
                poi_dict[poi_id].zone_id = zone_dict[zone_name].id
                zone_dict[zone_name].poi_id_list.append(poi_id)
                break
    return {"poi_dict": poi_dict, "zone_dict": zone_dict}


@func_running_time
def calc_zone_od_matrix(zone_dict: dict) -> dict[tuple[str, str], dict]:
    """Calculate the zone-to-zone distance matrix

    Args:
        zone_dict (dict): Zone cells

    Returns:
        dict: the zone-to-zone distance matrix
    """

    # convert zone_dict to dataframe
    df_zone = pd.DataFrame(zone_dict.values())

    # convert centroid from str to shapely.geometry.Point
    if isinstance(df_zone.loc[0, 'centroid'], str):
        df_zone['centroid'] = df_zone["centroid"].apply(shapely.from_wkt)

    len_df_zone = len(df_zone)

    # calculate zone-to-zone distance matrix and save to dict
    dist_dict = {
        (df_zone.loc[i, 'name'], df_zone.loc[j, 'name']): {
            "o_zone_id": df_zone.loc[i, 'id'],
            "o_zone_name": df_zone.loc[i, 'name'],
            "d_zone_id": df_zone.loc[j, 'id'],
            "d_zone_name": df_zone.loc[j, 'name'],
            "dist_km": calc_distance_on_unit_sphere(
                df_zone.loc[i, 'centroid'],
                df_zone.loc[j, 'centroid'],
                unit='km'),
            "volume": 0,
            "geometry": shapely.LineString(
                [df_zone.loc[i, 'centroid'], df_zone.loc[j, 'centroid']]),
        }
        for i, j in itertools.product(range(len_df_zone), range(len_df_zone))
    }
    return dist_dict