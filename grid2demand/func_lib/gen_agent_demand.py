'''
# -*- coding:utf-8 -*-
##############################################################
# Created Date: Thursday, September 28th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from random import choice, uniform
import math
import re

import pandas as pd
from pyufunc import gmns_geo


def gen_agent_based_demand(node_dict: dict, zone_dict: dict,
                           path_demand: str = "",
                           df_demand: pd.DataFrame = "",
                           agent_type: str = "v",
                           time_period: str = "0000-2359",
                           verbose: bool = False) -> pd.DataFrame:
    """Generate agent-based demand data

    Args:
        node_dict (dict): dictionary of node objects
        zone_dict (dict): dictionary of zone objects
        path_demand (str): user provided demand data. Defaults to "".
        df_demand (pd.DataFrame): user provided demand dataframe. Defaults to "".
        agent_type (str): specify the agent type. Defaults to "v".
        verbose (bool): whether to print out processing message. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    # Validate time_period format
    time_period_pattern = re.compile(r"^\d{4}-\d{4}$")
    if not isinstance(time_period, str) or not time_period_pattern.match(time_period):
        raise ValueError(
            f"Error: time_period '{time_period}' must be a string in the format 'HHMM-HHMM'.")

    start_time_str, end_time_str = time_period.split('-')

    # Validate that the times are within the valid range
    try:
        start_time = int(start_time_str[:2]) * 60 + int(start_time_str[2:])
        end_time = int(end_time_str[:2]) * 60 + int(end_time_str[2:])
    except ValueError as e:
        raise ValueError("Error: time_period contains non-numeric values.") from e

    if not (0 <= start_time <= 1440) or not (0 <= end_time <= 1440):
        raise ValueError(
            "Error: time_period must be between '0000' and '2400'.")

    if start_time >= end_time:
        raise ValueError(
            "Error: start_time must be less than end_time in time_period.")

    # if path_demand is provided, read demand data from path_demand
    if path_demand:
        df_demand = pd.read_csv(path_demand)

    # if df_demand is provided, validate df_demand
    if df_demand.empty:
        print("Error: No demand data provided.")
        return pd.DataFrame()

    agent_lst = []
    for i in range(len(df_demand)):
        o_zone_id = df_demand.loc[i, 'o_zone_id']
        d_zone_id = df_demand.loc[i, 'd_zone_id']

        # o_node_id = choice(zone_dict[o_zone_id]["node_id_list"] + [""])
        # d_node_id = choice(zone_dict[d_zone_id]["node_id_list"] + [""])

        o_node_id_lst = zone_dict[o_zone_id]["node_id_list"]
        d_node_id_lst = zone_dict[d_zone_id]["node_id_list"]

        for o_node_id in o_node_id_lst:
            for d_node_id in d_node_id_lst:
                if o_node_id and d_node_id:
                    # Generate a random time within the specified time period
                    rand_time = math.ceil(uniform(start_time, end_time))

                    # Calculate hours and minutes from rand_time
                    hours = rand_time // 60
                    minutes = rand_time % 60

                    # Format departure_time as HHMM
                    departure_time = str(f"time:{hours:02d}{minutes:02d}")

                    agent_lst.append(
                        gmns_geo.Agent(
                            id=str(i + 1),
                            agent_type=agent_type,
                            o_zone_id=o_zone_id,
                            d_zone_id=d_zone_id,
                            o_node_id=o_node_id,
                            d_node_id=d_node_id,
                            geometry=(f"LINESTRING({node_dict[o_node_id]['x_coord']} {node_dict[o_node_id]['y_coord']},"
                                      f"{node_dict[d_node_id]['x_coord']} {node_dict[d_node_id]['y_coord']})"),
                            departure_time=departure_time
                        )
                    )

    if verbose:
        print("  :Successfully generated agent-based demand data.")

    return pd.DataFrame(agent_lst)
