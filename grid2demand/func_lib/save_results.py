'''
##############################################################
# Created Date: Friday, October 11th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from pyufunc import path2linux, generate_unique_filename
import os
import pandas as pd


def save_demand(self, overwrite_file: bool = True,
                is_demand_with_geometry: bool = False) -> None:
    """Generate demand.csv file"""

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "demand.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "demand.csv")))

    # check if df_demand exists
    if not hasattr(self, "df_demand"):
        print("  : Could not save demand file: df_demand does not exist. Please run run_gravity_model() first.")
    else:

        # df_demand_non_zero = self.df_demand[self.df_demand["volume"] > 0]

        col_name = ["o_zone_id", "d_zone_id", "volume"]

        # Re-generate demand based on mode type
        self.df_demand["volume"] = self.df_demand["volume"] * self.pkg_settings["mode_type"].get(self.mode_type, 1)

        df_demand_res = self.df_demand[col_name].copy()

        # fill name with 0
        df_demand_res.fillna(0, inplace=True)

        df_demand_res.to_csv(path_output, index=False)
        print(f"  : Successfully saved demand.csv to {self.output_dir}")
    return None


def save_agent(self, overwrite_file: bool = True) -> None:
    """Generate agent.csv file"""

    if not hasattr(self, "df_agent"):
        print("  : Could not save agent file: df_agent does not exist."
              " Please run gen_agent_based_demand() first.")
        return None

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "agent.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "agent.csv")))
    self.df_agent.to_csv(path_output, index=False)
    print(f"  : Successfully saved agent.csv to {self.output_dir}")
    return None


def save_zone(self, overwrite_file: bool = True) -> None:
    """Generate zone.csv file"""

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "zone.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "zone.csv")))

    # check if zone_dict exists
    if not hasattr(self, "zone_dict"):
        print("  : Could not save zone file: zone_dict does not exist. \
            Please run sync_geometry_between_zone_and_node_poi() first.")
    else:
        zone_df = pd.DataFrame(self.zone_dict.values())

        # change column name from id to node_id
        zone_df.rename(columns={"id": "zone_id"}, inplace=True)

        zone_df.to_csv(path_output, index=False)
        print(f"  : Successfully saved zone.csv to {self.output_dir}")
    return None


def save_node(self, overwrite_file: bool = True) -> None:
    """Generate node.csv file"""

    if not hasattr(self, "node_dict"):
        print("  : node_dict does not exist. Please run sync_geometry_between_zone_and_node_poi() first.")
        return None

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "node.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "node.csv")))

    # if activity type is used, merge node_dict and node_dict_activity_nodes
    if self.node_dict_activity_type:
        node_dict = {**self.node_dict, **self.node_dict_activity_nodes}
    else:
        node_dict = self.node_dict

    node_df = pd.DataFrame(node_dict.values())
    node_df.rename(columns={"id": "node_id"}, inplace=True)
    node_df.to_csv(path_output, index=False)
    print(f"  : Successfully saved updated node to node.csv to {self.output_dir}")
    return None


def save_poi(self, overwrite_file: bool = True) -> None:
    """Generate poi.csv file"""

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "poi.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "poi.csv")))

    # check if poi_dict exists
    if not hasattr(self, "poi_dict"):
        print("  : Could not save updated poi file: poi_dict does not exist. Please run load_poi() first.")
    else:
        poi_df = pd.DataFrame(self.poi_dict.values())

        # rename column name from id to poi_id
        poi_df.rename(columns={"id": "poi_id"}, inplace=True)
        poi_df.to_csv(path_output, index=False)
        print(f"  : Successfully saved updated poi to poi.csv to {self.output_dir}")
    return None


def save_zone_od_dist_table(self, overwrite_file: bool = True) -> None:
    """Generate zone_od_dist_table.csv file"""

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "zone_od_dist_table.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "zone_od_dist_table.csv")))

    # check if zone_od_dist_matrix exists
    if not hasattr(self, "zone_od_dist_matrix"):
        print("  : zone_od_dist_matrix does not exist. Please run calc_zone_od_distance_matrix() first.")
    else:
        od_dist_list = [[key[0], key[1], value] for key, value in self.zone_od_dist_matrix.items()]
        zone_od_dist_table_df = pd.DataFrame(od_dist_list)
        zone_od_dist_table_df = zone_od_dist_table_df[["o_zone_id", "d_zone_id", "dist_km", ]]
        zone_od_dist_table_df.to_csv(path_output, index=False)
        print(f"  : Successfully saved zone_od_dist_table.csv to {self.output_dir}")
    return None


def save_zone_od_dist_matrix(self, overwrite_file: bool = True) -> None:
    """Generate zone_od_dist_matrix.csv file"""

    if overwrite_file:
        path_output = path2linux(os.path.join(self.output_dir, "zone_od_dist_matrix.csv"))
    else:
        path_output = generate_unique_filename(path2linux(os.path.join(self.output_dir, "zone_od_dist_matrix.csv")))

    # check if zone_od_dist_matrix exists
    if not hasattr(self, "zone_od_dist_matrix"):
        print(
            "  : zone_od_dist_matrix does not exist. Please run calc_zone_od_distance_matrix() first.")
    else:
        od_dist_list = [[key[0], key[1], value] for key, value in self.zone_od_dist_matrix.items()]
        zone_od_dist_table_df = pd.DataFrame(od_dist_list)
        zone_od_dist_table_df = zone_od_dist_table_df[["o_zone_id", "d_zone_id", "dist_km", ]]
        zone_od_dist_matrix_df = zone_od_dist_table_df.pivot(index='o_zone_id',
                                                             columns='d_zone_id',
                                                             values='dist_km')

        zone_od_dist_matrix_df.to_csv(path_output)
        print(f"  : Successfully saved zone_od_dist_matrix.csv to {self.output_dir}")
    return None
