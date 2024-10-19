"""
# -*- coding:utf-8 -*-
##############################################################
# Created Date: Thursday, September 28th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
"""

import os

import pandas as pd
import shapely
from joblib import Parallel, delayed
from tqdm import tqdm
from pyufunc import (path2linux,
                     get_filenames_by_ext,
                     func_time)

from grid2demand.utils_lib.pkg_settings import pkg_settings
from grid2demand.utils_lib.utils import check_required_files_exist

from grid2demand.func_lib.read_node_poi import (read_node,
                                                read_poi,
                                                read_zone_by_geometry,
                                                read_zone_by_centroid)
from grid2demand.func_lib.gen_zone import (net2grid,
                                           map_zone_geometry_and_node,
                                           map_zone_geometry_and_poi,
                                           map_zone_centroid_and_node,
                                           map_zone_centroid_and_poi,
                                           calc_zone_od_matrix)
from grid2demand.func_lib.trip_rate_production_attraction import (gen_poi_trip_rate,
                                                                  gen_node_prod_attr)
from grid2demand.func_lib.gravity_model import (run_gravity_model,
                                                calc_zone_production_attraction)
from grid2demand.func_lib.gen_agent_demand import gen_agent_based_demand
from grid2demand.func_lib.save_results import (save_demand,
                                               save_agent,
                                               save_zone,
                                               save_node,
                                               save_poi,
                                               save_zone_od_dist_table,
                                               save_zone_od_dist_matrix)


class GRID2DEMAND:
    """A tool for generating zone-to-zone travel demand based on grid zones or TAZs"""

    def __init__(self,
                 input_dir: str = os.getcwd(),
                 *,
                 output_dir: str = "",
                 use_zone_id: bool = False,
                 mode_type: str = "auto",
                 verbose: bool = False,
                 **kwargs) -> None:
        """initialize GRID2DEMAND object

        Args:
            input_dir (str): input directory. Defaults to os.getcwd().
            output_dir (str): output directory. Defaults to "".
            use_zone_id (bool): whether to use zone_id. Defaults to False.
            verbose (bool): whether to print verbose information. Defaults to False.
            mode_type (str): the mode type. Defaults to "auto". Options: ["auto", "bike", "walk"]
            kwargs: additional keyword arguments
        """

        # initialize input parameters
        self.input_dir = path2linux(input_dir)
        self.output_dir = path2linux(output_dir) if output_dir else self.input_dir
        self.verbose = verbose
        self.use_zone_id = use_zone_id
        self.mode_type = mode_type
        self.kwargs = kwargs

        # check mode type is valid
        if self.mode_type not in ["auto", "bike", "walk"]:
            raise ValueError("Error: mode_type must be auto, bike or walk.")

        # update node_file, poi_file, zone_file if specified in kwargs
        if node_file := kwargs.get("node_file"):
            self.node_file = path2linux(node_file)

        if poi_file := kwargs.get("poi_file"):
            self.poi_file = path2linux(poi_file)

        if zone_file := kwargs.get("zone_file"):
            self.zone_file = path2linux(zone_file)

        # load default package settings, can be modified by user
        self.__load_pkg_settings()

        # # check input directory
        self.__check_input_dir()

    def __check_input_dir(self) -> None:
        """check input directory

        Raises:
            NotADirectoryError: Error: Input directory _input_dir_ does not exist.
            Exception: Error: Required files are not satisfied. Please check _required_files_ in _input_dir_.
        """

        if self.verbose:
            print("  : Checking input directory...")

        if not os.path.isdir(self.input_dir):
            raise NotADirectoryError(f"Error: Input directory {self.input_dir} does not exist.")

        # check if node, poi and zone files are in the input
        if hasattr(self, "node_file") and not os.path.exists(os.path.join(self.input_dir, self.node_file)):
            raise FileNotFoundError(f"Error: File {self.node_file} does not exist in {self.input_dir}.")

        if hasattr(self, "poi_file") and not os.path.exists(os.path.join(self.input_dir, self.poi_file)):
            raise FileNotFoundError(f"Error: File {self.poi_file} does not exist in {self.input_dir}.")

        if hasattr(self, "zone_file") and not os.path.exists(os.path.join(self.input_dir, self.zone_file)):
            raise FileNotFoundError(f"Error: File {self.zone_file} does not exist in {self.input_dir}.")

        if not (hasattr(self, "node_file") and hasattr(self, "poi_file")):
            # check required files in input directory
            dir_files = get_filenames_by_ext(self.input_dir, "csv")
            required_files = self.pkg_settings.get("required_files", [])

            is_required_files_exist = check_required_files_exist(required_files, dir_files, verbose=self.verbose)
            if not is_required_files_exist:
                raise Exception(
                    f"Error: Required files are not satisfied. Please check {required_files} in {self.input_dir}.")

            self.node_file = path2linux(os.path.join(self.input_dir, "node.csv"))
            self.poi_file = path2linux(os.path.join(self.input_dir, "poi.csv"))
        else:
            print(f"  : node_file and poi_file are specified from {self.node_file} and {self.poi_file}.")

        if hasattr(self, "zone_file"):
            print(f"  : zone_file is specified from {self.zone_file}.")
        else:
            # check optional files in input directory (zone.csv)
            optional_files = self.pkg_settings.get("optional_files", [])
            is_optional_files_exist = check_required_files_exist(optional_files, dir_files, verbose=False)

            if is_optional_files_exist:
                print(f"  : Optional file: {optional_files} found in {self.input_dir}.")
                print("  : Optional files could be used in the future steps.")

                self.zone_file = path2linux(os.path.join(self.input_dir, "zone.csv"))

        if self.verbose:
            print("  : Input directory is valid.\n")

    def __load_pkg_settings(self) -> None:
        """load default package settings and internal configuration variables"""
        if self.verbose:
            print("  : Loading default package settings...")
        self.pkg_settings = pkg_settings

        # add zone_id to node_dict if use_zone_id is True
        if self.use_zone_id and "zone_id" not in self.pkg_settings["node_fields"]:
            self.pkg_settings["node_fields"].append("zone_id")

        # set internal configuration variables to make sure the following steps are executed in order
        self.__config = {
            # set default zone is geometry or centroid as False
            "is_geometry": False,
            "is_centroid": False,

            # set default poi_trip_rate, node_prod_attr, zone_prod_attr as False
            "is_poi_trip_rate": False,
            "is_node_prod_attr": False,
            "is_zone_prod_attr": False,
            "is_zone_od_dist_matrix": False,
            "is_sync_geometry": False
        }

        if self.verbose:
            print("  : Package settings loaded successfully.\n")

    def load_network(self) -> None:
        """read node.csv and poi.csv and return network_dict

        Raises:
            FileExistsError: Error: Input directory {input_dir} does not exist.
        """

        if self.verbose:
            print("  :Loading network data...")

        if not self.input_dir:
            raise Exception("Error: Input directory is not specified. Please specify input directory")

        if not os.path.isdir(self.input_dir):
            raise FileExistsError(f"Error: Input directory {self.input_dir} does not exist.")

        # load node.csv
        node_dict = read_node(self.node_file, self.pkg_settings.get("set_cpu_cores"), verbose=self.verbose)

        # create zone.csv if use_zone_id is True
        if self.use_zone_id:
            # extract activity nodes from node.csv
            _activity_node_id_val = [
                node_id
                for node_id in node_dict
                if node_dict[node_id]["_zone_id"] != "-1"]

            # check if activity nodes exist, if not raise exception
            if not _activity_node_id_val:
                raise Exception("Error: No activity nodes found in node.csv (zone_id column is empty)."
                                " Please check node.csv.")

            # if activity nodes exist, create node_dict_activity_nodes
            self.node_dict_activity_nodes = {}
            for node_id in _activity_node_id_val:
                self.node_dict_activity_nodes[node_id] = node_dict[node_id]
                del node_dict[node_id]

            # Create zone.csv file from node.csv with zone_id, x_coord, y_coord as zone centroid
            if self.kwargs.get("node_as_zone_centroid"):
                _zone_id_val = [
                    [node_dict[node_id]["_zone_id"],
                     node_dict[node_id]["x_coord"],
                     node_dict[node_id]["y_coord"],
                     shapely.Point(
                         node_dict[node_id]["x_coord"],
                         node_dict[node_id]["y_coord"])]
                    for node_id in set(_activity_node_id_val)]

                _zone_col = ["zone_id", "x_coord", "y_coord", "geometry"]
                _zone_df = pd.DataFrame(_zone_id_val, columns=_zone_col)
                _zone_df = _zone_df.sort_values(by=["zone_id"])
                self.zone_file = path2linux(os.path.join(self.input_dir, "zone.csv"))
                _zone_df.to_csv(self.zone_file, index=False)
                print(f"  : zone.csv is generated (use_zone_id=True) based on node.csv in {self.input_dir}.\n")

        self.node_dict = node_dict

        # check if area field is in df_poi_chunk, if it's empty, raise error
        poi_df = pd.read_csv(self.poi_file, nrows=3)
        if "area" in poi_df.columns and poi_df["area"].isnull().any():
            raise Exception(
                "Error: poi.csv contains empty area field. Please fill in the area field in poi.csv.")

        self.poi_dict = read_poi(self.poi_file,
                                 self.pkg_settings.get("set_cpu_cores"),
                                 verbose=self.verbose)

        return None

    def net2grid(self, *,
                 num_x_blocks: int = 10,
                 num_y_blocks: int = 10,
                 cell_width: float = 0,
                 cell_height: float = 0,
                 unit: str = "km") -> None:
        """convert node_dict to zone_dict by grid.
        The grid can be defined by num_x_blocks and num_y_blocks, or cell_width and cell_height.
        if num_x_blocks and num_y_blocks are specified
            the grid will be divided into num_x_blocks * num_y_blocks.
        if cell_width and cell_height are specified
            the grid will be divided into cells with cell_width * cell_height.
        Note: num_x_blocks and num_y_blocks have higher priority to cell_width and cell_height.
              if num_x_blocks and num_y_blocks are specified, cell_width and cell_height will be ignored.

        Args:
            node_dict (dict[int, Node]): node_dict {node_id: Node}, default is self.node_dict.
            num_x_blocks (int): total number of blocks/grids from x direction. Defaults to 10.
            num_y_blocks (int): total number of blocks/grids from y direction. Defaults to 10.
            cell_width (float): the width for each block/grid . Defaults to 0. unit: km.
            cell_height (float): the height for each block/grid. Defaults to 0. unit: km.
            unit (str): the unit of cell_width and cell_height. Defaults to "km".
                Options: ["km", "meter", "mile"]
        """
        if self.verbose:
            print("  : Note: net2grid will generate grid-based zones from node_dict.")

        # check parameters
        if not isinstance(num_x_blocks, int):
            raise ValueError("Error: num_x_blocks must be an integer.")

        if not isinstance(num_y_blocks, int):
            raise ValueError("Error: num_y_blocks must be an integer.")

        if not isinstance(cell_width, (int, float)):
            raise ValueError("Error: cell_width must be a number.")

        if not isinstance(cell_height, (int, float)):
            raise ValueError("Error: cell_height must be a number.")

        if unit not in ["km", "meter", "mile"]:
            raise ValueError("Error: unit must be km, meter or mile.")

        print("  : Creating grid...")
        # generate zone based on zone_id in node.csv
        if self.node_dict:
            node_dict = self.node_dict
        elif hasattr(self, "node_dict_activity_nodes"):
            node_dict = self.node_dict_activity_nodes
        else:
            raise Exception("Error: node_dict is not valid. Please check your node.csv first.")

        zone_dict_with_gate = net2grid(node_dict,
                                       num_x_blocks,
                                       num_y_blocks,
                                       cell_width,
                                       cell_height,
                                       unit,
                                       verbose=self.verbose)
        zone_dict = {
            zone_name: zone for zone_name, zone in zone_dict_with_gate.items() if "gate" not in zone.name}
        self.__config["is_geometry"] = True

        # save zone to zone.csv
        zone_df = pd.DataFrame(zone_dict.values())
        zone_df.rename(columns={"id": "zone_id"}, inplace=True)
        path_output = path2linux(os.path.join(self.input_dir, "zone.csv"))
        self.zone_file = path_output
        zone_df.to_csv(path_output, index=False)

        print(f"  : net2grid saved grids as zone.csv to {path_output} \n")
        return None

    @func_time
    def taz2zone(self, zone_file: str = "") -> None:
        """generate zone dictionary from zone.csv (TAZs)

        Args:
            zone_file (str): external zone.csv. Defaults to "".
            return_value (bool): whether or not to return generated zone. Defaults to False.

        Raises:
            FileNotFoundError: Error: File {zone_file} does not exist.
            Exception: Error: Failed to read {zone_file}.
            Exception: Error: {zone_file} does not contain valid zone fields.
            Exception: Error: {zone_file} does not contain valid geometry fields.
            Exception: Error: {zone_file} contains both point and polygon geometry fields.
        """

        if self.verbose:
            print("  : Note: taz2zone will generate zones from zone.csv (TAZs). \n"
                  "  : If you want to use grid-based zones (generate zones from node_dict), \n"
                  "  : please skip this method and use net2zone() instead. \n")

        # update zone_file if specified
        if zone_file:
            self.zone_file = path2linux(zone_file)

        # check zone_file, geometry or centroid?
        if not os.path.exists(self.zone_file):
            raise FileNotFoundError(f"Error: File {self.zone_file} does not exist.")

        # load zone file column names
        zone_columns = []
        try:
            zone_df = pd.read_csv(self.zone_file, nrows=1)  # 1 row, reduce memory and time
            zone_columns = zone_df.columns
        except Exception as e:
            raise Exception(f"Error: Failed to read {self.zone_file}.") from e

        # update geometry or centroid
        if set(self.pkg_settings.get("zone_geometry_fields")).issubset(set(zone_columns)):
            # we need to consider whether the geometry field is point or polygon

            # check geometry fields is valid from zone_df
            if not any(zone_df["geometry"].isnull()):

                # check whether geometry is point,
                # if it is, then convert to centroid and update zone file
                zone_df = pd.read_csv(self.zone_file)  # reload zone file
                include_point = False
                include_polygon = False

                # Function to process a chunk of the DataFrame
                def process_chunk(chunk):
                    for index, row in chunk.iterrows():
                        try:
                            geo = shapely.from_wkt(row["geometry"])
                            if isinstance(geo, shapely.Point):
                                row["x_coord"] = geo.x
                                row["y_coord"] = geo.y
                            elif isinstance(geo, (shapely.Polygon, shapely.MultiPolygon)):
                                row["x_coord"] = geo.centroid.x
                                row["y_coord"] = geo.centroid.y
                            else:
                                print(f"  : Error: {row['geometry']} is not valid geometry.")
                        except Exception as e:
                            print(f"  : Error: {row['geometry']} is not valid geometry.")
                            print(f"  : Error: {e}")
                    return chunk

                chunk_size = pkg_settings["data_chunk_size"]
                chunks = [zone_df.iloc[i:i + chunk_size]
                          for i in range(0, len(zone_df), chunk_size)]

                # Process each chunk in parallel with progress tracking
                processed_chunks = Parallel(n_jobs=-1)(
                    delayed(process_chunk)(chunk) for chunk in tqdm(chunks, desc="  : Update zone geometry")
                )

                # Concatenate all the processed chunks back into a single DataFrame
                zone_df = pd.concat(processed_chunks, ignore_index=True)

                # check if is point or polygon
                single_geo = shapely.from_wkt(zone_df.loc[0, "geometry"])
                if isinstance(single_geo, shapely.Point):
                    include_point = True
                elif isinstance(single_geo, (shapely.Polygon, shapely.MultiPolygon)):
                    include_polygon = True
                else:
                    raise Exception(f"Error: {self.zone_file} does not contain valid geometry fields.")

                if include_point:
                    if include_polygon:
                        raise Exception(f"Error: {self.zone_file} contains both point and polygon geometry fields.")

                    # save zone_df to zone.csv
                    zone_df.to_csv(self.zone_file, index=False)
                    self.__config["is_centroid"] = True

                if include_polygon:
                    self.__config["is_geometry"] = True

        # use elif to prioritize geometry
        elif set(self.pkg_settings.get("zone_centroid_fields")).issubset(set(zone_columns)):
            self.__config["is_centroid"] = True

        if not self.__config["is_geometry"] and not self.__config["is_centroid"]:
            raise Exception(f"Error: {self.zone_file} does not contain valid zone fields.")

        if self.verbose:
            if self.__config["is_geometry"]:
                print("  : read zone by geometry.")
            else:
                print("  : read zone by centroid.")

            print("  : Generating zone dictionary...")

        # generate zone by centroid: zone_id, x_coord, y_coord
        # generate zone by geometry: zone_id, geometry
        if self.__config["is_geometry"]:
            zone_dict = read_zone_by_geometry(self.zone_file,
                                              self.pkg_settings.get("set_cpu_cores"),
                                              verbose=self.verbose)

        elif self.__config["is_centroid"]:
            zone_dict = read_zone_by_centroid(self.zone_file,
                                              self.pkg_settings.get("set_cpu_cores"),
                                              verbose=self.verbose)
        else:
            print(f"Error: {self.zone_file} does not contain valid zone fields.")
            return {}
        self.zone_dict = zone_dict
        return None

    def map_mapping_between_zone_and_node_poi(self) -> None:
        """Map mapping between zone and node/poi.

        Raises:
            Exception: Error in running _function_name_: not valid zone_dict or node_dict
            Exception: Error in running _function_name_: not valid zone_dict or poi_dict
        """

        # check zone_dict exists
        if not hasattr(self, "zone_dict"):
            raise Exception("Not valid zone_dict. Please generate zone_dict first.")

        # synchronize zone with node
        if hasattr(self, "node_dict"):
            print("  : Mapping zones with nodes...")
            if hasattr(self, "node_dict_activity_nodes"):
                _node_dict = self.node_dict_activity_nodes
            else:
                _node_dict = self.node_dict

            if self.__config["is_geometry"]:
                try:
                    print("  : zone geometry with node...")
                    zone_node_dict = map_zone_geometry_and_node(self.zone_dict,
                                                                _node_dict,
                                                                self.pkg_settings.get("set_cpu_cores"),
                                                                verbose=self.verbose)
                except Exception as e:
                    print("Could not map zone with node.")
                    print(f"The error occurred: {e}")

            elif self.__config["is_centroid"]:
                try:
                    print("  : zone centroid with node...")
                    zone_node_dict = map_zone_centroid_and_node(self.zone_dict,
                                                                _node_dict,
                                                                self.pkg_settings.get("set_cpu_cores"),
                                                                verbose=self.verbose)
                except Exception as e:
                    print("Could not map zone with node.")
                    print(f"The error occurred: {e}")

            node_dict = zone_node_dict.get('node_dict')

            if hasattr(self, "node_dict_activity_nodes"):
                self.node_dict_activity_nodes = node_dict
            else:
                self.node_dict = node_dict

            zone_dict = zone_node_dict.get('zone_dict')

        # synchronize zone with poi
        if hasattr(self, "poi_dict"):
            print("  : Mapping zones with pois...")

            if self.__config["is_geometry"]:
                try:
                    print("  : zone geometry with poi...")
                    zone_poi_dict = map_zone_geometry_and_poi(zone_dict,
                                                              self.poi_dict,
                                                              self.pkg_settings.get("set_cpu_cores"),
                                                              verbose=self.verbose)
                except Exception as e:
                    print("Could not synchronize zone with poi.")
                    print(f"The error occurred: {e}")

            elif self.__config["is_centroid"]:
                try:
                    print("  : zone centroid with poi...")
                    zone_poi_dict = map_zone_centroid_and_poi(zone_dict,
                                                              self.poi_dict,
                                                              self.pkg_settings.get("set_cpu_cores"),
                                                              verbose=self.verbose)
                except Exception as e:
                    print("Could not synchronize zone with poi.")
                    print(f"The error occurred: {e}")

            self.zone_dict = zone_poi_dict.get('zone_dict')
            self.poi_dict = zone_poi_dict.get('poi_dict')

        self.__config["is_sync_geometry"] = True
        return None

    def calc_zone_od_distance_matrix(self, zone_dict: dict = "",
                                     *,
                                     selected_zone_id: list = [],
                                     pct: float = 1.0) -> None:
        """calculate zone-to-zone od distance matrix

        Args:
            zone_dict (dict): the zone dictionary. Defaults to "".
                if not specified, use self.zone_dict.
        """

        # if not specified, use self.zone_dict as input
        if zone_dict:
            self.zone_dict = zone_dict

        self.zone_od_dist_matrix = calc_zone_od_matrix(self.zone_dict,
                                                       cpu_cores=self.pkg_settings.get("set_cpu_cores"),
                                                       selected_zone_id=selected_zone_id,
                                                       pct=pct,
                                                       verbose=self.verbose)
        self.__config["is_zone_od_dist_matrix"] = True
        return None

    def calc_zone_prod_attr(self,
                            trip_rate_file: str = "",
                            trip_purpose: int = 1) -> None:
        """calculate zone production and attraction based on node production and attraction

        Args:
            node_dict (dict): Defaults to "". if not specified, use self.node_dict.
            zone_dict (dict): Defaults to "". if not specified, use self.zone_dict.
        """

        # update input parameters if specified
        if trip_rate_file:
            if ".csv" not in trip_rate_file:
                raise Exception(f"  : Error: trip_rate_file {trip_rate_file} must be a csv file.")

            if not os.path.exists(trip_rate_file):
                raise FileNotFoundError(f"Error: File {trip_rate_file} does not exist.")

            self.pkg_settings["trip_rate_file"] = pd.read_csv(trip_rate_file)

        if trip_purpose not in [1, 2, 3]:
            raise ValueError('Error: trip_purpose must be 1, 2 or 3, '
                             'represent home-based work, home-based others, non home-based.')

        # generate poi trip rate for each poi if not generated
        if not self.__config["is_poi_trip_rate"]:
            self.poi_dict = gen_poi_trip_rate(self.poi_dict,
                                              trip_rate_file,
                                              trip_purpose,
                                              verbose=self.verbose)
            self.__config["is_poi_trip_rate"] = True

        # generate node production and attraction for each node based on poi_trip_rate if not generated
        if not self.__config["is_node_prod_attr"]:
            if hasattr(self, "node_dict_activity_nodes"):
                node_dict = self.node_dict_activity_nodes
            else:
                node_dict = self.node_dict
            node_dict = gen_node_prod_attr(node_dict, self.poi_dict, verbose=self.verbose)
            self.__config["is_node_prod_attr"] = True

        # calculate zone production and attraction based on node production and attraction
        self.zone_dict = calc_zone_production_attraction(node_dict,
                                                         self.poi_dict,
                                                         self.zone_dict,
                                                         verbose=self.verbose)
        if hasattr(self, "node_dict_activity_nodes"):
            self.node_dict_activity_nodes = node_dict
        else:
            self.node_dict = node_dict
        self.__config["is_zone_prod_attr"] = True
        return None

    def run_gravity_model(self,
                          *,
                          alpha: float = 28507,
                          beta: float = -0.02,
                          gamma: float = -0.123,
                          trip_rate_file: str = "",
                          trip_purpose: int = 1) -> None:
        """run gravity model to generate demand

        Args:
            zone_dict (dict): dict store zones info. Defaults to "".
            zone_od_dist_matrix (dict): OD distance matrix. Defaults to "".
            trip_purpose (int): purpose of trip. Defaults to 1.
            alpha (float): parameter alpha. Defaults to 28507.
            beta (float): parameter beta. Defaults to -0.02.
            gamma (float): parameter gamma. Defaults to -0.123.

        Returns:
            pd.DataFrame: the final demand dataframe
        """

        # update parameters if specified
        if trip_rate_file and not os.path.exists(trip_rate_file):
            raise FileNotFoundError(f"Error: File {trip_rate_file} does not exist.")

        if trip_purpose not in [1, 2, 3]:
            raise ValueError('Error: trip_purpose must be 1, 2 or 3, '
                             'represent home-based work, home-based others, non home-based.')

        # synchronize geometry between zone and node/poi
        if not self.__config["is_sync_geometry"]:
            self.map_mapping_between_zone_and_node_poi()

        # calculate od distance matrix if not exists
        if not self.__config["is_zone_od_dist_matrix"]:
            raise Exception("Error: zone_od_dist_matrix does not exist. \n"
                            "Please run net.calc_zone_od_distance_matrix()"
                            "before net.run_gravity_model().")

        # calculate zone production and attraction based on node production and attraction
        if not self.__config["is_zone_prod_attr"]:
            self.calc_zone_prod_attr(trip_rate_file=trip_rate_file,
                                     trip_purpose=trip_purpose)

        # run gravity model to generate demand
        zone_od_demand_matrix = run_gravity_model(self.zone_dict,
                                                  self.zone_od_dist_matrix,
                                                  trip_purpose,
                                                  alpha,
                                                  beta,
                                                  gamma,
                                                  verbose=self.verbose)

        # Converting dictionary to DataFrame
        od_list = [(key[0], key[1], value) for key, value in zone_od_demand_matrix.items()]
        self.df_demand = pd.DataFrame(od_list, columns=['o_zone_id', 'd_zone_id', 'volume'])
        # self.df_demand = pd.DataFrame(list(self.zone_od_demand_matrix.values()))

        print("  : Successfully generated OD demands.")
        return None

    def gen_agent_based_demand(self, time_periods: str = "0700-0800") -> None:
        """generate agent-based demand

        Args:
            node_dict (dict): _description_. Defaults to "".
            zone_dict (dict): _description_. Defaults to "".
            df_demand (pd.DataFrame): _description_. Defaults to "".
        """
        if hasattr(self, "node_dict_activity_nodes"):
            node_dict = self.node_dict_activity_nodes
        else:
            node_dict = self.node_dict

        self.df_agent = gen_agent_based_demand(node_dict, self.zone_dict,
                                               df_demand=self.df_demand,
                                               time_period=time_periods,
                                               verbose=self.verbose)
        return None

    def save_results_to_csv(self, output_dir: str = "",
                            demand: bool = True,
                            *,  # enforce keyword-only arguments
                            zone: bool = True,
                            node: bool = True,  # save updated node
                            poi: bool = True,  # save updated poi
                            agent: bool = False,  # save agent-based demand
                            agent_time_period: str = "0700-0800",
                            zone_od_dist_table: bool = False,
                            zone_od_dist_matrix: bool = False,
                            overwrite_file: bool = True) -> None:
        """save results to csv files

        Args:
            output_dir (str): the output dir to save files. Defaults to "", represent current folder.
            demand (bool): whether to save demand file. Defaults to True.
            node (bool): whether to save node file. Defaults to True.
            poi (bool): whether to save poi file. Defaults to True.
            agent (bool): whether to save agent file. Defaults to False.
            zone_od_dist_table (bool): whether to save zone od distance table. Defaults to False.
            zone_od_dist_matrix (bool): whether to save zone od distance matrix. Defaults to False.
            is_demand_with_geometry (bool): whether include geometry in demand file. Defaults to False.
            overwrite_file (bool): whether to overwrite existing files. Defaults to True.
        """

        # update output_dir if specified
        if output_dir:
            self.output_dir = path2linux(output_dir)

        if demand:
            save_demand(self, overwrite_file=overwrite_file)

        if zone:
            save_zone(self, overwrite_file=overwrite_file)

        if node:
            save_node(self, overwrite_file=overwrite_file)

        if poi:
            save_poi(self, overwrite_file=overwrite_file)

        if zone_od_dist_table:
            save_zone_od_dist_table(self, overwrite_file=overwrite_file)

        if zone_od_dist_matrix:
            save_zone_od_dist_matrix(self, overwrite_file=overwrite_file)

        if agent:
            if agent_time_period:
                self.gen_agent_based_demand(time_periods=agent_time_period)
            else:
                self.gen_agent_based_demand()
            save_agent(self, overwrite_file=overwrite_file)

        return None
