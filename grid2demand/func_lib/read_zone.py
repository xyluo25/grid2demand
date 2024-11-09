'''
##############################################################
# Created Date: Saturday, November 9th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

import os

import pandas as pd
import shapely
from joblib import Parallel, delayed
from tqdm import tqdm
from pyufunc import (path2linux,)

from grid2demand.utils_lib.pkg_settings import pkg_settings

from grid2demand.func_lib.read_node_poi import (read_zone_by_geometry,
                                                read_zone_by_centroid)


def taz2zone(zone_file: str, verbose: bool = False) -> dict:
    """Generate zone dictionary from zone.csv file

    Args:
        zone_file (str): Path to the zone.csv file

    Returns:
        dict: Zone dictionary
    """

    # convert path to linux format
    zone_file = path2linux(zone_file)

    # Check if the zone file exists
    if not os.path.exists(zone_file):
        raise FileNotFoundError(f"zone_file: {zone_file} does not exist")

    # Read zone file
    zone_columns = []
    try:
        zone_df = pd.read_csv(zone_file, nrows=1)  # load only the first row to get the column names
        zone_columns = zone_df.columns
    except Exception as e:
        raise Exception(f"Error reading zone_file: {zone_file}, {e}")

    include_point = False
    include_polygon = False

    # checi if the zone is geometry or centroid based
    if set(pkg_settings.get("zone_geometry_fields")).issubset(zone_columns):
        # Then, we need to check if the geometry field is point or polygon

        if not any(zone_df["geometry"].isnull()):
            # if geometry field is not null, we reload the zone file to get the geometry type

            zone_df = pd.read_csv(zone_file)

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
                delayed(process_chunk)(chunk) for chunk in tqdm(chunks, desc="  : Update zone geometry"))

            # Concatenate the processed chunks
            zone_df = pd.concat(processed_chunks, ignore_index=True)

            # check if is point or polygon
            single_geo = shapely.from_wkt(zone_df.loc[0, "geometry"])
            if isinstance(single_geo, shapely.Point):
                include_point = True
            elif isinstance(single_geo, (shapely.Polygon, shapely.MultiPolygon)):
                include_polygon = True
            else:
                raise Exception(
                    f"Error: {zone_file} does not contain valid geometry fields.")

            # if point in the geometry field
            if include_point:
                # if polygon in the geometry field
                if include_polygon:
                    raise Exception(f"Error: {zone_file} contains both point and polygon geometry fields.",
                                    "grid2demand allow only one type of geometry.")
                # save zone_df to zone.csv
                zone_df.to_csv(zone_file, index=False)

    elif set(pkg_settings.get("zone_centroid_fields")).issubset(zone_columns):
        include_point = True

    if not any([include_point, include_polygon]):
        raise Exception(f"Error: {zone_file} does not contain valid geometry fields.")

    if verbose:
        if include_point:
            print("  : Read zone by point geometry fields.")
        elif include_polygon:
            print("  : Read zone by polygon geometry fields.")
        print("  : Generating zone dictionary...")

    # generate zone by centroid: zone_id, x_coord, y_coord
    # generate zone by geometry: zone_id, geometry
    if include_polygon:
        zone_dict = read_zone_by_geometry(zone_file,
                                          pkg_settings.get("set_cpu_cores"),
                                          verbose=verbose)

    elif include_point:
        zone_dict = read_zone_by_centroid(zone_file,
                                          pkg_settings.get("set_cpu_cores"),
                                          verbose=verbose)
    else:
        zone_dict = {}
        print("  : Error: Invalid geometry fields in zone file.")

    return zone_dict
