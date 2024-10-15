'''
##############################################################
# Created Date: Friday, October 4th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import shapely
import pandas as pd
from pyproj import Transformer
import os


def calc_area_from_wkt_geometry(path: str) -> pd.DataFrame:
    df_poi = pd.read_csv(path)

    for i in range(len(df_poi)):
        # check if area is empty or not
        area = df_poi.loc[i, 'area']

        if pd.isna(area) or not area:
            geometry_shapely = shapely.from_wkt(df_poi.loc[i, 'geometry'])

            # Set up a Transformer to convert from WGS 84 to UTM zone 18N (EPSG:32618)
            transformer = Transformer.from_crs(
                "EPSG:4326", "EPSG:32618", always_xy=True)

            # Transform the polygon's coordinates to UTM
            if isinstance(geometry_shapely, shapely.MultiPolygon):
                transformed_polygons = []
                for polygon in geometry_shapely.geoms:
                    transformed_coords = [transformer.transform(
                        x, y) for x, y in polygon.exterior.coords]
                    transformed_polygons.append(
                        shapely.Polygon(transformed_coords))
                transformed_geometry = shapely.MultiPolygon(
                    transformed_polygons)
            else:
                transformed_coords = [transformer.transform(
                    x, y) for x, y in geometry_shapely.exterior.coords]
                transformed_geometry = shapely.Polygon(transformed_coords)

            # square meters
            area_sqm = transformed_geometry.area

            # square feet
            # area = area_sqm * 10.7639104

            area = area_sqm
            df_poi.loc[i, "area"] = area
    return df_poi


if __name__ == "__main__":
    in_dir = r"C:\Users\xyluo25\anaconda3_workspace\001_GitHub\grid2demand\datasets\demand_from_grid_use_zone_id_in_node\UMD"
    filename = "poi.csv"
    in_file = os.path.join(in_dir, filename)

    df_poi = calc_area_from_wkt_geometry(in_file)
    df_poi.to_csv(in_file, index=False)
