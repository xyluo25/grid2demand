'''
##############################################################
# Created Date: Saturday, October 19th 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import matplotlib.pyplot as plt
import shapely.wkt as wkt
from shapely.geometry import Polygon, MultiPolygon


def plot_gd(net: object, *,
            demand: bool = False, node: bool = False,
            poi: bool = False, zone: bool = False,
            save_fig: bool = False, fig_name: str = "grid2demand_net.png",
            fig_size: tuple = (10, 10), fig_dpi: int = 600,
            return_fig: bool = False) -> plt.figure:
    """Plot the network and demand data"""

    # crate a figure
    is_add_data_to_plot = False
    fig, ax = plt.subplots(figsize=fig_size)

    title_str = "GRID2DEMAND NETWORK ("

    # demand data separate from node, zone and poi data
    if demand:
        # check if net have demand data and zone data
        if hasattr(net, 'df_demand') and hasattr(net, 'zone_dict'):
            df_demand = net.df_demand
            for i in range(len(df_demand)):
                o_zone_id = df_demand.loc[i, "o_zone_id"]
                d_zone_id = df_demand.loc[i, "d_zone_id"]
                demand_val = df_demand.loc[i, "volume"]
                line_x = net.zone_dict[o_zone_id]["x_coord"], net.zone_dict[d_zone_id]["x_coord"]
                line_y = net.zone_dict[o_zone_id]["y_coord"], net.zone_dict[d_zone_id]["y_coord"]
                ax.plot(line_x, line_y, c='k', alpha=demand_val / df_demand["volume"].max())
            is_add_data_to_plot = True
            title_str += " DEMANDS"
    else:
        # plot node data
        if node:
            # check if net have node data
            if hasattr(net, 'node_dict'):
                for node_val in net.node_dict.values():
                    ax.scatter(node_val["x_coord"], node_val["y_coord"], s=10, c='b', alpha=0.5)
                is_add_data_to_plot = True
                title_str += " NODES"

        # plot poi data
        if poi:
            # check if net have poi data
            if hasattr(net, 'poi_dict'):
                for poi_val in net.poi_dict.values():
                    ax.scatter(poi_val["x_coord"], poi_val["y_coord"], s=10, c='r', alpha=0.5)
                is_add_data_to_plot = True
                title_str += " POIs"

        # plot zone data
        if zone:
            # check if net have zone data
            if hasattr(net, 'zone_dict'):
                for zone_val in net.zone_dict.values():
                    if isinstance(zone_val["geometry"], str):
                        __geometry = wkt.loads(zone_val["geometry"])
                    else:
                        __geometry = zone_val["geometry"]

                    if isinstance(__geometry, Polygon):
                        x, y = __geometry.exterior.xy
                        ax.fill(x, y, alpha=0.5, fc='gray', ec='white')
                    elif isinstance(__geometry, MultiPolygon):
                        for polygon in __geometry.geoms:
                            x, y = polygon.exterior.xy
                            ax.fill(x, y, alpha=0.5, fc='gray', ec='white')
                is_add_data_to_plot = True
                title_str += " ZONES"

    if is_add_data_to_plot:

        # Set equal scaling
        ax.set_aspect('equal')

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title_str + " )")

        # save figure
        if save_fig:
            fig.savefig(fig_name, dpi=fig_dpi)

        plt.show()

        if return_fig:
            return fig
    else:
        print("No data added to plot")
