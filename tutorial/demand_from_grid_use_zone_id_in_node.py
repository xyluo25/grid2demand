"""
# -*- coding:utf-8 -*-
##############################################################
# Created Date: Monday, September 11th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
"""

from __future__ import absolute_import
from pathlib import Path
import os

try:
    import grid2demand as gd
except ImportError:
    root_path = Path(os.path.abspath(__file__)).parent.parent
    os.chdir(root_path)
    import grid2demand as gd

if __name__ == "__main__":

    # Step 0: Specify input directory
    INPUT_DIR = r"datasets\demand_from_grid_use_zone_id_in_node\ASU\auto"

    # Initialize a GRID2DEMAND object, and specify the mode_type as "auto" in default
    net = gd.GRID2DEMAND(INPUT_DIR, use_zone_id=True, mode_type="auto")

    # Step 1: Load node and poi data from input directory
    net.load_network()
    # visualize node and poi
    gd.plot_gd(net, node=True)
    gd.plot_gd(net, poi=True)

    # Step 2: create grids by specifying number of x blocks and y blocks
    net.net2grid()

    # Step 3: Generate zone dictionary from zone.csv
    net.taz2zone()
    # visualize the zone
    gd.plot_gd(net, zone=True, node=True)

    # Step 4: Map the zone id in node and poi, vise versa
    net.map_mapping_between_zone_and_node_poi()

    # Step 5: Calculate zone-to-zone travel time matrix
    net.calc_zone_od_distance_matrix(pct=1)

    # Step 6: Run gravity model to generate agent-based demand
    net.run_gravity_model()

    # Step 7: Output demand, agent, zone, zone_od_dist_table, zone_od_dist_matrix files
    net.save_results_to_csv(zone=True, node=True, poi=True, agent=True, overwrite_file=False)
