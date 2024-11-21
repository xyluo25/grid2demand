# -*- coding:utf-8 -*-
##############################################################
# Created Date: Monday, September 11th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

from __future__ import absolute_import
import grid2demand as gd


if __name__ == "__main__":

    # Step 0: Specify input directory
    INPUT_DIR = r"datasets\demand_from_TAZ\SF_TAZ_are_centroids"

    # Initialize a GRID2DEMAND object
    net = gd.GRID2DEMAND(input_dir=INPUT_DIR, verbose=False)

    # Step 1: Load node and poi data from input directory
    net.load_network()

    # Step 2: Generate zone dictionary from zone.csv file
    net.taz2zone()

    net.map_zone_node_poi()

    net.calc_zone_od_distance(pct=1)

    # Step 3: Run gravity model to generate agent-based demand
    net.run_gravity_model()

    # Step 4: Output demand, agent, zone, zone_od_dist_table, zone_od_dist_matrix files
    net.save_results_to_csv(node=True, poi=False, zone=False, overwrite_file=False)
