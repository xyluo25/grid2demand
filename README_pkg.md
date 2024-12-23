## Project Description

GRID2DEMAND: A tool for generating zone-to-zone travel demand based on grid cells or TAZs and gravity model

## Introduction

Grid2demand is an open-source quick demand generation tool based on the trip generation and trip distribution methods of the standard 4-step travel model.

By taking advantage of OSM2GMNS tool to obtain route-able transportation network from OpenStreetMap, Grid2demand aims to further utilize Point of Interest (POI) data to construct trip demand matrix aligned with standard travel models.

You can get access to the introduction video with the link: [https://www.youtube.com/watch?v=EfjCERQQGTs&amp;t=1021s](https://www.youtube.com/watch?v=EfjCERQQGTs&t=1021s)

You can find base-knowledge tutorial with the link: [Base Knowledge such as transportation 4 stages planning](https://github.com/asu-trans-ai-lab/grid2demand/tree/main/docs)

You can find the tutorial code witht the link: [How To Use Grid2demand](https://github.com/xyluo25/grid2demand/tree/main/tutorial)

## Installation

```python
pip install grid2demand
```

If you meet installation issues, please reach out to our [developers](mailto:luoxiangyong01@gmail.com) for solutions.

## Demand Generation

[!IMPORTANT]
node.csv and poi.csv should follow the [GMNS](https://github.com/zephyr-data-specs/GMNS) standard and you can generate node.csv and poi.csv using [osm2gmns](https://osm2gmns.readthedocs.io/en/latest/quick-start.html).

### Generate demands from node.csv and poi.csv (zone_id as activity nodes)

1. Create zone from node.csv (the boundary of nodes), this will generate grid cells (num_x_blocks, num_y_blocks, or x length and y length in km for each grid cell)
2. Generate demands for between zones (utilize nodes and pois)

```python
from __future__ import absolute_import
import grid2demand as gd

if __name__ == "__main__":

    # Specify input directory
    input_dir = "your-data-folder"

    # Initialize a GRID2DEMAND object
    net = gd.GRID2DEMAND(input_dir=input_dir, use_zone_id=True, mode_type="auto")

    # load network: node and poi
    net.load_network()

    # Generate zone.csv from node boundary by specifying number of x blocks and y blocks
    net.net2grid(num_x_blocks=10, num_y_blocks=10)
    # net.net2grid(cell_width=10, cell_height=10, unit="km")

    # Load zone from zone.csv
    net.taz2zone()

    # Map zones with nodes and pois, viseversa
    net.map_mapping_between_zone_and_node_poi()

    # Calculate zone-to-zone distance matrix
    net.calc_zone_od_distance_matrix(pct=1)

    # Calculate demand by running gravity model
    net.run_gravity_model()

    # Save demand, zone, updated node, updated poi to csv
    net.save_results_to_csv(agent=True, overwrite_file=False)
```

### Generate demands from node.csv, poi.csv and zone.csv (from TAZ)

```python
from __future__ import absolute_import
import grid2demand as gd

if __name__ == "__main__":

    # Specify input directory
    input_dir = "your-data-folder"

    # Initialize a GRID2DEMAND object
    net = gd.GRID2DEMAND(input_dir=input_dir, use_zone_id=True, mode_type="auto")

    # Load network: node and poi
    net.load_network()

    # Load zone from zone.csv
    net.taz2zone()

    # Map zones with nodes and pois, viseversa
    net.map_mapping_between_zone_and_node_poi()

    # Calculate zone-to-zone distance matrix
    net.calc_zone_od_distance_matrix(pct=1)

    # Calculate demand by running gravity model
    net.run_gravity_model()

    # Save demand, zone, updated node, updated poi to csv
    net.save_results_to_csv(node=Frale, zone=False, agent=False, overwrite_file=False)
```

## Call for Contributions

The grid2demand project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact us through email: [Xiangyong Luo](mailto:luoxiangyong01@gmail.com), [Dr. Xuesong Simon Zhou](mailto:xzhou74@asu.edu)

Writing code isn't the only way to contribute to grid2demand. You can also:

* review pull requests
* help us stay on top of new and old issues
* develop tutorials, presentations, and other educational materials
* develop graphic design for our brand assets and promotional materials
* translate website content
* help with outreach and onboard new contributors
* write grant proposals and help with other fundraising efforts

For more information about the ways you can contribute to grid2demand, visit [our GitHub](https://github.com/asu-trans-ai-lab/grid2demand). If you' re unsure where to start or how your skills fit in, reach out! You can ask by opening a new issue or leaving a comment on a relevant issue that is already open on GitHub.

## Citing Grid2demand

If you use grid2demand in your research please use the following BibTeX entry:

Xiangyong Luo, Xuesiong Simon Zhou (2023). [xyluo25/grid2demand](https://github.com/xyluo25/grid2demand/): Zenodo. https://doi.org/10.5281/zenodo.11212556
