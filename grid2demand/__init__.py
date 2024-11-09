'''
# -*- coding:utf-8 -*-
##############################################################
# Created Date: Monday, September 4th 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import sys

from .func_lib.read_node_poi import (read_node,
                                     read_poi,
                                     read_network)

from .func_lib.gen_zone import net2grid
from .func_lib.read_zone import taz2zone

from .func_lib.plot_net import plot_gd

from .utils_lib.pkg_settings import pkg_settings
from ._grid2demand import GRID2DEMAND


def check_python_version() -> tuple:
    """Check if the Python version is greater than 3.10

    Raises:
        EnvironmentError: grid2demand, supports Python 3.10 or higher

    Returns:
        tuple: Python version tuple
    """

    # Split the version string and convert to tuple of integers
    version_tuple = tuple(map(int, sys.version.split()[0].split('.')))

    # Check if the version is greater than 3.10
    try:
        if version_tuple < (3, 10):
            raise EnvironmentError("grid2demand, supports Python 3.10 or higher")
    except Exception:
        print("grid2demand, supports Python 3.10 or higher")
    return version_tuple
check_python_version()

# print('grid2demand, version 0.4.8, supports Python 3.10 or higher')

__all__ = ["read_node", "read_poi", "read_network",
           "net2grid",
           "taz2zone",
           "plot_gd",
           "pkg_settings",
           "GRID2DEMAND"]
