# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import sys
import setuptools
from setuptools import setup
import unittest

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pkgs = {
    "required": [
        "grid2op>=1.9",
        "lightsim2grid>=0.7"
    ]
}

setup(name='TopologySuperpositionTheorem',
      version='0.0.1',
      description='A package for efficient combinatorial topological actions power flow computation based on the extended superposition theorem for powersystems',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Antoine MAROT',
      python_requires='>=3.8',
      url="https://github.com/marota/Topology_Superposition_Theorem",
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=pkgs["required"],
      zip_safe=False,
      )