# Tango: An AGV Path Planning and Area Coverage Project

## Overview

Tango is a Python-based library designed to provide an advanced path planning and area coverage solution for Automated Guided Vehicles (AGVs), primarily focusing on efficiently managing solar park maintenance tasks. It utilizes unique partitioning strategies and adaptive path planning to maximize efficiency and accuracy in complex environments.

## Project Structure

The main components of the project are spread across several Python files:

- `solver.py`: This file implements the coverage path planning algorithm for navigating a robot in a grid-based environment. Utilizing the Unified Planning Framework (UPF), the module defines planning problems in PDDL (Planning Domain Definition Language), solves them optimally, and generates action plans for the robot. The solver keeps track of the robot's position, visited cells, and manages movement decisions to ensure all accessible areas are covered.

- `map.py`: This module includes classes and methods for map management and conversion. It reads map and robot configuration files, dilates the map based on the robot's dimensions to designate accessible areas, and transforms the high-resolution map into a low-resolution grid, marking cells as occupied if an obstacle's coverage exceeds a certain threshold. It can also select a polygonal region of the original map for localized operation.

- `router.py`: This module projects a computed path from a simplified grid back to the original high-resolution map. It includes functions that aid in handling partial obstacle coverage, avoiding collisions, and converting diagonal lines to orthogonal paths, ensuring smooth and safe navigation for the AGV.

- `utilities.py`: This file holds various utility functions used across the project. These functions handle file operations, data conversions, and other common tasks that don't specifically belong to the other modules.

- `plot_utils.py`: This module is used for visualization purposes. It provides functions to plot the environment map, the planned path for the AGV, and the actual path taken, which can be useful for debugging and understanding the AGV's movements.

## Installation

Tango is designed as a Python package, ensuring easy installation alongside its required dependencies.

To install the package, use a package manager like pip:

```bash
pip install cpp_solver
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.