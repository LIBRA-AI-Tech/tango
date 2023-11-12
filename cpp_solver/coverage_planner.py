import os
from shutil import rmtree
from typing import Union, List
import numpy as np
from cpp_solver.map import Parser
from cpp_solver.solver import Solver
from cpp_solver.router import Router

class CoveragePlanner:

    def __init__(self, hard_map: str, soft_map: str, robot_config: str, **kwargs) -> None:
        self._map = Parser(hard_map, soft_map, robot_config, **kwargs)
        self._solver = None

    @property
    def map(self):
        return self._map

    @property
    def solver(self):
        return self._solver

    def run(self, polygon: Union[np.ndarray, List], initial_position=[0, 0], direction='dnorth', unit: str = 'meter'):
        self._map.select(polygon, unit=unit)
        self._solver = Solver(self.map.grid_downscaled, position=initial_position, direction=direction)
        self._solver.run()

        router = Router(self._map)
        rescaled_path = router.rescale_path(self._solver._path)
        self._rescaled_path = rescaled_path

        path = [d[0] for d in rescaled_path]
        path.append(rescaled_path[-1][1])
        path = np.array(path)

        x0, y0 = np.nanmin(self.map._polygon, axis=0)
        ymax, _ = self.map._grid.shape

        abs_path = [[c[0]+x0, ymax - (c[1]+y0)] for c in path]
        if unit == 'meter':
            abs_path = np.array([self.map.pixel_to_meter(c) for c in abs_path])
        try:
            rmtree(self.solver._working_path)
        except:
            pass

        return abs_path
