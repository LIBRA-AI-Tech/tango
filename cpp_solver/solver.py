import math
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import jinja2
import numpy as np
import unified_planning
from cpp_solver.utilities import get_unreachable_points, patch_array
from scipy.spatial import KDTree
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

unified_planning.shortcuts.get_environment().credits_stream = None

class ProblemNotSolvable(Exception):
    """Raised when the problem is unsolvable"""

class NoActionException(Exception):
    """Raised when solution involves no action"""

def parse_cell(cell):
    cell = str(cell)
    regex = r"cell-(?P<x>[0-9]*)-(?P<y>[0-9]*)"
    m = re.match(regex, cell)
    x = int(m.group('x'))
    y = int(m.group('y'))
    return (x, y)

class CoverageSolver():

    def __init__(self, grid: List[List[float]], output_path: str, template_path: str, position: List[int] = [0, 0], direction: str = 'deast') -> None:
        """Class representing a coverage solver.

        Parameters
        ----------
            grid (List[List[float]]): The grid representing the coverage area.
            output_path (str): The path to the output directory.
            template_path (str): The path to the Jinja template directory.
            position (List[int], optional): The starting position of the robot on the grid. Defaults to [0, 0].
            direction (str, optional): The initial direction of the robot. Must be one of ['deast', 'dwest', 'dsouth', 'dnorth']. Defaults to 'deast'.

        Raises
        ------
            ValueError: If the `direction` parameter is not recognized.
        """
        if direction not in ['deast', 'dwest', 'dsouth', 'dnorth']:
            raise ValueError('`direction` not recognized')
        x0, y0 = position
        self._grid = get_unreachable_points(np.array(grid).astype(int), ([y0, x0])).astype(bool)
        self._subgrid = np.array([])
        self._boundary = []
        self._current_pos = position
        self._direction = [direction]
        self._visited = {tuple(position)}
        self._path = []
        self._turns = 0
        self._counter = 0
        self._offsets = [0, 0, 0, 0] # [xwest, ynorth, xeast, ysouth]
        self._template_path = template_path
        self._working_path = os.path.join(output_path, f"{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(self._working_path)
        shutil.copy2(os.path.join(template_path, 'domain.pddl'), os.path.join(self._working_path, 'domain.pddl'))

    def info(self) -> Dict[str, Union[str, int]]:
        """
        Returns information about the coverage solver.

        Returns
        -------
            Dict[str, Union[str, int]]: A dictionary containing the length of the path, coverage percentage, total length, and number of turns.
        """
        total_length = self._grid.size - np.count_nonzero(self._grid)
        return {'length': len(self._path), 'covered': f"{round(len(self._visited) / total_length * 100, 1)}%", 'totalLength': total_length, 'turns': self._turns}

    @property
    def grid(self) -> np.ndarray:
        """
        Property representing the grid.

        Returns
        -------
            np.ndarray: The grid.
        """
        if len(self._boundary) == 0:
            return self._grid
        xmin, ymin, _, _ = self._boundary
        return patch_array(self._subgrid, self._grid, xmin, ymin)

    @property
    def direction(self) -> str:
        """
        Property representing the current direction.

        Returns
        -------
            str: The current direction.
        """
        direction = self._direction[-1]
        return direction

    def is_visited(self, boundaries: List[int]) -> bool:
        """
        Checks if all cells within the given boundaries have been visited.

        Parameters
        ----------
            boundaries (List[int]): The boundaries to check.

        Returns
        -------
            bool: True if all cells have been visited, False otherwise.
        """
        grid = self.grid
        xmin, ymin, xmax, ymax = boundaries
        visited = True
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                if x >= grid.shape[1] or y >= grid.shape[0]:
                    continue
                if grid[y][x]:
                    continue
                if (x, y) in self._visited:
                    continue
                visited = False
                break
        return visited

    def occupied_grid(self) -> np.ndarray:
        """
        Generates a grid where visited and cells with obstacles are marked as True.

        Returns
        -------
            np.ndarray: The occupied grid.
        """
        grid = self.grid
        visited = np.zeros(grid.shape, dtype=bool)
        x, y = zip(*self._visited)
        visited[y, x] = True
        visited = np.logical_or(visited, grid)
        return visited

    def check_offsets(self) -> None:
        """
        Updates the offsets of the grid.
        
        Offsets are calculated based on the current position, visited cells and position of obstacles.
        """
        xwest, ynorth, xeast, ysouth = self._offsets
        visited = self.occupied_grid()
        x0, y0 = self._current_pos

        for y in range(ynorth, y0):
            if np.all(visited[y]):
                ynorth += 1
            else:
                break
        for y in range(visited.shape[0] - ysouth - 1, y0, -1):
            if np.all(visited[y]):
                ysouth += 1
            else:
                break

        visited = visited.T
        for x in range(xwest, x0):
            if np.all(visited[x]):
                xwest += 1
            else:
                break
        for x in range(visited.shape[0] - xeast - 1, x0, -1):
            if np.all(visited[x]):
                xeast += 1
            else:
                break
        self._offsets = [xwest, ynorth, xeast, ysouth]

    def proceed(self, subgrid_dim: int=5, offsets: Optional[List[int]] = None) -> Tuple[Union[np.array, bool]]:
        """
        Advances the coverage solver by generating a subgrid and solving the corresponding problem.

        Parameters
        ----------
            subgrid_dim (int, optional): The dimension of the subgrid. Defaults to 5.

        Returns
        -------
            Tuple[Union[np.array, bool]]: A tuple containing the path and a flag indicating whether the coverage should continue.
        """
        if offsets is not None:
            self._offsets = offsets
        else:
            self.check_offsets()
        relaxed = False
        boundaries, grid = self.partition(subgrid_dim=subgrid_dim)
        xmin, ymin, xmax, ymax = boundaries
        free_sides, free_cells = self.find_free_boundaries(boundaries)
        inside = np.array([self.grid[y][x] or (x, y) in self._visited for x in range(xmin, xmax) for y in range(ymin, ymax)])
        number_of_unvisited = np.size(self.grid) - np.count_nonzero(self.grid) - len(self._visited) - inside.size + np.count_nonzero(inside)
        if free_sides == 0 and number_of_unvisited > 0:
            relaxed = True
            free_sides, free_cells = self.find_free_boundaries(boundaries, relaxed=relaxed)
            if len(free_cells) == 0:
                free_cells = [self._current_pos]
            problem_type = 'complete'
        else:
            problem_type = 'complete' if len(free_cells) <= 1 else 'partial'
        if not relaxed:
            goal = self.find_last_in_direction(boundaries, self.direction) if problem_type == 'partial' else []
        else:
            goal = self.find_most_distant(boundaries, self.direction)
        problem = self.define_problem(boundaries, grid, problem_type=problem_type, free_boundaries=free_cells, goal=goal)
        result_status = PlanGenerationResultStatus.SOLVED_OPTIMALLY if problem_type == 'complete' else PlanGenerationResultStatus.SOLVED_SATISFICING
        try:
            path = self.solve(problem, result_status=result_status)
        except NoActionException:
            return self.proceed(subgrid_dim=subgrid_dim+1, offsets=offsets)
        except ProblemNotSolvable:
            offx_min, offy_min, offx_max, offy_max = self._offsets
            offsets = [max(offx_min-1, 0), max(offy_min-1, 0), max(offx_max-1, 0), max(offy_max-1, 0)]
            return self.proceed(subgrid_dim=subgrid_dim+1, offsets=offsets)
        else:
            should_continue = (self._grid.size - np.count_nonzero(self._grid) - len(self._visited)) > 0
        return path, should_continue

    def partition(self, subgrid_dim: int=5) -> Tuple[List[int], np.ndarray]:
        """Partitions the grid into subgrids based on the current position

        Creates all possible partitions for the subsequent solution and elects
        one based on scores taking into account the current direction,
        the covered area, the position of the obstacles and the relative to
        the current position direction of the nearest uncovered area.

        Parameters
        ----------
            subgrid_dim (int, optional): The dimension of the subgrid. Defaults to 5.

        Returns
        -------
            Tuple[List[int], np.ndarray]: A tuple containing the boundaries of the selected partition and the created subgrid.
        """
        grid = self.grid
        x0, y0 = self._current_pos
        offx_min, offy_min, offx_max, offy_max = self._offsets
        xranges = []
        yranges = []

        for x in range(x0 + 1 - subgrid_dim, x0 + 1):
            xdim = min(subgrid_dim, grid.shape[1] - offx_max - offx_min)
            if x < offx_min or x + xdim > grid.shape[1] - offx_max:
                continue
            xranges.append([x, x + xdim])
        for y in range(y0 + 1 - subgrid_dim, y0 + 1):
            ydim = min(subgrid_dim, grid.shape[0] - offy_max - offy_min)
            if y < offy_min or y + ydim > grid.shape[0] - offy_max:
                continue
            yranges.append([y, y + ydim])

        ranges = [(xmin, ymin, xmax, ymax) for xmin, xmax in xranges for ymin, ymax in yranges]
        ranges = list(set(ranges))

        scores = []
    
        yf, xf = self._locate_closest_free()
        theta = np.arctan2(yf - y0, xf - x0)
        pi = math.pi
        self._theta = theta

        for boundary in ranges:
            free_sides, free_exits = self.find_free_boundaries(boundary)
            score = 0
            xmin, ymin, xmax, ymax = boundary
            self._generate_subgrid(boundary, x0, y0)
            # Free remaining cells in grid
            if theta >= -pi / 2 and theta < pi / 2:
                score += (xmax - 1 - x0) / subgrid_dim
            else:
                score += (x0 - xmin) / subgrid_dim
            if theta >= 0:
                score += (ymax - 1 - y0) / subgrid_dim
            else:
                score += (y0 - ymin) / subgrid_dim
            # Direction
            dscore = 5
            if self.direction == 'dwest':
                score += (x0 - xmin)*dscore / (xmax - xmin)
            elif self.direction == 'deast':
                score += (xmax - x0)*dscore / (xmax - xmin)
            elif self.direction == 'dnorth':
                score += (y0 - ymin)*dscore / (ymax - ymin)
            elif self.direction == 'dsouth':
                score += (ymax - y0)*dscore / (ymax - ymin)
            # Free cells in boundaries
            if free_sides == 0:
                if not self.is_visited(boundary):
                    score += 100
            else:
                score += (1 - len(free_exits) / 2 / (xmax - xmin + ymax - ymin - 2))/free_sides
            scores.append(score)
        index = np.argmax(scores)
        boundary = ranges[index]
        subgrid = self._generate_subgrid(boundary, x0, y0)
        return boundary, subgrid

    def _generate_subgrid(self, boundary: List[int], x0: int, y0: int) -> np.ndarray:
        """Generates a subgrid based on the given boundary and current position.

        Parameters
        ----------
            boundary (List[int]): The boundary of the subgrid
            x0 (int): The current x-coordinate.
            y0 (int): The current y-coordinate.

        Returns
        -------
            np.ndarray: The generated subgrid.
        """
        grid = self._grid
        xmin, ymin, xmax, ymax = boundary
        subgrid = np.array([grid[y][x] for x in range(xmin, xmax) for y in range(ymin, ymax)])
        subgrid = np.reshape(subgrid, (-1, ymax-ymin))
        subgrid = get_unreachable_points(subgrid.astype(int), ([y0 - ymin, x0 - xmin])).astype(bool)
        self._subgrid = subgrid.T
        self._boundary = boundary
        return subgrid
    
    def _locate_closest_free(self) -> Tuple[int]:
        """Finds the closest uncovered free cell to the current position.

        Returns
        -------
            Tuple[int]: The coordinates of the closest uncovered free cell.
        """
        x0, y0 = self._current_pos
        grid = self.occupied_grid()
        free = np.array([[i, j] for i, j in zip(*np.where(grid==False))])
        tree = KDTree(free)
        return free[tree.query([y0, x0])[1]]

    def find_most_distant(self, boundaries: List[int], direction: str) -> List[int]:
        """Finds the most distant unoccupied cell within the given boundaries in the specified direction.

        Parameters
        ----------
            boundaries (List[int]): The boundaries to search within.
            direction (str): The direction to search in.

        Raises:
            ValueError: If the `direction` parameter is not recognized.

        Returns
        -------
            List[int]: The coordinates of the most distant cell.
        """
        xmin, ymin, xmax, ymax = boundaries
        grid = self.grid
        x, y = self._current_pos
        if direction in ['deast', 'dwest']:
            rangey = range(y, ymax) if ymax - y > y - ymin else range(y, ymin - 1, -1)
            rangex = [x]*len(rangey)
        elif direction in ['dnorth', 'dsouth']:
            rangex = range(x, xmax) if xmax - x > x - xmin else range(x, xmin - 1, -1)
            rangey = [y]*len(rangex)
        else:
            raise ValueError('`direction` not recognized')
        last = [x, y]
        for i, j in zip(rangex, rangey):
            if i == x and j == y:
                continue
            if (i, j) in self._visited:
                continue
            if grid[j][i]:
                break
            last = [i, j]
        return last

    def find_last_in_direction(self, boundaries: List[int], direction: str) -> List[int]:
        """Find the last cell (without obstacle) in the specified direction within the given boundaries.

        Parameters
        ----------
            boundaries (List[int]): The boundaries to search within.
            direction (str): The direction to search in.

        Returns
        -------
            List[int]: The coordinates of the last cell in the specified direction.
        """
        xmin, ymin, xmax, ymax = boundaries
        grid = self.grid
        directions = ['deast', 'dsouth', 'dwest', 'dnorth']
        steps = [+1, +1, -1, -1]
        x, y = self._current_pos
        step = steps[directions.index(direction)]
        if direction in ['deast', 'dwest']:
            rangex = range(x, xmax) if step == 1 else range(x, xmin - 1, step)
            rangey = [y]*len(rangex)
        else:
            rangey = range(y, ymax) if step == 1 else range(y, ymin - 1, step)
            rangex = [x]*len(rangey)
        last = [x, y]
        for i, j in zip(rangex, rangey):
            if i == x and j == y:
                continue
            if (i, j) in self._visited:
                continue
            if grid[j][i]:
                break
            last = [i, j]
        return last

    def find_free_boundaries(self, boundaries: List[int], relaxed=False) -> Tuple[int, List[int]]:
        """Finds the free edges within the given boundaries.

        Parameters
        ----------
            boundaries (List[int]): The boundaries to search within.
            relaxed (bool, optional): Flag indicating whether to perform a relaxed search.
                In relaxed mode, visited cells are considered free, otherwise visited and
                cells containing obstacles are considered occupied. Defaults to False.

        Returns
        -------
            Tuple[Union[int, List[int]]]: A tuple containing the number of free sides and the list of free cells.
        """
        free_count = 0
        free = []
        grid = self.grid
        visited = self._visited
        xmin, ymin, xmax, ymax = boundaries
        outer = self.is_visited(boundaries)
        at_x, at_y = self._current_pos

        theta = self._theta
        pi = math.pi
        
        # Check North
        free_north = []
        y0 = ymin if not outer else ymin - 1
        for x in range(xmin, xmax):
            if outer and theta > 0:
                break
            if y0 - 1 < 0 or (x == at_x and ymin == at_y):
                continue
            if grid[y0 - 1][x] or grid[y0][x]:
                continue
            if not relaxed and ((x, y0 - 1) in visited or (x, y0) in visited):
                continue
            free_north.append((x, ymin))
        if len(free_north) > 0:
            free_count += 1
            free.extend(free_north)
        # Check West
        free_west = []
        x0 = xmin if not outer else xmin - 1
        for y in range(ymin, ymax):
            if outer and theta > -pi / 2 and theta < pi/2:
                break
            if x0 - 1 < 0 or (xmin == at_x and y == at_y):
                continue
            if grid[y][x0 - 1] or grid[y][x0]:
                continue
            if not relaxed and ((x0 - 1) in visited or (x0, y) in visited):
                continue
            free_west.append((xmin, y))
        if len(free_west) > 0:
            free_count += 1
            free.extend(free_west)
        # Check South
        free_south = []
        y0 = ymax if not outer else ymax + 1
        for x in range(xmin, xmax):
            if outer and theta < 0:
                break
            if y0 >= grid.shape[0] or (x == at_x and ymax - 1 == at_y):
                continue
            if grid[y0][x] or grid[y0 - 1][x]:
                continue
            if not relaxed and ((x, y0) in visited or (x, y0 - 1) in visited):
                continue
            free_south.append((x, ymax - 1))
        if len(free_south) > 0:
            free_count += 1
            free.extend(free_south)
        # Check East
        free_east = []
        x0 = xmax if not outer else xmax + 1
        for y in range(ymin, ymax):
            if outer and (theta < -pi / 2 or theta > pi / 2):
                break
            if x0 >= grid.shape[1] or (xmax - 1 == at_x and y == at_y):
                continue
            if grid[y][x0] or grid[y][x0 - 1]:
                continue
            if not relaxed and ((x0, y) in visited or (x0 - 1, y) in visited):
                continue
            free_east.append((xmax - 1, y))
        if len(free_east) > 0:
            free_count += 1
            free.extend(free_east)
        return free_count, list(set(free))

    def define_problem(self, boundaries: List[int], subgrid: np.ndarray, problem_type: str='complete', free_boundaries: List[Optional[List[int]]]=[], goal: List[Optional[List[int]]]=[]):
        """Defines the problem to be solved by the planner

        Parameters
        ----------
            boundaries (List[int]): The boundaries of the problem.
            subgrid (np.ndarray): The subgrid representing the problem area.
            problem_type (str, optional): The type of the problem; one of 'complete', 'partial'. Defaults to 'complete'.
            free_boundaries (List[Optional[List[int]]], optional): The list of free boundaries. Defaults to [].
            goal (List[Optional[List[int]]], optional): The goal cell(s) to be reached. Defaults to [].

        Returns
        -------
            Problem: The defined problem
        """
        self._counter += 1
        xmin, ymin, xmax, ymax = boundaries
        template_path = os.path.join(self._template_path, problem_type)
        loader = jinja2.FileSystemLoader(searchpath=template_path)
        environment = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        template = environment.get_template('problem.j2')
        content = template.render(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            current=self._current_pos,
            direction=self.direction,
            grid=self.grid,
            subgrid=subgrid,
            enumerate=enumerate,
            len=len,
            free_boundaries=free_boundaries,
            goal=goal,
            visited=[p for p in self._visited if p[0] < xmax and p[0] >= xmin and p[1] < ymax and p[1] >= ymin]
        )
        problem_pddl = os.path.join(self._working_path, f'problem-{self._counter}.pddl')
        with open(problem_pddl, 'w', encoding="utf-8") as f:
            f.write(content)
        # Read Problem/Domain
        domain_pddl = os.path.join(self._working_path, 'domain.pddl')

        reader = PDDLReader()
        problem = reader.parse_problem(domain_pddl, problem_pddl)
        problem.add_quality_metric(unified_planning.model.metrics.MinimizeActionCosts({problem.action("move"): 0, problem.action("revisit"): 5, problem.action("turn"): 10}))
        return problem

    def solve(self, problem, result_status=PlanGenerationResultStatus.SOLVED_OPTIMALLY):
        solver = 'enhsp-opt' if result_status == PlanGenerationResultStatus.SOLVED_OPTIMALLY else 'enhsp-opt'
        with OneshotPlanner(
            name=solver,
            problem_kind=problem.kind,
            optimality_guarantee=result_status,
        ) as planner:
            result = planner.solve(problem)
        if result.status not in unified_planning.engines.results.POSITIVE_OUTCOMES:
            raise ProblemNotSolvable
        if len(result.plan.actions) == 0:
            raise NoActionException
        path = [(parse_cell(a.actual_parameters[0]), parse_cell(a.actual_parameters[1])) for a in result.plan.actions if not str(a).startswith('turn')]
        self._path.extend(path)
        self._result = result
        for a in result.plan.actions:
            if str(a).startswith('turn'):
                self._turns += 1
                self._direction.append(str(a.actual_parameters[-1]))
            else:
                self._visited.add(parse_cell(a.actual_parameters[1]))
        self._current_pos = list(parse_cell(result.plan.actions[-1].actual_parameters[1]))
        return path
