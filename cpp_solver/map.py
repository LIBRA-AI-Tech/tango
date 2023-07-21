import os
from dataclasses import dataclass
from typing import Union, Optional, Tuple

import cv2
import numpy as np
import pygeos as pg
import yaml
from PIL import Image, ImageOps


@dataclass
class ImageDefs:
    image: str
    resolution: float
    origin: list
    occupied_thresh: float
    free_thresh: float
    negate: bool
    
    @classmethod
    def from_yaml(cls, yaml_file):
        location = os.path.dirname(yaml_file)
        with open(yaml_file, 'r') as f:
            defs = yaml.safe_load(f)
        image = defs['image'] if os.path.isabs(defs['image']) else os.path.join(location, defs['image'])
        return cls(
            image=image, 
            resolution=defs['resolution'], 
            origin=defs['origin'], 
            occupied_thresh=defs['occupied_thresh'],
            free_thresh=defs['free_thresh'],
            negate=defs['negate']
        )
    
@dataclass
class RobotDefs:
    robot_dim: float
    mower_dim: float

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            defs = yaml.safe_load(f)
        return cls(
            robot_dim=defs['robot_dim'],
            mower_dim=defs['mower_dim']
        )

class Parser:
    
    def __init__(self, hard_map: str, soft_map: str, robot_config: str, cell_dim: Optional[int] = None, occupied_cell_threshold: float = 0.5, use_disc: bool = False) -> None:

        self._obs_defs = ImageDefs.from_yaml(hard_map)
        self._soft_defs = ImageDefs.from_yaml(soft_map)
        self._robot_defs = RobotDefs.from_yaml(robot_config)


        self._cell_dim = self.default_cell_dim if cell_dim is None else cell_dim
        self._occupied_cell_threshold = occupied_cell_threshold
        self._use_disc = use_disc

        hard = 1*self._read(self._obs_defs)

        kernel = np.ones((50, 50), np.uint8)
        hard = cv2.morphologyEx(hard.astype('uint8'), cv2.MORPH_CLOSE, kernel)
        hard = hard.astype('int')

        soft = 1*self._read(self._soft_defs)
        hard[(soft==1) & (hard!=1)] = -1
        
        self._grid = hard
        self._grid_dilated = self.dilate()
        self._grid_eroded = self.erode()
        self._grid_downscaled = None
        self._free_cell_areas = None
        
        self._selection = False
        self._polygon = None
        self._selected_grid = None
        self._selected_grid_dilated = None
        self._selected_grid_eroded = None
        self._selected_grid_downscaled = None
        self._selected_free_cell_areas = None

    @property
    def grid(self) -> np.ndarray:
        grid = self._grid if not self._selection else self._selected_grid
        return grid
    
    @property
    def grid_dilated(self) -> np.ndarray:
        grid_dilated = self._grid_dilated if not self._selection else self._selected_grid_dilated
        return grid_dilated
    
    @property
    def grid_eroded(self) -> np.ndarray:
        grid_eroded = self._grid_eroded if not self._selection else self._selected_grid_eroded
        return grid_eroded
    
    @property
    def grid_downscaled(self) -> np.ndarray:
        grid_downscaled = self._grid_downscaled if not self._selection else self._selected_grid_downscaled
        return grid_downscaled
    
    @property
    def default_cell_dim(self) -> int:
        map_resolution = self._obs_defs.resolution
        mower_dim = self._robot_defs.mower_dim
        return np.ceil(mower_dim/map_resolution).astype(int)
    
    def _read(self, defs: ImageDefs) -> np.ndarray:
        im = ImageOps.grayscale(Image.open(defs.image))
        arr = (255 - np.array(im)) / 255.0 if not defs.negate else np.array(im) / 255.0
        im.close()
        return arr>=defs.occupied_thresh

    def show(self, reverse: bool = False) -> Image.Image:
        arr = 1.*self.grid
        arr[arr==-1] = 0.5
        arr = ((1 - arr)*255).astype(np.uint8) if reverse else (arr*255).astype(np.uint8)
        im = Image.fromarray(arr)
        return im
    
    def select(self, polygon: Union[np.ndarray, list]) -> None:

        def select_grid(grid, polygon: np.ndarray):      
            xmin, ymin = np.nanmin(polygon, axis=0)
            xmax, ymax = np.nanmax(polygon, axis=0)
            polygon_geo = pg.polygons(polygon - np.array([[xmin, ymin]*polygon.shape[0]]).reshape(polygon.shape))
            selected_grid = grid[ymin:ymax,xmin:xmax]
            selected_grid = np.array([
                [
                    selected_grid[j][i] if selected_grid[j][i] == 1 or pg.intersects(pg.points([i, j]), polygon_geo) else 0
                    for i in range(0, selected_grid.shape[1])
                ] for j in range(0, selected_grid.shape[0])
            ])
            return selected_grid

        if isinstance(polygon, list):
            polygon = np.array(polygon)

        self._selected_grid = select_grid(self._grid, polygon)
        self._selected_grid_dilated = select_grid(self._grid_dilated, polygon)
        self._selected_grid_eroded = select_grid(self._grid_eroded, polygon) 
        self._selection = True
        self._polygon = polygon

    def dilate(self) -> np.ndarray:
        
        boolean_grid = (self._grid > 0).astype(np.uint8)
        soft = 1*self._read(self._soft_defs)
        
        map_resolution = self._obs_defs.resolution
        robot_dim = self._robot_defs.robot_dim

        dim = np.ceil(robot_dim/map_resolution).astype(int)
        if self._use_disc:
            struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
        else:
            struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (dim, dim))
        dilated_binary = cv2.dilate(boolean_grid, struct_elem)

        non_hard = np.where((dilated_binary == 0) & (soft == 1))
        dilated_grid = dilated_binary.copy().astype(np.int64)
        dilated_grid[non_hard[0], non_hard[1]] = -1

        return dilated_grid
    
    def erode(self) -> np.ndarray:
        
        boolean_grid = (self._grid_dilated > 0).astype(np.uint8)
        soft = 1*self._read(self._soft_defs)

        map_resolution = self._obs_defs.resolution
        mower_dim = self._robot_defs.mower_dim

        dim = np.ceil(mower_dim/map_resolution).astype(int)
        if self._use_disc:
            struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
        else:
            struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (dim, dim))
        eroded_binary = cv2.erode(boolean_grid, struct_elem)

        non_hard = np.where((eroded_binary == 0) & (soft == 1))
        eroded_grid = eroded_binary.copy().astype(np.int64)
        eroded_grid[non_hard[0], non_hard[1]] = -1

        return eroded_grid

    def pad_and_expand(self, selected_grid: np.ndarray, pad_height: int, pad_width: int, pad_grid: Optional[np.ndarray] = None, polygon: Optional[np.ndarray] = None) -> np.ndarray:

        height, width = selected_grid.shape
  
        if pad_grid is None:
            padded_image = np.pad(selected_grid, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
            return padded_image
        
        if polygon is None:
            raise Exception("polygon must be provided if pad_grid is provided")

        if isinstance(polygon, list):
            polygon = np.array(polygon)

        xmin, ymin = np.nanmin(polygon, axis=0)
        xmax, ymax = np.nanmax(polygon, axis=0)

        padded_image = np.zeros((height + pad_height, width + pad_width))
        padded_image[:height, :width] = selected_grid

        if pad_height > 0:
            padded_image[height:, :width] = pad_grid[ymax: ymax + pad_height, xmin: xmin + width]
        if pad_width > 0:
            padded_image[:height + pad_height, width:] = pad_grid[ymin: ymin + height + pad_height, xmax: xmax + pad_width]

        return padded_image

    def split_to_cells(self, grid: np.ndarray, grid_pad: Optional[np.ndarray] = None, polygon: Optional[np.ndarray] = None) -> np.ndarray:

        height, width = grid.shape
        pad_height = self._cell_dim - (height % self._cell_dim) if height % self._cell_dim != 0 else 0
        pad_width = self._cell_dim - (width % self._cell_dim) if width % self._cell_dim != 0 else 0
        
        if grid_pad is None:
            padded_image = self.pad_and_expand(grid, pad_height, pad_width)
        else:
            padded_image = self.pad_and_expand(grid, pad_height, pad_width, grid_pad, polygon)

        padded_height, _ = padded_image.shape
        cells = padded_image.reshape(padded_height//self._cell_dim, self._cell_dim, -1, self._cell_dim).swapaxes(1,2)
        
        return cells

    def downscale_grid(self, occupied_cell_threshold: Optional[float] = None) -> None:
        
        if occupied_cell_threshold is None:
            occupied_cell_threshold = self._occupied_cell_threshold

        dilated_cells = self.split_to_cells(self.grid_dilated, self._grid_dilated, self._polygon)

        cell_obstacle_count = np.count_nonzero(dilated_cells==1, axis=(2,3))
        cell_obstacle_perc = cell_obstacle_count / self._cell_dim**2
        downscaled_grid = (cell_obstacle_perc > occupied_cell_threshold)

        if self._selection:
            self._selected_grid_downscaled = downscaled_grid
        else:
            self._grid_downscaled = downscaled_grid
