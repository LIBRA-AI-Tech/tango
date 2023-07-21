import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from typing import Optional, Tuple, List
from .map import Parser

def get_maps(map_parser: Parser, grid_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:  
    if grid_type == 'dilated':
        return map_parser.grid_dilated, map_parser._grid_dilated
    elif grid_type == 'eroded':
        return map_parser.grid_eroded, map_parser._grid_eroded
    elif grid_type == 'original':
        return map_parser.grid, map_parser._grid
    else:
        raise ValueError(f'Invalid grid type: {grid_type}')

def get_padding(height: int, width: int, cell_dim: int) -> Tuple[int, int]:
    pad_height = cell_dim - (height % cell_dim) if height % cell_dim != 0 else 0
    pad_width = cell_dim - (width % cell_dim) if width % cell_dim != 0 else 0
    return pad_height, pad_width

def plot_config(ax: Axes, grid_shape: Tuple[int,int], gridlines_alpha: float = 1) -> None:
    ax.grid(color='green', linestyle='-', linewidth=1.5, alpha=gridlines_alpha)
    ax.set_xticks(np.linspace(0, 1, grid_shape[1] + 1))
    ax.set_yticks(np.linspace(0, 1, grid_shape[0] + 1))
    ax.set_xticks(np.linspace(1/(2*grid_shape[1]), 1 - 1/(2*grid_shape[1]), grid_shape[1]), minor=True)
    ax.set_yticks(np.linspace(1/(2*grid_shape[0]), 1 - 1/(2*grid_shape[0]), grid_shape[0]), minor=True)
    ax.set_xticklabels(list(range(0, grid_shape[1])), fontsize=6, minor=True)
    ax.set_yticklabels(list(range(grid_shape[0]-1, -1, -1)), fontsize=6, minor=True)
    ax.tick_params(axis='x', which='major', labelbottom=False, top=True, bottom=False)
    ax.tick_params(axis='y', which='major', labelleft=False)
    ax.tick_params(axis='x', which='minor', labelbottom=False, labeltop=True, top=True, length=0)
    ax.tick_params(axis='y', which='minor', length=0)

def plot_grid(map_parser: Parser, grid_type: Optional[str] = None, draw_plot: bool = True) -> Axes:
    '''
    Plot the grid generated from the map.
    '''
    grid = map_parser.grid_downscaled
    cell_dim = map_parser._cell_dim
    
    grid_alpha = 1 if grid_type is None else 0.5
    grid_zorder = None if draw_plot else 1
    map_alpha = 1 if draw_plot else 0.5
    map_zorder = None if draw_plot else 10
    gridlines_alpha = 0.6 if draw_plot else 0.4

    _, ax = plt.subplots()

    if grid_type is not None:
        image_map, pad_map = get_maps(map_parser, grid_type)
        pad_height, pad_width = get_padding(image_map.shape[0], image_map.shape[1], cell_dim)
        padded_image = map_parser.pad_and_expand(image_map, pad_height, pad_width, pad_map, map_parser._polygon)
        ax.imshow(padded_image==1, cmap='gray_r', extent=[0, 1, 0, 1], alpha=map_alpha, zorder=map_zorder)

    grid_alpha = 1 if grid_type is None else 0.5
    grid_zorder = None if draw_plot else 1
    map_alpha = 1 if draw_plot else 0.5

    ax.imshow(grid, cmap='gray_r', extent=[0, 1, 0, 1], alpha=grid_alpha, zorder=grid_zorder) 
    plot_config(ax, grid.shape, gridlines_alpha)
    
    if draw_plot:
        plt.show()
    
    return ax

def plot_map(map_parser: Parser, grid_type: str = 'original', draw_plot: bool = True) -> Axes:
    '''
    Plot the map
    '''
    image_map, pad_map = get_maps(map_parser, grid_type)
    cell_dim = map_parser._cell_dim
    grid_shape = map_parser.grid_downscaled.shape

    _, ax = plt.subplots()

    pad_height, pad_width = get_padding(image_map.shape[0], image_map.shape[1], cell_dim)
    padded_image = map_parser.pad_and_expand(image_map, pad_height, pad_width, pad_map, map_parser._polygon)

    if draw_plot:
        ax.imshow(padded_image==1, cmap='gray_r', extent=[0,1,0,1], alpha=1)
    else:
        ax.imshow(padded_image==1, cmap='gray_r', alpha=1)

    if draw_plot:
        plot_config(ax, grid_shape, 0.6)
        plt.show()

    return ax

def plot_grid_with_path(map_parser: Parser, path: List[Tuple], grid_type: str = 'original', cmap: Optional[str] = None, arrow_dims: Tuple[float,float] = (0.3, 0.3)) -> None:

    ax = plot_grid(map_parser, grid_type, draw_plot=False)
    
    grid_shape = map_parser.grid_downscaled.shape
    dx, dy = 1/(2*grid_shape[1]), 1/(2*grid_shape[0])
    rescaled_path = []
    for (x1,y1),(x2,y2) in path:
        x1, x2 = [x/grid_shape[1] for x in (x1, x2)]
        y1, y2 = [y/grid_shape[0] for y in (y1, y2)]
        rescaled_path.append(((x1+dx,1-(y1+dy)),(x2+dx,1-(y2+dy))))
        
    
    if cmap is not None:
        colormap = mpl.colormaps[cmap]
    else:
        default_color = 'red'

    norm = mpl.cm.colors.Normalize(vmin=0, vmax=len(rescaled_path)-1)
    head_width, head_length = (arrow_dim/np.max(grid_shape) for arrow_dim in arrow_dims)
    for i, ((x1, y1), (x2, y2)) in enumerate(rescaled_path):
        dx, dy = x2 - x1, y2 - y1
        color = colormap(norm(i)) if cmap is not None else default_color
        ax.arrow(x1, y1, dx, dy, head_width=head_width, head_length=head_length, length_includes_head=True, color=color, alpha=0.95, zorder=20)

    plt.show()

def plot_path(map_parser: Parser, path: List[Tuple], grid_type: str = 'original', cmap: Optional[str] = None, arrow_dims: Tuple[float,float] = (0.3, 0.3)) -> None:
    
    ax = plot_map(map_parser, grid_type, draw_plot=False)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.grid(False)

    if cmap is not None:
        colormap = mpl.colormaps[cmap]
    else:
        default_color = 'red'

    norm = mpl.cm.colors.Normalize(vmin=0, vmax=len(path)-1)
    head_width, head_length = arrow_dims
    for i, ((x1, y1), (x2, y2)) in enumerate(path):
        dx, dy = x2 - x1, y2 - y1
        color = colormap(norm(i)) if cmap is not None else default_color
        ax.arrow(x1, y1, dx, dy, head_width=head_width, head_length=head_length, length_includes_head=True, color=color, overhang=1, alpha=0.95)
    
    plt.show()