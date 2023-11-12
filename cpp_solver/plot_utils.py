import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from typing import Optional, Tuple, List
from cpp_solver.map import Parser

class PlotUtils:

    def __init__(self, map_parser: Parser) -> None:
        """

        Parameters
        ----------
        map_parser : Parser
            Parser object which contains the map data.
        """
        self._map_parser = map_parser

    def get_maps(self, map_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:  
        """
        Get the specified map type.

        Parameters
        ----------
        map_type : Optional[str], default=None
            Type of the map grid. It can be 'dilated', 'eroded', 'original'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Desired cropped map and its uncropped version.

        Raises
        ------
        ValueError
            If the input grid type is not valid.
        """
        map_parser = self._map_parser
        if map_type == 'dilated':
            return map_parser.grid_dilated, map_parser._grid_dilated
        elif map_type == 'eroded':
            return map_parser.grid_eroded, map_parser._grid_eroded
        elif map_type == 'original':
            return map_parser.grid, map_parser._grid
        else:
            raise ValueError(f'Invalid grid type: {map_type}')

    def _get_padding(self, height: int, width: int, cell_dim: int) -> Tuple[int, int]:
        """
        Get the padding for the given dimensions.

        Parameters
        ----------
        height : int
            The height of the grid.
        width : int
            The width of the grid.
        cell_dim : int
            The dimension of each cell.

        Returns
        -------
        Tuple[int, int]
            Padding for height and width.
        """
        pad_height = cell_dim - (height % cell_dim) if height % cell_dim != 0 else 0
        pad_width = cell_dim - (width % cell_dim) if width % cell_dim != 0 else 0
        return pad_height, pad_width

    def _plot_config(self, ax: Axes, grid_shape: Tuple[int,int], gridlines_alpha: float = 1) -> None:
        """
        Set the configuration for the plot.

        Parameters
        ----------
        ax : Axes
            The axes of the plot to be configured.
        grid_shape : Tuple[int, int]
            The shape of the grid (number of cells in height and width).
        gridlines_alpha : float, default=1
            The transparency of grid lines.
        """
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

    def plot_grid(self, map_type: Optional[str] = None, draw_plot: bool = True) -> Axes:
        """
        Plot the grid generated from the map.

        Parameters
        ----------
        map_type : Optional[str], default=None
            Type of the map which overlayed on the grid. It can be 'dilated', 'eroded', 'original'.
        draw_plot : bool, default=True
            If True, the plot will be displayed.

        Returns
        -------
        Axes
            The Axes object with the plot.
        """
        map_parser = self._map_parser
        grid = map_parser.grid_downscaled
        cell_dim = map_parser._cell_dim
        
        grid_alpha = 1 if map_type is None else 0.5
        grid_zorder = None if draw_plot else 1
        map_alpha = 1 if draw_plot else 0.5
        map_zorder = None if draw_plot else 10
        gridlines_alpha = 0.6 if draw_plot else 0.4

        _, ax = plt.subplots()

        if map_type is not None:
            image_map, pad_map = self.get_maps(map_parser, map_type)
            pad_height, pad_width = self._get_padding(image_map.shape[0], image_map.shape[1], cell_dim)
            padded_image = map_parser.pad_and_expand(image_map, pad_height, pad_width, pad_map, map_parser._polygon)
            ax.imshow(padded_image==1, cmap='gray_r', extent=[0, 1, 0, 1], alpha=map_alpha, zorder=map_zorder)

        grid_alpha = 1 if map_type is None else 0.5
        grid_zorder = None if draw_plot else 1
        map_alpha = 1 if draw_plot else 0.5

        ax.imshow(grid, cmap='gray_r', extent=[0, 1, 0, 1], alpha=grid_alpha, zorder=grid_zorder) 
        self._plot_config(ax, grid.shape, gridlines_alpha)
        
        if draw_plot:
            plt.show()
        
        return ax

    def plot_map(self, map_type: str = 'original', draw_plot: bool = True) -> Axes:
        """
        Plot the map.

        Parameters
        ----------
        map_type : str, default='original'
            Type of the map to plot. It can be 'dilated', 'eroded', 'original'.
        draw_plot : bool, default=True
            If True, the plot will be displayed.

        Returns
        -------
        Axes
            The Axes object with the plot.
        """
        map_parser = self._map_parser
        image_map, pad_map = self._get_maps(map_parser, map_type)
        cell_dim = map_parser._cell_dim
        grid_shape = map_parser.grid_downscaled.shape

        _, ax = plt.subplots()

        pad_height, pad_width = self._get_padding(image_map.shape[0], image_map.shape[1], cell_dim)
        padded_image = map_parser.pad_and_expand(image_map, pad_height, pad_width, pad_map, map_parser._polygon)

        if draw_plot:
            ax.imshow(padded_image==1, cmap='gray_r', extent=[0,1,0,1], alpha=1)
        else:
            ax.imshow(padded_image==1, cmap='gray_r', alpha=1)

        if draw_plot:
            self._plot_config(ax, grid_shape, 0.6)
            plt.show()

        return ax

    def plot_grid_with_path(self, path: List[Tuple], map_type: str = 'original', cmap: Optional[str] = None, arrow_dims: Tuple[float,float] = (0.3, 0.3)) -> None:
        """
        Plot the path on the grid.

        Parameters
        ----------
        path : List[Tuple]
            The path to be plotted.
        map_type : str, default='original'
            Type of the map which is overlayed on the grid. It can be 'dilated', 'eroded', 'original'.
        cmap : Optional[str], default=None
            Colormap for the path.
        arrow_dims : Tuple[float,float], default=(0.3, 0.3)
            Dimensions of the arrow.
        """
        map_parser = self._map_parser
        ax = self._plot_grid(map_parser, map_type, draw_plot=False)
        
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

    def plot_path(self, path: List[Tuple], map_type: str = 'original', cmap: Optional[str] = None, arrow_dims: Tuple[float,float] = (0.3, 0.3)) -> None:
        """
        Plot the path on the map area.

        Parameters
        ----------
        path : List[Tuple]
            The path to be plotted.
        map_type : str, default='original'
            Type of the map overlayed on the grid. It can be 'dilated', 'eroded', 'original'.
        cmap : Optional[str], default=None
            Colormap for the path.
        arrow_dims : Tuple[float,float], default=(0.3, 0.3)
            Dimensions of the arrow.
        """
        map_parser = self._map_parser
        ax = self._plot_map(map_parser, map_type, draw_plot=False)

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
