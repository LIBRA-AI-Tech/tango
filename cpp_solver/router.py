import numpy as np
from typing import List, Tuple
from cpp_solver.utilities import find_central_point
from cpp_solver.map import Parser

def get_dilated_padded_grid(map_parser: Parser) -> np.ndarray:
    grid_dilated = map_parser.grid_dilated
    grid_dilated_pad = map_parser._grid_dilated
    cell_dim = map_parser._cell_dim

    height, width = grid_dilated.shape
    pad_height = cell_dim - (height % cell_dim) if height % cell_dim != 0 else 0
    pad_width = cell_dim - (width % cell_dim) if width % cell_dim != 0 else 0
    grid_dilated_padded = map_parser.pad_and_expand(grid_dilated, pad_height, pad_width, grid_dilated_pad, map_parser._polygon)

    return grid_dilated_padded

def rescale_path(map_parser: Parser, path: List[Tuple]) -> List[Tuple]:
    
    cell_dim = map_parser._cell_dim
    grid_dilated_padded = get_dilated_padded_grid(map_parser)
    
    rescaled_path = []
    for p1, p2 in path:
        
        cell1 = map_parser.split_to_cells(grid_dilated_padded)[p1[1],p1[0]]
        dy1, dx1 = find_central_point(cell1)
        nx1 = int(p1[0]*cell_dim + dx1)
        ny1 = int(p1[1]*cell_dim + dy1) 
        
        cell2 = map_parser.split_to_cells(grid_dilated_padded)[p2[1],p2[0]]
        dy2, dx2 = find_central_point(cell2)
        nx2 = int(p2[0]*cell_dim + dx2)
        ny2 = int(p2[1]*cell_dim + dy2)

        rescaled_path.append(((nx1, ny1), (nx2, ny2)))

    return rescaled_path

def orthogonalize_path_line(img_map: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[Tuple, Tuple]:
    (xs,ys), (xe,ye) = start, end

    if img_map[int(ys),int(xe)] != 1:
        mid = (xe,ys)
    elif img_map[int(ye),int(xs)] != 1:
        mid = (xs,ye)
    else:
        print("Couldn't navigate around obstacle. Going through it.")
        return (start, end), (start, end)

    return (start, mid), (mid, end)

def orthogonalize_path(map_parser: Parser, path: List[Tuple]) -> List[Tuple]:
    
    cell_dim = map_parser._cell_dim
    grid_dilated_padded = get_dilated_padded_grid(map_parser)

    orthogonalized_path = []
    for ((xs,ys), (xd,yd)) in path:
        if xs != xd and ys != yd:
            line1, line2 = orthogonalize_path_line(grid_dilated_padded, (xs,ys), (xd,yd))
            orthogonalized_path.append(line1)
            orthogonalized_path.append(line2)
        else:
            orthogonalized_path.append(((xs,ys),(xd,yd)))

    return orthogonalized_path

def get_distance(p1: int, p2: int) -> int:
    """Helper function to get the absolute distance between two points"""
    return abs(p1 - p2)

def is_same_sign(num1: int, num2: int) -> int:
    """Helper function to check if two numbers have the same sign"""
    return num1 * num2 > 0

def normalize_path(path: List[Tuple], threshold: float) -> List[Tuple]:
    # Initialize the normalized path
    normalized_path = []
    last_edge = path[0]
    i = 1

    while i < len(path):
        point2, point3 = path[i]
        point1, _ = last_edge

        # When point1, point2, point3 are on the same vertical line
        if point1[0] == point2[0] == point3[0]:
            if get_distance(point2[1], point1[1]) >= threshold and not is_same_sign(point2[1] - point1[1], point3[1] - point1[1]):
                normalized_path.append((point1, point2))
                last_edge = (point2, point3)
            elif point1[1] != point3[1]:
                normalized_path.append((point1, point3))
                last_edge = (point1, point3)
            else:
                last_edge = normalized_path[-1]

        # When point1, point2, point3 are on the same horizontal line
        elif point1[1] == point2[1] == point3[1]:
            if get_distance(point2[0], point1[0]) >= threshold and not is_same_sign(point2[0] - point1[0], point3[0] - point1[0]):
                normalized_path.extend([(point1, point2), (point2, point3)])
                last_edge = (point2, point3)
            elif point1[0] != point3[0]:
                normalized_path.append((point1, point3))
                last_edge = (point1, point3)
            else:
                last_edge = normalized_path[-1]

        # When point1, point2, point3 are not on the same line
        else:
            normalized_path.append((point1, point2))
            last_edge = (point2, point3)

        i += 1

    return normalized_path