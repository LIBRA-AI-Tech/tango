import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from queue import Queue

def flood_fill(arr, x, y, marker):
    """
    Perform flood-fill starting from position (x, y) in the array.
    """
    if x < 0 or x >= arr.shape[0] or y < 0 or y >= arr.shape[1]:
        return
    if arr[x, y] != 0:
        return
    
    arr[x, y] = marker
    
    flood_fill(arr, x+1, y, marker)
    flood_fill(arr, x-1, y, marker)
    flood_fill(arr, x, y+1, marker)
    flood_fill(arr, x, y-1, marker)
    
    return

def get_unreachable_points(arr, position):
    """
    Identify and return a NumPy array of the same dimension as the initial array,
    filled with 1s for all the unreachable points from the current position.
    """
    marker = 2  # Start marker for labeling unreachable areas
    
    # Copy the original array to avoid modifying it
    x, y = position
    arr_copy = arr.copy()
    
    # Perform flood-fill from the current position
    flood_fill(arr_copy, y, x, marker)
    
    # Replace reachable points (0s) with 1s in the copied array
    arr_copy[arr_copy == 0] = 1
    arr_copy[arr_copy == marker] = 0
    
    return arr_copy

def patch_array(arr1, arr2, x0, y0):
    arr2 = arr2.copy()
    n, m = arr1.shape
    i_range = slice(x0, x0 + m)
    j_range = slice(y0, y0 + n)
    try:
        arr2[j_range, i_range] = arr1
    except Exception as e:
        print(str(e))

    return arr2

def is_valid_move(x, y, grid):
    rows, cols = grid.shape
    return (0 <= x < rows) and (0 <= y < cols) and (grid[x][y] != 1)

def bfs(grid, border_points):
    rows, cols = grid.shape
    distance_map = np.full((rows, cols), np.inf)
    queue = Queue()
    
    for point in border_points:
        queue.put(point)
        distance_map[point] = 0

    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    while not queue.empty():
        x, y = queue.get()
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if is_valid_move(new_x, new_y, grid) and distance_map[x][y] + 1 < distance_map[new_x][new_y]:
                distance_map[new_x][new_y] = distance_map[x][y] + 1
                queue.put((new_x, new_y))
    return distance_map

def find_central_point(grid):
    rows, cols = grid.shape
    center = (rows // 2, cols // 2)
    border_points = list((x, y) for x in range(rows) for y in [0, cols-1]) + list((x, y) for x in [0, rows-1] for y in range(cols))
    distance_map = bfs(grid, border_points)
    valid_points = np.argwhere(grid != 1)
    _, closest_point = min((distance.euclidean(center, point), tuple(point)) for point in valid_points if np.isfinite(distance_map[tuple(point)]))
    return closest_point
