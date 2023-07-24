import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from queue import Queue

def flood_fill(arr, x, y, marker):
    """
    Performs a flood-fill operation on a given 2D numpy array starting from the position (x, y).

    Parameters
    ----------
    arr : np.ndarray
        The 2D numpy array to perform the flood-fill operation on.
    x : int
        The x-coordinate of the starting point of the flood-fill.
    y : int
        The y-coordinate of the starting point of the flood-fill.
    marker : int
        The marker used to label the connected area during the flood-fill operation.
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
    Identifies and returns a NumPy array of the same dimension as the initial array, 
    filled with 1s for all the unreachable points from the current position.

    Parameters
    ----------
    arr : np.ndarray
        The 2D numpy array from which unreachable points are identified.
    position : tuple
        The starting position (x, y) from where unreachable points are identified.

    Returns
    -------
    np.ndarray
        A numpy array with 1s at unreachable points and 0s elsewhere.
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
    """
    Patches a given 2D array onto another one at the specified position.

    Parameters
    ----------
    arr1 : np.ndarray
        The 2D numpy array to be patched.
    arr2 : np.ndarray
        The 2D numpy array to be patched onto.
    x0 : int
        The x-coordinate of the starting position to patch arr1 onto arr2.
    y0 : int
        The y-coordinate of the starting position to patch arr1 onto arr2.

    Returns
    -------
    np.ndarray
        The resulting 2D numpy array after patching arr1 onto arr2.
    """

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
    '''
    Checks if a move to the position (x, y) is valid within the given grid.
    '''
    rows, cols = grid.shape
    return (0 <= x < rows) and (0 <= y < cols) and (grid[x][y] != 1)

def bfs(grid, border_points):
    """
    Performs a breadth-first search on a grid from a set of starting points, 
    computing the distance from each starting point to every other point in the grid.

    Parameters
    ----------
    grid : np.ndarray
        The 2D numpy array representing the grid.
    border_points : list
        A list of tuples where each tuple represents a starting point for the BFS.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each cell contains the minimum distance to one of the starting points.
    """

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
    """
    Finds the most centrally located reachable point in a given 2D grid.

    Parameters
    ----------
    grid : np.ndarray
        The 2D numpy array representing the grid.

    Returns
    -------
    tuple
        The coordinates (x, y) of the most centrally located reachable point in the grid.
    """

    rows, cols = grid.shape
    center = (rows // 2, cols // 2)
    border_points = list((x, y) for x in range(rows) for y in [0, cols-1]) + list((x, y) for x in [0, rows-1] for y in range(cols))
    distance_map = bfs(grid, border_points)
    valid_points = np.argwhere(grid != 1)
    _, closest_point = min((distance.euclidean(center, point), tuple(point)) for point in valid_points if np.isfinite(distance_map[tuple(point)]))
    return closest_point
