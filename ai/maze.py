"""
maze.py - Region Detection and Navigation Tools for Grid-Based Mazes

This module provides utilities for analyzing and navigating 2D mazes
represented as NumPy arrays. It includes functions for:

- Calculating Manhattan and Euclidean distances.
- Detecting and labeling connected open regions (e.g. rooms, corridors).
- Building a region connectivity graph using door positions.
- Finding shortest paths between regions using BFS.

Primarily designed for dungeon-style map layouts such as Diablo.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

from collections import deque
from scipy.ndimage import label
import heapq
import math
import numpy as np

def manhattan_dist(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_centroid(region) -> tuple[int, int]:
    # Extract all points and convert to (y, x) -> (x, y)
    y, x = np.where(region == 1)
    points = np.column_stack((y, x))

    # geometric centroid (average)
    cx = sum(x) / len(x)
    cy = sum(y) / len(y)

    # snap to nearest region cell
    center = min(points, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
    return tuple(center)

def detect_regions(maze):
    """Finds open areas (rooms/corridors) and labels them uniquely"""
    structure = np.ones((3, 3))
    labeled_maze, num_features = label(maze, structure)
    return labeled_maze, num_features

def get_regions_graph(doors, labeled_maze, num_regions):
    """
    Returns a dict where each region is a key, and the value is a set of
    adjacent regions.
    Returns a dict where each region is a key, and the value is another
    dict which holds door positions and boolean value, describing if the
    door leads to the goal. By default set to false.
    Returns a matrix that describes door positions by regions.
    """
    rows, cols = labeled_maze.shape
    regions_graph = {i: set() for i in range(1, num_regions + 1)}
    regions_doors = {i: {} for i in range(1, num_regions + 1)}
    doors_matrix = np.zeros((num_regions + 1, num_regions + 1, 2), dtype=int)

    for (x, y) in doors:
        regions = set()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                r = labeled_maze[nx, ny]
                if r > 0:
                    regions.add(r)
                    regions_doors[r][(x, y)] = False

        # In Diablo, it is quite possible that a door does not lead to
        # a new closed region but actually stands in the same region
        # and does not connect anything.
        if len(regions) != 2:
            continue

        r1, r2 = regions

        regions_graph[r1].add(r2)
        regions_graph[r2].add(r1)
        doors_matrix[r1, r2] = (x, y)
        doors_matrix[r2, r1] = (x, y)

    return regions_graph, regions_doors, doors_matrix

def bfs_regions_path(regions_graph, start_region, goal_region):
    """Finds the shortest path in the regions graph."""
    queue = deque([(start_region, [start_region])])
    visited = set()

    while queue:
        current_region, path = queue.popleft()
        if current_region == goal_region:
            return path

        if current_region in visited:
            continue
        visited.add(current_region)

        for neighbor in regions_graph[current_region]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None

def shortest_path(start, goal, cost_fn, valid_path=None) -> list[tuple[int, int]]:
    """Finds the shortest path on a 2D grid (dungeon). Expects an
    environment where @cost_fn returns 0 if tile is not walkable"""
    dist = {start: 0}
    pq = [(0, start)]
    prev = {}
    dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if not (dr == dc == 0)]

    while pq:
        d, (r, c) = heapq.heappop(pq)
        if (r, c) == goal:
            path = [(r, c)]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1]
        if d > dist[(r, c)]:
            continue
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            cost = cost_fn((nr, nc))
            if cost <= 0:
                continue

            if abs(dr) + abs(dc) == 2:
                # Diagonal move. Prevent squeezing through diagonally
                # adjacent tiles. Consider this case:
                #
                #    #.         .       #.
                #   ↗ #  or  ↗ #   or  ↗
                #
                if cost_fn((r, nc)) != 1 or cost_fn((nr, c)) != 1:
                    continue
            step = math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1
            nd = d + step * cost
            if nd < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, (nr, nc)))
    return []

def visibility_frontier(start, visibility_fn) -> tuple[set[tuple], set[tuple]]:
    """Returns a cloud of visible points and its frontier.
    @visibility_fn returns -1 if the point is beyond the environment,
    1 if it is visible, and 0 otherwise. """
    q = deque([start])
    visited = {start}
    frontier = set()

    while q:
        r, c = q.popleft()
        # 4-neighbors
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited:
                vis = visibility_fn((nr, nc))
                if vis < 0:
                    continue
                elif vis == 1:
                    visited.add((nr, nc))
                    q.append((nr, nc))
                else:
                    frontier.add((r, c))

    return visited - frontier, frontier


if __name__ == "__main__":
    grid_text = """
    1 1 0 0 0 0 0 0
    1 1 0 1 1 1 0 0
    1 1 0 0 1 1 0 0
    1 1 0 0 1 1 0 0
    1 1 0 3 1 1 0 0
    1 2 1 1 1 1 0 0
    1 1 0 0 0 0 0 0
    1 1 0 0 0 0 0 0
    """

    start = (5, 1)
    end = (4, 3)

    grid = np.array([[int(x) for x in line.split()] for line in grid_text.strip().splitlines()])

    def cost_fn(p):
        shape = grid.shape
        if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
            return -1
        return 1 if (grid[p] or p == end) else 0

    path = shortest_path(start, end, cost_fn)
    for p in path:
        grid[p] = 8

    print("Grid:")
    print(grid)
    print()
    print(f"Path: {path}")
