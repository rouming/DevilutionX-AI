"""
maze.py — Region Detection and Navigation Tools for Grid-Based Mazes

This module provides utilities for analyzing and navigating 2D mazes
represented as NumPy arrays. It includes functions for:

- Calculating Manhattan and Euclidean distances.
- Detecting and labeling connected open regions (e.g. rooms, corridors).
- Building a region connectivity graph using door positions.
- Finding shortest paths between regions using BFS.

Primarily designed for dungeon-style map layouts such as Diablo.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

import numpy as np
from scipy.ndimage import label
from collections import deque

def manhattan_dist(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
