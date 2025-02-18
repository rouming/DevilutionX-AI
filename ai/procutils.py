"""
procutils.py

This module provides utility functions for discovering and filtering
running processes based on various criteria, such as shared files,
PIDs, or memory mappings.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

import os
import psutil
import re

def get_maps_of_pid(pid):
    """Get a list of memory-mapped files from /proc/<pid>/maps."""
    maps = []
    maps_path = f"/proc/{pid}/maps"

    if os.path.exists(maps_path):
        try:
            with open(maps_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    start, end = parts[0].split("-")
                    maps.append([start, end, *parts[1:]])
        except (OSError, IOError):
            # Ignore unreadable files
            pass

    return maps

def get_mapped_file_and_offset_of_pid(pid, path_or_filename):
    maps = get_maps_of_pid(pid)

    # Match the ending of the mapped file so that either a full path
    # or a filename can work
    matching_files = [m for m in maps
                      if len(m) > 5 and m[-1].endswith(path_or_filename)]

    if matching_files:
        base_addr = int(maps[0][0], 16)
        mapped_addr = int(matching_files[0][0], 16)
        offset = mapped_addr - base_addr
        return matching_files[0][-1], offset

    return None, None

def procs_natural_sort(p, _nsre=re.compile(r'(\d+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(p['mshared_path'])]

def get_processes_by_binary(binary_path):
    """Find all processes matching the specified command binary name."""
    matching_processes = []
    for proc in psutil.process_iter(attrs=['pid', 'name', 'exe', 'cmdline']):
        if binary_path == proc.info['exe']:
            matching_processes.append(proc.info)
    return matching_processes

def find_processes_with_mapped_file(binary_path, path_or_filename):
    """Find processes with mapped filename"""
    matching_processes = get_processes_by_binary(binary_path)
    result = []

    for proc in matching_processes:
        pid = proc['pid']
        mshared_path, offset = get_mapped_file_and_offset_of_pid(
            pid, path_or_filename)

        if mshared_path:
            result.append({'pid': pid,
                           'exe': proc['exe'],
                           'cmdline': proc['cmdline'],
                           'offset': offset,
                           'mshared_path': mshared_path})

    return sorted(result, key=procs_natural_sort)
