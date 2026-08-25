#!/usr/bin/env python3
"""
tile-info.py - Print automap tile info for a given dungeon coordinate.

Usage:
    ./tile-info.py <shmem-path> <x> <y>

    shmem-path  path to the devilutionX shared memory file
    x, y        game tile coordinates (e.g. 73 39)

Prints:
    - megatile coords
    - AutomapView status
    - dungeon tile ID
    - AutomapTypeTile type and flags (named)
    - the 4 dPiece values and their SOLData flags
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import devilutionx as dx
import diablo_state
import procutils


def flags_names(enum_cls, value):
    names = []
    for member in enum_cls:
        if member.value and (value & member.value) == member.value:
            names.append(member.name)
    return names if names else ["None"]


def type_name(value):
    try:
        return dx.Types(value).name
    except ValueError:
        return "Unknown(%d)" % value


def sol_flags_names(value):
    return flags_names(dx.TileProperties, value)


def main():
    if len(sys.argv) != 4:
        print("Usage: %s <shmem-path> <x> <y>" % sys.argv[0])
        sys.exit(1)

    shmem_path = sys.argv[1]
    gx = int(sys.argv[2])
    gy = int(sys.argv[3])

    # find the devilutionX process that has this file mapped and get its offset
    import psutil
    min_addr = min(v['addr'] for v in dx.VARS)
    offset = None
    for proc in psutil.process_iter(attrs=['pid', 'exe']):
        _, off = procutils.get_mapped_file_and_offset_of_pid(proc.info['pid'], shmem_path)
        if off is not None and off <= min_addr:
            offset = off
            break
    if offset is None:
        print("error: no devilutionX process found with %s mapped" % shmem_path)
        sys.exit(1)

    d = diablo_state.map_devilutionx_state(shmem_path, offset)

    mx = (gx - 16) // 2
    my = (gy - 16) // 2

    print("game tile  : %d, %d" % (gx, gy))
    print("megatile   : %d, %d" % (mx, my))
    print()

    # AutomapView
    am_view = int(d.AutomapView[mx, my])
    print("AutomapView: %d (%s)" % (am_view, "explored" if am_view > 0 else "unexplored"))

    # dungeon tile ID and AutomapTypeTile
    tile_id = int(d.dungeon[mx, my])
    am_tile = d.AutomapTypeTiles[tile_id]
    tile_type  = int(am_tile['type'])
    tile_flags = int(am_tile['flags'])

    print("dungeon ID : %d" % tile_id)
    print("amtype     : %d (%s)" % (tile_type, type_name(tile_type)))
    print("amflags    : %d (%s)" % (tile_flags, ", ".join(flags_names(dx.Flags, tile_flags))))

    # 4 game tiles of the megatile
    print()
    print("pieces (2x2 game tiles, dx=0..1, dy=0..1):")
    solid_bit = dx.TileProperties.Solid.value
    for dy in range(2):
        for dx_ in range(2):
            tx = 16 + mx * 2 + dx_
            ty = 16 + my * 2 + dy
            piece = int(d.dPiece[tx, ty])
            sol   = int(d.SOLData[piece])
            print("  dPiece[%d,%d] = %4d  SOLData = %3d (%s)" % (
                tx, ty, piece, sol,
                ", ".join(sol_flags_names(sol)) if sol else "None"))

    # Full AutomapView 40x40 grid (x=col, y=row; mark target megatile with '*')
    print()
    print("AutomapView[40x40]  (. explored, * target, space unexplored):")
    for y in range(40):
        row = []
        for x in range(40):
            if x == mx and y == my:
                row.append('*')
            elif d.AutomapView[x, y] > 0:
                row.append('.')
            else:
                row.append(' ')
        print("  %2d |%s|" % (y, "".join(row)))

    # Flags set for this megatile
    print()
    print("amflags set:")
    for name in flags_names(dx.Flags, tile_flags):
        print("  %s" % name)


if __name__ == "__main__":
    main()
