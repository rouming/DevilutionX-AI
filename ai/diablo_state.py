"""
diablo_state.py - Provides high-level access and analysis tools for
                  the live Diablo (DevilutionX engine) game state.

This module includes utilities to map and interpret shared memory from
a running Diablo process. It provides functions to inspect dungeon
tiles, player status, monsters, items, and interactive objects, as
well as utilities for environmental flagging, region labeling, and
pathfinding.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

from types import SimpleNamespace
import ctypes
import enum
import mmap
import numpy as np
import os
import subprocess
import tempfile
import time

import dbg2ctypes
import devilutionx as dx
import maze
import procutils
import ring

class DoorState(enum.Enum):
    DOOR_CLOSED   = 0
    DOOR_OPEN     = 1,
    DOOR_BLOCKED  = 2

def round_up_int(i, d):
    assert type(i) == int
    assert type(d) == int
    return (i + d - 1) // d * d

def get_ctypes_array_shape(array):
    shape = []
    t = type(array)
    while issubclass(t, ctypes.Array):
        shape.append(t._length_)
        t = t._type_
    return tuple(shape)

def dungeon_dim(d):
    return get_ctypes_array_shape(d.dObject)

def to_object(d, pos):
    obj_id = d.dObject_np[pos]
    if obj_id != 0:
        return d.Objects[abs(obj_id) - 1]
    return None

def is_interactable(obj):
    return obj.selectionRegion != 0

def is_breakable(obj):
    return obj._oBreak == 1

def is_door_closed(obj):
    return obj._oVar4 == DoorState.DOOR_CLOSED.value

def is_door(obj):
    return obj._oDoorFlag

def is_barrel(obj):
    return obj._otype in (dx._object_id.OBJ_BARREL.value,
                          dx._object_id.OBJ_BARRELEX.value,
                          dx._object_id.OBJ_POD.value,
                          dx._object_id.OBJ_PODEX.value,
                          dx._object_id.OBJ_URN.value,
                          dx._object_id.OBJ_URNEX.value)

def is_crucifix(obj):
    return obj._otype in (dx._object_id.OBJ_CRUX1.value,
                          dx._object_id.OBJ_CRUX2.value,
                          dx._object_id.OBJ_CRUX3.value)

def is_chest(obj):
    return obj._otype in (dx._object_id.OBJ_CHEST1.value,
                          dx._object_id.OBJ_CHEST2.value,
                          dx._object_id.OBJ_CHEST3.value,
                          dx._object_id.OBJ_TCHEST1.value,
                          dx._object_id.OBJ_TCHEST2.value,
                          dx._object_id.OBJ_TCHEST3.value,
                          dx._object_id.OBJ_SIGNCHEST.value)

def is_sarcophagus(obj):
    return obj._otype in (dx._object_id.OBJ_SARC.value,
                          dx._object_id.OBJ_L5SARC.value)

def is_floor(d, pos):
    return not (d.SOLData[d.dPiece_np[pos]] & \
                (dx.TileProperties.Solid.value | dx.TileProperties.BlockMissile.value))

def is_arch(d, pos):
    return d.dSpecial_np[pos] > 0

def is_wall(d, pos):
    return not is_floor(d, pos) and not is_arch(d, pos)

def is_trigger(d, pos):
    for trig in d.trigs[:d.numtrigs.value]:
        if trig.position.x == pos[0] and trig.position.y == pos[1]:
            return True
    return False

def is_game_paused(d):
    return d.PauseMode.value != 0

def is_player_dead(d):
    return d.player._pmode == dx.PLR_MODE.PM_DEATH.value

def player_position(d):
    return (d.player.position.tile.x, d.player.position.tile.y)

def player_direction(d):
    # Compensate dungeon 45CW rotation
    return (d.player._pdir - 1) % (len(dx.Direction) - 1)

def count_active_objects(d):
    def interesting_objects(oid):
        obj = d.Objects[oid]
        if is_barrel(obj):
            return 1 if obj._oSolidFlag else 0
        elif is_chest(obj) or is_sarcophagus(obj) or is_crucifix(obj):
            return 1 if is_interactable(obj) else 0
        return 0
    return sum(map(interesting_objects, d.ActiveObjects))

def get_closed_doors_ids(d):
    closed_doors = []
    for oid in d.ActiveObjects:
        obj = d.Objects[oid]
        if is_door(obj) and is_door_closed(obj):
            closed_doors.append(oid)
    return closed_doors

def count_active_items(d):
    return d.ActiveItemCount.value

def count_active_monsters(d):
    return d.ActiveMonsterCount.value

def count_active_monsters_total_hp(d):
    return sum(map(lambda mid: d.Monsters[mid].hitPoints, d.ActiveMonsters))

def count_explored_tiles(d):
    bits = dx.DungeonFlag.Explored.value
    return np.sum((d.dFlags_np & bits) == bits)

def find_trigger(d, tmsg):
    for trig in d.trigs:
        if trig._tmsg == tmsg.value:
            return trig
    return None

class Rect:
    # Top left
    lt = None
    width  = 0
    height = 0

class EnvironmentRect:
    # Source rectangle
    srect = Rect()
    # Destination rectangle
    drect = Rect()

    def __init__(self, d, radius=None):
        dundim = dungeon_dim(d)
        if radius is not None:
            pos = player_position(d)

            x_min = max(pos[0] - radius[0], 0)
            x_max = min(pos[0] + radius[0], dundim[0])
            y_min = max(pos[1] - radius[1], 0)
            y_max = min(pos[1] + radius[1], dundim[1])

            self.srect.lt     = np.array([x_min, y_min])
            self.srect.width  = x_max - self.srect.lt[0]
            self.srect.height = y_max - self.srect.lt[1]

            # Place player position in the center of a destination rectangle
            self.drect.lt     = radius - (pos - self.srect.lt)
            self.drect.width  = radius[0] * 2
            self.drect.height = radius[1] * 2
        else:
            self.srect.lt     = np.array([0, 0])
            self.srect.width  = dundim[0]
            self.srect.height = dundim[1]
            self.drect        = self.srect

class EnvironmentFlag(enum.Enum):
    Player         = 1<<0
    Wall           = 1<<1
    Trigger        = 1<<2
    DoorOpened     = 1<<3
    DoorClosed     = 1<<4
    Missile        = 1<<5
    Monster        = 1<<6
    UnknownObject  = 1<<7
    Crucifix       = 1<<8
    Barrel         = 1<<9
    Chest          = 1<<10
    Sarcophagus    = 1<<11
    Item           = 1<<12
    Explored       = 1<<13
    Visible        = 1<<14
    Interactable   = 1<<15

def get_environment(d, radius=None, ignore_explored_visible=False):
    """Returns the environment, either the whole dungeon or windowed
    if a radius is specified. Setting @ignore_explored_visible to True
    is used when the entire dungeon needs to be revealed. However, be
    careful, as this can be CPU intensive, so @ignore_explored_visible
    set to False is the default.
    """
    env_rect = EnvironmentRect(d, radius)
    # Transpose to Diablo indexing: (width, height), instead of numpy
    # (height, weight)
    env = np.zeros((env_rect.drect.width, env_rect.drect.height),
                   dtype=np.uint16)

    for j in range(env_rect.srect.height):
        for i in range(env_rect.srect.width):
            spos = (env_rect.srect.lt[0] + i, env_rect.srect.lt[1] + j)
            obj = to_object(d, spos)
            s = 0

            if d.dFlags_np[spos] & dx.DungeonFlag.Explored.value:
                s |= EnvironmentFlag.Explored.value
            if d.dFlags_np[spos] & dx.DungeonFlag.Visible.value:
                s |= EnvironmentFlag.Visible.value

            if ignore_explored_visible or s & EnvironmentFlag.Explored.value:
                if is_wall(d, spos):
                    s |= EnvironmentFlag.Wall.value
                if is_trigger(d, spos):
                    s |= EnvironmentFlag.Trigger.value
                if obj is not None and is_door(obj):
                    if is_door_closed(obj):
                        s |= EnvironmentFlag.DoorClosed.value
                    else:
                        s |= EnvironmentFlag.DoorOpened.value
            if ignore_explored_visible or s & EnvironmentFlag.Visible.value:
                if d.dFlags_np[spos] & dx.DungeonFlag.Missile.value:
                    s |= EnvironmentFlag.Missile.value
                if d.dMonster_np[spos] > 0:
                    s |= EnvironmentFlag.Monster.value

                if obj is not None:
                    if is_barrel(obj):
                        if is_breakable(obj):
                            s |= EnvironmentFlag.Barrel.value
                    elif is_crucifix(obj):
                        s |= EnvironmentFlag.Crucifix.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                    elif is_chest(obj):
                        s |= EnvironmentFlag.Chest.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                    elif is_sarcophagus(obj):
                        s |= EnvironmentFlag.Sarcophagus.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                    elif is_door(obj):
                        # Handled above by the explored 'if' branch
                        pass
                    else:
                        s |= EnvironmentFlag.UnknownObject.value
                        if is_interactable(obj):
                            s |= EnvironmentFlag.Interactable.value
                if d.dItem_np[spos] > 0:
                    s |= EnvironmentFlag.Item.value

            if spos == player_position(d):
                s |= EnvironmentFlag.Player.value

            # Transpose to Diablo indexing: (x, y), instead of numpy (y, x)
            dpos = (env_rect.drect.lt[0] + i, env_rect.drect.lt[1] + j)
            env[dpos] = s

    return env

def get_surroundings(d, radius):
    env = get_environment(d, radius)
    surroundings = np.full(env.shape, ' ', dtype=str)

    for j, row in enumerate(env):
        for i, tile in enumerate(row):
            if tile == 0:
                continue
            if tile & EnvironmentFlag.Explored.value:
                s = ' '
            if tile & EnvironmentFlag.Visible.value:
                s = '.'
            if tile & EnvironmentFlag.Wall.value:
                s = '#'
            if tile & EnvironmentFlag.Trigger.value:
                s = '$'
            if tile & EnvironmentFlag.DoorClosed.value:
                s = 'D'
            if tile & EnvironmentFlag.DoorOpened.value:
                s = 'd'
            if tile & EnvironmentFlag.Barrel.value:
                s = 'B'
            if tile & EnvironmentFlag.UnknownObject.value:
                s = 'O' if tile & EnvironmentFlag.Interactable.value else 'o'
            if tile & EnvironmentFlag.Chest.value:
                s = 'C' if tile & EnvironmentFlag.Interactable.value else 'c'
            if tile & EnvironmentFlag.Sarcophagus.value:
                s = 'S' if tile & EnvironmentFlag.Interactable.value else 's'
            if tile & EnvironmentFlag.Crucifix.value:
                s = 'U' if tile & EnvironmentFlag.Interactable.value else 'u'
            if tile & EnvironmentFlag.Item.value:
                s = 'I'
            if tile & EnvironmentFlag.Missile.value:
                s = '%'
            if tile & EnvironmentFlag.Monster.value:
                s = '@'
            if tile & EnvironmentFlag.Player.value:
                if is_player_dead(d):
                    s = 'X'
                else:
                    s = '*'
                    match player_direction(d):
                        case dx.Direction.North.value:
                            s = "\u2191"
                        case dx.Direction.NorthEast.value:
                            s = "\u2197"
                        case dx.Direction.East.value:
                            s = "\u2192"
                        case dx.Direction.SouthEast.value:
                            s = "\u2198"
                        case dx.Direction.South.value:
                            s = "\u2193"
                        case dx.Direction.SouthWest.value:
                            s = "\u2199"
                        case dx.Direction.West.value:
                            s = "\u2190"
                        case dx.Direction.NorthWest.value:
                            s = "\u2196"
            surroundings[j, i] = s

    return surroundings

def get_dungeon_graph_and_path(d, start, goal):
    # Environment of the whole dungeon
    env = get_environment(d, ignore_explored_visible=True)
    # 0 - walls
    # 1 - empty areas, probably occupied by player, monsters, etc
    empty_env = \
        (env & EnvironmentFlag.Player.value) | \
        (env & EnvironmentFlag.Monster.value) | \
        (env & EnvironmentFlag.Barrel.value) | \
        (env & EnvironmentFlag.Item.value) | \
        (env & EnvironmentFlag.Trigger.value) | \
        (env == 0) | \
        (env == EnvironmentFlag.Explored.value) | \
        (env == EnvironmentFlag.Visible.value) | \
        (env == (EnvironmentFlag.Explored.value | \
                 EnvironmentFlag.Visible.value))
    # Doors positions
    doors = np.argwhere(env & (EnvironmentFlag.DoorOpened.value | \
                               EnvironmentFlag.DoorClosed.value))

    # Label independent regions
    labeled_regions, num_regions = maze.detect_regions(empty_env)
    # Build graph of connected regions
    regions_graph, regions_doors, doors_matrix = \
        maze.get_regions_graph(doors, labeled_regions, num_regions)

    start_region = labeled_regions[start]
    goal_region = labeled_regions[goal]

    assert start_region != 0
    assert goal_region != 0

    # Shortest path between regions
    regions_path = maze.bfs_regions_path(regions_graph, start_region,
                                         goal_region)
    assert regions_path is not None

    # Doors between regions on the shortest path. We could use set()
    # here, but we need to keep an order
    path_doors = []
    for i, region in enumerate(regions_path):
        if i < len(regions_path) - 1:
            next_region = regions_path[i + 1]
            # Get the door coordinates, which leads to the goal region
            x, y = doors_matrix[region, next_region]
            assert x != 0 and y != 0
            if (x, y) not in path_doors:
                path_doors.append((x, y))
            regions_doors[region][(x, y)] = True

    return regions_doors, labeled_regions, regions_path, path_doors

def map_shared_state(path, offset):
    f = open(path, "r+b")
    mmapped = mmap.mmap(f.fileno(), 0)
    f.close()

    vars_dict = {}

    for var in dx.VARS:
        addr = var['addr']
        assert offset <= addr
        obj = var['type'].from_buffer(mmapped, addr - offset)
        name = dbg2ctypes.strip_namespaces(var['name'])
        vars_dict[name] = obj

        if isinstance(obj, ctypes.Array):
            # Add numpy array view
            np_view = np.ctypeslib.as_array(obj)
            vars_dict[name + "_np"] = np_view

    state = SimpleNamespace(**vars_dict)

    return mmapped, state

def map_shared_state_by_pid(pid, mshared_path):
    for attempt in range(0, 10):
        try:
            # Get offset of mapped file
            _, offset = procutils.get_mapped_file_and_offset_of_pid(
                pid, mshared_path)
            if not offset:
                # Wait until remapped
                time.sleep(0.1)
                continue
            # Open the file and map it to memory
            mmapped, state = map_shared_state(mshared_path, offset)
            return mmapped, state
        except FileNotFoundError:
            time.sleep(0.1)
    else:
        raise FileNotFoundError(mshared_path)

class DiabloGame:
    def __init__(self, state_dir, proc, log_file, mshared_path,
                 mmapped, state):
        self.state_dir = state_dir
        self.proc = proc
        self.log_file = log_file
        self.mshared_path = mshared_path
        self.mmapped = mmapped
        self.state = state
        self.last_tick = 0
        self.acceptance_tick = 0

        # Catch up with the events queue. The @read_idx from the queue
        # is not used to support other processes that might want to
        # attach.
        self.events_queue_read_idx = state.events_queue.write_idx

    def __del__(self):
        self.stop_or_detach()

    def stop_or_detach(self):
        if self.proc:
            self.proc.terminate()
        if self.log_file:
            self.log_file.close()
        if self.state_dir:
            self.state_dir.cleanup()

    def ticks(self, d=None):
        t = self.state.game_ticks.value if d is None else d.game_ticks.value
        # There are two phases of tick updates:
        #     "odd" phase - before keys are processed
        #    "even" phase - after keys are processed
        # Divide on two to have a full game update cycle
        return t // 2

    def update_ticks(self):
        missed = self.ticks() - self.last_tick
        self.last_tick += missed
        return missed

    def same_ticks(self):
        diff = self.ticks() - self.last_tick
        return diff == 0

    def retrieve_event(self):
        read_idx = self.events_queue_read_idx
        entry = ring.get_entry_to_retrieve(self.state.events_queue, read_idx)
        if entry == None:
            return None
        self.events_queue_read_idx += 1
        return entry

    def submit_key(self, key):
        # Busy-loop for the next tick. It's important to understand that every
        # key has press and release phases, so we can't submit the same
        # two keys sequentially if one was not released; otherwise, the
        # second key will be lost. With the following loop, we
        # introduce a 1-tick delay from the previous key acceptance.
        # Also be aware that each tick has two phases, hence the +2.
        while self.state.game_ticks.value < self.acceptance_tick + 2:
            time.sleep(0.01)

        assert ring.has_capacity_to_submit(self.state.input_queue)
        entry = ring.get_entry_to_submit(self.state.input_queue)
        entry.type = key
        entry.data = 0

        # Submit key
        ring.submit(self.state.input_queue)

        # Busy-loop for actual key acceptance
        while ring.nr_submitted_entries(self.state.input_queue) != 0:
            time.sleep(0.01)

        # Acceptance of a key is always the last "even" phase, thus
        # round up on 2.
        self.acceptance_tick = round_up_int(self.state.game_ticks.value, 2)

    @staticmethod
    def run(config):
        cfg_file = open("diablo.ini.template", "r")
        cfg = cfg_file.read()
        cfg_file.close()
        cfg = cfg.format(seed=config["seed"],
                         headless=1,
                         mshared_filename=config["mshared-filename"],
                         no_monsters=1 if config["no-monsters"] else 0)

        prefix = "diablo-%d-" % config["seed"]
        state_dir = tempfile.TemporaryDirectory(prefix=prefix)
        cfg_file = open(state_dir.name + "/diablo.ini", "w")
        cfg_file.write(cfg)
        cfg_file.close()

        log_file = open(state_dir.name + "/diablo.log", "w", buffering=1)

        cmd = [
            config["diablo-bin-path"],
            '--config-dir', state_dir.name,
            '--save-dir', state_dir.name,
        ]
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        mshared_path = os.path.abspath(state_dir.name + "/" + config["mshared-filename"])
        mmapped, state = map_shared_state_by_pid(proc.pid, mshared_path)
        return DiabloGame(state_dir, proc, log_file, mshared_path, mmapped, state)

    @staticmethod
    def attach(config):
        mshared_path = config['attach-path']
        offset = config['attach-offset']
        mmapped, state = map_shared_state(mshared_path, offset)
        return DiabloGame(None, None, None, mshared_path, mmapped, state)

    @staticmethod
    def run_or_attach(config):
        if 'attach-path' in config:
            return DiabloGame.attach(config)
        return DiabloGame.run(config)
