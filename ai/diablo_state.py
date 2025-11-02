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
import copy
import enum
import mmap
import numpy as np
import os
import subprocess
import tempfile
import time

from numba import types, njit
from numba.experimental import jitclass, structref

import dbg2numpy
import devilutionx as dx
import maze
import procutils
import ring


# Define a StructRef used in njit. `structref.register` associates the
# type with the default data model.  This will also install getters
# and setters to the fields of the StructRef.
@structref.register
class DevilutionStateType(types.StructRef):
    def preprocess_fields(self, fields):
        # This method is called by the type constructor for additional
        # preprocessing on the fields.
        # Here, we don't want the struct to take Literal types.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

# Define a Python type that can be use as a proxy to the StructRef
# used inside njit.
class DevilutionState(structref.StructRefProxy):
    def __new__(cls, vars_dict):
        this = structref.StructRefProxy.__new__(cls, *vars_dict.values())
        # Set attributes to access vars_dict from Python code
        for k, v in vars_dict.items():
            setattr(this, k, v)
        return this


# This associates the proxy with DevilutionStateType for the given set
# of fields
structref.define_proxy(DevilutionState, DevilutionStateType,
                       [var['short_name'] for var in dx.VARS])


AgentState = np.dtype([
	("goal_pos", dx.PointOf_int_),
], align=True)

class DoorState(enum.Enum):
    DOOR_CLOSED   = 0
    DOOR_OPEN     = 1
    DOOR_BLOCKED  = 2

@njit(cache=True)
def round_up_int(i, d):
    assert type(i) == int
    assert type(d) == int
    return (i + d - 1) // d * d

@njit(cache=True)
def dungeon_dim(d):
    return d.dObject.shape

@njit(cache=True)
def to_object(d, pos):
    obj_id = d.dObject[pos]
    if obj_id != 0:
        return d.Objects[abs(obj_id) - 1]
    return None

@njit(cache=True)
def is_interactable(obj):
    return obj.selectionRegion != 0

@njit(cache=True)
def is_breakable(obj):
    return obj._oBreak == 1

@njit(cache=True)
def is_door_closed(obj):
    return obj._oVar4 == DoorState.DOOR_CLOSED.value

@njit(cache=True)
def is_door_open(obj):
    return obj._oVar4 == DoorState.DOOR_OPEN.value

@njit(cache=True)
def is_door(obj):
    return obj._oDoorFlag

@njit(cache=True)
def is_barrel(obj):
    return obj._otype in (dx._object_id.OBJ_BARREL.value,
                          dx._object_id.OBJ_BARRELEX.value,
                          dx._object_id.OBJ_POD.value,
                          dx._object_id.OBJ_PODEX.value,
                          dx._object_id.OBJ_URN.value,
                          dx._object_id.OBJ_URNEX.value)

@njit(cache=True)
def is_crucifix(obj):
    return obj._otype in (dx._object_id.OBJ_CRUX1.value,
                          dx._object_id.OBJ_CRUX2.value,
                          dx._object_id.OBJ_CRUX3.value)

@njit(cache=True)
def is_chest(obj):
    return obj._otype in (dx._object_id.OBJ_CHEST1.value,
                          dx._object_id.OBJ_CHEST2.value,
                          dx._object_id.OBJ_CHEST3.value,
                          dx._object_id.OBJ_TCHEST1.value,
                          dx._object_id.OBJ_TCHEST2.value,
                          dx._object_id.OBJ_TCHEST3.value,
                          dx._object_id.OBJ_SIGNCHEST.value)

@njit(cache=True)
def is_sarcophagus(obj):
    return obj._otype in (dx._object_id.OBJ_SARC.value,
                          dx._object_id.OBJ_L5SARC.value)

@njit(cache=True)
def is_floor(d, pos):
    return not (d.SOLData[d.dPiece[pos]] & \
                (dx.TileProperties.Solid.value | dx.TileProperties.BlockMissile.value))

@njit(cache=True)
def is_arch(d, pos):
    return d.dSpecial[pos] > 0

@njit(cache=True)
def is_wall(d, pos):
    return not is_floor(d, pos) and not is_arch(d, pos)


@njit(cache=True)
def to_trigger(d, pos):
    for trig in d.trigs[:d.numtrigs.value]:
        if trig.position.x == pos[0] and trig.position.y == pos[1]:
            return trig
    return None

@njit(cache=True)
def is_trigger_to_prev_level(trig):
    return trig._tmsg == dx.interface_mode.WM_DIABPREVLVL.value

@njit(cache=True)
def is_trigger_to_next_level(trig):
    return trig._tmsg == dx.interface_mode.WM_DIABNEXTLVL.value

@njit(cache=True)
def is_trigger_warp(trig):
    return trig._tmsg == dx.interface_mode.WM_DIABTWARPUP.value

@njit(cache=True)
def is_game_paused(d):
    return d.PauseMode.value != 0

@njit(cache=True)
def is_player_dead(d):
    return d.player._pmode == dx.PLR_MODE.PM_DEATH.value

@njit(cache=True)
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

@njit(cache=True)
def get_closed_doors_ids(d):
    closed_doors = []
    for oid in d.ActiveObjects:
        obj = d.Objects[oid]
        if is_door(obj) and is_door_closed(obj):
            closed_doors.append(oid)
    return closed_doors

@njit(cache=True)
def count_active_items(d):
    return d.ActiveItemCount.value

@njit(cache=True)
def count_active_monsters(d):
    return d.ActiveMonsterCount.value

def count_active_monsters_total_hp(d):
    return sum(map(lambda mid: d.Monsters[mid].hitPoints, d.ActiveMonsters))

@njit(cache=True)
def count_explored_tiles(d):
    bits = dx.DungeonFlag.Explored.value
    return np.sum((d.dFlags & bits) == bits)

@njit(cache=True)
def find_trigger(d, tmsg):
    for trig in d.trigs:
        if trig._tmsg == tmsg.value:
            return trig
    return None


spec = [ ('lt', types.uint64[:]) ]
@jitclass(spec)
class Rect:
    lt: np.ndarray
    width: int
    height: int

    def __init__(self):
        self.lt = np.zeros(2, dtype=np.uint64)
        self.width = 0
        self.height = 0


@jitclass
class EnvRect:
    # Source rectangle
    srect: Rect
    # Destination rectangle
    drect: Rect

    def __init__(self, d, radius_=None):
        # Source rectangle
        self.srect = Rect()
        # Destination rectangle
        self.drect = Rect()

        dundim = dungeon_dim(d)
        if radius_ is not None:
            # Safe cast
            radius = radius_.view(np.uint64)
            pos = np.array(player_position(d), dtype=np.uint64)

            x_min = max(pos[0] - radius[0], 0)
            x_max = min(pos[0] + radius[0] + 1, dundim[0])
            y_min = max(pos[1] - radius[1], 0)
            y_max = min(pos[1] + radius[1] + 1, dundim[1])

            self.srect.lt     = np.array([x_min, y_min], dtype=np.uint64)
            self.srect.width  = x_max - self.srect.lt[0]
            self.srect.height = y_max - self.srect.lt[1]

            # Place player position in the center of a destination rectangle
            self.drect.lt     = radius - (pos - self.srect.lt)
            self.drect.width  = radius[0] * 2 + 1
            self.drect.height = radius[1] * 2 + 1
        else:
            self.srect.lt     = np.array([0, 0])
            self.srect.width  = dundim[0]
            self.srect.height = dundim[1]
            self.drect        = self.srect


# Do not change the order of environment bits, as this is likely to break
# the pretrained model. All new flags should be only appended and carefully
# handled in the gymnasium environment class.
class EnvironmentFlag(enum.Enum):
    Player         = 1<<0
    Wall           = 1<<1
    PrevTrigger    = 1<<2
    NextTrigger    = 1<<3
    WarpTrigger    = 1<<4
    Door           = 1<<5
    Missile        = 1<<6
    Monster        = 1<<7
    UnknownObject  = 1<<8
    Crucifix       = 1<<9
    Barrel         = 1<<10
    Chest          = 1<<11
    Sarcophagus    = 1<<12
    Item           = 1<<13
    Explored       = 1<<14
    Visible        = 1<<15
    Interactable   = 1<<16
    Open           = 1<<17
    Goal           = 1<<18


@njit(cache=True)
def get_environment(d, radius=None, goal_pos=None,
                    show_invisible=False, show_unexplored=False):
    """Returns the environment, either the whole dungeon or windowed
    if a radius is specified. Setting @show_invisible and
    @show_unexplored to True is used when the entire dungeon needs to
    be revealed. However, be careful, as this can be CPU intensive, so
    @show_invisible and @show_unexplored set to False is the default.

    """
    env_rect = EnvRect(d, radius)
    # Transpose to Diablo indexing: (width, height), instead of numpy
    # (height, weight)
    env = np.zeros((env_rect.drect.width, env_rect.drect.height),
                   dtype=np.uint32)

    for j in range(env_rect.srect.height):
        for i in range(env_rect.srect.width):
            spos = (env_rect.srect.lt[0] + i, env_rect.srect.lt[1] + j)
            obj = to_object(d, spos)
            trig = to_trigger(d, spos)
            s = 0

            if d.dFlags[spos] & dx.DungeonFlag.Explored.value:
                s |= EnvironmentFlag.Explored.value
            if d.dFlags[spos] & dx.DungeonFlag.Visible.value:
                s |= EnvironmentFlag.Visible.value

            if show_unexplored or s & EnvironmentFlag.Explored.value:
                if is_wall(d, spos):
                    s |= EnvironmentFlag.Wall.value
                if trig is not None:
                    if is_trigger_to_next_level(trig):
                        s |= EnvironmentFlag.NextTrigger.value
                    elif is_trigger_to_prev_level(trig):
                        s |= EnvironmentFlag.PrevTrigger.value
                    elif is_trigger_warp(trig):
                        s |= EnvironmentFlag.WarpTrigger.value
                if obj is not None and is_door(obj):
                    s |= EnvironmentFlag.Door.value
                    if is_door_open(obj):
                        s |= EnvironmentFlag.Open.value
            if show_invisible or s & EnvironmentFlag.Visible.value:
                if goal_pos and spos == goal_pos:
                    s |= EnvironmentFlag.Goal.value
                if d.dFlags[spos] & dx.DungeonFlag.Missile.value:
                    s |= EnvironmentFlag.Missile.value
                if d.dMonster[spos] > 0:
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
                if d.dItem[spos] > 0:
                    s |= EnvironmentFlag.Item.value

            if spos == player_position(d):
                s |= EnvironmentFlag.Player.value

            # Transpose to Diablo indexing: (x, y), instead of numpy (y, x)
            dpos = (env_rect.drect.lt[0] + i, env_rect.drect.lt[1] + j)
            env[dpos] = s

    return env

def get_surroundings(d, radius, goal_pos):
    env = get_environment(d, radius, goal_pos=goal_pos)
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
            if tile & EnvironmentFlag.NextTrigger.value:
                s = 'v'
            if tile & EnvironmentFlag.PrevTrigger.value:
                s = '^'
            if tile & EnvironmentFlag.WarpTrigger.value:
                s = '$'
            if tile & EnvironmentFlag.Door.value:
                s = 'd' if tile & EnvironmentFlag.Open.value else 'D'
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
            if tile & EnvironmentFlag.Goal.value:
                s = "\u2691"
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

def pick_random_empty_tile_pos(env):
    """Randomly select a tile in a random room. Since all rooms vary
    in size, we first select a room with equal probability and then
    randomly choose a tile coordinate within that room."""
    # 1 - empty, not occupied cell
    # 0 - everything else
    empty_env = \
        (env == 0) | \
        (env == EnvironmentFlag.Explored.value) | \
        (env == EnvironmentFlag.Visible.value) | \
        (env == (EnvironmentFlag.Explored.value | \
                 EnvironmentFlag.Visible.value))

    # Label independent regions (rooms)
    labeled_regions, num_regions = maze.detect_regions(empty_env)

    # Randomly select a region
    region_nr = np.random.randint(num_regions) + 1

    xs, ys = np.where(labeled_regions == region_nr)
    # Randomly select a position in the labeled region
    i = np.random.randint(len(xs))
    return (xs[i], ys[i])

def get_dungeon_graph_and_path(env, start, goal):
    # 0 - walls
    # 1 - empty areas, probably occupied by player, monsters, etc
    # Interesting fact: a closed door has a `wall` flag, while an open
    # door does not, so treat `door` as a wall.
    empty_env = \
        (env & (EnvironmentFlag.Wall.value | \
                EnvironmentFlag.Door.value)) == 0

    # Doors positions
    doors = np.argwhere(env & EnvironmentFlag.Door.value)

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

def map_agent_state(path):
    size = AgentState.itemsize
    f = open(path, "a+b")
    f.truncate(size)
    mmapped = mmap.mmap(f.fileno(), 0)
    f.close()

    # Create a 1-element view and return the scalar object. The
    # `view(np.recarray)` is needed to allow access using dot
    # notation. `AgentState` dtype is a structured type, so the
    # `state_array[0]` is still backed by a memory buffer, and not a
    # copy.
    state_array = np.frombuffer(mmapped, dtype=AgentState, count=1).view(np.recarray)
    state = state_array[0]

    return state

def map_devilutionx_state(path, offset):
    f = open(path, "r+b")
    mmapped = mmap.mmap(f.fileno(), 0)
    f.close()

    vars_dict = {}
    for var in dx.VARS:
        addr = var['addr']
        dtype = var['type']
        assert offset <= addr, "Address offset is larger than variable address"
        var_offset = addr - offset

        # Create a 1-element NumPy array view at the specified offset
        # This view points directly into the mmap buffer. The
        # `view(np.recarray)` is needed to allow access using dot
        # notation.
        #
        # Be careful! We use `obj_array[0]`, which does not produce a
        # copy for structured types, but for primitives, this will be a
        # copy and not backed by a memory buffer. However, the
        # `dbg2numpy` should handle this.
        obj_array = np.frombuffer(mmapped, dtype=dtype, count=1, offset=var_offset).view(np.recarray)
        obj = obj_array[0]
        vars_dict[var['short_name']] = obj

    return DevilutionState(vars_dict)


def map_devilutionx_state_by_pid(pid, mshared_path):
    for attempt in range(0, 50):
        try:
            # Get offset of mapped file
            _, offset = procutils.get_mapped_file_and_offset_of_pid(
                pid, mshared_path)
            if not offset:
                # Wait until remapped
                time.sleep(0.1)
                continue
            # Open the file and map it to memory
            return map_devilutionx_state(mshared_path, offset)
        except FileNotFoundError:
            time.sleep(0.1)
    else:
        raise FileNotFoundError(mshared_path)

class DiabloGame:
    def __init__(self, state_path, state, state_dir=None, proc=None,
                 log_file=None, game_ticks_per_step=None):
        self.state_dir = state_dir
        self.state_path = state_path
        self.proc = proc
        self.log_file = log_file
        self.state = state
        self.last_tick = 0
        self.agent_state = map_agent_state(state_path + "/agent-shared.mem")

        if game_ticks_per_step is None:
            # Read shared option in case of the attach
            self.game_ticks_per_step = self.state.GameTicksPerStep.value
        else:
            self.game_ticks_per_step = game_ticks_per_step

        # Catch up with the events queue. The @read_idx from the queue
        # is not used to support other processes that might want to
        # attach.
        self.events_queue_read_idx1 = self.events_queue_read_idx2 = \
            state.events_queue.write_idx

    def __del__(self):
        self.stop_or_detach()

    @property
    def goal_pos(self):
        return (self.agent_state.goal_pos.x, self.agent_state.goal_pos.y)

    @goal_pos.setter
    def goal_pos(self, goal_pos):
        """Goal position is set from the training environment"""
        self.agent_state.goal_pos.x = goal_pos[0]
        self.agent_state.goal_pos.y = goal_pos[1]

    @property
    def safe_state(self):
        if not self.game_ticks_per_step:
            # Perform a deepcopy only if this is a run-time mode, in which
            # we need to persist the state; otherwise, the state may
            # change in the middle of some calculation, which can lead
            # to an assertion failure. Unfortunately, the deepcopy is
            # very inefficient (performance-wise).
            return copy.deepcopy(self.state)
        return self.state

    def stop_or_detach(self):
        if self.proc:
            self.proc.terminate()
        if self.log_file:
            self.log_file.close()
        if self.state_dir:
            self.state_dir.cleanup()

    def ticks(self, d=None):
        t = self.state.game_ticks.value if d is None else d.game_ticks.value
        return t

    def update_ticks(self):
        missed = self.ticks() - self.last_tick
        self.last_tick += missed
        return missed

    def same_ticks(self):
        diff = self.ticks() - self.last_tick
        return diff == 0

    def retrieve_event(self):
        read_idx = self.events_queue_read_idx1
        entry = ring.get_entry_to_retrieve(self.state.events_queue, read_idx)
        if entry == None:
            return None
        self.events_queue_read_idx1 += 1
        return entry

    def submit_key(self, key):
        request_tag = (time.time_ns() & 0xffffffff)
        assert ring.has_capacity_to_submit(self.state.input_queue)
        entry = ring.get_entry_to_submit(self.state.input_queue)
        entry.en_type = key
        entry.en_data = request_tag

        # Submit key
        ring.submit(self.state.input_queue)

        feedback_events = [
            # Released keys event
            key & ~ring.RingEntryType.RING_ENTRY_FLAGS,
            # `STEP_FINISHED` - only if `game_ticks_per_step` is not 0
            ring.RingEntryType.RING_ENTRY_EVENT_STEP_FINISHED,
        ]
        # Get read index
        read_idx = self.events_queue_read_idx2
        # Wait for feedback events that come one after another
        event_idx = 0
        while event_idx < len(feedback_events):
            ring.wait_any_submitted(self.state.events_queue, read_idx)
            entry = ring.get_entry_to_retrieve(self.state.events_queue, read_idx)
            read_idx += 1
            assert entry != None
            if entry.en_data == request_tag:
                event_type = feedback_events[event_idx]
                event_idx += 1
                assert entry.en_type == event_type

                if not self.game_ticks_per_step:
                    # For run-time mode we receive only released keys
                    break

        # Update read index
        self.events_queue_read_idx2 = read_idx

    @staticmethod
    def run(config):
        cfg_file = open("diablo.ini.template", "r")
        cfg = cfg_file.read()
        cfg_file.close()

        mshared_filename = config["mshared-filename"]
        game_ticks_per_step = config["game-ticks-per-step"]

        cfg = cfg.format(seed=config["seed"],
                         fixed_seed=1 if config["fixed-seed"] else 0,
                         automap_active=1 if config["gui"] else 0,
                         skip_progress=1 if config["gui"] else 0,
                         skip_animation=0 if config["gui"] else 1,
                         headless=0 if config["gui"] else 1,
                         game_ticks_per_step=game_ticks_per_step,
                         mshared_filename=mshared_filename,
                         no_monsters=1 if config["no-monsters"] else 0)

        prefix = "diablo-%d-%d-" % (config["seed"], os.getpid())
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
        env = os.environ.copy()
        # It was observed that after `import gymnasium`, any attempt
        # to initialize SDL with audio support failed with the `dsp:
        # No such audio device` error.  It turned out that gymnasium
        # pulls in pygame, which modifies the current environment and
        # sets `SDL_AUDIODRIVER=dsp`. This change removes the variable
        # and helps to init SDL properly in a child app (devilutionx).
        if "SDL_AUDIODRIVER" in env:
            del env["SDL_AUDIODRIVER"]
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
        state_path = state_dir.name
        mshared_path = os.path.abspath(state_dir.name + "/" + mshared_filename)
        state = map_devilutionx_state_by_pid(proc.pid, mshared_path)
        return DiabloGame(state_path, state, state_dir=state_dir, proc=proc,
                          log_file=log_file, game_ticks_per_step=game_ticks_per_step)

    @staticmethod
    def attach(config):
        mshared_path = config['attach-path']
        offset = config['attach-offset']
        game_ticks_per_step = config["game-ticks-per-step"]
        state = map_devilutionx_state(mshared_path, offset)
        state_path = os.path.dirname(mshared_path)
        return DiabloGame(state_path, state)

    @staticmethod
    def run_or_attach(config):
        if 'attach-path' in config:
            return DiabloGame.attach(config)
        return DiabloGame.run(config)
