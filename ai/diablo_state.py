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
import enum
import mmap
import numpy as np
import os
import subprocess
import tempfile
import time

from numba import types, njit
from numba.experimental import jitclass, structref

DISABLE_NJIT = False

if DISABLE_NJIT:
    def _nop_decorator(*dargs, **dkwargs):
        # If used as "@decorator" (no parentheses)
        if len(dargs) == 1 and callable(dargs[0]):
            return dargs[0]
        # If used as "@decorator(...)" (with arguments)
        def wrapper(func):
            return func
        return wrapper

    njit = _nop_decorator
    jitclass = _nop_decorator

import dbg2numpy
import devilutionx as dx
import maze
import procutils
import ring


def inv_slot(inv_list_index):
    """Return item slot for InvList[inv_list_index]: INVITEM_INV_FIRST + inv_list_index."""
    return dx.inv_item.INVITEM_INV_FIRST.value + inv_list_index

def belt_slot(belt_index):
    """Return item slot for SpdList[belt_index]: INVITEM_BELT_FIRST + belt_index (0-7)."""
    return dx.inv_item.INVITEM_BELT_FIRST.value + belt_index

def find_inv_item(player, misc_id, spell_id=None):
    """Return item slot of first matching item in inventory then belt, or -1."""
    misc_val  = misc_id.value  if hasattr(misc_id,  'value') else misc_id
    spell_val = spell_id.value if hasattr(spell_id, 'value') else spell_id
    itype_none = dx.ItemType.None_.value
    for i in range(int(player._pNumInv)):
        item = player.InvList[i]
        if item._iMiscId == misc_val and (spell_val is None or item._iSpell == spell_val):
            return inv_slot(i)
    for i in range(8):
        item = player.SpdList[i]
        # RemoveSpdBarItem only clears _itype; skip ghost slots.
        if int(item._itype) == itype_none:
            continue
        if item._iMiscId == misc_val and (spell_val is None or item._iSpell == spell_val):
            return belt_slot(i)

    return -1

# Cap for potion-count observation scalars. Matches the starting-potion
# injection range so the model sees the full [0, 1] normalised range from
# episode 1.
POTION_CAP = 5.0


@njit(cache=True)
def player_pot_counts(p):
    """Walk belt (SpdList[0..7]) and inventory (InvList[0..pNumInv-1]) once
    each, tallying the 7 potion categories in observation order:
    small_hp, scroll_heal, full_hp, small_mana, full_mana, rejuv, full_rejuv.
    Returns a float32 array of length 7, each entry normalised against
    POTION_CAP and clipped to 1.0."""
    imisc_heal      = dx.item_misc_id.IMISC_HEAL.value
    imisc_scroll    = dx.item_misc_id.IMISC_SCROLL.value
    imisc_fullheal  = dx.item_misc_id.IMISC_FULLHEAL.value
    imisc_mana      = dx.item_misc_id.IMISC_MANA.value
    imisc_fullmana  = dx.item_misc_id.IMISC_FULLMANA.value
    imisc_rejuv     = dx.item_misc_id.IMISC_REJUV.value
    imisc_fullrejuv = dx.item_misc_id.IMISC_FULLREJUV.value
    spellid_healing = dx.SpellID.Healing.value
    itype_none      = dx.ItemType.None_.value

    counts = np.zeros(7, dtype=np.int32)

    for k in range(8):
        it = p.SpdList[k]
        # RemoveSpdBarItem only clears _itype; other fields are left stale.
        # Skip cleared slots so ghost entries don't inflate the count.
        if it._itype == itype_none:
            continue
        misc = it._iMiscId
        if   misc == imisc_heal:       counts[0] += 1
        elif misc == imisc_scroll and it._iSpell == spellid_healing: counts[1] += 1
        elif misc == imisc_fullheal:   counts[2] += 1
        elif misc == imisc_mana:       counts[3] += 1
        elif misc == imisc_fullmana:   counts[4] += 1
        elif misc == imisc_rejuv:      counts[5] += 1
        elif misc == imisc_fullrejuv:  counts[6] += 1

    for k in range(p._pNumInv):
        it = p.InvList[k]
        misc = it._iMiscId
        if   misc == imisc_heal:       counts[0] += 1
        elif misc == imisc_scroll and it._iSpell == spellid_healing: counts[1] += 1
        elif misc == imisc_fullheal:   counts[2] += 1
        elif misc == imisc_mana:       counts[3] += 1
        elif misc == imisc_fullmana:   counts[4] += 1
        elif misc == imisc_rejuv:      counts[5] += 1
        elif misc == imisc_fullrejuv:  counts[6] += 1

    cap = POTION_CAP
    out = np.empty(7, dtype=np.float32)
    for i in range(7):
        c = counts[i]
        out[i] = (c if c < cap else cap) / cap
    return out

def get_pot_counts(d):
    """Public wrapper -- returns the 7 normalised potion counts as a plain
    Python list (for diablo-ai.py / TUI code that expects list semantics).
    Delegates to player_pot_counts(d.player) under @njit for the actual work."""
    return list(player_pot_counts(d.player))


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
	("goal_pos",       dx.PointOf_int_),
	("episode_steps",  np.int32),
	("stuck_steps",    np.int32),
], align=True)

class DoorState(enum.Enum):
    DOOR_CLOSED   = 0
    DOOR_OPEN     = 1
    DOOR_BLOCKED  = 2

def fmix32(h):
    """MurmurHash3 finalizer: bijective 32-bit mixer with full avalanche."""
    h &= 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x85ebca6b) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 0xc2b2ae35) & 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFFFFFF

def make_episode_seed(base_seed, index, counter):
    """Deterministic per-episode seed: fmix32(fmix32(base+index) + counter)."""
    initial = fmix32(base_seed + index)
    return fmix32(initial + counter)

def parse_dungeon_level_spec(spec):
    """Parse a dungeon level spec string like '1=5,2-4=45,5=35' into
    a list of (level, weight) pairs as used by sample_dungeon_level."""
    result = []
    for part in spec.split(','):
        part = part.strip()
        if '=' in part:
            levels_str, weight_str = part.split('=', 1)
            weight = int(weight_str)
        else:
            levels_str, weight = part, 1
        if '-' in levels_str:
            lo, hi = levels_str.split('-', 1)
            for lvl in range(int(lo), int(hi) + 1):
                result.append((lvl, weight))
        else:
            result.append((int(levels_str), weight))
    return result

def read_suggested_dungeon_level(env_stats_path):
    """Parse the suggested --dungeon-level spec from an env-stats file.
    Returns the spec string or None if not found."""
    try:
        with open(env_stats_path) as f:
            for line in f:
                if line.startswith('# --dungeon-level '):
                    return line[len('# --dungeon-level '):].strip()
    except OSError:
        pass
    return None

CURRICULUM_WEIGHT_STEP = 5  # granularity for suggested dungeon-level weights


class DungeonLevelSpec:
    """Dungeon level sampling spec; __str__ returns canonical level-sorted spec string.

    Levels with weight == auto use stats-derived weights at runtime; their
    nominal weight is CURRICULUM_WEIGHT_STEP and __str__ renders them as =auto.
    """
    def __init__(self, levels, auto_levels=frozenset(), stats_path=None):
        self._levels = levels
        self.auto_levels = frozenset(auto_levels)
        self.stats_path = stats_path  # set by training loop when model_dir is known
        self._spec = self._make_spec(levels, self.auto_levels)

    @staticmethod
    def _make_spec(levels, auto_levels):
        parts = []
        i = 0
        while i < len(levels):
            level, weight = levels[i]
            is_auto = level in auto_levels
            j = i + 1
            while (j < len(levels)
                   and levels[j][0] == levels[j-1][0] + 1
                   and levels[j][1] == weight
                   and (levels[j][0] in auto_levels) == is_auto):
                j += 1
            part = str(level) if j - i == 1 else "%d-%d" % (level, levels[j-1][0])
            if is_auto:
                part += "=auto"
            elif weight != 1:
                part += "=%d" % weight
            parts.append(part)
            i = j
        return ",".join(parts)

    def __str__(self):   return self._spec
    def __repr__(self):  return repr(self._spec)
    def __iter__(self):  return iter(self._levels)
    def __getitem__(self, idx): return self._levels[idx]
    def __len__(self):   return len(self._levels)


DUNGEON_LEVEL_DEFAULT = DungeonLevelSpec([(1, 1)])


def _reversed_lines(path, chunk=65536):
    """Yield lines of a file in reverse order without loading it fully."""
    with open(path, 'rb') as f:
        f.seek(0, 2)
        pos = f.tell()
        buf = b''
        while pos > 0:
            read_size = min(chunk, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + buf
            lines = data.split(b'\n')
            buf = lines[0]
            for line in reversed(lines[1:]):
                yield line.decode('utf-8', errors='replace')
        if buf:
            yield buf.decode('utf-8', errors='replace')

def env_stats(args):
    import glob, re, math
    from collections import defaultdict

    TERMINAL = {'Goal', 'Diablo killed', 'Death', 'Escape', 'Stuck', 'Timedout'}

    RE_LEVEL = re.compile(r'dungeon_level=(\d+)')
    RE_DONE  = re.compile(r'EPISODE DONE=(true|false) steps=(\d+)')
    RE_EVENT = re.compile(r'^(.+),\s*R\s*(\[[-\d.,\s]+\]|[-\d.]+)\s*$')

    counts    = {}
    sums      = {}
    lvl_out   = defaultdict(lambda: defaultdict(int))
    lvl_steps = defaultdict(lambda: {'succ': [], 'fail': []})

    pattern = env_log_glob(eval=args.eval_runners)
    files = sorted(glob.glob(pattern))
    if not files:
        print("No files matched: %s" % pattern)
        return 1

    level_keep = None
    if args.level_filter:
        level_keep = set()
        for part in args.level_filter.split(','):
            part = part.strip()
            if '-' in part:
                lo, hi = part.split('-', 1)
                level_keep.update(range(int(lo), int(hi) + 1))
            else:
                level_keep.add(int(part))

    episodes_scanned = 0
    succ_scanned = 0

    if args.last_episodes == 0:
        cur_level = None
        for path in files:
            try:
                with open(path) as f:
                    for line in f:
                        line = line.rstrip()
                        m = RE_LEVEL.search(line)
                        if m:
                            cur_level = int(m.group(1))
                            continue
                        m = RE_DONE.search(line)
                        if m:
                            if cur_level is not None and (level_keep is None or cur_level in level_keep):
                                episodes_scanned += 1
                                key = 'succ' if m.group(1) == 'true' else 'fail'
                                if key == 'succ':
                                    succ_scanned += 1
                                lvl_steps[cur_level][key].append(int(m.group(2)))
                            continue
                        m = RE_EVENT.match(line)
                        if not m:
                            continue
                        if level_keep is not None and cur_level not in level_keep:
                            continue
                        label = m.group(1).strip()
                        vs    = m.group(2)
                        val   = sum(float(x) for x in vs.strip('[]').split(',')) \
                                if vs.startswith('[') else float(vs)
                        counts[label] = counts.get(label, 0) + 1
                        sums[label]   = sums.get(label, 0.0) + val
                        if label in TERMINAL and cur_level is not None:
                            lvl_out[cur_level][label] += 1
            except OSError:
                pass
    else:
        quota = (args.last_episodes + len(files) - 1) // len(files)
        for path in files:
            ep_events = []
            ep_done   = None
            in_ep     = False
            ep_count  = 0
            try:
                for line in _reversed_lines(path):
                    line = line.rstrip()
                    m = RE_LEVEL.search(line)
                    if m:
                        if in_ep:
                            level = int(m.group(1))
                            if level_keep is None or level in level_keep:
                                episodes_scanned += 1
                                for label, val in ep_events:
                                    counts[label] = counts.get(label, 0) + 1
                                    sums[label]   = sums.get(label, 0.0) + val
                                    if label in TERMINAL:
                                        lvl_out[level][label] += 1
                                if ep_done is not None:
                                    key = 'succ' if ep_done[0] else 'fail'
                                    if key == 'succ':
                                        succ_scanned += 1
                                    lvl_steps[level][key].append(ep_done[1])
                            ep_count  += 1
                            ep_events  = []
                            ep_done    = None
                            in_ep      = False
                            if ep_count >= quota:
                                break
                        continue
                    m = RE_DONE.search(line)
                    if m:
                        ep_done = (m.group(1) == 'true', int(m.group(2)))
                        in_ep   = True
                        continue
                    if not in_ep:
                        continue
                    m = RE_EVENT.match(line)
                    if not m:
                        continue
                    label = m.group(1).strip()
                    vs    = m.group(2)
                    val   = sum(float(x) for x in vs.strip('[]').split(',')) \
                            if vs.startswith('[') else float(vs)
                    ep_events.append((label, val))
            except OSError:
                pass

    if not counts:
        print("No events found.")
        return 0

    total_ev = sum(counts.values())
    if args.sort == 'count':
        order = sorted(counts, key=lambda k: counts[k], reverse=True)
    elif args.sort == 'sum':
        order = sorted(counts, key=lambda k: abs(sums[k]), reverse=True)
    else:
        order = sorted(counts)

    freq_s  = {k: "%.1f%%(%d)" % (100.0 * counts[k] / total_ev, counts[k]) for k in order}
    if episodes_scanned:
        per_ep_s = {k: "%.2f" % (counts[k] / episodes_scanned) for k in order}
    else:
        per_ep_s = {k: "" for k in order}
    w_freq   = max(max(len(s) for s in freq_s.values()), len("freq"))
    w_per_ep = max(max(len(s) for s in per_ep_s.values()), len("per_ep")) if episodes_scanned else 0
    w_sum    = max(max(len("%.1f" % sums[k]) for k in order), len("sum_R"))

    MIN_COUNT = 4
    shown = [k for k in order if counts[k] >= MIN_COUNT]
    hidden = [k for k in order if counts[k] < MIN_COUNT]

    kind = "eval" if args.eval_runners else "train"
    window = "all" if args.last_episodes == 0 else "last"
    ep_desc = "%d %s %s episodes" % (episodes_scanned, window, kind) if episodes_scanned else ""
    run_id = getattr(args, 'run_id', None)
    run_suffix = ", run %s" % run_id if run_id else ""
    ep_suffix = ", %s" % ep_desc if ep_desc else ""
    print("Event frequency (%d total events, %d files%s%s):" % (total_ev, len(files), ep_suffix, run_suffix))
    if w_per_ep:
        print("%-*s  %-*s  %*s  label" % (w_freq, "freq", w_per_ep, "per_ep", w_sum, "sum_R"))
        print("%s  %s  %s  %s" % ("-" * w_freq, "-" * w_per_ep, "-" * w_sum, "-" * 30))
    else:
        print("%-*s  %*s  label" % (w_freq, "freq", w_sum, "sum_R"))
        print("%s  %s  %s" % ("-" * w_freq, "-" * w_sum, "-" * 30))
    for k in shown:
        if w_per_ep:
            print("%-*s  %-*s  %*.1f  %s" % (w_freq, freq_s[k], w_per_ep, per_ep_s[k], w_sum, sums[k], k))
        else:
            print("%-*s  %*.1f  %s" % (w_freq, freq_s[k], w_sum, sums[k], k))
    if hidden:
        hid_cnt = sum(counts[k] for k in hidden)
        hid_sum = sum(sums[k] for k in hidden)
        hid_freq = "%.1f%%(%d)" % (100.0 * hid_cnt / total_ev, hid_cnt)
        if w_per_ep:
            hid_per_ep = "%.2f" % (hid_cnt / episodes_scanned)
            print("%-*s  %-*s  %*.1f  ... %d labels skipped (count < %d)" % (
                w_freq, hid_freq, w_per_ep, hid_per_ep, w_sum, hid_sum, len(hidden), MIN_COUNT))
        else:
            print("%-*s  %*.1f  ... %d labels skipped (count < %d)" % (
                w_freq, hid_freq, w_sum, hid_sum, len(hidden), MIN_COUNT))

    if not lvl_out:
        return 0

    OUTCOME_COLS = ['Goal', 'Diablo killed', 'Death', 'Escape', 'Stuck', 'Timedout']
    COL_ABBREV   = {
        'Goal': 'Goal', 'Diablo killed': 'Diablo', 'Death': 'Death',
        'Escape': 'Escape', 'Stuck': 'Stuck', 'Timedout': 'Timeout',
    }
    cols = [c for c in OUTCOME_COLS if any(lvl_out[lvl].get(c) for lvl in lvl_out)]

    def outcome_fmt(cnt, tot):
        return "%.1f%%(%d)" % (100.0 * cnt / tot, cnt) if tot else "-"

    def steps_fmt(lst):
        if not lst:
            return "-"
        n    = len(lst)
        mean = sum(lst) / n
        std  = math.sqrt(sum((x - mean) ** 2 for x in lst) / n)
        return "μ=%d σ=%d [%d..%d]" % (mean, std, min(lst), max(lst))

    all_levels = sorted(lvl_out)
    col_w = {c: max(len(COL_ABBREV[c]),
                    max(len(outcome_fmt(lvl_out[l].get(c, 0), sum(lvl_out[l].values())))
                        for l in all_levels))
             for c in cols}
    step_w = max(
        len("steps(succ)"),
        max((len(steps_fmt(lvl_steps[l]['succ'])) for l in all_levels), default=1),
        max((len(steps_fmt(lvl_steps[l]['fail'])) for l in all_levels), default=1),
    )

    hdr = "  ".join(
        ["%6s" % "Level"] +
        ["%*s" % (col_w[c], COL_ABBREV[c]) for c in cols] +
        ["%*s" % (step_w, "steps(succ)"), "%*s" % (step_w, "steps(fail)")]
    )
    print()
    total_succ_pct = 100.0 * succ_scanned / episodes_scanned if episodes_scanned else 0.0
    level_succ_rates = [
        100.0 * len(lvl_steps[level]['succ']) / sum(lvl_out[level].values())
        for level in all_levels
        if sum(lvl_out[level].values())
    ]
    uniform_succ_pct = sum(level_succ_rates) / len(level_succ_rates) if level_succ_rates else 0.0
    succ_line = "success: %.1f%% total / %.1f%% uniform" % (total_succ_pct, uniform_succ_pct)
    print("Per-level outcomes (%s, %s%s):" % (ep_desc, succ_line, run_suffix))
    print(hdr)
    print("-" * len(hdr))
    for level in all_levels:
        tot = sum(lvl_out[level].values())
        row = "  ".join(
            ["%6d" % level] +
            ["%*s" % (col_w[c], outcome_fmt(lvl_out[level].get(c, 0), tot)) for c in cols] +
            ["%*s" % (step_w, steps_fmt(lvl_steps[level]['succ'])),
             "%*s" % (step_w, steps_fmt(lvl_steps[level]['fail']))]
        )
        print(row)

    WEIGHT_STEP = CURRICULUM_WEIGHT_STEP
    sugg = {}
    for level in all_levels:
        tot = sum(lvl_out[level].values())
        if not tot:
            continue
        succ = lvl_out[level].get('Goal', 0) + lvl_out[level].get('Diablo killed', 0)
        sugg[level] = max(1, round(100.0 * (tot - succ) / tot / WEIGHT_STEP) * WEIGHT_STEP)

    if sugg:
        parts = []
        lvls = sorted(sugg)
        i = 0
        while i < len(lvls):
            w = sugg[lvls[i]]
            j = i + 1
            while j < len(lvls) and lvls[j] == lvls[j-1] + 1 and sugg[lvls[j]] == w:
                j += 1
            if j - i == 1:
                parts.append("%d=%d" % (lvls[i], w))
            else:
                parts.append("%d-%d=%d" % (lvls[i], lvls[j-1], w))
            i = j
        print()
        print("# Suggested (weight = 100 - success%, use with --dungeon-level):")
        print("# --dungeon-level %s" % ",".join(parts))

    return 0


def write_env_stats(path, eval_runners, last_episodes, run_id=None):
    import io, contextlib, argparse, os
    buf = io.StringIO()
    fake_args = argparse.Namespace(
        eval_runners=eval_runners,
        last_episodes=last_episodes,
        sort="count",
        level_filter=None,
        run_id=run_id,
    )
    with contextlib.redirect_stdout(buf):
        env_stats(fake_args)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def sample_dungeon_level(dungeon_level_config, seed):
    total = sum(w for _, w in dungeon_level_config)
    r = seed % total
    cumul = 0
    for level, weight in dungeon_level_config:
        cumul += weight
        if r < cumul:
            return level
    return dungeon_level_config[-1][0]

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
    return obj._oVar4 == DoorState.DOOR_OPEN.value or \
           obj._oVar4 == DoorState.DOOR_BLOCKED.value

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
    # Be aware! We use future position solely for the purpose of agent
    # evaluation in GUI mode, when the game loop is not frozen and
    # does not wait for input, but runs freely (real-time mode). Future
    # positions are updated earlier by the DevilutionX engine (thus
    # future), so the agent can get its current state earlier,
    # minimizing the annoying hero jitter while moving (though it
    # still exists, sigh). For the training, future positions are
    # irrelevant because the game loop is frozen (in step mode) and
    # waits for further input.
    #
    # Why does hero movement jitter? Once an action is submitted, the
    # agent waits for frames to be skipped to get the actual
    # observation state after the action. Consequently, a hero
    # finishes its movement and idles before a new action begins. This
    # transition: idle -> move -> idle -> ... is visible and
    # annoying. To minimize it, we skip fewer frames compared to the
    # training mode (where animation is also skipped, contributing to
    # overall frame skipping), but use future positions, so the game
    # animation continues while the agent still gets its relevant
    # state update. This is not perfect, but it works for now.
    #
    # Cast to signed integer to prevent signed arithmetic on
    # `np.uint8` types; otherwise, a warning occurs.
    #
    return (int(d.player.position.future.x), int(d.player.position.future.y))

def player_direction(d):
    # Compensate dungeon 45CW rotation. Cast to signed integer to
    # prevent signed arithmetic on `np.uint8` types; otherwise, a
    # warning occurs.
    return (int(d.player._pdir) - 1) % (len(dx.Direction) - 1)

@njit(cache=True)
def count_active_objects(d):
    count = 0
    for oid in d.ActiveObjects:
        obj = d.Objects[oid]
        if is_barrel(obj):
            if obj._oSolidFlag:
                count += 1
        elif is_chest(obj) or is_sarcophagus(obj) or is_crucifix(obj):
            if is_interactable(obj):
                count += 1
    return count

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

@njit(cache=True)
def player_spell_bits(d):
    """Bit SpellID(i) set iff SpellID(i) is castable.
    The engine's GetSpellBitmask stores spell id N at bit N-1 (1-indexed).
    Shift left by 1 so callers can check with (1 << SpellID.value)."""
    p = d.player
    raw = p._pMemSpells | p._pAblSpells | p._pISpells | p._pScrlSpells
    return raw << np.uint64(1)

@njit(cache=True)
def count_active_monsters_total_hp(d):
    total = 0
    for i in range(d.ActiveMonsterCount.value):
        total += d.Monsters[d.ActiveMonsters[i]].hitPoints
    return total

def alloc_monster_hp(d):
    """Allocate a MaxMonsters-sized int32 buffer for HP snapshots."""
    return np.full(len(d.Monsters), -1, dtype=np.int32)

@njit(cache=True)
def snapshot_monster_hp(d, out):
    """Fill out[mid] = current hitPoints for each active monster; -1 elsewhere."""
    for i in range(len(out)):
        out[i] = -1
    for i in range(d.ActiveMonsterCount.value):
        mid = d.ActiveMonsters[i]
        out[mid] = d.Monsters[mid].hitPoints

@njit(cache=True)
def count_monsters_hit(d, prev_hp):
    """Count monsters whose hitPoints dropped since prev_hp snapshot.
    Iterates all MaxMonsters slots directly so monsters that died this step
    (swapped past ActiveMonsterCount) are still counted.
    prev_hp[i] >= 0 limits to monsters that were active at the last snapshot."""
    count = 0
    for i in range(len(d.Monsters)):
        if prev_hp[i] >= 0 and d.Monsters[i].hitPoints < prev_hp[i]:
            count += 1
    return count

@njit(cache=True)
def count_visible_monsters(env):
    return np.sum((env & EnvironmentFlag.Monster.value) != 0)

@njit(cache=True)
def player_has_adjacent(env, view_radius, mask, exclude_mask):
    """Return True if any of the 8 tiles adjacent to the player matches:
      (t & mask) != 0 and (t & exclude_mask) == 0.
    Player is always at (view_radius, view_radius) in the windowed env.

    Typical calls:
      player_has_adjacent(env, r, Monster,                    0)    - melee target
      player_has_adjacent(env, r, Item|Interactable|Door, Open)    - pickup/open target
    """
    rows, cols = env.shape
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            x = view_radius + dx
            y = view_radius + dy
            if 0 <= x < rows and 0 <= y < cols:
                t = env[x, y]
                if t & mask and not (t & exclude_mask):
                    return True
    return False

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

# Source: monster.cpp AiProc[] - types whose handler calls StartRangedAttack:
#   AiRanged/AiRangedAvoidance directly, SkeletonBowAi, CounselorAi (also used by
#   Lazarus/LazarusSuccubus), and FireMan (Hellfire, nullptr AI but ranged by design).
# Lazy-init so dx.MonsterAIID attribute resolution is deferred until first use.
# Python `x in np.array` works element-wise, so the single array view serves
# both Python membership checks and @njit callers (frozensets don't survive
# @njit, arrays do).
_RANGED_AI_IDS_ARR = None
def ranged_ai_ids_array():
    """Numpy int64 array of MonsterAIID values that use ranged behavior."""
    global _RANGED_AI_IDS_ARR
    if _RANGED_AI_IDS_ARR is None:
        _RANGED_AI_IDS_ARR = np.array(sorted([
            dx.MonsterAIID.SkeletonRanged.value, dx.MonsterAIID.GoatRanged.value,
            dx.MonsterAIID.Magma.value,          dx.MonsterAIID.Succubus.value,
            dx.MonsterAIID.Storm.value,          dx.MonsterAIID.FireMan.value,
            dx.MonsterAIID.Acid.value,           dx.MonsterAIID.AcidUnique.value,
            dx.MonsterAIID.Counselor.value,      dx.MonsterAIID.Lazarus.value,
            dx.MonsterAIID.LazarusSuccubus.value,dx.MonsterAIID.FireBat.value,
            dx.MonsterAIID.Torchant.value,       dx.MonsterAIID.Lich.value,
            dx.MonsterAIID.ArchLich.value,       dx.MonsterAIID.Psychorb.value,
            dx.MonsterAIID.Necromorb.value,      dx.MonsterAIID.BoneDemon.value,
            dx.MonsterAIID.Diablo.value,
        ]), dtype=np.int64)
    return _RANGED_AI_IDS_ARR

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

    def __init__(self, d, radius=None):
        # Source rectangle
        self.srect = Rect()
        # Destination rectangle
        self.drect = Rect()

        dundim = dungeon_dim(d)
        if radius is not None:
            radius = np.array([radius, radius], dtype=np.uint64)
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
                # dMonster keeps the monster id during the death animation;
                # the engine only zeroes it in MonsterDeath() at isLastFrame().
                # Filter on hitPoints > 0 so a dead-but-animating monster does
                # not leave a stale '@' on the grid, matching what
                # nearest_monster_info reports.
                mid = d.dMonster[spos]
                if mid > 0 and d.Monsters[mid - 1].hitPoints > 0:
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


@njit(cache=True)
def compute_monster_attrs(d, view_radius, max_level, max_walk, max_attack,
                          ranged_ids):
    """njit compute of the (W, H, 9) monster attribute grid used by the v2
    observation. Mirrors the Python implementation in DiabloEnvV2Mixin
    but stays in jit land for the inner loop."""
    env_rect = EnvRect(d, view_radius)
    attrs = np.zeros((env_rect.drect.width, env_rect.drect.height, 9),
                     dtype=np.float32)

    visible_flag = dx.DungeonFlag.Visible.value
    unique_none  = dx.UniqueMonsterType.None_.value
    imm_fire     = dx.monster_resistance.IMMUNE_FIRE.value
    res_fire     = dx.monster_resistance.RESIST_FIRE.value
    imm_light    = dx.monster_resistance.IMMUNE_LIGHTNING.value
    res_light    = dx.monster_resistance.RESIST_LIGHTNING.value
    imm_magic    = dx.monster_resistance.IMMUNE_MAGIC.value
    res_magic    = dx.monster_resistance.RESIST_MAGIC.value

    for j in range(env_rect.srect.height):
        for i in range(env_rect.srect.width):
            sx = env_rect.srect.lt[0] + i
            sy = env_rect.srect.lt[1] + j
            if not (d.dFlags[sx, sy] & visible_flag):
                continue
            mid = d.dMonster[sx, sy]
            if mid <= 0:
                continue
            m = d.Monsters[mid - 1]
            if m.hitPoints <= 0:
                continue
            ti = d.monster_type_info[m.levelType]
            r  = m.resistance
            ox = env_rect.drect.lt[0] + i
            oy = env_rect.drect.lt[1] + j

            fire_val  = 1.0 if (r & imm_fire)  else (0.5 if (r & res_fire)  else 0.0)
            light_val = 1.0 if (r & imm_light) else (0.5 if (r & res_light) else 0.0)
            magic_val = 1.0 if (r & imm_magic) else (0.5 if (r & res_magic) else 0.0)

            ai = m.ai
            is_ranged = False
            for k in range(len(ranged_ids)):
                if ai == ranged_ids[k]:
                    is_ranged = True
                    break

            denom = m.maxHitPoints
            if m.hitPoints > denom:
                denom = m.hitPoints
            if denom < 1:
                denom = 1

            attrs[ox, oy, 0] = m.hitPoints / denom
            attrs[ox, oy, 1] = ti.level / max_level
            attrs[ox, oy, 2] = 0.0 if m.uniqueType == unique_none else 1.0
            attrs[ox, oy, 3] = 1.0 - ti.walk_frames / max_walk
            attrs[ox, oy, 4] = 1.0 - ti.attack_frames / max_attack
            attrs[ox, oy, 5] = fire_val
            attrs[ox, oy, 6] = light_val
            attrs[ox, oy, 7] = magic_val
            attrs[ox, oy, 8] = 1.0 if is_ranged else 0.0
    return attrs


@njit(cache=True)
def compute_scalars(d):
    """njit compute of the 46-float scalar observation vector used by the
    v2 observation. Layout (matches DiabloEnvV2Mixin._get_scalars):

    Player progression context:
        [ 0]      dungeon_level / 16
        [ 1]      char_level / 50

    Vital state (current resources + facing):
        [ 2]      hp / max_hp
        [ 3]      mana / max_mana
        [ 4]      hero_dir / 8

    Base stats (normalised by class cap, clipped to 1.0):
        [ 5..8]   strength / magic / dexterity / vitality

    Offense (weapon damage range):
        [ 9]      weapon_dam_min / max_weapon_dam
        [10]      weapon_dam_max / max_weapon_dam

    Defense:
        [11]      armor class: max(0, GetArmor()) / 150, clipped to 1.0.
                  GetArmor() = _pIBonusAC + _pIAC + _pDexterity/5.
                  Hero injection sets _pIAC; _pArmorClass stays 0.
        [12..14]  elemental resists (fire / lightning / magic):
                  max(0, _pFireResist | _pLghtResist | _pMagResist) /
                  MaxResistance, clipped to 1.0. Engine caps damage reduction
                  at MaxResistance (75%), so 1.0 here = full effective
                  immunity. Negative resist (cursed gear) clamped to 0.

    Active buff:
        [15]      mana_shield (0/1) -- when on, incoming damage drains mana
                  instead of HP.

    Inventory consumables:
        [16..22]  potion counts (small_hp, scroll_heal, full_hp, small_mana,
                  full_mana, rejuv, full_rejuv), normalised to [0,1] with
                  cap POTION_CAP.

    Special level type:
        [23..31]  setlvlnum one-hot (9 entries: SL_NONE .. SL_ARENA_*).

    Spellbook (paired blocks, in ActionEnum Cast* order: Firebolt,
    ChargedBolt, FireWall, StoneCurse, ManaShield, Phasing, Fireball):
        [32..38]  spell availability bits (binary 0/1). Set when the spell
                  appears in any of _pMemSpells | _pAblSpells | _pISpells |
                  _pScrlSpells (learned + class ability + staff + scroll).
        [39..45]  spell levels: _pSplLvl[spell_id] / MaxSpellLevel (15). 0
                  means "not learned by the player" (still castable via
                  scroll / class ability / staff with their own levels)."""
    # Normalisation caps. Mirrored from the engine (player.h) where
    # applicable, so 1.0 here corresponds to the engine-bound maximum.
    SPELL_LEVEL_CAP   = 15.0   # devilution::MaxSpellLevel (player.h:39)
    RESIST_CAP        = 75.0   # devilution::MaxResistance (player.h:38)
    ARMOR_CLASS_CAP   = 150.0  # injected max ~114; real gear clips above (Godly Full Plate + helm + shield ~370)
    DUNGEON_LEVEL_CAP = 16.0   # Diablo I has 16 dungeon floors (4 cathedral +
                               # 4 catacombs + 4 caves + 4 hell). NUMLEVELS=25
                               # in the engine includes town and set levels.
    CHAR_LEVEL_CAP    = 50.0   # max character level in Diablo I. Not a compile-
                               # time constant in the engine; ultimately driven
                               # by Experience.tsv (see getMaxCharacterLevel()).
    HERO_DIR_CAP      = 8.0    # Direction enum has 8 compass headings; divide
                               # by 8 (not 7) because direction is circular --
                               # 7 (NW) is not an endpoint.

    p  = d.player
    ca = d.player_class_attrs

    str_cap = ca.maxStr
    if str_cap < 1: str_cap = 1
    mag_cap = ca.maxMag
    dex_cap = ca.maxDex
    if dex_cap < 1: dex_cap = 1
    vit_cap = ca.maxVit
    if vit_cap < 1: vit_cap = 1

    out = np.zeros(46, dtype=np.float32)

    # [0..1] progression context
    out[0] = d.currlevel.value / DUNGEON_LEVEL_CAP
    out[1] = p._pLevel         / CHAR_LEVEL_CAP

    # [2..4] vital state + facing
    hp_denom = p._pMaxHP
    if p._pHitPoints > hp_denom: hp_denom = p._pHitPoints
    if hp_denom < 1: hp_denom = 1
    out[2] = p._pHitPoints / hp_denom

    mana_denom = p._pMaxMana
    if p._pMana > mana_denom: mana_denom = p._pMana
    if mana_denom < 1: mana_denom = 1
    out[3] = p._pMana / mana_denom

    out[4] = p._pdir / HERO_DIR_CAP

    # [5..8] base stats
    v = p._pStrength / str_cap
    out[5] = v if v < 1.0 else 1.0
    if mag_cap > 0:
        v = p._pMagic / mag_cap
        out[6] = v if v < 1.0 else 1.0
    else:
        out[6] = 0.0
    v = p._pDexterity / dex_cap
    out[7] = v if v < 1.0 else 1.0
    v = p._pVitality / vit_cap
    out[8] = v if v < 1.0 else 1.0

    # [9..10] offense: weapon damage range (min, then max)
    # max_weapon_dam covers only base weapon iMaxDam from itemdat.tsv, but
    # _pIMaxDam includes additive bonuses from prefixes/suffixes and jewelry,
    # so clip to 1.0 the same way stats are handled above.
    weapon_denom = d.max_weapon_dam.value
    if weapon_denom < 1: weapon_denom = 1
    v = p._pIMinDam / weapon_denom
    out[9]  = v if v < 1.0 else 1.0
    v = p._pIMaxDam / weapon_denom
    out[10] = v if v < 1.0 else 1.0

    # [11..14] defense: armor class + elemental resists
    ac = p._pIBonusAC + p._pIAC + p._pDexterity // 5
    if ac < 0: ac = 0
    v = ac / ARMOR_CLASS_CAP
    out[11] = v if v < 1.0 else 1.0
    fr = p._pFireResist
    if fr < 0: fr = 0
    v = fr / RESIST_CAP
    out[12] = v if v < 1.0 else 1.0
    lr = p._pLghtResist
    if lr < 0: lr = 0
    v = lr / RESIST_CAP
    out[13] = v if v < 1.0 else 1.0
    mr = p._pMagResist
    if mr < 0: mr = 0
    v = mr / RESIST_CAP
    out[14] = v if v < 1.0 else 1.0

    # [15] active buff
    out[15] = 1.0 if p.pManaShield else 0.0

    # [16..22] inventory consumables (belt + inv, capped at POTION_CAP)
    pot = player_pot_counts(p)
    for i in range(7):
        out[16 + i] = pot[i]

    # [23..31] special level type
    out[23 + d.setlvlnum] = 1.0

    # Spell availability (binary), one bit per Cast action in ActionEnum order.
    # Castable when any of memorised / class ability / staff / scroll has the bit set.
    spell_bits = player_spell_bits(d)
    one = np.uint64(1)
    sp_firebolt    = dx.SpellID.Firebolt.value
    sp_chargedbolt = dx.SpellID.ChargedBolt.value
    sp_firewall    = dx.SpellID.FireWall.value
    sp_stonecurse  = dx.SpellID.StoneCurse.value
    sp_manashield  = dx.SpellID.ManaShield.value
    sp_phasing     = dx.SpellID.Phasing.value
    sp_fireball    = dx.SpellID.Fireball.value
    # [32..38] spell availability bits
    out[32] = 1.0 if (spell_bits & (one << sp_firebolt))    else 0.0
    out[33] = 1.0 if (spell_bits & (one << sp_chargedbolt)) else 0.0
    out[34] = 1.0 if (spell_bits & (one << sp_firewall))    else 0.0
    out[35] = 1.0 if (spell_bits & (one << sp_stonecurse))  else 0.0
    out[36] = 1.0 if (spell_bits & (one << sp_manashield))  else 0.0
    out[37] = 1.0 if (spell_bits & (one << sp_phasing))     else 0.0
    out[38] = 1.0 if (spell_bits & (one << sp_fireball))    else 0.0

    # [39..45] spell levels (one per Cast action), normalised against the engine cap.
    out[39] = p._pSplLvl[sp_firebolt]    / SPELL_LEVEL_CAP
    out[40] = p._pSplLvl[sp_chargedbolt] / SPELL_LEVEL_CAP
    out[41] = p._pSplLvl[sp_firewall]    / SPELL_LEVEL_CAP
    out[42] = p._pSplLvl[sp_stonecurse]  / SPELL_LEVEL_CAP
    out[43] = p._pSplLvl[sp_manashield]  / SPELL_LEVEL_CAP
    out[44] = p._pSplLvl[sp_phasing]     / SPELL_LEVEL_CAP
    out[45] = p._pSplLvl[sp_fireball]    / SPELL_LEVEL_CAP

    return out


def get_surroundings_by_env(d, env):
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

def get_surroundings(d, radius, goal_pos):
    env = get_environment(d, radius, goal_pos=goal_pos)
    return get_surroundings_by_env(d, env)

def pick_random_clean_goal(env, start, rng):
    """Pick a goal tile that doesn't match any known dungeon-layout issue
    (see goal_known_issue). Iterates over the picker's strict-empty regions
    in random order WITHOUT REPLACEMENT - once a region is proved sealed
    (or otherwise known-bad) it is dropped from the candidate pool and
    never re-tried within this call.

    Returns (goal_pos, retried_issues) on success.
    Raises AssertionError if every strict-empty region in the dungeon
    matches a known issue (means we have a new scenario to investigate
    or the dungeon really has no walkable goal beyond the player tile).
    """
    empty_env = \
        (env == 0) | \
        (env == EnvironmentFlag.Explored.value) | \
        (env == EnvironmentFlag.Visible.value) | \
        (env == (EnvironmentFlag.Explored.value | \
                 EnvironmentFlag.Visible.value))

    # Label independent regions (rooms)
    labeled_regions, num_regions = maze.detect_regions(empty_env)

    # Pool of picker regions still untested. We pop from this list each
    # iteration so the same region is never sampled twice in one call.
    remaining = list(range(1, num_regions + 1))
    retried_issues = []

    while remaining:
        i = rng.integers(len(remaining))
        r = remaining.pop(i)
        xs, ys = np.where(labeled_regions == r)
        j = rng.integers(len(xs))
        goal_pos = (int(xs[j]), int(ys[j]))

        issue = goal_known_issue(env, start, goal_pos)
        if issue is None:
            return goal_pos, retried_issues
        retried_issues.append((r, goal_pos, issue))

    raise AssertionError(
        f"pick_random_clean_goal: exhausted all {num_regions} picker regions; "
        f"every region matched a known issue. retried_issues={retried_issues}")

def goal_known_issue(env, start, goal):
    """Return "isolated_subgraph" if goal is unreachable from start via
    the door graph, or None if no known issue applies.

    Two dungeon layouts produce unreachable goals:
      - Catacombs (lvl 5-8): the generator occasionally creates orphan
        sub-areas whose doors only connect to each other, forming a
        component disconnected from the main playable space.
      - Hell (lvl 13-16): certain quest chambers (e.g. Lazarus altar
        room) have no walkable door at all - access is scripted-only.
    Both reduce to the same graph property: goal's floor region is not
    reachable from start's floor region via BFS over door-bridged
    region neighbours. A sealed room (no doors) trivially fails the
    same BFS, so one check covers both cases.

    Callers use the non-None return as a repick signal. If no known
    issue matches but the path-finder still reports unreachable, the
    assertion in get_dungeon_graph_and_path fires - that means a new
    layout variant to investigate.
    """
    empty_env = \
        (env & (EnvironmentFlag.Wall.value | EnvironmentFlag.Door.value)) == 0
    labeled_regions, num_regions = maze.detect_regions(empty_env)
    start_region = labeled_regions[start]
    goal_region  = labeled_regions[goal]
    if start_region == 0 or goal_region == 0:
        return None
    if start_region == goal_region:
        return None
    doors = np.argwhere(env & EnvironmentFlag.Door.value)
    regions_graph, _, _ = maze.get_regions_graph(
        doors, labeled_regions, num_regions)
    reachable = {int(start_region)}
    queue = [int(start_region)]
    while queue:
        r = queue.pop()
        for nbr in regions_graph.get(r, set()):
            nbr = int(nbr)
            if nbr not in reachable:
                reachable.add(nbr)
                queue.append(nbr)
    if int(goal_region) not in reachable:
        return "isolated_subgraph"
    return None

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
    if regions_path is None:
        # Dump reproduction info before tripping the assert so logs show
        # the broken state. Caller is expected to also print the seed +
        # dungeon level above this line.
        import sys
        print(f"get_dungeon_graph_and_path: NO PATH "
              f"start={tuple(int(s) for s in start)} (region={int(start_region)}) -> "
              f"goal={tuple(int(g) for g in goal)} (region={int(goal_region)}); "
              f"num_regions={int(num_regions)}, "
              f"graph_edges={sum(len(v) for v in regions_graph.values()) // 2}, "
              f"start_region_neighbors={sorted(regions_graph.get(int(start_region), set()))}",
              file=sys.stderr, flush=True)
        assert False, "regions_path is None - unreachable goal"

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

def movement_to_true_north(keys):
    """Transforms movement keys so that UP corresponds to true North
    (compensates for 45° Diablo dungeon rotation)"""

    movement_bits = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                     ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                     ring.RingEntryType.RING_ENTRY_KEY_LEFT |
                     ring.RingEntryType.RING_ENTRY_KEY_RIGHT)

    # Copy except movement bits
    reskeys = ~movement_bits & keys
    movement_keys = movement_bits & keys

    if movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_UP):
        # N
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_UP |
                    ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                           ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
        # NE
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
        # E
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                    ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                           ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
        # SE
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_DOWN)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN):
        # S
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                    ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                           ring.RingEntryType.RING_ENTRY_KEY_LEFT):
        # SW
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_LEFT):
        # W
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_UP |
                    ring.RingEntryType.RING_ENTRY_KEY_LEFT)
    elif movement_keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                           ring.RingEntryType.RING_ENTRY_KEY_LEFT):
        # NW
        reskeys |= (ring.RingEntryType.RING_ENTRY_KEY_UP)
    elif movement_keys == 0:
        # Nothing
        pass
    else:
        assert 0, f"Can't remap these movement keys combination '0b{movement_keys:b}'"

    return reskeys

def env_log_glob(eval=False):
    """Return glob pattern for env.log files of training or eval runners."""
    kind = "eval" if eval else "run"
    return os.path.join(tempfile.gettempdir(), "diablo-%s-*" % kind, "env.log")

class DiabloGame:
    def __init__(self, state_path, state, state_dir=None, proc=None,
                 log_file=None, game_ticks_per_step=None, step_mode=None):
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

        if step_mode is None:
            # Read shared option in case of the attach
            self.step_mode = self.state.StepMode.value
        else:
            self.step_mode = step_mode

        # Catch up with the events queue. The @read_idx from the queue
        # is not used to support other processes that might want to
        # attach.
        self.events_queue_read_idx = state.events_queue.write_idx

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
        if not self.step_mode:
            # Perform a deep copy only if this is a real-time mode, in
            # which we need to persist the state; otherwise, the state
            # may change in the middle of some calculation, which can
            # lead to an assertion failure. Unfortunately, the deep
            # copy is very inefficient (performance-wise).
            vars_dict = {}
            for var in dx.VARS:
                short_name = var['short_name']
                # Perform np.array() copy
                vars_dict[short_name] = getattr(self.state, short_name).copy()
            return DevilutionState(vars_dict)
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
        read_idx = self.events_queue_read_idx
        entry = ring.get_entry_to_retrieve(self.state.events_queue, read_idx)
        if entry == None:
            return None
        self.events_queue_read_idx += 1
        return entry

    def submit_key(self, key, data=(0, 0)):
        key = movement_to_true_north(key)
        request_tag = (time.time_ns() & 0xffffffff)
        assert ring.has_capacity_to_submit(self.state.input_queue)
        entry = ring.get_entry_to_submit(self.state.input_queue)
        entry.en_type = key
        entry.en_tag = request_tag
        entry.en_data1, entry.en_data2 = data

        # Catch up with the events queue. The @read_idx from the queue
        # is not used to support other processes that might want to
        # attach.
        read_idx = self.state.events_queue.write_idx

        # Submit key
        ring.submit(self.state.input_queue)

        feedback_events = [
            # Released keys event
            key & ~ring.RingEntryType.RING_ENTRY_FLAGS,
            # `STEP_FINISHED` - only if `game_ticks_per_step` is not 0
            ring.RingEntryType.RING_ENTRY_EVENT_STEP_FINISHED,
        ]
        # Wait for feedback events that come one after another
        event_idx = 0
        while event_idx < len(feedback_events):
            ring.wait_any_submitted(self.state.events_queue, read_idx)
            entry = ring.get_entry_to_retrieve(self.state.events_queue, read_idx)
            read_idx += 1
            assert entry != None
            if entry.en_tag == request_tag:
                event_type = feedback_events[event_idx]
                event_idx += 1
                assert entry.en_type == event_type

                if not self.game_ticks_per_step:
                    # We receive only release event
                    break

    def find_restore_item(self, candidates):
        """Search belt then inventory for the first matching item; return its
        inv_item slot (cii) or -1 if no candidate is present.

        candidates: iterable of item_misc_id values searched in priority order.
        IMISC_SCROLL is matched as a healing scroll (SpellID::Healing).

        Pure search -- no ring submission. The caller decides whether to act
        on the result by submitting INV_USE_ITEM(slot). Keeping this side-
        effect-free lets ALGO code peek ("would this potion be available?")
        without committing a tick.
        """
        player = self.state.player
        for misc_id in candidates:
            spell_id = dx.SpellID.Healing if misc_id == dx.item_misc_id.IMISC_SCROLL else None
            slot = find_inv_item(player, misc_id, spell_id)
            if slot >= 0:
                return slot
        return -1

    @staticmethod
    def run(config):
        cfg_file = open("diablo.ini.template", "r")
        cfg = cfg_file.read()
        cfg_file.close()

        mshared_filename = config["mshared-filename"]
        game_ticks_per_step = config["game-ticks-per-step"]
        step_mode = config["step-mode"]

        cfg = cfg.format(seed=config["seed"],
                         fixed_seed=1 if config["fixed-seed"] else 0,
                         dungeon_level=config.get("dungeon-level", DUNGEON_LEVEL_DEFAULT)[0][0],
                         automap_active=1 if config["gui"] else 0,
                         skip_progress=1 if config["gui"] else 0,
                         skip_animation=0 if config["gui"] else 1,
                         headless=0 if config["gui"] else 1,
                         game_ticks_per_step=game_ticks_per_step,
                         step_mode=0 if config["gui"] else 1 if step_mode else 0,
                         mshared_filename=mshared_filename,
                         invincible_player=1 if config["invincible-player"] else 0,
                         no_monsters=1 if config["no-monsters"] else 0,
                         blind_monsters=1 if config["blind-monsters"] else 0,
                         harmless_barrels=1 if config["harmless-barrels"] else 0,
                         no_butcher=1 if config.get("no_butcher") else 0,
                         spell_potency=int(config.get("spell-potency", 0.0) * 100),
                         no_auto_walk_on_seconday_action=
                         1 if config["no-auto-walk-on-seconday-action"] else 0,
                         hero_hp_min_pct=config.get("hero-hp-min-pct", 100),
                         hero_hp_max_pct=config.get("hero-hp-max-pct", 100),
                         hero_mana_min_pct=config.get("hero-mana-min-pct", 100),
                         hero_mana_max_pct=config.get("hero-mana-max-pct", 100),
                         hero_potions_min=config.get("hero-potions-min", 2),
                         hero_potions_max=config.get("hero-potions-max", 20),
                         )

        kind   = "eval" if config.get("eval") else "run"
        prefix = "diablo-%s-%d-%d-" % (kind, config["index"], os.getpid())
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
                          log_file=log_file, game_ticks_per_step=game_ticks_per_step,
                          step_mode=step_mode)

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
