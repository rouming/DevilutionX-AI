#!/usr/bin/env python3

"""
agent.py - Algorithmic supervisor (AgentAI) for the full Diablo game run.

Combines a trained RL model (combat/exploration) with deterministic control
(town navigation, stair pathfinding, level transitions).

State machine:
  TOWN      - in town (currlevel == 0): pathfind to cathedral entrance
  DUNGEON   - model drives hero; agent monitors kills and stair visibility
  PATHFIND  - algorithm walks to stairs (after kill threshold met)
  DEAD      - hero died, loop ends
  DONE      - all 16 levels complete or Diablo killed

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

import enum
import re
import sys
import time

import numpy as np
import torch

import devilutionx as dx
import diablo_env
import diablo_state
import maze
import ring


# ---------------------------------------------------------------------------
# Item evaluation helpers
# ---------------------------------------------------------------------------

_SKIP_ITYPE = frozenset({
    dx.ItemType.None_.value,
    dx.ItemType.Gold.value,
    dx.ItemType.Misc.value,
})

_SKIP_ILOC = frozenset({
    dx.item_equip_type.ILOC_NONE.value,
    dx.item_equip_type.ILOC_UNEQUIPABLE.value,
    dx.item_equip_type.ILOC_BELT.value,
    dx.item_equip_type.ILOC_INVALID.value,
})

_WEAPON_ILOC = frozenset({
    dx.item_equip_type.ILOC_ONEHAND.value,
    dx.item_equip_type.ILOC_TWOHAND.value,
})

_JEWELRY_ILOC = frozenset({
    dx.item_equip_type.ILOC_RING.value,
    dx.item_equip_type.ILOC_AMULET.value,
})

_ARMOR_ILOC = frozenset({
    dx.item_equip_type.ILOC_ARMOR.value,
    dx.item_equip_type.ILOC_HELM.value,
})

# One point of shield AC counts as this many avg_dmg points when comparing
# a one-hander+shield configuration against a two-hander.
_AC_WEIGHT = 0.5


def _item_name(item):
    return bytes(item._iIName).rstrip(b'\x00').decode('ascii', errors='replace')


def _item_body_slot(item):
    """Return the primary body slot cii for item, or None if not equippable."""
    iloc = int(item._iLoc)
    if iloc in _SKIP_ILOC:
        return None
    ILOC = dx.item_equip_type
    II   = dx.inv_item
    if iloc == ILOC.ILOC_TWOHAND.value:
        return II.INVITEM_HAND_LEFT.value
    if iloc == ILOC.ILOC_ONEHAND.value:
        return (II.INVITEM_HAND_RIGHT.value
                if int(item._itype) == dx.ItemType.Shield.value
                else II.INVITEM_HAND_LEFT.value)
    if iloc == ILOC.ILOC_ARMOR.value:
        return II.INVITEM_CHEST.value
    if iloc == ILOC.ILOC_HELM.value:
        return II.INVITEM_HEAD.value
    if iloc == ILOC.ILOC_RING.value:
        return II.INVITEM_RING_LEFT.value
    if iloc == ILOC.ILOC_AMULET.value:
        return II.INVITEM_AMULET.value
    return None


def _can_equip(item, player):
    # Diablo 1 has no level requirement on items; stat minimums are the gate.
    return (int(item._iMinStr) <= int(player._pBaseStr) and
            int(item._iMinMag) <= int(player._pBaseMag) and
            int(item._iMinDex) <= int(player._pBaseDex))


def _pl_stats_warrior(item):
    """Stat bonus contribution for warrior: STR + VIT.
    Only meaningful when item is identified; caller is responsible for that check."""
    return int(item._iPLStr) + int(item._iPLVit)


def _item_score(item, hero_class):
    """Scalar score + label for armor and jewelry (used inside class-specific functions)."""
    iloc = int(item._iLoc)
    identified = bool(item._iIdentified)
    if iloc in _ARMOR_ILOC:
        ac = int(item._iAC)
        if identified:
            # GetBonusAC: _iAC * _iPLAC / 100 (clamped to sign if rounds to 0)
            pl_ac = int(item._iPLAC)
            bonus = ac * pl_ac // 100
            if bonus == 0 and pl_ac != 0:
                bonus = 1 if pl_ac > 0 else -1
            ac += bonus
            if hero_class == dx.HeroClass.Warrior.value:
                ac += _pl_stats_warrior(item)
        score = ac
        return score, f"AC={score}"
    # Jewelry: stat bonus priority depends on class.
    if hero_class == dx.HeroClass.Warrior.value:
        score = int(item._iPLStr) + int(item._iPLVit)
        return score, f"str+vit={score}"
    if hero_class == dx.HeroClass.Rogue.value:
        score = int(item._iPLDex) + int(item._iPLVit)
        return score, f"dex+vit={score}"
    # Sorcerer / others
    score = int(item._iPLMag) + int(item._iPLVit)
    return score, f"mag+vit={score}"


def _is_scroll_misc(imisc):
    """Return True for both IMISC_SCROLL (21) and IMISC_SCROLLT (22)."""
    return (imisc == dx.item_misc_id.IMISC_SCROLL.value or
            imisc == dx.item_misc_id.IMISC_SCROLLT.value)


def _find_scroll_of_identify(player):
    """Return cii of first scroll of identify in inventory or belt, or None."""
    spellid_identify = dx.SpellID.Identify.value
    INV_FIRST        = dx.inv_item.INVITEM_INV_FIRST.value
    BELT_FIRST       = dx.inv_item.INVITEM_BELT_FIRST.value
    itype_none       = dx.ItemType.None_.value
    for i in range(int(player._pNumInv)):
        it = player.InvList[i]
        if (_is_scroll_misc(int(it._iMiscId)) and
                int(it._iSpell) == spellid_identify):
            return INV_FIRST + i
    for i in range(8):
        it = player.SpdList[i]
        if (int(it._itype) != itype_none and
                _is_scroll_misc(int(it._iMiscId)) and
                int(it._iSpell) == spellid_identify):
            return BELT_FIRST + i
    return None


def _find_unidentified_equipped(player):
    """Return (seed, name) of first unidentified magical item in body slots, or None."""
    none_type      = dx.ItemType.None_.value
    quality_normal = dx.item_quality.ITEM_QUALITY_NORMAL.value
    for cii in range(dx.inv_item.INVITEM_INV_FIRST.value):
        item = player.InvBody[cii]
        if (int(item._itype) != none_type
                and int(item._iMagical) != quality_normal
                and not item._iIdentified):
            return int(item._iSeed), _item_name(item)
    return None


def _count_scroll_spellid(player, spellid):
    """Count scrolls with the given spellid in inventory and belt."""
    itype_none = dx.ItemType.None_.value
    count = 0
    for i in range(int(player._pNumInv)):
        it = player.InvList[i]
        if _is_scroll_misc(int(it._iMiscId)) and int(it._iSpell) == spellid:
            count += 1
    for i in range(8):
        it = player.SpdList[i]
        if (int(it._itype) != itype_none and
                _is_scroll_misc(int(it._iMiscId)) and
                int(it._iSpell) == spellid):
            count += 1
    return count


def _is_combat_spell_scroll(spellid):
    """Return True if this scroll carries a combat/utility spell worth keeping.
    TODO: decide per-spell policy - Phasing (SpellID.Phasing) is the primary
    candidate for Warrior at levels 2-4; others (Firebolt etc.) likely not.
    Currently returns False so all unrecognised scrolls are dropped by caller.
    """
    # TODO: return True for spells the agent should stockpile.
    return False


def _identified_label(item):
    """Return a compact string of the revealed magical properties of an item."""
    parts = []
    pl_dam  = int(item._iPLDam)
    pl_ac   = int(item._iPLAC)
    pl_str  = int(item._iPLStr)
    pl_vit  = int(item._iPLVit)
    pl_dex  = int(item._iPLDex)
    pl_mag  = int(item._iPLMag)
    pl_tohit= int(item._iPLToHit)
    pl_hp   = int(item._iPLHP)
    pl_mana = int(item._iPLMana)
    pl_fr   = int(item._iPLFR)
    pl_lr   = int(item._iPLLR)
    pl_mr   = int(item._iPLMR)
    if pl_dam   != 0: parts.append(f"dmg%={pl_dam:+d}")
    if pl_tohit != 0: parts.append(f"tohit={pl_tohit:+d}")
    if pl_ac    != 0: parts.append(f"ac%={pl_ac:+d}")
    if pl_str   != 0: parts.append(f"str={pl_str:+d}")
    if pl_vit   != 0: parts.append(f"vit={pl_vit:+d}")
    if pl_dex   != 0: parts.append(f"dex={pl_dex:+d}")
    if pl_mag   != 0: parts.append(f"mag={pl_mag:+d}")
    if pl_hp    != 0: parts.append(f"hp={pl_hp:+d}")
    if pl_mana  != 0: parts.append(f"mana={pl_mana:+d}")
    if pl_fr    != 0: parts.append(f"FR={pl_fr:+d}")
    if pl_lr    != 0: parts.append(f"LR={pl_lr:+d}")
    if pl_mr    != 0: parts.append(f"MR={pl_mr:+d}")
    return " ".join(parts) if parts else "no bonuses"


def _dur_ratio(item):
    """Current durability as a fraction of max. 255 = indestructible -> 1.0."""
    max_dur = int(item._iMaxDur)
    if max_dur == 0 or max_dur == 255:
        return 1.0
    return int(item._iDurability) / max_dur


def _is_better_for_warrior(item, player, body_cii, pending):
    """Is item an upgrade for warrior at body_cii?
    pending: (seed, score, name) tuple if an equip is in-flight for this slot, else None.
    Returns (is_better, new_score, new_label, old_score, old_label, old_name).
    old_name is None for class-filtered items (no comparison logged).
    """
    iloc = int(item._iLoc)

    # Warriors do not use bows; ranged weapons are useless for melee fighters.
    if int(item._itype) == dx.ItemType.Bow.value:
        return False, 0, "bow", 0, "", None

    if iloc in _WEAPON_ILOC:
        # Shields go to HAND_RIGHT and are scored by AC, not damage.
        if int(item._itype) == dx.ItemType.Shield.value:
            item_id = bool(item._iIdentified)
            new_ac  = int(item._iAC)
            if item_id:
                pl_ac = int(item._iPLAC)
                b = new_ac * pl_ac // 100
                if b == 0 and pl_ac != 0:
                    b = 1 if pl_ac > 0 else -1
                new_ac += b + _pl_stats_warrior(item)
            new_score = new_ac * _AC_WEIGHT
            new_label = f"score={new_score:.1f}"
            if pending is not None:
                _, old_score, old_name = pending
                return new_score > old_score, new_score, new_label, old_score, f"score={old_score:.1f}", old_name
            eq = player.InvBody[body_cii]
            if int(eq._itype) == dx.ItemType.None_.value:
                return True, new_score, new_label, 0, "empty", ""
            eq_id  = bool(eq._iIdentified)
            old_ac = int(eq._iAC)
            if eq_id:
                pl_ac = int(eq._iPLAC)
                b = old_ac * pl_ac // 100
                if b == 0 and pl_ac != 0:
                    b = 1 if pl_ac > 0 else -1
                old_ac += b + _pl_stats_warrior(eq)
            old_score = old_ac * _AC_WEIGHT
            is_better = new_score > old_score or (
                new_score == old_score and _dur_ratio(item) > _dur_ratio(eq))
            return is_better, new_score, new_label, old_score, f"score={old_score:.1f}", _item_name(eq)

        # Weapon in HAND_LEFT: score = avg_dmg + shield_AC * AC_WEIGHT + stat_bonus.
        # A two-hander displaces the shield, so its AC contribution drops to 0.
        hand_r    = player.InvBody[dx.inv_item.INVITEM_HAND_RIGHT.value]
        shield_ac = (int(hand_r._iAC)
                     if int(hand_r._itype) == dx.ItemType.Shield.value else 0)
        item_id      = bool(item._iIdentified)
        new_dmg      = (int(item._iMinDam) + int(item._iMaxDam)) / 2
        if item_id:
            pl_dam  = int(item._iPLDam)
            new_dmg *= (1 + pl_dam / 100)
        keeps_shield = (iloc != dx.item_equip_type.ILOC_TWOHAND.value)
        new_stat     = _pl_stats_warrior(item) if item_id else 0
        new_score    = new_dmg + (shield_ac * _AC_WEIGHT if keeps_shield else 0) + new_stat
        new_label    = f"score={new_score:.1f}"

        if pending is not None:
            _, old_score, old_name = pending
            return new_score > old_score, new_score, new_label, old_score, f"score={old_score:.1f}", old_name

        eq = player.InvBody[body_cii]
        if int(eq._itype) == dx.ItemType.None_.value:
            return True, new_score, new_label, 0, "empty", ""
        eq_id     = bool(eq._iIdentified)
        eq_iloc   = int(eq._iLoc)
        old_dmg   = (int(eq._iMinDam) + int(eq._iMaxDam)) / 2
        if eq_id:
            old_pl_dam = int(eq._iPLDam)
            old_dmg   *= (1 + old_pl_dam / 100)
        old_keeps = (eq_iloc != dx.item_equip_type.ILOC_TWOHAND.value)
        old_stat  = _pl_stats_warrior(eq) if eq_id else 0
        old_score = old_dmg + (shield_ac * _AC_WEIGHT if old_keeps else 0) + old_stat
        is_better = new_score > old_score or (
            new_score == old_score and _dur_ratio(item) > _dur_ratio(eq))
        return is_better, new_score, new_label, old_score, f"score={old_score:.1f}", _item_name(eq)

    # Armor and jewelry: standard scoring.
    new_score, new_label = _item_score(item, dx.HeroClass.Warrior.value)
    if pending is not None:
        _, old_score, old_name = pending
        return new_score > old_score, new_score, new_label, old_score, f"score={old_score:.1f}", old_name
    eq = player.InvBody[body_cii]
    if int(eq._itype) == dx.ItemType.None_.value:
        return True, new_score, new_label, 0, "empty", ""
    old_score, old_label = _item_score(eq, dx.HeroClass.Warrior.value)
    is_better = new_score > old_score or (
        new_score == old_score and _dur_ratio(item) > _dur_ratio(eq))
    return is_better, new_score, new_label, old_score, old_label, _item_name(eq)


def _is_better_for_rogue(item, player, body_cii, pending):
    assert False, "rogue item evaluation not implemented"


def _is_better_for_sorcerer(item, player, body_cii, pending):
    assert False, "sorcerer item evaluation not implemented"


_IS_BETTER_FOR_CLASS = {
    dx.HeroClass.Warrior.value:  _is_better_for_warrior,
    dx.HeroClass.Rogue.value:    _is_better_for_rogue,
    dx.HeroClass.Sorcerer.value: _is_better_for_sorcerer,
}


# ---------------------------------------------------------------------------
# Butcher door masking
# ---------------------------------------------------------------------------

def find_butcher_door(d):
    """Return the world (x, y) position of the Butcher's room exit door on
    level 2, or None if the Butcher is already dead or not on level 2.

    The Butcher is a unique boss on level 2 that is far too strong to fight
    early. We mask his door so the model treats it as a wall and does not
    wander in. The door is kept stored so it can be unmasked later when the
    hero returns stronger (level 3-4) to collect his drop (The Butcher's
    Cleaver).

    BFS from the Butcher's tile through non-wall, non-door tiles (his room
    interior). The first adjacent door tile found is the room's only exit.
    """
    if int(d.currlevel.value) != 2:
        return None

    butcher_pos = None
    for i in range(int(d.ActiveMonsterCount.value)):
        mid = int(d.ActiveMonsters[i])
        m = d.Monsters[mid]
        if int(m.ai) == dx.MonsterAIID.Butcher.value:
            butcher_pos = (int(m.position.tile.x), int(m.position.tile.y))
            break

    if butcher_pos is None:
        return None

    EF = diablo_state.EnvironmentFlag
    env_full = diablo_state.get_environment(
        d, show_invisible=True, show_unexplored=True)
    w, h = env_full.shape

    def passable(pos):
        x, y = pos
        if not (0 <= x < w and 0 <= y < h):
            return False
        tile = int(env_full[x, y])
        return not (tile & (EF.Wall.value | EF.Door.value))

    def target(pos):
        x, y = pos
        if not (0 <= x < w and 0 <= y < h):
            return False
        return bool(int(env_full[x, y]) & EF.Door.value)

    door_pos = maze.bfs_find(butcher_pos, passable, target)
    if door_pos is not None:
        tile = int(env_full[door_pos[0], door_pos[1]])
        assert tile & EF.Door.value, \
            f"find_butcher_door: {door_pos} has no Door flag (tile=0x{tile:x})"
    return door_pos


# ---------------------------------------------------------------------------
# Pathfinder
# ---------------------------------------------------------------------------

class Pathfinder:
    """Goal-directed A* pathfinder over the dungeon map.
    Recomputes path whenever the hero drifts off the cached route.
    """

    def __init__(self, game, view_radius):
        self.game        = game
        self.view_radius = view_radius
        self._goal       = None
        self._path       = []
        self._walkable   = None  # bool array indexed [x, y]
        self._skip_items = set()  # world positions where item pickup failed; step over instead

    def set_goal(self, goal_pos):
        self._goal = goal_pos
        self._path = []
        self._skip_items = set()
        self._rebuild_walkable()

    def skip_item(self, pos):
        """Mark pos as an item to step over rather than interact with."""
        self._skip_items.add(pos)
        self._path = []  # force replan

    def _rebuild_walkable(self):
        d = self.game.safe_state
        env_full = diablo_state.get_environment(
            d, show_invisible=True, show_unexplored=True)
        EF = diablo_state.EnvironmentFlag
        # Strip transient flags before structural check: monsters and missiles
        # are not physical obstacles - _tick_pathfind attacks monsters in the way.
        TRANSIENT = np.uint32(EF.Monster.value | EF.Missile.value)
        ef = env_full & ~TRANSIENT
        self._walkable = (
            (ef == 0) |
            (ef == EF.Explored.value) |
            (ef == EF.Visible.value) |
            (ef == (EF.Explored.value | EF.Visible.value)) |
            ((ef & (EF.Open.value | EF.Item.value)) != 0))

    def next_action(self):
        """Return an ActionEnum int, or None if goal reached / unreachable."""
        d = self.game.safe_state
        player = diablo_state.player_position(d)

        if player == self._goal:
            return None

        if player not in self._path:
            self._rebuild_walkable()
            walkable = self._walkable
            goal     = self._goal

            def cost(p):
                x, y = p
                if not (0 <= x < walkable.shape[0] and
                        0 <= y < walkable.shape[1]):
                    return 0
                # Goal tile may not be walkable (stairs), allow it anyway
                if (x, y) == goal:
                    return 1
                return 1 if walkable[x, y] else 0

            self._path = maze.shortest_path(player, goal, cost)
            if not self._path:
                return None

        try:
            idx = self._path.index(player)
        except ValueError:
            self._path = []
            return self.next_action()

        self._path = self._path[idx:]
        if len(self._path) < 2:
            return None

        next_pos  = self._path[1]
        delta     = (next_pos[0] - player[0], next_pos[1] - player[1])
        diag_move = (abs(delta[0]) + abs(delta[1]) == 2)

        EF = diablo_state.EnvironmentFlag
        env_local = diablo_state.get_environment(d, radius=self.view_radius)
        vr = self.view_radius
        px, py = player

        def local(pos):
            return (pos[0] - px + vr, pos[1] - py + vr)

        def tile_at(pos):
            lx, ly = local(pos)
            if 0 <= lx < env_local.shape[0] and 0 <= ly < env_local.shape[1]:
                return int(env_local[lx, ly])
            return 0

        skip = self._skip_items

        def is_barrel_or_item(pos, tile):
            if pos in skip:
                return bool(tile & EF.Barrel.value)
            return bool(tile & (EF.Barrel.value | EF.Item.value))

        def is_closed_door(tile):
            return (tile & EF.Door.value) and not (tile & EF.Open.value)

        next_tile = tile_at(next_pos)
        ipos1 = (px + delta[0], py)
        ipos2 = (px, py + delta[1])

        needs_interact = (
            is_closed_door(next_tile) or
            is_barrel_or_item(next_pos, next_tile) or
            (diag_move and (
                is_barrel_or_item(ipos1, tile_at(ipos1)) or
                is_barrel_or_item(ipos2, tile_at(ipos2)))))

        if needs_interact:
            # Face the target first; only send SecondaryAction once aligned
            player_dir = diablo_env.player_direction_delta(d)
            if delta != player_dir:
                return diablo_env.direction_to_action(delta)
            return diablo_env.ActionEnum.SecondaryAction.value

        return diablo_env.direction_to_action(delta)


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class ModelRunner:
    """Wraps an AC model for step-by-step inference, maintaining LSTM state."""

    def __init__(self, acmodel, preprocess_obss, device, argmax=False):
        self.acmodel         = acmodel
        self.preprocess_obss = preprocess_obss
        self.device          = device
        self.argmax          = argmax
        self.memory          = None
        self.reset()

    def reset(self):
        if self.acmodel.recurrent:
            self.memory = self.acmodel.reset_memory(self.device)
        else:
            self.memory = None

    def act(self, obs):
        """obs: dict matching the env observation_space. Returns int action."""
        preprocessed = self.preprocess_obss([obs], device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memory = self.acmodel(
                    preprocessed, self.memory, noise=None)
            else:
                dist, _ = self.acmodel(preprocessed, noise=None)
        # dist is a list (one entry per hierarchy level); use the first
        if self.argmax:
            return dist[0].probs.argmax(dim=1).item()
        return dist[0].sample().item()


# ---------------------------------------------------------------------------
# AgentAI
# ---------------------------------------------------------------------------

class AgentAI:
    """Algorithmic supervisor for a full Diablo game run (all 16 levels).

    Combines:
    - TOWN:     pathfind to cathedral entrance
    - DUNGEON:  model controls hero; agent monitors kills and stair sightings
    - PATHFIND: walk to stairs once kill threshold is met or time limit hit
    """

    class State(enum.Enum):
        TOWN     = "town"
        DUNGEON  = "dungeon"
        PATHFIND = "pathfind"
        DEAD     = "dead"
        DONE     = "done"

    def __init__(self, game, model_runner, view_radius=10,
                 kill_threshold=0.5, repair_threshold=0.25,
                 max_steps_per_level=3000, no_gear_management=False,
                 keep_mana_potions=False,
                 safe_radius=2, pause=0.0, log=None, stat_strategy='dex-rush'):
        self.game          = game
        self.model         = model_runner
        if stat_strategy in AgentAI._STAT_STRATEGIES:
            self._stat_strategy = AgentAI._STAT_STRATEGIES[stat_strategy]
        else:
            self._stat_strategy = AgentAI._compile_stat_strategy(stat_strategy)
        self.view_radius = view_radius
        self.kill_threshold      = kill_threshold
        self.repair_threshold    = repair_threshold
        self.max_steps_per_level = max_steps_per_level
        self.no_gear_management  = no_gear_management
        self.keep_mana_potions   = keep_mana_potions
        self.safe_radius         = safe_radius
        self.pause       = pause
        self.log         = log or sys.stdout

        self.state       = AgentAI.State.TOWN
        self.cur_level   = -1
        self.level_steps = 0

        self.initial_monster_cnt = 0
        self.stairs_pos          = None
        # {level: monster count on first entry} - persists across re-entries so
        # the kill threshold is always relative to the original population.
        self._level_monsters     = {}
        # {level: stairs position} - persists so re-entry restores known stairs.
        self._level_stairs       = {}

        # {level: {(x,y): (clear_mask, set_flags)}} - per-tile obs overrides.
        # apply: tile = (tile & ~clear_mask) | set_flags
        # Only the specified bits change; all other flags (Explored, Visible,
        # Monster, Player, ...) are preserved.
        # Door -> Wall:  clear=Door|Open, set=Wall
        # Item -> clear: clear=Item,      set=0  (tile keeps explored/visible etc.)
        self._masked_tiles       = {}
        # Set of levels already searched for Butcher door (avoids repeat BFS).
        self._butcher_searched   = set()
        # (level, pos) of item tile we pressed X for; checked next tick.
        self._pending_item_at        = None
        self._pathfind_stall_logged  = 0.0  # time.time() of last stall log
        self._tick_count             = 0     # monotonically increasing tick counter for logging
        self._inv_prev               = None  # previous tick full snapshot (cii,seed,name)
        # body_cii -> (seed, score, name): set at queue-time so subsequent evaluations
        # for the same slot compare against the queued winner, not the live InvBody item.
        # Cleared only when engine confirms the equip (InvBody seed match) - never on
        # intermediate equips - so the winning seed always governs comparisons.
        self._pending_equip          = {}
        # ordered list of gear decisions pending execution:
        #   {'action': 'equip'|'drop', 'seed': int, 'name': str, 'body_cii': int (equip)}
        self._action_queue           = []
        # seeds currently in _action_queue; prevents re-detection of already-decided items
        self._queued_seeds           = set()
        # seeds that were just identified this tick; re-evaluated next tick
        self._identify_pending_seeds = set()
        # True if new InvList seeds were detected this tick; blocks queue execution
        self._inv_changed            = False

        self._pathfinder = Pathfinder(game, view_radius)
        # Agent-owned set of masked drop positions (level-specific). Stable across
        # pathfinder goal resets - set_goal() rebinds _skip_items, so we re-sync
        # from here each tick in _locate_dropped_items.
        self._masked_positions = set()

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def run(self, max_steps=None):
        step = 0
        while self.state not in (AgentAI.State.DEAD, AgentAI.State.DONE):
            if max_steps is not None and step >= max_steps:
                print(f"agent {self._tick_count}: max_steps {max_steps} reached", file=self.log)
                break
            self._tick()
            step += 1
            if self.pause:
                time.sleep(self.pause)
        print(f"agent {self._tick_count}: done - state={self.state.value} steps={step}",
              file=self.log)

    # -----------------------------------------------------------------------
    # Tick
    # -----------------------------------------------------------------------

    def _inv_snapshot(self, player):
        """Return a frozenset of (cii, seed, name) for all non-empty slots."""
        items = set()
        none_type = dx.ItemType.None_.value
        for cii in range(dx.inv_item.INVITEM_INV_FIRST.value):  # body slots 0-6
            item = player.InvBody[cii]
            if int(item._itype) != none_type:
                items.add((cii, int(item._iSeed), _item_name(item)))
        for i in range(int(player._pNumInv)):
            item = player.InvList[i]
            if int(item._itype) != none_type:
                cii = dx.inv_item.INVITEM_INV_FIRST.value + i
                items.add((cii, int(item._iSeed), _item_name(item)))
        for i, item in enumerate(player.SpdList):
            if int(item._itype) != none_type:
                cii = dx.inv_item.INVITEM_BELT_FIRST.value + i
                items.add((cii, int(item._iSeed), _item_name(item)))
        return frozenset(items)

    def _log_inv_diff(self, prev, curr):
        removed     = prev - curr
        added       = curr - prev
        added_seeds = {s for (_, s, _) in added}
        inv_first   = dx.inv_item.INVITEM_INV_FIRST.value
        dead        = diablo_state.is_player_dead(self.game.safe_state)
        for cii, seed, name in sorted(removed):
            # Body-slot item gone without seed reappearing: broke in combat.
            # Suppress on death - all equipped items leave simultaneously when hero dies.
            destroyed = not dead and cii < inv_first and seed not in added_seeds
            suffix = ' (destroyed)' if destroyed else ''
            print(f"agent {self._tick_count}: inv[-] cii={cii} seed={seed} '{name}'{suffix}", file=self.log)
        for cii, seed, name in sorted(added):
            print(f"agent {self._tick_count}: inv[+] cii={cii} seed={seed} '{name}'", file=self.log)

    def _tick(self):
        d = self.game.safe_state

        self._tick_count += 1
        self._inv_changed = False

        inv_curr = self._inv_snapshot(d.player)
        if self._inv_prev is not None and inv_curr != self._inv_prev:
            self._log_inv_diff(self._inv_prev, inv_curr)

        if self._inv_prev is not None:
            # Detect new InvList and belt items and evaluate them.
            # Body slots (cii < INVITEM_INV_FIRST) excluded; belt included.
            # Level-entry evaluation of carried items happens in _on_level_change, not here.
            inv_first  = dx.inv_item.INVITEM_INV_FIRST.value
            belt_first = dx.inv_item.INVITEM_BELT_FIRST.value
            prev_list = {s for (c, s, _) in self._inv_prev if inv_first <= c}
            curr_list = {s for (c, s, _) in inv_curr    if inv_first <= c}
            skip      = self._queued_seeds
            new_seeds = curr_list - prev_list - skip
            # Re-evaluate inventory items that were just identified this tick
            # (e.g. jewelry identified in-place before equip decision).
            pending_id = self._identify_pending_seeds
            new_seeds |= (pending_id & curr_list) - skip
            if pending_id:
                self._log_identified_items(d, pending_id, inv_curr)
            self._identify_pending_seeds = set()
            # No rescan needed for equipped items identified in body slots: all
            # inventory gear was already evaluated on pickup and either queued
            # (seed in _queued_seeds, skipped) or dropped (gone from inv).
            # New pickups after identification compare against the live InvBody
            # state which already reflects the identified (possibly cursed) item.
            if new_seeds and not self.no_gear_management:
                self._inv_changed = True
                for seed in new_seeds:
                    self._evaluate_and_queue(d, seed)

        self._inv_prev = inv_curr
        self._locate_dropped_items(d)

        if diablo_state.is_player_dead(d):
            mode = {'DUNGEON': 'model', 'PATHFIND': 'algo'}.get(self.state.name, self.state.name.lower())
            print(f"agent {self._tick_count}: hero died at level {self.cur_level} [{mode}]", file=self.log)
            self.state = AgentAI.State.DEAD
            return

        if d.player._pmode == dx.PLR_MODE.PM_QUIT.value:
            print(f"agent {self._tick_count}: Diablo killed, game complete", file=self.log)
            self.state = AgentAI.State.DONE
            return

        cur_level = int(d.currlevel.value)
        if cur_level != self.cur_level:
            self._on_level_change(d, cur_level)

        self._assign_stat_points(d)
        if not self.no_gear_management:
            sr  = self.safe_radius
            pos = diablo_state.player_position(d)
            safe = (sr == 0 or not _monster_within_radius(d, pos, sr))
            # Resolve pending equips confirmed by engine this tick.
            for bc in list(self._pending_equip.keys()):
                if int(d.player.InvBody[bc]._iSeed) == self._pending_equip[bc][0]:
                    del self._pending_equip[bc]
            if safe:
                self._repair_gear(d)
            if self._action_queue and safe and not self._inv_changed:
                self._execute_queued(d)

        if self.state == AgentAI.State.TOWN:
            self._tick_town(d)
        elif self.state == AgentAI.State.DUNGEON:
            self._tick_dungeon(d)
        elif self.state == AgentAI.State.PATHFIND:
            self._tick_pathfind(d)

    def _on_level_change(self, d, new_level):
        self.cur_level          = new_level
        self.level_steps        = 0
        self.stairs_pos         = self._level_stairs.get(new_level)
        self._action_queue           = []
        self._queued_seeds           = set()
        self._identify_pending_seeds = set()
        self._inv_changed            = False
        self._pending_equip          = {}
        # Drop tile-skip set: positions are level-specific; stale entries from the
        # previous level could block valid item pickups on the new level.
        self._masked_positions.clear()
        self._pathfinder._skip_items.clear()
        self.model.reset()

        if new_level == 0:
            self.state = AgentAI.State.TOWN
            print(f"agent {self._tick_count}: in town", file=self.log)
            # Snapshot now so town entry doesn't re-evaluate carried items next tick.
            self._inv_prev = self._inv_snapshot(d.player)
            return

        cur_cnt = diablo_state.count_active_monsters(d)
        if new_level not in self._level_monsters:
            self._level_monsters[new_level] = cur_cnt
        self.initial_monster_cnt = self._level_monsters[new_level]
        print(f"agent {self._tick_count}: now on level {new_level}, monsters {cur_cnt}"
              f" (initial {self.initial_monster_cnt})", file=self.log)
        self._init_butcher_door_mask(d)
        self.state = AgentAI.State.DUNGEON

        # Evaluate all InvList items carried into this level. Set _inv_prev to
        # the current snapshot so the next tick does not treat them as new again.
        if not self.no_gear_management:
            inv_first  = dx.inv_item.INVITEM_INV_FIRST.value
            inv_now    = self._inv_snapshot(d.player)
            for seed in {s for (c, s, _) in inv_now if inv_first <= c}:
                self._evaluate_and_queue(d, seed)
            self._inv_prev = inv_now
        else:
            self._inv_prev = self._inv_snapshot(d.player)

    # -----------------------------------------------------------------------
    # State handlers
    # -----------------------------------------------------------------------

    def _tick_town(self, d):
        trig = diablo_state.find_trigger(d, dx.interface_mode.WM_DIABNEXTLVL)
        if trig is None:
            self._submit(diablo_env.ActionEnum.Stand.value)
            return

        goal = (int(trig.position.x), int(trig.position.y))
        player = diablo_state.player_position(d)

        if player == goal:
            self._submit(diablo_env.ActionEnum.Stand.value)
            return

        if self._pathfinder._goal != goal:
            self._pathfinder.set_goal(goal)

        action = self._pathfinder.next_action()
        self._submit(action if action is not None
                     else diablo_env.ActionEnum.Stand.value)

    def _tick_dungeon(self, d):
        self.level_steps += 1

        # Record stairs once they become visible/explored
        if self.stairs_pos is None:
            trig = diablo_state.find_trigger(
                d, dx.interface_mode.WM_DIABNEXTLVL)
            if trig is not None:
                pos = (int(trig.position.x), int(trig.position.y))
                if d.dFlags[pos] & dx.DungeonFlag.Explored.value:
                    self.stairs_pos = pos
                    self._level_stairs[self.cur_level] = pos
                    print(f"agent {self._tick_count}: found stairs to level {self.cur_level + 1} at {pos}",
                          file=self.log)

        # Switch to PATHFIND once kill threshold or step budget reached
        kills    = (self.initial_monster_cnt -
                    diablo_state.count_active_monsters(d))
        kill_pct = kills / max(self.initial_monster_cnt, 1)
        timeout  = self.level_steps >= self.max_steps_per_level

        if self.stairs_pos is not None and (kill_pct >= self.kill_threshold
                                            or timeout):
            reason = f"timeout, kills {kill_pct:.0%}" if timeout else f"kills {kill_pct:.0%}"
            print(f"agent {self._tick_count}: switching to PATHFIND ({reason})", file=self.log)
            self._pathfinder.set_goal(self.stairs_pos)
            self.state = AgentAI.State.PATHFIND
            return

        obs = self._build_obs(d)
        self._submit(self.model.act(obs))

    def _tick_pathfind(self, d):
        player = diablo_state.player_position(d)

        if player == self.stairs_pos:
            # Stepped on stairs; level change will be caught next tick
            self._submit(diablo_env.ActionEnum.Stand.value)
            return

        # If last tick pressed X for an item and item is still there, give up
        if self._pending_item_at is not None:
            level, pos = self._pending_item_at
            self._pending_item_at = None
            if level == self.cur_level and self._tile_has_item(d, pos):
                print(f"agent {self._tick_count}: item pickup failed at {pos}, skipping",
                      file=self.log)
                self._pathfinder.skip_item(pos)

        # Attack any adjacent monster before moving
        if _monster_adjacent(d, player):
            self._submit(diablo_env.ActionEnum.PrimaryAction.value)
            return

        action = self._pathfinder.next_action()
        if action is None:
            now = time.time()
            if now - self._pathfind_stall_logged >= 10:
                stairs = self.stairs_pos
                if stairs is None:
                    reason = "stairs not discovered"
                elif player == stairs:
                    reason = f"on stairs {stairs}, waiting for level change"
                else:
                    reason = f"no route from {player} to stairs {stairs}"
                print(f"agent {self._tick_count}: pathfinder stalled: {reason}",
                      file=self.log)
                self._pathfind_stall_logged = now
                self.state = AgentAI.State.DONE
            self._submit(diablo_env.ActionEnum.Stand.value)
            return

        # Track item pickup attempts to detect failure next tick
        if action == diablo_env.ActionEnum.SecondaryAction.value:
            path = self._pathfinder._path
            if len(path) >= 2 and self._tile_has_item(d, path[1]):
                self._pending_item_at = (self.cur_level, path[1])

        self._submit(action)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _tile_has_item(self, d, pos):
        """Return True if pos has the Item bit set in the local environment."""
        px, py = diablo_state.player_position(d)
        vr = self.view_radius
        env = diablo_state.get_environment(d, radius=vr)
        lx, ly = pos[0] - px + vr, pos[1] - py + vr
        if 0 <= lx < env.shape[0] and 0 <= ly < env.shape[1]:
            return bool(int(env[lx, ly]) & diablo_state.EnvironmentFlag.Item.value)
        return False

    def _mask_tile(self, level, pos, clear, set_flags=0):
        """Register a tile override for level at pos.
        Obs tile becomes: (original & ~clear) | set_flags."""
        self._masked_tiles.setdefault(level, {})[pos] = (clear, set_flags)

    def _init_butcher_door_mask(self, d):
        """Called once per level-2 visit. Finds the Butcher's room door and
        adds it to _masked_tiles so the model never tries to open it."""
        level = int(d.currlevel.value)
        if level != 2 or level in self._butcher_searched:
            return
        self._butcher_searched.add(level)
        pos = find_butcher_door(d)
        if pos is not None:
            EF = diablo_state.EnvironmentFlag
            self._mask_tile(level, pos,
                            clear=EF.Door.value | EF.Open.value,
                            set_flags=EF.Wall.value)
            print(f"agent {self._tick_count}: masking butcher door at {pos}", file=self.log)
        else:
            print(f"agent {self._tick_count}: butcher not on level 2 (quest inactive or already dead)",
                  file=self.log)

    def _apply_masked_tiles(self, d, env):
        """Apply per-tile obs overrides for the current level."""
        level = int(d.currlevel.value)
        tiles = self._masked_tiles.get(level)
        if not tiles:
            return
        px, py = diablo_state.player_position(d)
        vr = self.view_radius
        for (wx, wy), (clear, set_flags) in tiles.items():
            lx = wx - px + vr
            ly = wy - py + vr
            if 0 <= lx < env.shape[0] and 0 <= ly < env.shape[1]:
                env[lx, ly] = (int(env[lx, ly]) & ~clear) | set_flags

    def _build_obs(self, d):
        env = diablo_state.get_environment(d, radius=self.view_radius)
        self._apply_masked_tiles(d, env)
        scalars = diablo_state.compute_scalars(d)
        max_level  = max(int(d.max_monster_level.value),  1)
        max_walk   = max(int(d.max_walk_frames.value),    1)
        max_attack = max(int(d.max_attack_frames.value),  1)
        monster_attrs = diablo_state.compute_monster_attrs(
            d, self.view_radius, max_level, max_walk, max_attack,
            diablo_state.ranged_ai_ids_array())
        return {"env": env, "scalars": scalars, "monster_attrs": monster_attrs}

    def _submit(self, action):
        RE = ring.RingEntryType
        ae = diablo_env.ActionEnum(int(action))
        data = (0, 0)
        if ae == diablo_env.ActionEnum.RestoreHealth:
            slot = self.game.find_restore_item(diablo_env._RESTORE_HEALTH_CANDIDATES)
            key = RE.RING_ENTRY_KEY_INV_USE_ITEM if slot >= 0 else RE.RING_ENTRY_KEY_NOOP
            data = (slot, 0) if slot >= 0 else (0, 0)
        elif ae == diablo_env.ActionEnum.RestoreMana:
            slot = self.game.find_restore_item(diablo_env._RESTORE_MANA_CANDIDATES)
            key = RE.RING_ENTRY_KEY_INV_USE_ITEM if slot >= 0 else RE.RING_ENTRY_KEY_NOOP
            data = (slot, 0) if slot >= 0 else (0, 0)
        elif (spell := diablo_env._ACTION_TO_SPELL.get(ae)) is not None:
            key, data = RE.RING_ENTRY_KEY_CAST_SPELL, (int(spell.value), 0)
        else:
            key = diablo_env.DiabloEnv.action_to_key(action)
        self.game.submit_key(key | RE.RING_ENTRY_F_SINGLE_TICK_PRESS, data=data)

    def _assign_stat_points(self, d):
        pts = int(d.player._pStatPts)
        if pts <= 0:
            return

        hero_class = int(d.player._pClass)
        assert hero_class == dx.HeroClass.Warrior.value, \
            f"unsupported hero class {hero_class}, only Warrior is supported"

        strategy = getattr(self, '_stat_strategy', self._warrior_dexterity_rush)
        str_pts, dex_pts, vit_pts = strategy(d, pts)

        data1 = (str_pts & 0xFF) | ((dex_pts & 0xFF) << 16) | ((vit_pts & 0xFF) << 24)
        self.game.submit_key(
            ring.RingEntryType.RING_ENTRY_KEY_STAT_ASSIGN |
            ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS,
            data=(data1, 0))
        print(f"agent {self._tick_count}: level up {int(d.player._pLevel)}, assign exp points to "
              f"str {int(d.player._pBaseStr)}+{str_pts}, "
              f"mag {int(d.player._pBaseMag)}+0, "
              f"dex {int(d.player._pBaseDex)}+{dex_pts}, "
              f"vit {int(d.player._pBaseVit)}+{vit_pts}",
              file=self.log)

    @staticmethod
    def _warrior_dexterity_rush(d, pts):
        """Warrior stat allocation - dexterity rush variant.
        Levels 2-11:  fill Dex to cap (60) - improves to-hit.
        Levels 12-30: 3 Str + 2 Vit per level - armor reqs + HP.
        Levels 31+:   fill Str to cap (250), remainder to Vit."""
        lvl     = int(d.player._pLevel)
        dex     = int(d.player._pBaseDex)
        str_val = int(d.player._pBaseStr)
        vit_val = int(d.player._pBaseVit)

        str_pts = dex_pts = vit_pts = 0
        remaining = pts

        if lvl <= 11:
            # Fill Dex first; any overflow falls through to Str/Vit below
            dex_pts   = min(remaining, max(0, 60 - dex))
            remaining -= dex_pts

        if lvl <= 30 and remaining > 0:
            # 3 Str : 2 Vit ratio per 5 points
            s         = min(3, remaining, max(0, 250 - str_val))
            v         = min(remaining - s, max(0, 100 - vit_val))
            str_pts  += s
            vit_pts  += v
            remaining -= s + v
        elif remaining > 0:
            # Level 31+: fill Str to cap, remainder to Vit
            s         = min(remaining, max(0, 250 - str_val))
            str_pts  += s
            remaining -= s
            vit_pts  += min(remaining, max(0, 100 - vit_val))

        return str_pts, dex_pts, vit_pts

    @staticmethod
    def _warrior_str_vit(d, pts):
        """Warrior stat allocation - strength and vitality split.
        3 Str + 2 Vit every level until Vit caps at 100, then all Str.
        Skips Dex entirely - trades to-hit for early HP and faster gear access."""
        str_val = int(d.player._pBaseStr)
        vit_val = int(d.player._pBaseVit)

        str_pts = vit_pts = 0
        remaining = pts

        vit_room = max(0, 100 - vit_val)
        if vit_room > 0:
            v        = min(remaining, vit_room, 2)
            s        = min(remaining - v, max(0, 250 - str_val), 3)
            vit_pts  = v
            str_pts  = s
            remaining -= v + s

        str_pts += min(remaining, max(0, 250 - str_val - str_pts))

        return str_pts, 0, vit_pts

    _STAT_ALLOC_RE = re.compile(r'(\d+)([smdv])')

    @staticmethod
    def validate_stat_strategy(s):
        """Validate a stat strategy string; return it unchanged or raise ValueError.

        Named presets pass through.  Per-level spec format:
          'RANGE=ALLOC[,RANGE=ALLOC...]'
          RANGE - 'N-M', 'N+', or 'N'
          ALLOC - concatenated stat specs: '2s3v', '5s', etc.
                  letters: s=str m=mag d=dex v=vit
                  each count 1-5; all counts must sum to 5
        """
        if s in AgentAI._STAT_STRATEGIES:
            return s
        covered = set()
        for entry in s.split(','):
            if '=' not in entry:
                raise ValueError("stat-strategy: missing '=' in segment %r" % entry)
            lvl_part, alloc_str = entry.split('=', 1)
            lvl_part  = lvl_part.strip()
            alloc_str = alloc_str.strip()
            try:
                if lvl_part.endswith('+'):
                    lo = int(lvl_part[:-1])
                    hi = 16
                elif '-' in lvl_part:
                    a, b = lvl_part.split('-', 1)
                    lo, hi = int(a), int(b)
                    if lo > hi:
                        raise ValueError("stat-strategy: bad range %r" % lvl_part)
                else:
                    lo = hi = int(lvl_part)
            except ValueError as exc:
                if 'stat-strategy' in str(exc):
                    raise
                raise ValueError("stat-strategy: bad level range %r" % lvl_part) from exc
            covered.update(range(lo, min(hi, 16) + 1))
            specs = AgentAI._STAT_ALLOC_RE.findall(alloc_str)
            if not specs:
                raise ValueError("stat-strategy: no stat specs in %r" % alloc_str)
            for n_str, _ in specs:
                n = int(n_str)
                if not (1 <= n <= 5):
                    raise ValueError(
                        "stat-strategy: each count must be 1-5, got %d in %r" % (n, alloc_str))
            total = sum(int(n) for n, _ in specs)
            if total != 5:
                raise ValueError(
                    "stat-strategy: counts must sum to 5, got %d in %r" % (total, alloc_str))
        missing = sorted(set(range(1, 17)) - covered)
        if missing:
            raise ValueError("stat-strategy: levels not covered: %s" % missing)
        return s

    @staticmethod
    def _compile_stat_strategy(spec):
        """Parse a validated per-level-range stat strategy string; return a strategy function.

        Format: 'RANGE=ALLOC[,RANGE=ALLOC...]'  (see validate_stat_strategy)
        Example: '1-7=2s2v1d,8+=5s'
        """
        CAPS = {'s': 250, 'm': 100, 'd': 250, 'v': 100}
        FIELD = {'s': '_pBaseStr', 'm': '_pBaseMag', 'd': '_pBaseDex', 'v': '_pBaseVit'}

        ranges = []
        for entry in spec.split(','):
            range_str, alloc_str = entry.split('=', 1)
            range_str = range_str.strip()
            alloc_str = alloc_str.strip()

            if range_str.endswith('+'):
                lo, hi = int(range_str[:-1]), None
            elif '-' in range_str:
                a, b = range_str.split('-', 1)
                lo, hi = int(a), int(b)
            else:
                lo = hi = int(range_str)

            alloc = [(letter, int(n))
                     for n, letter in AgentAI._STAT_ALLOC_RE.findall(alloc_str)]
            ranges.append((lo, hi, alloc))

        def strategy(d, pts):
            lvl = int(d.player._pLevel)
            alloc = None
            for lo, hi, a in ranges:
                if lvl >= lo and (hi is None or lvl <= hi):
                    alloc = a
                    break
            if alloc is None:
                # No range matched - fallback: all to str, then vit
                alloc = [('s', pts), ('v', pts)]

            result = {'s': 0, 'm': 0, 'd': 0, 'v': 0}
            remaining = pts
            for letter, count in alloc:
                cap = CAPS[letter]
                cur = int(getattr(d.player, FIELD[letter]))
                give = min(count, remaining, max(0, cap - cur))
                result[letter] += give
                remaining -= give

            # Spill remainder to each stat in alloc order that still has room
            if remaining > 0:
                for letter, _ in alloc:
                    cap = CAPS[letter]
                    cur = int(getattr(d.player, FIELD[letter]))
                    give = min(remaining, max(0, cap - cur - result[letter]))
                    result[letter] += give
                    remaining -= give
                    if remaining == 0:
                        break

            return result['s'], result['d'], result['v']

        return strategy

    _STAT_STRATEGIES = {
        'dex-rush': _warrior_dexterity_rush.__func__,
        'str-vit':  _warrior_str_vit.__func__,
    }

    _REPAIR_SLOTS = (
        (dx.inv_item.INVITEM_HAND_LEFT.value,  "in left hand"),
        (dx.inv_item.INVITEM_HAND_RIGHT.value, "in right hand"),
        (dx.inv_item.INVITEM_CHEST.value,      "on chest"),
        (dx.inv_item.INVITEM_HEAD.value,       "on head"),
    )

    def _repair_gear(self, d):
        RE = ring.RingEntryType
        for cii, slot_name in self._REPAIR_SLOTS:
            item = d.player.InvBody[cii]
            # Item::clear() only sets _itype=None; durability fields stay as
            # stale garbage after the item breaks in combat. Skip empty slots.
            if int(item._itype) == dx.ItemType.None_.value:
                continue
            max_dur = int(item._iMaxDur)
            cur_dur = int(item._iDurability)
            if max_dur <= 0 or cur_dur >= max_dur:
                continue
            if cur_dur / max_dur < self.repair_threshold:
                self.game.submit_key(
                    RE.RING_ENTRY_KEY_INV_REPAIR_ITEM |
                    RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
                    data=(cii, 0))
                name = bytes(item._iName).rstrip(b'\x00').decode('ascii', errors='replace')
                print(f"agent {self._tick_count}: repairing '{name}' {slot_name}, dur {cur_dur}/{max_dur}",
                      file=self.log)


    def _drop_and_mask(self, d, cii):
        """Drop inventory item cii. data2=1 tells the engine to set _iMasked on the
        floor item after a successful drop."""
        RE = ring.RingEntryType
        self.game.submit_key(
            RE.RING_ENTRY_KEY_INV_DROP_ITEM | RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
            data=(cii, 1))

    def _evaluate_misc_item(self, d, item, seed, name):
        """Evaluate a Misc-type item and queue an action or keep it silently."""
        player       = d.player
        imisc      = int(item._iMiscId)
        imisc_book = dx.item_misc_id.IMISC_BOOK.value

        if imisc == imisc_book:
            if int(player._pMagic) >= int(item._iMinMag):
                print(f"agent {self._tick_count}: queue read book '{name}' seed={seed}", file=self.log)
                self._queued_seeds.add(seed)
                self._action_queue.append({'action': 'use', 'seed': seed, 'name': name})
            else:
                print(f"agent {self._tick_count}: queue drop book '{name}' seed={seed}"
                      f" - magic {int(player._pMagic)} < {int(item._iMinMag)}", file=self.log)
                self._queued_seeds.add(seed)
                self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
            return

        if _is_scroll_misc(imisc):
            spellid          = int(item._iSpell)
            spellid_healing  = dx.SpellID.Healing.value
            spellid_identify = dx.SpellID.Identify.value
            spellid_portal   = dx.SpellID.TownPortal.value

            if spellid == spellid_healing:
                # Keep all - primary survival kit, never drop.
                return

            if spellid == spellid_identify:
                cap = 2
                if _count_scroll_spellid(player, spellid) > cap:
                    print(f"agent {self._tick_count}: queue drop '{name}' seed={seed}"
                          f" - scroll surplus (>{cap})", file=self.log)
                    self._queued_seeds.add(seed)
                    self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
                else:
                    # New identify scroll: immediately identify an equipped unidentified
                    # item if any, so magical bonuses take effect right away.
                    self._try_identify_with_new_scroll(d)
                return

            if spellid == spellid_portal:
                cap = 2
                if _count_scroll_spellid(player, spellid) > cap:
                    print(f"agent {self._tick_count}: queue drop '{name}' seed={seed}"
                          f" - scroll surplus (>{cap})", file=self.log)
                    self._queued_seeds.add(seed)
                    self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
                return

            if _is_combat_spell_scroll(spellid):
                # TODO: once policy is set, keep up to N per spell type.
                return

            # All other scrolls: drop.
            print(f"agent {self._tick_count}: queue drop '{name}' seed={seed}"
                  f" - unused scroll spellid={spellid}", file=self.log)
            self._queued_seeds.add(seed)
            self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
            return

        imisc_heal     = dx.item_misc_id.IMISC_HEAL.value
        imisc_fullheal = dx.item_misc_id.IMISC_FULLHEAL.value
        imisc_rejuv    = dx.item_misc_id.IMISC_REJUV.value
        imisc_fullrejuv= dx.item_misc_id.IMISC_FULLREJUV.value
        imisc_mana     = dx.item_misc_id.IMISC_MANA.value
        imisc_fullmana = dx.item_misc_id.IMISC_FULLMANA.value

        if imisc in (imisc_heal, imisc_fullheal, imisc_rejuv, imisc_fullrejuv):
            # Always keep healing and rejuvenation potions.
            return

        if imisc in (imisc_mana, imisc_fullmana):
            # TODO: enable keep_mana_potions once the agent uses spells heavily.
            if self.keep_mana_potions:
                return
            print(f"agent {self._tick_count}: queue drop '{name}' seed={seed}"
                  f" - mana potion, not keeping", file=self.log)
            self._queued_seeds.add(seed)
            self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
            return

        # All other Misc types (gold, runes, quest items, etc.): skip silently.

    def _try_identify_with_new_scroll(self, d):
        """Called when a new scroll of identify is acquired.
        Queues an identify action for the first unidentified equipped item, if any."""
        player = d.player
        target = _find_unidentified_equipped(player)
        if target is None:
            return
        target_seed, target_name = target
        if target_seed in self._queued_seeds:
            return
        scroll_cii = _find_scroll_of_identify(player)
        if scroll_cii is None:
            return
        print(f"agent {self._tick_count}: queue identify equipped '{target_name}'"
              f" seed={target_seed} (identify scroll acquired)", file=self.log)
        self._queued_seeds.add(target_seed)
        self._action_queue.append(
            {'action': 'identify', 'seed': target_seed, 'name': target_name,
             'scroll_cii': scroll_cii})

    def _locate_dropped_items(self, d):
        """Scan all active floor items for _iMasked set (dropped by agent).
        Apply observation mask and pathfinder-skip so the model ignores them.
        _masked_positions is the stable source; pathfinder._skip_items may be
        rebound by set_goal(), so we re-sync it from _masked_positions each tick."""
        item_flag = diablo_state.EnvironmentFlag.Item.value
        # Re-sync in case pathfinder reset its skip set (set_goal rebinds, not .clear())
        self._pathfinder._skip_items.update(self._masked_positions)
        for i in range(int(d.ActiveItemCount.value)):
            idx = int(d.ActiveItems[i])
            item = d.Items[idx]
            if not item._iMasked:
                continue
            p = (int(item.position.x), int(item.position.y))
            newly_skipped = p not in self._masked_positions
            self._masked_positions.add(p)
            self._pathfinder._skip_items.add(p)
            self._mask_tile(self.cur_level, p, clear=item_flag)
            if newly_skipped:
                self._pathfinder._path = []
                seed = int(item._iSeed)
                name = _item_name(item)
                print(f"agent {self._tick_count}: masked drop '{name}' seed={seed} at {p}", file=self.log)

    def _equip(self, inv_cii, body_cii):
        """Move inventory item inv_cii to body slot body_cii (engine handles swap)."""
        RE = ring.RingEntryType
        self.game.submit_key(
            RE.RING_ENTRY_KEY_INV_MOVE_ITEM | RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
            data=(inv_cii, body_cii))

    def _log_identified_items(self, d, seeds, inv_curr):
        """Log revealed magical properties for all items in seeds that are now identified."""
        player     = d.player
        itype_none = dx.ItemType.None_.value
        INV_FIRST  = dx.inv_item.INVITEM_INV_FIRST.value
        all_seeds  = {s for (_, s, _) in inv_curr}
        for seed in seeds:
            if seed not in all_seeds:
                continue
            item = None
            for i in range(INV_FIRST):
                eq = player.InvBody[i]
                if int(eq._itype) != itype_none and int(eq._iSeed) == seed:
                    item = eq
                    break
            if item is None:
                for i in range(int(player._pNumInv)):
                    it = player.InvList[i]
                    if int(it._iSeed) == seed:
                        item = it
                        break
            if item is None:
                for i in range(8):
                    it = player.SpdList[i]
                    if int(it._itype) != itype_none and int(it._iSeed) == seed:
                        item = it
                        break
            if item is None:
                continue
            print(f"agent {self._tick_count}: revealed '{_item_name(item)}'"
                  f" seed={seed} [{_identified_label(item)}]", file=self.log)

    def _evaluate_and_queue(self, d, seed):
        """Evaluate one InvList or belt item immediately and enqueue an equip or drop action."""
        player       = d.player
        INV_FIRST    = dx.inv_item.INVITEM_INV_FIRST.value
        BELT_FIRST   = dx.inv_item.INVITEM_BELT_FIRST.value
        itype_none   = dx.ItemType.None_.value
        hero_class   = int(player._pClass)
        is_better_fn = _IS_BETTER_FOR_CLASS[hero_class]

        item = None
        for i in range(int(player._pNumInv)):
            if int(player.InvList[i]._iSeed) == seed:
                item = player.InvList[i]
                break
        if item is None:
            for i in range(8):
                it = player.SpdList[i]
                if int(it._itype) != itype_none and int(it._iSeed) == seed:
                    item = it
                    break
        if item is None:
            return

        name = _item_name(item)

        if int(item._itype) == dx.ItemType.Misc.value:
            self._evaluate_misc_item(d, item, seed, name)
            return

        if int(item._itype) in _SKIP_ITYPE:
            return

        body_cii = _item_body_slot(item)
        if body_cii is None:
            return

        max_dur = int(item._iMaxDur)
        cur_dur = int(item._iDurability)
        if 0 < max_dur and cur_dur / max_dur < self.repair_threshold:
            print(f"agent {self._tick_count}: queue drop '{name}' seed={seed}"
                  f" - dur {cur_dur}/{max_dur} below threshold", file=self.log)
            self._queued_seeds.add(seed)
            self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
            return

        if not _can_equip(item, player):
            req = (f"MinStr={int(item._iMinStr)} MinMag={int(item._iMinMag)}"
                   f" MinDex={int(item._iMinDex)}")
            print(f"agent {self._tick_count}: queue drop '{name}' seed={seed} - stat req not met ({req})", file=self.log)
            self._queued_seeds.add(seed)
            self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
            return

        if (int(item._iMagical) != dx.item_quality.ITEM_QUALITY_NORMAL.value
                and not item._iIdentified):
            if int(item._iLoc) in _JEWELRY_ILOC:
                # Jewelry has no base stats; all value is in _iPL* which only apply
                # after identification. Must identify to know if worth wearing.
                scroll_cii = _find_scroll_of_identify(player)
                if scroll_cii is None:
                    print(f"agent {self._tick_count}: queue drop '{name}' seed={seed}"
                          f" - jewelry not identified, no scroll", file=self.log)
                    self._queued_seeds.add(seed)
                    self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})
                else:
                    print(f"agent {self._tick_count}: queue identify '{name}' seed={seed}", file=self.log)
                    self._queued_seeds.add(seed)
                    self._action_queue.append(
                        {'action': 'identify', 'seed': seed, 'name': name,
                         'scroll_cii': scroll_cii})
                return
            # Non-jewelry: base damage (_iMinDam/_iMaxDam) and base AC (_iAC) apply
            # even without identification. Scoring uses only these base stats so the
            # comparison is valid - fall through to normal scoring.

        # For rings: if left slot occupied, try right slot instead.
        if (int(item._iLoc) == dx.item_equip_type.ILOC_RING.value and
                int(player.InvBody[body_cii]._itype) != dx.ItemType.None_.value):
            body_cii = dx.inv_item.INVITEM_RING_RIGHT.value

        pending = self._pending_equip.get(body_cii)
        is_better, new_score, new_label, old_score, old_label, old_name = \
            is_better_fn(item, player, body_cii, pending)

        if is_better:
            id_tag = "" if item._iIdentified else " unidentified"
            if old_label == "empty":
                print(f"agent {self._tick_count}: queue equip '{name}' seed={seed} [{new_label}]{id_tag} - empty slot",
                      file=self.log)
            else:
                print(f"agent {self._tick_count}: queue equip '{name}' seed={seed} [{new_label}]{id_tag}"
                      f" over '{old_name}' [{old_label}]", file=self.log)
            # Set _pending_equip at queue-time so subsequent evaluations compare against
            # this winner. Not added to any exclusion set: displaced gear must re-enter evaluation.
            self._pending_equip[body_cii] = (seed, new_score, name)
            self._queued_seeds.add(seed)
            self._action_queue.append(
                {'action': 'equip', 'seed': seed, 'name': name, 'body_cii': body_cii})
        else:
            if old_name is None:
                print(f"agent {self._tick_count}: queue drop '{name}' seed={seed} - class filter [{new_label}]",
                      file=self.log)
            elif new_score == old_score:
                print(f"agent {self._tick_count}: queue drop '{name}' seed={seed} [{new_label}]"
                      f" - tied with '{old_name}' [{old_label}]", file=self.log)
            else:
                print(f"agent {self._tick_count}: queue drop '{name}' seed={seed} [{new_label}]"
                      f" - worse than '{old_name}' [{old_label}]", file=self.log)
            self._queued_seeds.add(seed)
            self._action_queue.append({'action': 'drop', 'seed': seed, 'name': name})

    def _execute_queued(self, d):
        """Execute the next viable action from the queue (one submission per tick).
        Both equips and drops pop immediately on submission."""
        player    = d.player
        INV_FIRST = dx.inv_item.INVITEM_INV_FIRST.value

        while self._action_queue:
            entry = self._action_queue[0]
            seed  = entry['seed']
            name  = entry['name']

            # Find item by seed; slot may have shifted due to auto-sort.
            cii = None
            for i in range(int(player._pNumInv)):
                if int(player.InvList[i]._iSeed) == seed:
                    cii = INV_FIRST + i
                    break
            if cii is None:
                itype_none = dx.ItemType.None_.value
                BELT_FIRST = dx.inv_item.INVITEM_BELT_FIRST.value
                for i in range(8):
                    it = player.SpdList[i]
                    if int(it._itype) != itype_none and int(it._iSeed) == seed:
                        cii = BELT_FIRST + i
                        break
            if cii is None and entry['action'] != 'identify':
                print(f"agent {self._tick_count}: queue discard '{name}' seed={seed} - not in inv/belt", file=self.log)
                self._action_queue.pop(0)
                self._queued_seeds.discard(seed)
                continue

            if entry['action'] == 'equip':
                body_cii = entry['body_cii']
                print(f"agent {self._tick_count}: equip '{name}' seed={seed}", file=self.log)
                self._action_queue.pop(0)
                self._queued_seeds.discard(seed)
                # _pending_equip[body_cii] is NOT cleared here - only when engine confirms
                # via InvBody seed match. Intermediate equips must not clear it prematurely.
                self._equip(cii, body_cii)
                return
            elif entry['action'] == 'identify':
                # cii may be None if item is in a body slot (already equipped).
                target_cii = cii
                if target_cii is None:
                    itype_none = dx.ItemType.None_.value
                    for i in range(INV_FIRST):  # body slots 0..INV_FIRST-1
                        eq = player.InvBody[i]
                        if int(eq._itype) != itype_none and int(eq._iSeed) == seed:
                            target_cii = i
                            break
                if target_cii is None:
                    print(f"agent {self._tick_count}: queue discard '{name}' seed={seed}"
                          f" - identify: not found", file=self.log)
                    self._action_queue.pop(0)
                    self._queued_seeds.discard(seed)
                    continue
                scroll_cii = _find_scroll_of_identify(player)
                if scroll_cii is None:
                    print(f"agent {self._tick_count}: drop '{name}' seed={seed}"
                          f" - identify: scroll gone", file=self.log)
                    self._action_queue.pop(0)
                    self._queued_seeds.discard(seed)
                    if cii is not None:  # only drop if item is in inv, not if equipped
                        self._drop_and_mask(d, cii)
                    return
                print(f"agent {self._tick_count}: identify '{name}' seed={seed}", file=self.log)
                self._action_queue.pop(0)
                self._queued_seeds.discard(seed)
                self._identify_pending_seeds.add(seed)
                RE = ring.RingEntryType
                self.game.submit_key(
                    RE.RING_ENTRY_KEY_INV_IDENTIFY_ITEM | RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
                    data=(scroll_cii, target_cii))
                return
            elif entry['action'] == 'use':
                print(f"agent {self._tick_count}: use '{name}' seed={seed}", file=self.log)
                self._action_queue.pop(0)
                self._queued_seeds.discard(seed)
                RE = ring.RingEntryType
                self.game.submit_key(
                    RE.RING_ENTRY_KEY_INV_USE_ITEM | RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
                    data=(cii, 0))
                return
            else:
                print(f"agent {self._tick_count}: drop '{name}' seed={seed}", file=self.log)
                self._action_queue.pop(0)
                self._queued_seeds.discard(seed)
                self._drop_and_mask(d, cii)
                return


def _monster_within_radius(d, pos, radius):
    """Return True if any live monster tile is within a square of @radius around pos."""
    for dx_val in range(-radius, radius + 1):
        for dy_val in range(-radius, radius + 1):
            if dx_val == 0 and dy_val == 0:
                continue
            p = (pos[0] + dx_val, pos[1] + dy_val)
            try:
                if d.dMonster[p] > 0:
                    return True
            except Exception:
                pass
    return False


def _monster_adjacent(d, pos):
    return _monster_within_radius(d, pos, 1)
