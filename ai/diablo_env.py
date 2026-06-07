"""
diablo_env.py - Diablo Gymnasium Environment

This module provides a custom Gymnasium environment that integrates
with a running Diablo (DevilutionX) instance for reinforcement
learning. It enables an agent to navigate, interact, and act
strategically within the game world, using structured observations and
rewards based on exploration, combat, and game state. Supports
training and evaluation with reproducible seeding and action masking
for constrained movement.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

from abc import ABC, abstractmethod
import collections
import copy
import enum
import gymnasium as gym
import numpy as np
import os
import sys
import time

import devilutionx as dx
import diablo_state
import maze
import ring

class ActionEnum(enum.Enum):
    # Indices are stable; pre-trained models rely on them. Append only.
    # v1 set: 8 movement + Stand + PrimaryAction + SecondaryAction.
    Walk_N          = 0
    Walk_NE         = enum.auto()
    Walk_E          = enum.auto()
    Walk_SE         = enum.auto()
    Walk_S          = enum.auto()
    Walk_SW         = enum.auto()
    Walk_W          = enum.auto()
    Walk_NW         = enum.auto()
    Stand           = enum.auto()
    # Attack monsters, talk to towners, lift and place inventory items.
    PrimaryAction   = enum.auto()
    # Open chests, interact with doors, pick up items.
    SecondaryAction = enum.auto()
    # v2 set: 2 restore + 7 cast spells.
    RestoreHealth   = enum.auto()
    RestoreMana     = enum.auto()
    CastFirebolt    = enum.auto()
    CastChargedBolt = enum.auto()
    CastFireWall    = enum.auto()
    CastStoneCurse  = enum.auto()
    CastManaShield  = enum.auto()
    CastPhasing     = enum.auto()
    CastFireball    = enum.auto()

# Priority order for the two restore actions. Searched belt then inventory
# inside DiabloGame.find_restore_item; smallest-first by convention so the
# agent does not waste a full-heal when a small_hp would do.
_RESTORE_HEALTH_CANDIDATES = [
    dx.item_misc_id.IMISC_HEAL,
    dx.item_misc_id.IMISC_SCROLL,      # matched as Scroll of Healing
    dx.item_misc_id.IMISC_FULLHEAL,
    dx.item_misc_id.IMISC_REJUV,
    dx.item_misc_id.IMISC_FULLREJUV,
]
_RESTORE_MANA_CANDIDATES = [
    dx.item_misc_id.IMISC_MANA,
    dx.item_misc_id.IMISC_FULLMANA,
    dx.item_misc_id.IMISC_REJUV,
    dx.item_misc_id.IMISC_FULLREJUV,
]

# ActionEnum -> SpellID for the seven cast actions.
_ACTION_TO_SPELL = {
    ActionEnum.CastFirebolt:    dx.SpellID.Firebolt,
    ActionEnum.CastChargedBolt: dx.SpellID.ChargedBolt,
    ActionEnum.CastFireWall:    dx.SpellID.FireWall,
    ActionEnum.CastStoneCurse:  dx.SpellID.StoneCurse,
    ActionEnum.CastManaShield:  dx.SpellID.ManaShield,
    ActionEnum.CastPhasing:     dx.SpellID.Phasing,
    ActionEnum.CastFireball:    dx.SpellID.Fireball,
}
# SpellID int value -> ActionEnum (reverse of _ACTION_TO_SPELL)
_SPELL_TO_ACTION = {int(v.value): k for k, v in _ACTION_TO_SPELL.items()}

class ActionMask(enum.Enum):
    MASK_TRIGGERS      = 1<<0
    MASK_CLOSED_DOORS  = 1<<1
    MASK_WALLS         = 1<<2
    MASK_OTHER_SOLIDS  = 1<<3


class DiabloEnv(gym.Env):
    MASK_EVERYTHING = (ActionMask.MASK_TRIGGERS.value |
                       ActionMask.MASK_CLOSED_DOORS.value |
                       ActionMask.MASK_WALLS.value |
                       ActionMask.MASK_OTHER_SOLIDS.value)

    @staticmethod
    @abstractmethod
    def tune_config(env_config):
        """Tune configuration before instantiation"""
        pass

    @staticmethod
    def action_to_key(action):
        match ActionEnum(action):
            case ActionEnum.Walk_NE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case ActionEnum.Walk_E:
                key = (ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case ActionEnum.Walk_SE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case ActionEnum.Walk_S:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN)
            case ActionEnum.Walk_SW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case ActionEnum.Walk_W:
                key = (ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case ActionEnum.Walk_NW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case ActionEnum.Walk_N:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP)
            case ActionEnum.Stand:
                key = (ring.RingEntryType.RING_ENTRY_KEY_NOOP)
            case ActionEnum.PrimaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_A)
            case ActionEnum.SecondaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_X)
            case ActionEnum.RestoreHealth | ActionEnum.RestoreMana:
                # Dispatched in step() via DiabloGame.find_restore_item.
                # NOOP is returned as a fallback so static callers (e.g. the
                # bot, which only emits v1 actions anyway) do not crash if
                # a v2 action ever leaks through.
                key = (ring.RingEntryType.RING_ENTRY_KEY_NOOP)
            case ActionEnum.CastFirebolt | ActionEnum.CastChargedBolt \
                 | ActionEnum.CastFireWall | ActionEnum.CastStoneCurse \
                 | ActionEnum.CastManaShield | ActionEnum.CastPhasing \
                 | ActionEnum.CastFireball:
                # step() packs SpellID into data1 before submitting; this
                # branch only returns the key so the fallback above stays
                # consistent.
                key = (ring.RingEntryType.RING_ENTRY_KEY_CAST_SPELL)
        return key

    # Return neighbours in cardinal CW directions:
    # N, NE, E, SE, S, SW, W, NW
    @staticmethod
    def get_matrix_neighbors(matrix, i, j):
        rows, cols = matrix.shape
        neighbors = np.full(8, -1, dtype=np.int32)
        ind = 0
        for y in range(j-1, j+2):
            for x in range(i-1, i+2):
                # Exclude center element
                if (x, y) == (i, j):
                    continue
                ind += 1
                if x < 0 or x >= rows or y < 0 or y >= cols:
                    continue
                neighbors[ind-1] = matrix[y, x]

        # Remap array from:
        # "NW, N, NE, W, E, SW, S, SE" to "N, NE, E, SE, S, SW, W, NW"
        # to have a simple mapping to ActionEnum
        remap = [1, 2, 4, 7, 6, 5, 3, 0]
        return np.take(neighbors, remap)

    @staticmethod
    def action_mask(d, env, what):
        """
        Computes an action mask for the action space using the state information.
        For example can be called to mask all actions to go to triggers:
             DiabloEnv.action_mask(d, env,
                                   ActionMask.MASK_TRIGGERS.value)
        """
        mask = np.full(len(ActionEnum), 1, dtype=np.int8)
        player_pos = diablo_state.player_position(d)

        neighbors = DiabloEnv.get_matrix_neighbors(env, *player_pos)
        for i, tile in enumerate(neighbors):
            if tile < 0:
                # Block in the direction beyond map
                mask[i] = 0
                continue

            if what & ActionMask.MASK_TRIGGERS.value and \
               tile & diablo_state.EnvironmentFlag.PrevTrigger.value:
                # Block in the direction of previous trigger, so forbid
                # escaping to the town
                # TODO: this blocks all the trigger including stairs to the
                # next level
                mask[i] = 0
            elif what & ActionMask.MASK_WALLS.value and \
                 tile & diablo_state.EnvironmentFlag.Wall.value:
                mask[i] = 0
            elif what & ActionMask.MASK_CLOSED_DOORS.value and \
                 tile & diablo_state.EnvironmentFlag.DoorClosed.value:
                mask[i] = 0
            elif what & ActionMask.MASK_OTHER_SOLIDS.value and \
                 (tile & diablo_state.EnvironmentFlag.Barrel.value or \
                  tile & diablo_state.EnvironmentFlag.Chest.value or \
                  tile & diablo_state.EnvironmentFlag.Sarcophagus.value or \
                  tile & diablo_state.EnvironmentFlag.Crucifix.value):
                mask[i] = 0

        return mask

    def __init__(self, env_config, game, **kwargs):
        if env_config is None:
            raise ValueError("env_config must be provided!")
        if game is None:
            raise ValueError("game must be provided!")
        # No hierarchy
        self.num_hierarchy_levels = 1
        self.resets_cnt = 0
        self.config = env_config
        self.game = game
        self.seed = self.config['seed']
        self.initial_seed = diablo_state.fmix32(self.seed + self.config['index'])
        self.auto_reset_counter = 0
        self.paused = False
        self.view_radius = None
        self.used_goal = None
        if self.config['view-radius']:
            self.view_radius = self.config['view-radius']

        self.log_to_stdout = self.config['log-to-stdout'] \
            if 'log-to-stdout' in self.config else 0
        self.no_actions = self.config['no-actions'] \
            if 'no-actions' in self.config else 0

        if self.log_to_stdout:
            self.log = sys.stdout
        else:
            logfile = os.path.join(self.game.state_path, "env.log")
            self.log = open(logfile, "w", buffering=1)

        print(f"INSTANCE seed={self.seed}", file=self.log)

        # Initialize the rest of the state
        self.reset(seed=self.seed)

        d = self.game.safe_state
        env = diablo_state.get_environment(d, radius=self.view_radius)

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = self._build_observation_space(d, env)


    @property
    def nr_env_channels(self):
        # Explicit Goal channel for v1
        EnvFlag = diablo_state.EnvironmentFlag
        return list(EnvFlag).index(EnvFlag.Goal) + 1

    @property
    def num_actions(self):
        """Width of the action head. Old env classes (FindNextLevel, FindRandomGoal,
        ClearTheLevel, their HRL twins) use the original 8 movement + Stand +
        PrimaryAction + SecondaryAction set. Future env classes that add restore /
        spell actions override this to extend the head."""
        return ActionEnum.SecondaryAction.value + 1

    @property
    def obs_includes_old_status(self):
        """Old broadcast-plane status (monsters_cnt, hp, mode, player_x, player_y)
        concatenated as five constant H x W planes in the rl preprocessor.
        Pre-existing models depend on this being present and 5-wide."""
        return True

    @property
    def obs_includes_monster_attrs(self):
        """Per-tile monster attribute grid (W, H, 9) -- new env classes only."""
        return False

    @property
    def obs_includes_scalars(self):
        """Flat scalar vector concatenated after the CNN embedding before the
        LSTM -- new env classes only."""
        return False

    def _submit_action(self, action):
        """Translate a discrete action into a (key, data) pair and submit once.

        The dispatch lives in this single helper so every action goes through
        exactly one submit_key call, and ring-protocol knowledge (key bits,
        data1/data2 layout) does not leak into find_* helpers in
        diablo_state.

          - RestoreHealth / RestoreMana: ask DiabloGame.find_restore_item for
            the best matching slot (pure search). If found, submit
            INV_USE_ITEM(slot); else fall back to NOOP so the tick still
            advances (otherwise submit_key would deadlock waiting for the
            step-finished event).
          - Cast* actions: submit RING_ENTRY_KEY_CAST_SPELL with the SpellID
            packed into data1.
          - All other actions go through action_to_key (movement, Stand,
            PrimaryAction, SecondaryAction).
        """
        RE = ring.RingEntryType
        data = (0, 0)
        ae = ActionEnum(int(action))
        if ae == ActionEnum.RestoreHealth:
            slot = self.game.find_restore_item(_RESTORE_HEALTH_CANDIDATES)
            if slot >= 0:
                key, data = RE.RING_ENTRY_KEY_INV_USE_ITEM, (slot, 0)
            else:
                key = RE.RING_ENTRY_KEY_NOOP
        elif ae == ActionEnum.RestoreMana:
            slot = self.game.find_restore_item(_RESTORE_MANA_CANDIDATES)
            if slot >= 0:
                key, data = RE.RING_ENTRY_KEY_INV_USE_ITEM, (slot, 0)
            else:
                key = RE.RING_ENTRY_KEY_NOOP
        elif (spell := _ACTION_TO_SPELL.get(ae)) is not None:
            key, data = RE.RING_ENTRY_KEY_CAST_SPELL, (int(spell.value), 0)
        else:
            if ae == ActionEnum.PrimaryAction:
                right_hand = self.game.state.player.InvBody[dx.inv_item.INVITEM_HAND_RIGHT.value]
                assert right_hand._itype != dx.ItemType.Bow.value, \
                    "bow equipped - PrimaryAction range assumption (radius 1) is invalid"
            key = DiabloEnv.action_to_key(action)
        self.game.submit_key(key | RE.RING_ENTRY_F_SINGLE_TICK_PRESS, data=data)

    def _get_monster_attrs(self, d):
        """Per-tile monster attribute grid. Default returns None; new env classes
        that set obs_includes_monster_attrs override this."""
        return None

    def _get_scalars(self, d):
        """Flat scalar vector (hp, mana, potion counts, etc.). Default returns
        None; new env classes that set obs_includes_scalars override this."""
        return None

    def _build_observation_space(self, d, env):
        """Assemble the gym observation_space dict from the obs_includes_* flags.
        Computes sample arrays once at init to derive shapes."""
        spaces = {
            "env": gym.spaces.Box(low=0,
                                  high=(1 << self.nr_env_channels) - 1,
                                  shape=env.shape,
                                  dtype=np.uint32),
        }
        if self.obs_includes_old_status:
            env_status = self.get_env_status(d)
            spaces["env-status"] = gym.spaces.Box(low=0, high=0xfffff,
                                                  shape=env_status.shape,
                                                  dtype=np.uint32)
        if self.obs_includes_monster_attrs:
            ma = self._get_monster_attrs(d)
            spaces["monster_attrs"] = gym.spaces.Box(low=0.0, high=1.0,
                                                     shape=ma.shape,
                                                     dtype=np.float32)
        if self.obs_includes_scalars:
            sc = self._get_scalars(d)
            spaces["scalars"] = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=sc.shape,
                                                dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def _build_obs(self, d, env):
        """Assemble the per-step / per-reset observation dict using the same
        include flags. Old env classes get exactly the original two-key dict."""
        obss = {"env": env}
        if self.obs_includes_old_status:
            obss["env-status"] = self.get_env_status(d)
        if self.obs_includes_monster_attrs:
            obss["monster_attrs"] = self._get_monster_attrs(d)
        if self.obs_includes_scalars:
            obss["scalars"] = self._get_scalars(d)
        return obss

    def pause_game(self, pause=True):
        if self.paused != pause:
            self.paused = pause
            game_paused = diablo_state.is_game_paused(self.game.state)

            if pause ^ game_paused:
                key = ring.RingEntryType.RING_ENTRY_KEY_PAUSE | \
                    ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
                self.game.submit_key(key)

                # Wait for the PAUSE to take effect
                while game_paused == diablo_state.is_game_paused(self.game.state):
                    time.sleep(0.01)

                print("PAUSED, total R %.1f" % self.total_reward
                      if pause else "CONTINUED", file=self.log)


    def close(self):
        print("CLOSE INSTANCE", file=self.log)
        # Reset the goal position for game state
        self.game.goal_pos = (0, 0)
        self.game.stop_or_detach()

    def get_env_status(self, d):
        monsters_cnt = diablo_state.count_active_monsters(d)
        hp = d.player._pHitPoints
        mode = d.player._pmode
        player_pos = diablo_state.player_position(d)
        return np.array([monsters_cnt, hp, mode, *player_pos], dtype=np.uint32)

    def init_goal(self, d):
        # Environment of the whole dungeon
        env_whole = diablo_state.get_environment(d, show_invisible=True, show_unexplored=True)
        start_pos = diablo_state.player_position(d)

        # Goal generation depends on the environment
        goal_pos = self.generate_goal_pos(d, env_whole)

        # Get dungeon graph, doors and path from start to goal
        graph_and_path = diablo_state.get_dungeon_graph_and_path(env_whole, start_pos, goal_pos)
        regions_doors, labeled_regions, regions_path, _ = graph_and_path

        # Share goal position via game state
        self.game.goal_pos = goal_pos
        self.goal_pos = goal_pos

        self.regions_doors = regions_doors
        self.labeled_regions = labeled_regions
        self.regions_path = regions_path

    @abstractmethod
    def generate_goal_pos(self, d, env_whole):
        """Must be implemented in subclasses"""
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            episode_seed = seed
            self.seed = seed
        else:
            # Auto-reset path: hash (initial_seed + counter) via fmix32.
            # Eval resets always supply an explicit seed and never touch
            # auto_reset_counter, so the training RNG sequence is completely
            # isolated from eval resets.
            episode_seed = diablo_state.fmix32(self.initial_seed + self.auto_reset_counter)
            self.auto_reset_counter += 1

        if seed is not None or self.config.get("fixed-seed", False):
            self.resets_cnt = 0
        else:
            # When a seed is not available, we need to distinguish
            # between two resets to perform different probability
            # sampling. See determenistic_sample() and its callers for
            # details
            self.resets_cnt += 1

        if self.paused:
            # Resume first
            self.pause_game(False)

        base_config = self.config.get('dungeon-level', diablo_state.DUNGEON_LEVEL_DEFAULT)
        auto_levels = base_config.auto_levels
        stats_path  = base_config.stats_path
        if auto_levels and stats_path:
            spec = diablo_state.read_suggested_dungeon_level(stats_path)
            suggested = dict(diablo_state.parse_dungeon_level_spec(spec)) if spec else {}
            dungeon_level_config = [
                (lvl, suggested.get(lvl, diablo_state.CURRICULUM_WEIGHT_STEP)
                      if lvl in auto_levels else w)
                for lvl, w in base_config
            ]
        else:
            dungeon_level_config = list(base_config)
        dungeon_level = diablo_state.sample_dungeon_level(dungeon_level_config, episode_seed)

        if seed is not None:
            print(f"RESET seed={episode_seed} dungeon_level={dungeon_level}", file=self.log)
        else:
            print(f"RESET auto_seed={episode_seed:08x} dungeon_level={dungeon_level}", file=self.log)

        # Bits 0: seed present; bits 5:1: dungeon level for stat injection.
        seed_data = ((dungeon_level << 1) | 1, episode_seed)

        # Start new game
        key = ring.RingEntryType.RING_ENTRY_KEY_NEW | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.game.submit_key(key, data=seed_data)

        d = self.game.safe_state

        # First init all goal related members
        self.init_goal(d)

        # Keep compatibility with old environments
        goal_pos = None
        if self.used_goal:
            goal_pos = self.goal_pos
            if self.config.get('gui'):
                # Set the goal in the game for GUI representation
                key = ring.RingEntryType.RING_ENTRY_KEY_SET_GOAL | \
                      ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
                self.game.submit_key(key, data=goal_pos)
        env = diablo_state.get_environment(d, radius=self.view_radius,
                                           goal_pos=goal_pos)

        obj_cnt = diablo_state.count_active_objects(d)
        closed_doors_ids = diablo_state.get_closed_doors_ids(d)
        items_cnt = diablo_state.count_active_items(d)
        monsters_cnt = diablo_state.count_active_monsters(d)
        mon_total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_state.count_explored_tiles(d)
        hp = d.player._pHitPoints
        pos = diablo_state.player_position(d)

        self.prev_obj_cnt = obj_cnt
        self.prev_closed_doors_ids = closed_doors_ids
        self.opened_doors_ids = []
        self.prev_items_cnt = items_cnt
        self.prev_monsters_cnt = monsters_cnt
        self.prev_mon_total_hp = mon_total_hp
        self.prev_explored_cnt = explored_cnt
        self.prev_hp = hp
        self.total_reward = 0.0
        self.exploration_reward = 0.0
        self.episode_success = False
        self.hist_player_pos = collections.deque([pos], maxlen=3)
        self.last_player_pos = pos
        self.last_steps_cnt = 0
        self.steps_cnt = 0

        # Starting dungeon level
        self.start_dungeon_level = d.player.plrlevel

        obss = self._build_obs(d, env)
        info = {"env-counters": (self.resets_cnt, self.steps_cnt)}
        return obss, info

    EPISODE_TIMEOUT = 3000
    STUCK_TIMEOUT   = 300

    def is_agent_timedout(self):
        # Should cover most of the cases
        return self.steps_cnt >= self.EPISODE_TIMEOUT

    def is_agent_stuck(self, d, was_exploring):
        if self.is_agent_timedout():
            return True

        p = diablo_state.player_position(d)

        # Update counter and position if the agent was exploring or moved
        # a significant distance away
        if was_exploring or np.any(np.abs(self.last_player_pos - np.asarray(p)) > 10):
            self.last_steps_cnt = self.steps_cnt
            self.last_player_pos = p
            return False

        # Check if the agent was "doing nothing" for some time
        return self.steps_cnt - self.last_steps_cnt >= self.STUCK_TIMEOUT

    def get_exploration_reward_coef(self, d, min_reward):
        pos = diablo_state.player_position(d)
        current_region = self.labeled_regions[pos]
        if current_region == 0:
            # Region is invalid, when the player is in the doorway,
            # so use the previous position from the history
            prev_pos = self.hist_player_pos[1]
            current_region = self.labeled_regions[prev_pos]
            assert current_region != 0

        doors_here = self.regions_doors[current_region]

        if True:
            # Leave only the door that leads to the goal;
            # no multi-pole attraction.
            doors_here = { k:v for k,v in doors_here.items() if v }
            assert len(doors_here) <= 1
            if len(doors_here) == 0:
                # No doors in the region leading to the goal. Return
                # minimal reward if we are in the region which is not
                # on the path
                if self.regions_path[-1] != current_region:
                    return min_reward

        # Add the goal to the doors if player is on the last region on
        # the path
        if self.regions_path[-1] == current_region:
            # Add the goal
            doors_here = copy.deepcopy(doors_here)
            doors_here[self.goal_pos] = True
        assert len(doors_here)

        # Controls decay smoothness
        lambda_factor = 0.3
        # Only door which is on the path does not have any discount
        discount = lambda door: 0 if doors_here.get(door, False) else 2
        dists = np.array([(maze.euclidean_dist(pos, door) + discount(door)) * lambda_factor
                          for door in doors_here])
        reward = 1 / (1 + dists.min())
        # Every region that is on the path and closer to the goal
        # provides more reward
        reward += self.regions_path.index(current_region) \
            if current_region in self.regions_path else 0.0

        return max(min_reward, reward)

    def evaluate_step(self, d, env, action):
        obj_cnt = diablo_state.count_active_objects(d)
        closed_doors_ids = diablo_state.get_closed_doors_ids(d)
        items_cnt = diablo_state.count_active_items(d)
        monsters_cnt = diablo_state.count_active_monsters(d)
        mon_total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_state.count_explored_tiles(d)
        hp = d.player._pHitPoints
        player_pos = diablo_state.player_position(d)

        # Exploration reward
        # +1. calculate number of explored tiles, compare with a previous step
        # 2. find a way to estimate full exploration
        # 3. penalty for revisiting explored tiles (needed?)

        # Combat reward
        # +1. calculate number of all monsters HP, compare with previous step,
        #     reward for reduced, i.e. monster damage
        # +2.  reward for killing monster
        # 2.1. huge reward once last monster is killed

        # Survival & Caution Reward
        # +1.  penalty for taking damage
        # +1.1 higher penalty when HP drops beyond threshold
        # 2.  reward for staying at high health

        # Strategic Encouragement
        # +1. reward for collecting items
        # +2. penalty for attacking walls or wasting actions
        # 3. reward for reaching the next level stairs after full explorations

        # Endgame conditions
        # +1. Huge penalty for dying
        # +2. Huge penalty for escaping to the town
        # +3. Huge penalty for descending if current dungeon is not cleared


        #### ITEMS PICK:   ON PRESS
        #### DOORS OPEN:   ON PRESS
        #### OBJECTS OPEN: ON RELEASE
        ####  PM_ATTACK:   ON PRESS
        #### TODO: handle PM_GOTHIT, PM_ATTACK
        ####

        truncated = False
        done = False
        # The initial value must be a zero integer. I need a simple
        # marker to indicate that @reward was changed in many if-blocks below.
        # It seems the easiest way is to set it to an integer initially and
        # propagate it to a float on any update. This will be an ideal
        # marker that @reward was updated and that the agent was exploring.
        reward = int(0)

        if diablo_state.is_player_dead(d):
            # We are dead, game over
            reward = -100.0
            done = True
            print("Death, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel < self.start_dungeon_level:
            # Done with this episode with 0 reward if agent has
            # stepped into a trigger to escape
            reward = 0.0
            done = True
            print("Escape, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel > self.start_dungeon_level:
            reward = 50.0
            done = True
            self.episode_success = True
            print("Goal, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel != self.start_dungeon_level:
            # Done with this episode with 0 reward if agent has
            # stepped into a trigger to escape
            reward = 0.0
            done = True
            print("Escape, R %.1f" % reward, file=self.log)
        else:
            if hp < self.prev_hp:
                # More damage to health - more penalty
                # from 0 to -100
                change = (hp - self.prev_hp) / self.prev_hp
                reward += change * 100.0
                print("Damage taken, R %.1f" % reward, file=self.log)
                self.prev_hp = hp
            if explored_cnt > self.prev_explored_cnt:
                # Exploration
                if self.config["exploration-door-attraction"]:
                    # Reward increases when the agent moves closer to
                    # unexplored doors.
                    min_reward = 0.1
                    exploration_reward = self.get_exploration_reward_coef(d, min_reward)
                    # Scale
                    exploration_reward *= 10.0
                    if self.config["exploration-door-backtrack-penalty"]:
                        # Convert exploration reward to a penalty if
                        # reward starts decreasing, in other words
                        # proximity to unexplored door increases.
                        if exploration_reward >= self.exploration_reward:
                            self.exploration_reward = exploration_reward
                        else:
                            exploration_reward = -min_reward
                else:
                    # Flat exploration reward
                    exploration_reward = 1.0
                reward += exploration_reward
                self.prev_explored_cnt = explored_cnt
                print("Exploration, R %.1f" % reward, file=self.log)
            if obj_cnt < self.prev_obj_cnt:
                # Reward for opening chests, crashing barrels, etc
                reward += (self.prev_obj_cnt - obj_cnt) * 5.0
                self.prev_obj_cnt = obj_cnt
                print("Activate object, R %.1f" % reward, file=self.log)
            if len(closed_doors_ids) != len(self.prev_closed_doors_ids):
                if len(closed_doors_ids) < len(self.prev_closed_doors_ids):
                    opened = list(set(self.prev_closed_doors_ids) - set(closed_doors_ids))
                    # Exclude reopened doors
                    opened = [o for o in opened if o not in self.opened_doors_ids]
                    self.opened_doors_ids.extend(opened)
                    if len(opened):
                        reward += len(opened) * 5.0
                        print("Open door, R %.1f" % reward, file=self.log)
                self.prev_closed_doors_ids = closed_doors_ids
            if items_cnt != self.prev_items_cnt:
                if items_cnt < self.prev_items_cnt:
                    # Reward for collecting items. Item number can increase if
                    # dropped from a chest, but actions can't be applied on a
                    # closed chest and picking up an item, so object counter
                    # should be checked first
                    reward += 5.0
                    print("Collect item, R %.1f" % reward, file=self.log)
                self.prev_items_cnt = items_cnt
            if mon_total_hp < self.prev_mon_total_hp:
                # Monster took damage
                reward += 10.0
                self.prev_mon_total_hp = mon_total_hp
                print("Attack monster, R %.1f" % reward, file=self.log)
            if monsters_cnt < self.prev_monsters_cnt:
                # Monsters killed
                reward += (self.prev_monsters_cnt - monsters_cnt) * 20.0
                self.prev_monsters_cnt = monsters_cnt
                print("Kill monster, R %.1f" % reward, file=self.log)

        # See the definition of @reward: initially, it is set to
        # the integer zero, so we can safely check for type changes
        # if the agent was exploring and @reward has changed to float.
        was_exploring = (type(reward) != int)

        if self.is_agent_stuck(d, was_exploring):
            # Cut this episode, agent is stuck
            truncated = True
            reward = -5.0
            if self.is_agent_timedout():
                print("Timedout, R %.1f" % reward, file=self.log)
            else:
                print("Stuck, R %.1f" % reward, file=self.log)
        elif not was_exploring:
            # Penalty for NOP
            reward = -0.1

        return [reward], done, truncated

    def _opt_changed(self, action):
        return False

    def step(self, action):
        self.steps_cnt += 1

        # HRL-awareness: (L,) shape
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        assert len(action) == self.num_hierarchy_levels
        worker_action = action[0]

        if self.paused:
            # Resume first
            self.pause_game(False)

        if self.no_actions:
            # We still submit NOOP and synchronize with the diablo
            # instance game ticks
            key = ring.RingEntryType.RING_ENTRY_KEY_NOOP \
                  | ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            self.game.submit_key(key)
        else:
            self._submit_action(worker_action)

        d = self.game.safe_state

        # Maintain history of positions
        pos = diablo_state.player_position(d)
        if self.hist_player_pos[0] != pos:
            self.hist_player_pos.appendleft(pos)

        # Keep compatibility with old environments
        goal_pos = self.goal_pos if self.used_goal is not None else None
        env = diablo_state.get_environment(d, radius=self.view_radius,
                                           goal_pos=goal_pos)

        rewards, done, truncated = self.evaluate_step(d, env, action)
        self.total_reward += rewards[0]

        if done or truncated:
            print("EPISODE DONE=%s steps=%d total R %.1f" % (
                "true" if self.episode_success else "false",
                self.steps_cnt, self.total_reward), file=self.log)

        self.game.agent_state.episode_steps = self.steps_cnt
        self.game.agent_state.stuck_steps   = self.steps_cnt - self.last_steps_cnt

        obss = self._build_obs(d, env)
        info = {"hierarchy/opt-changed": self._opt_changed(action),
                "hierarchy/reward": rewards,
                "env-counters": (self.resets_cnt, self.steps_cnt),
                "success": self.episode_success if done else False}
        return obss, rewards[0], done, truncated, info

class DiabloEnv_FindNextLevel_v0(DiabloEnv):
    @staticmethod
    def tune_config(env_config):
        # Keep compatibility with pre-trained models
        env_config['no-auto-walk-on-seconday-action'] = False

    @property
    def nr_env_channels(self):
        # No explicit Goal channel for v0
        EnvFlag = diablo_state.EnvironmentFlag
        return list(EnvFlag).index(EnvFlag.Open) + 1

    def generate_goal_pos(self, d, env_whole):
        # Our goal is to reach the stairs (trigger) to the next level
        nxtlvl_trig = diablo_state.find_trigger(d, dx.interface_mode.WM_DIABNEXTLVL)
        assert nxtlvl_trig is not None
        goal_pos = (nxtlvl_trig.position.x, nxtlvl_trig.position.y)
        return goal_pos

    def evaluate_step(self, d, env, action):
        player_pos = diablo_state.player_position(d)

        truncated = False
        done = False
        # The initial value must be a zero integer. I need a simple
        # marker to indicate that @reward was changed in many if-blocks below.
        # It seems the easiest way is to set it to an integer initially and
        # propagate it to a float on any update. This will be an ideal
        # marker that @reward was updated and that the agent was exploring.
        reward = int(0)

        if diablo_state.is_player_dead(d):
            # We are dead, game over
            reward = 0.0
            done = True
            print("Death, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel < self.start_dungeon_level:
            # Done with this episode with 0 reward if agent has
            # stepped into a trigger to escape
            reward = 0.0
            done = True
            print("Escape, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel > self.start_dungeon_level:
            reward = 20.0
            done = True
            self.episode_success = True
            print("Goal, R %.1f" % reward, file=self.log)

        # See the definition of @reward: initially, it is set to
        # the integer zero, so we can safely check for type changes
        # if the agent was exploring and @reward has changed to float.
        was_exploring = (type(reward) != int)

        if self.is_agent_stuck(d, was_exploring):
            # Cut this episode, agent is stuck
            truncated = True
            reward = 0.0
            if self.is_agent_timedout():
                print("Timedout, R %.1f" % reward, file=self.log)
            else:
                print("Stuck, R %.1f" % reward, file=self.log)
        elif not was_exploring:
            # Penalty for NOP
            reward = 0.0

        return [reward], done, truncated

class DiabloEnv_FindRandomGoal_v0(DiabloEnv):
    @staticmethod
    def tune_config(env_config):
        """Tune configuration before instantiation"""
        pass

    def __init__(self, env_config, **kwargs):
        super().__init__(env_config, **kwargs)
        self.used_goal = "random"

    def generate_goal_pos(self, d, env_whole):
        start_pos = diablo_state.player_position(d)
        goal_pos, _ = diablo_state.pick_random_clean_goal(
            env_whole, start_pos, self.np_random)
        return goal_pos

    def evaluate_step(self, d, env, action):
        player_pos = diablo_state.player_position(d)

        truncated = False
        done = False
        # The initial value must be a zero integer. I need a simple
        # marker to indicate that @reward was changed in many if-blocks below.
        # It seems the easiest way is to set it to an integer initially and
        # propagate it to a float on any update. This will be an ideal
        # marker that @reward was updated and that the agent was exploring.
        reward = int(0)

        if diablo_state.is_player_dead(d):
            # We are dead, game over
            reward = 0.0
            done = True
            print("Death, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel < self.start_dungeon_level or \
             (self.used_goal == "random" and d.player.plrlevel != self.start_dungeon_level):
            # Done with this episode with 0 reward if agent has
            # stepped into a trigger to escape
            reward = 0.0
            done = True
            print("Escape, R %.1f" % reward, file=self.log)
        elif player_pos == self.goal_pos or \
             (self.used_goal == "next-level" and d.player.plrlevel > self.start_dungeon_level):
            reward = 20.0
            done = True
            print("Goal, R %.1f" % reward, file=self.log)

        # See the definition of @reward: initially, it is set to
        # the integer zero, so we can safely check for type changes
        # if the agent was exploring and @reward has changed to float.
        was_exploring = (type(reward) != int)

        if self.is_agent_stuck(d, was_exploring):
            # Cut this episode, agent is stuck
            truncated = True
            reward = 0.0
            if self.is_agent_timedout():
                print("Timedout, R %.1f" % reward, file=self.log)
            else:
                print("Stuck, R %.1f" % reward, file=self.log)
        elif not was_exploring:
            # Penalty for NOP
            reward = 0.0

        return [reward], done, truncated

class DiabloEnv_FindNextLevel_v1(DiabloEnv_FindRandomGoal_v0):
    @staticmethod
    def tune_config(env_config):
        """Tune configuration before instantiation"""
        pass

    def __init__(self, env_config, **kwargs):
        super().__init__(env_config, **kwargs)
        self.used_goal = "next-level"

    def generate_goal_pos(self, d, env_whole):
        # Our goal is to reach the stairs (trigger) to the next level
        nxtlvl_trig = diablo_state.find_trigger(d, dx.interface_mode.WM_DIABNEXTLVL)
        assert nxtlvl_trig is not None
        goal_pos = (nxtlvl_trig.position.x, nxtlvl_trig.position.y)
        return goal_pos

class DiabloEnv_ClearTheLevel_v0(DiabloEnv):
    @staticmethod
    def tune_config(env_config):
        """Tune configuration before instantiation"""
        pass

    def __init__(self, env_config, **kwargs):
        super().__init__(env_config, **kwargs)
        self.used_goal = "random"

    def generate_goal_pos(self, d, env_whole):
        start_pos = diablo_state.player_position(d)
        goal_pos, _ = diablo_state.pick_random_clean_goal(
            env_whole, start_pos, self.np_random)
        return goal_pos

    def evaluate_step(self, d, env, action):
        monsters_cnt = diablo_state.count_active_monsters(d)
        mon_total_hp = diablo_state.count_active_monsters_total_hp(d)
        player_pos = diablo_state.player_position(d)
        hp = d.player._pHitPoints

        truncated = False
        done = False
        # The initial value must be a zero integer. I need a simple
        # marker to indicate that @reward was changed in many if-blocks below.
        # It seems the easiest way is to set it to an integer initially and
        # propagate it to a float on any update. This will be an ideal
        # marker that @reward was updated and that the agent was exploring.
        reward = int(0)

        if diablo_state.is_player_dead(d):
            # We are dead, game over
            reward = -10.0
            done = True
            print("Death, R %.2f" % reward, file=self.log)
        elif d.player.plrlevel < self.start_dungeon_level or \
             (self.used_goal == "random" and d.player.plrlevel != self.start_dungeon_level):
            # Done with this episode with 0 reward if agent has
            # stepped into a trigger to escape
            reward = 0.0
            done = True
            print("Escape, R %.2f" % reward, file=self.log)
        elif player_pos == self.goal_pos or \
             (self.used_goal == "next-level" and d.player.plrlevel > self.start_dungeon_level):
            reward = 20.0
            done = True
            self.episode_success = True
            print("Goal, R %.2f" % reward, file=self.log)
        else:
            if hp < self.prev_hp:
                # Player took damage
                reward -= (self.prev_hp - hp) / d.player._pMaxHP * 5.0
                self.prev_hp = hp
                print("Damage taken, R %.2f" % reward, file=self.log)
            if mon_total_hp < self.prev_mon_total_hp:
                # Monster took damage
                reward += 0.02
                self.prev_mon_total_hp = mon_total_hp
                print("Attack monster, R %.2f" % reward, file=self.log)
            if monsters_cnt < self.prev_monsters_cnt:
                # Monsters killed
                reward += (self.prev_monsters_cnt - monsters_cnt) * 0.1
                self.prev_monsters_cnt = monsters_cnt
                print("Kill monster, R %.2f" % reward, file=self.log)

        # See the definition of @reward: initially, it is set to
        # the integer zero, so we can safely check for type changes
        # if the agent was exploring and @reward has changed to float.
        was_exploring = (type(reward) != int)

        if self.is_agent_stuck(d, was_exploring):
            # Cut this episode, agent is stuck
            truncated = True
            reward = 0.0
            if self.is_agent_timedout():
                print("Timedout, R %.2f" % reward, file=self.log)
            else:
                print("Stuck, R %.2f" % reward, file=self.log)
        elif not was_exploring:
            # Penalize only movement that didn't accomplish anything.
            # Stand/PrimaryAction/SecondaryAction get no penalty: the agent
            # should be free to attempt attacks or interact without being
            # punished for a miss or a failed interaction.
            if action < ActionEnum.Stand.value:
                reward -= 0.01

        return [reward], done, truncated

### HRL Environment Classes

class DiabloEnvHRL_ClearTheLevel_v0(DiabloEnv):
    @staticmethod
    def tune_config(env_config):
        pass

    def __init__(self, env_config, **kwargs):
        super().__init__(env_config, **kwargs)
        self.num_hierarchy_levels = 2
        self.used_goal = "random"
        self.prev_option = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_option = None
        return obs, info

    def _opt_changed(self, action):
        opt = int(action[1])
        changed = (opt != self.prev_option)
        self.prev_option = opt
        return bool(changed)

    def generate_goal_pos(self, d, env_whole):
        start_pos = diablo_state.player_position(d)
        goal_pos, _ = diablo_state.pick_random_clean_goal(
            env_whole, start_pos, self.np_random)
        return goal_pos

    def evaluate_step(self, d, env, action):
        worker_action = int(action[0])
        manager_option = int(action[1])
        monsters_cnt = diablo_state.count_active_monsters(d)
        mon_total_hp = diablo_state.count_active_monsters_total_hp(d)
        player_pos = diablo_state.player_position(d)
        hp = d.player._pHitPoints

        truncated = False
        done = False
        # int(0) sentinel: reward becomes float only when something happens,
        # used to detect whether the agent was active this step.
        worker_reward = int(0)
        manager_reward = int(0)

        if diablo_state.is_player_dead(d):
            # We are dead, game over
            worker_reward = -10.0
            manager_reward = -10.0
            done = True
            print("Death, R [%.2f, %.2f]" % (worker_reward, manager_reward), file=self.log)
        elif d.player.plrlevel < self.start_dungeon_level or \
             (self.used_goal == "random" and d.player.plrlevel != self.start_dungeon_level):
            # Done with this episode with 0 reward if agent has
            # stepped into a trigger to escape
            worker_reward = 0.0
            manager_reward = 0.0
            done = True
            print("Escape, R [%.2f, %.2f]" % (worker_reward, manager_reward), file=self.log)
        elif player_pos == self.goal_pos or \
             (self.used_goal == "next-level" and d.player.plrlevel > self.start_dungeon_level):
            worker_reward = 20.0
            manager_reward = 20.0
            done = True
            self.episode_success = True
            print("Goal, R [%.2f, %.2f]" % (worker_reward, manager_reward), file=self.log)
        elif manager_option == 0:
            # Explorer selected: keep state in sync so no stale delta
            # fires when manager later switches to combat.
            self.prev_hp = hp
            self.prev_mon_total_hp = mon_total_hp
            self.prev_monsters_cnt = monsters_cnt
        elif manager_option == 1:
            # Combat selected

            # Manager hint: penalize choosing fight with no visible targets
            if diablo_state.count_visible_monsters(env) == 0:
                manager_reward = -0.5

            if hp < self.prev_hp:
                # Player took damage
                worker_reward -= (self.prev_hp - hp) / d.player._pMaxHP * 5.0
                self.prev_hp = hp
                print("Damage taken, R %.2f" % worker_reward, file=self.log)
            if mon_total_hp < self.prev_mon_total_hp:
                # Monster took damage
                worker_reward += 0.02
                self.prev_mon_total_hp = mon_total_hp
                print("Attack monster, R %.2f" % worker_reward, file=self.log)
            if monsters_cnt < self.prev_monsters_cnt:
                # Monsters killed
                worker_reward += (self.prev_monsters_cnt - monsters_cnt) * 0.1
                self.prev_monsters_cnt = monsters_cnt
                print("Kill monster, R %.2f" % worker_reward, file=self.log)

        # See the definition of @worker_reward: initially, it is set
        # to the integer zero, so we can safely check for type changes
        # if the agent was exploring and @worker_reward has changed to
        # float.
        was_exploring = (type(worker_reward) != int)

        if self.is_agent_stuck(d, was_exploring):
            # Cut this episode, agent is stuck
            truncated = True
            worker_reward = 0.0
            manager_reward = 0.0
            if self.is_agent_timedout():
                print("Timedout, R [%.2f, %.2f]" % (worker_reward, manager_reward), file=self.log)
            else:
                print("Stuck, R [%.2f, %.2f]" % (worker_reward, manager_reward), file=self.log)
        elif not was_exploring:
            # Penalize only movement that didn't accomplish anything.
            # Stand/PrimaryAction/SecondaryAction get no penalty: the agent
            # should be free to attempt attacks or interact without being
            # punished for a miss or a failed interaction.
            if worker_action < ActionEnum.Stand.value:
                worker_reward -= 0.01

        return [worker_reward, manager_reward], done, truncated


### v2 Environment Classes (extended action + observation contract)

class DiabloEnvV2Mixin:
    """Switches an env class to the v2 contract:
    - full action set (movement + Stand + primary/secondary + 2 restore + 7 cast)
    - drops the legacy broadcast env-status planes
    - adds the per-tile monster_attrs (W,H,9) grid
    - adds a flat scalars vector (see compute_scalars in diablo_state.py)
    Apply before any DiabloEnv-derived class in the MRO so the overrides win."""

    @property
    def num_actions(self):
        return ActionEnum.CastFireball.value + 1

    @property
    def obs_includes_old_status(self):
        return False

    @property
    def obs_includes_monster_attrs(self):
        return True

    @property
    def obs_includes_scalars(self):
        return True

    def _get_monster_attrs(self, d):
        # Thin wrapper; the hot loop lives in diablo_state.compute_monster_attrs
        # under @njit. Module-level helper takes ints (not enum values) and a
        # numpy array of ranged AI ids (frozensets don't survive @njit).
        max_level  = max(int(d.max_monster_level.value), 1)
        max_walk   = max(int(d.max_walk_frames.value),   1)
        max_attack = max(int(d.max_attack_frames.value), 1)
        return diablo_state.compute_monster_attrs(
            d, self.view_radius, max_level, max_walk, max_attack,
            diablo_state.ranged_ai_ids_array(),
        )

    def _get_scalars(self, d):
        # Thin wrapper; assembly + pot-count loop live in
        # diablo_state.compute_scalars under @njit.
        return diablo_state.compute_scalars(d)


class RewardEvent(enum.Enum):
    """Keys for the REWARDS dict used by DiabloEnv_ClearAllLevels."""
    # Terminal outcomes
    Death              = enum.auto()
    DiabloKilled       = enum.auto()
    Escape             = enum.auto()
    Goal               = enum.auto()
    # Episode end by inactivity
    Stuck              = enum.auto()
    Timedout           = enum.auto()
    # Per-step shaping (flat unless noted)
    DamageTaken        = enum.auto()  # coeff: reward -= (delta_hp/max_hp)*coeff
    AttackMonster      = enum.auto()  # per distinct monster hit
    KillMonster        = enum.auto()  # per kill
    ActivateObject     = enum.auto()  # per object activated
    OpenDoor           = enum.auto()  # per first-time door open
    CollectItem        = enum.auto()  # per item picked up
    RestoreHpCorrect   = enum.auto()
    RestoreHpWasteful  = enum.auto()
    RestoreHpNoPot     = enum.auto()
    RestoreManaCorrect   = enum.auto()
    RestoreManaWasteful  = enum.auto()
    RestoreManaNoPot     = enum.auto()
    SpellUnavailable   = enum.auto()
    SpellSuccessful    = enum.auto()
    SpellWasteful      = enum.auto()
    SpellRedundantShield = enum.auto()
    SpellFirstUse      = enum.auto()
    MovementPenalty    = enum.auto()
    WastedPrimary      = enum.auto()
    WastedSecondary    = enum.auto()


class DiabloEnv_ClearAllLevels_v0(DiabloEnvV2Mixin, DiabloEnv_ClearTheLevel_v0):
    """Combat + exploration across all dungeon levels with the v2 action and
    observation set. Reward magnitudes live in the REWARDS class dict;
    subclasses tune shaping by overriding individual entries."""

    REWARDS = {
        RewardEvent.Death:               -10.0,
        RewardEvent.DiabloKilled:        +20.0,
        RewardEvent.Escape:              -10.0,
        RewardEvent.Goal:                +20.0,
        RewardEvent.Stuck:               -10.0,
        RewardEvent.Timedout:            -10.0,
        RewardEvent.DamageTaken:         +5.0,
        RewardEvent.AttackMonster:       +0.02,
        RewardEvent.KillMonster:         +0.1,
        RewardEvent.ActivateObject:      +0.05,
        RewardEvent.OpenDoor:            +0.02,
        RewardEvent.CollectItem:         +0.02,
        RewardEvent.RestoreHpCorrect:    +0.05,
        RewardEvent.RestoreHpWasteful:   -0.10,
        RewardEvent.RestoreHpNoPot:      -0.10,
        RewardEvent.RestoreManaCorrect:  +0.05,
        RewardEvent.RestoreManaWasteful: -0.10,
        RewardEvent.RestoreManaNoPot:    -0.10,
        RewardEvent.SpellUnavailable:    -0.10,
        RewardEvent.SpellSuccessful:     +0.15,
        RewardEvent.SpellWasteful:       -0.10,
        RewardEvent.SpellRedundantShield:-0.10,
        RewardEvent.SpellFirstUse:       +0.10,
        RewardEvent.MovementPenalty:     -0.01,
        RewardEvent.WastedPrimary:       -0.05,
        RewardEvent.WastedSecondary:     -0.05,
    }

    @staticmethod
    def _hp_pot_sum(d):
        """Sum of all HP-restoring pots; decreases iff one was consumed this step."""
        pots = diablo_state.player_pot_counts(d.player)
        return pots[0] + pots[1] + pots[2] + pots[5] + pots[6]

    @staticmethod
    def _mana_pot_sum(d):
        """Sum of all mana-restoring pots; decreases iff one was consumed this step."""
        pots = diablo_state.player_pot_counts(d.player)
        return pots[3] + pots[4] + pots[5] + pots[6]

    def is_agent_stuck(self, d, made_progress):
        """Override: drop the position-based counter reset.
        Counter resets on combat/item progress (made_progress) or when
        the agent discovers new dungeon tiles (active exploration)."""
        if self.is_agent_timedout():
            return True
        explored = diablo_state.count_explored_tiles(d)
        if explored > self.prev_explored_cnt:
            made_progress = True
            self.prev_explored_cnt = explored
        if made_progress:
            self.last_steps_cnt = self.steps_cnt
            self.last_player_pos = diablo_state.player_position(d)
            return False
        return self.steps_cnt - self.last_steps_cnt >= self.STUCK_TIMEOUT

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        d = self.game.state
        self.prev_mana = int(d.player._pMana)
        self.prev_mana_shield = bool(d.player.pManaShield)
        self.prev_pmode = int(d.player._pmode)
        self.prev_hp_pots = self._hp_pot_sum(d)
        self.prev_mana_pots = self._mana_pot_sum(d)
        self.v2_spells_used = set()
        self.prev_mon_hp = diablo_state.alloc_monster_hp(d)
        diablo_state.snapshot_monster_hp(d, self.prev_mon_hp)
        return obs, info

    def evaluate_step(self, d, env, action):
        R                = self.REWARDS
        action           = int(action)
        monsters_cnt     = diablo_state.count_active_monsters(d)
        obj_cnt          = diablo_state.count_active_objects(d)
        closed_doors_ids = diablo_state.get_closed_doors_ids(d)
        items_cnt        = diablo_state.count_active_items(d)
        player_pos       = diablo_state.player_position(d)
        hp               = d.player._pHitPoints
        mana             = int(d.player._pMana)
        curr_pmode       = int(d.player._pmode)
        max_hp           = max(int(d.player._pMaxHP),   1)
        max_mana         = max(int(d.player._pMaxMana), 1)

        truncated = False
        done = False
        # int(0) sentinel: becomes float only when a reward fires.
        # was_exploring below uses type(reward) != int to detect activity.
        reward = int(0)
        made_progress = False

        if diablo_state.is_player_dead(d):
            reward = R[RewardEvent.Death]
            done = True
            print("Death, R %.2f" % reward, file=self.log)
        elif d.player._pmode == dx.PLR_MODE.PM_QUIT.value:
            # PrepDoEnding() fired: Diablo (final boss) was killed.
            reward = R[RewardEvent.DiabloKilled]
            done = True
            self.episode_success = True
            print("Diablo killed, R %.2f" % reward, file=self.log)
        elif d.player.plrlevel < self.start_dungeon_level or \
             (self.used_goal == "random" and d.player.plrlevel != self.start_dungeon_level):
            reward = R[RewardEvent.Escape]
            done = True
            print("Escape, R %.2f" % reward, file=self.log)
        elif player_pos == self.goal_pos or \
             (self.used_goal == "next-level" and d.player.plrlevel > self.start_dungeon_level):
            reward = R[RewardEvent.Goal]
            done = True
            self.episode_success = True
            print("Goal, R %.2f" % reward, file=self.log)
        else:
            # Count distinct monsters that took damage this step.
            monsters_hit = diablo_state.count_monsters_hit(d, self.prev_mon_hp)
            monster_damaged = monsters_hit > 0

            if hp < self.prev_hp:
                # Player took damage. prev_hp updated unconditionally
                # at end of step so damage right after a heal still
                # credits the full drop.
                r = R[RewardEvent.DamageTaken]
                if r:
                    reward -= (self.prev_hp - hp) / d.player._pMaxHP * r
                print("Damage taken, R %.2f" % reward, file=self.log)
            if monster_damaged:
                # Monsters took damage
                r = R[RewardEvent.AttackMonster]
                if r:
                    reward += monsters_hit * r
                made_progress = True
                print("Attack %d monster(s), R %.2f" % (monsters_hit, reward), file=self.log)
            if monsters_cnt < self.prev_monsters_cnt:
                # Monsters killed
                r = R[RewardEvent.KillMonster]
                if r:
                    reward += (self.prev_monsters_cnt - monsters_cnt) * r
                self.prev_monsters_cnt = monsters_cnt
                made_progress = True
                print("Kill monster, R %.2f" % reward, file=self.log)
            if obj_cnt < self.prev_obj_cnt:
                # Chests, sarcophagi, barrels, crucifixes, doors etc.
                r = R[RewardEvent.ActivateObject]
                if r:
                    reward += (self.prev_obj_cnt - obj_cnt) * r
                self.prev_obj_cnt = obj_cnt
                made_progress = True
                print("Activate object, R %.2f" % reward, file=self.log)
            if len(closed_doors_ids) != len(self.prev_closed_doors_ids):
                if len(closed_doors_ids) < len(self.prev_closed_doors_ids):
                    opened = list(set(self.prev_closed_doors_ids) - set(closed_doors_ids))
                    # Exclude doors we previously opened (re-closed -> re-opened cycle).
                    opened = [o for o in opened if o not in self.opened_doors_ids]
                    self.opened_doors_ids.extend(opened)
                    if opened:
                        r = R[RewardEvent.OpenDoor]
                        if r:
                            reward += len(opened) * r
                        made_progress = True
                        print("Open door, R %.2f" % reward, file=self.log)
                self.prev_closed_doors_ids = closed_doors_ids
            if items_cnt != self.prev_items_cnt:
                if items_cnt < self.prev_items_cnt:
                    # Items can also appear (chest spill); only decrease credits a pickup.
                    r = R[RewardEvent.CollectItem]
                    if r:
                        reward += (self.prev_items_cnt - items_cnt) * r
                    made_progress = True
                    print("Collect item, R %.2f" % reward, file=self.log)
                self.prev_items_cnt = items_cnt

            # Restore actions: pot-count change detects actual consumption
            # (net HP can drop even after healing when taking damage same step).
            hp_pots   = self._hp_pot_sum(d)
            mana_pots = self._mana_pot_sum(d)
            if action == ActionEnum.RestoreHealth.value:
                if self.prev_hp / max_hp >= 0.9:
                    r = R[RewardEvent.RestoreHpWasteful]
                    if r:
                        reward += r
                    print("Wasteful restore HP, R %.2f" % reward, file=self.log)
                elif hp_pots < self.prev_hp_pots:
                    r = R[RewardEvent.RestoreHpCorrect]
                    if r:
                        reward += r
                    made_progress = True
                    print("Correct restore HP, R %.2f" % reward, file=self.log)
                else:
                    r = R[RewardEvent.RestoreHpNoPot]
                    if r:
                        reward += r
                    print("No-potion restore HP, R %.2f" % reward, file=self.log)
            elif action == ActionEnum.RestoreMana.value:
                if self.prev_mana / max_mana >= 0.9:
                    r = R[RewardEvent.RestoreManaWasteful]
                    if r:
                        reward += r
                    print("Wasteful restore mana, R %.2f" % reward, file=self.log)
                elif mana_pots < self.prev_mana_pots:
                    r = R[RewardEvent.RestoreManaCorrect]
                    if r:
                        reward += r
                    made_progress = True
                    print("Correct restore mana, R %.2f" % reward, file=self.log)
                else:
                    r = R[RewardEvent.RestoreManaNoPot]
                    if r:
                        reward += r
                    print("No-potion restore mana, R %.2f" % reward, file=self.log)
            # Cast spells: unavailable penalty is action-tied.
            elif (ActionEnum.CastFirebolt.value <= action
                  <= ActionEnum.CastFireball.value):
                spell_id = _ACTION_TO_SPELL[ActionEnum(action)]
                if not (diablo_state.player_spell_bits(d) & (1 << int(spell_id.value))):
                    r = R[RewardEvent.SpellUnavailable]
                    if r:
                        reward += r
                    print("Unavailable %s spell, R %.2f" % (spell_id.name, reward), file=self.log)

            # Spell cast reward: animation skipped so PM_SPELL lasts 1 tick;
            # may fire later than the action was submitted.
            if curr_pmode == dx.PLR_MODE.PM_SPELL.value:
                spell_id = dx.SpellID(int(d.player.executedSpell['spellId']))
                ae = _SPELL_TO_ACTION.get(int(spell_id.value))
                if ae is not None:
                    if ae == ActionEnum.CastPhasing:
                        # Escape spell: reward when monsters are visible.
                        if diablo_state.count_visible_monsters(env) > 0:
                            r = R[RewardEvent.SpellSuccessful]
                            if r:
                                reward += r
                            made_progress = True
                            print("Successful %s spell, R %.2f" % (spell_id.name, reward), file=self.log)
                    elif ae == ActionEnum.CastManaShield:
                        # Casting when already active wastes mana with no benefit.
                        if self.prev_mana_shield:
                            r = R[RewardEvent.SpellRedundantShield]
                            if r:
                                reward += r
                            print("Redundant %s spell, R %.2f" % (spell_id.name, reward), file=self.log)
                        else:
                            r = R[RewardEvent.SpellSuccessful]
                            if r:
                                reward += r
                            made_progress = True
                            print("Successful %s spell, R %.2f" % (spell_id.name, reward), file=self.log)
                    else:
                        if diablo_state.count_visible_monsters(env) == 0:
                            r = R[RewardEvent.SpellWasteful]
                            if r:
                                reward += r
                            print("Wasteful %s spell, R %.2f" % (spell_id.name, reward), file=self.log)
                        else:
                            r = R[RewardEvent.SpellSuccessful]
                            if r:
                                reward += r
                            made_progress = True
                            print("Successful %s spell, R %.2f" % (spell_id.name, reward), file=self.log)
                    if ae.value not in self.v2_spells_used and made_progress:
                        self.v2_spells_used.add(ae.value)
                        r = R[RewardEvent.SpellFirstUse]
                        if r:
                            reward += r
                        print("Successful %s spell, R %.2f" % (spell_id.name, reward), file=self.log)

        # Update per-monster HP snapshot every step so new spawns and
        # regeneration are baselined correctly.
        diablo_state.snapshot_monster_hp(d, self.prev_mon_hp)

        # prev_hp and prev_mana updated unconditionally so the pre-step value
        # is visible to shaping branches and damage after a heal is credited.
        self.prev_hp          = hp
        self.prev_mana        = mana
        self.prev_mana_shield = bool(d.player.pManaShield)
        self.prev_pmode       = curr_pmode
        self.prev_hp_pots     = self._hp_pot_sum(d)
        self.prev_mana_pots   = self._mana_pot_sum(d)

        # made_progress gates the stuck counter; was_exploring (reward became
        # float) gates the idle penalty. Damage alone must not reset the stuck
        # counter - otherwise corner-dancing under monster fire loops until the
        # 3000-step hard timeout.
        was_exploring = (type(reward) != int)

        if not done and self.is_agent_stuck(d, made_progress):
            truncated = True
            if self.is_agent_timedout():
                reward = R[RewardEvent.Timedout]
                print("Timedout, R %.2f" % reward, file=self.log)
            else:
                reward = R[RewardEvent.Stuck]
                print("Stuck, R %.2f" % reward, file=self.log)
        elif not was_exploring:
            if action <= ActionEnum.Stand.value:
                r = R[RewardEvent.MovementPenalty]
                if r:
                    reward += r
            elif self.view_radius is not None:
                EF = diablo_state.EnvironmentFlag
                if action == ActionEnum.PrimaryAction.value:
                    # Penalize attack with no adjacent monster (melee assumed).
                    if not diablo_state.player_has_adjacent(
                            env, self.view_radius,
                            EF.Monster.value, 0):
                        r = R[RewardEvent.WastedPrimary]
                        if r:
                            reward += r
                        print("Wasted primary, R %.2f" % reward, file=self.log)
                elif action == ActionEnum.SecondaryAction.value:
                    if not diablo_state.player_has_adjacent(
                            env, self.view_radius,
                            EF.Item.value | EF.Interactable.value, 0):
                        r = R[RewardEvent.WastedSecondary]
                        if r:
                            reward += r
                        print("Wasted secondary, R %.2f" % reward, file=self.log)

        return [reward], done, truncated


class DiabloEnv_ClearAllLevels_v1(DiabloEnv_ClearAllLevels_v0):
    """Sparse-reward variant: only terminal and stuck/timeout rewards fire;
    all intermediate shaping terms are zeroed."""
    ZEROED_EVENTS = frozenset(RewardEvent) - {
        RewardEvent.Death,
        RewardEvent.DiabloKilled,
        RewardEvent.Escape,
        RewardEvent.Goal,
        RewardEvent.Stuck,
        RewardEvent.Timedout,
    }
    REWARDS = {
        **DiabloEnv_ClearAllLevels_v0.REWARDS,
        **{e: 0.0 for e in ZEROED_EVENTS},
    }


class DiabloEnv_ClearAllLevels_v2(DiabloEnv_ClearAllLevels_v0):
    """Like v1 (sparse) but object-activation and item-pickup rewards are
    restored to their v0 values."""
    REWARDS = {
        **DiabloEnv_ClearAllLevels_v1.REWARDS,
        RewardEvent.ActivateObject: +0.15,
        RewardEvent.CollectItem:    +0.15,
    }


class DiabloEnv_ClearAllLevels_v3(DiabloEnv_ClearAllLevels_v0):
    """Like v2 but with rebalanced rewards: higher goal/penalty magnitudes and
    reduced item-shaping to discourage farming over level completion."""
    REWARDS = {
        **DiabloEnv_ClearAllLevels_v1.REWARDS,
        RewardEvent.Goal:           +40.0,
        RewardEvent.DiabloKilled:   +40.0,
        RewardEvent.Death:          -20.0,
        RewardEvent.Escape:         -25.0,
        RewardEvent.Stuck:          -25.0,
        RewardEvent.Timedout:       -25.0,
        RewardEvent.ActivateObject: +0.08,
        RewardEvent.CollectItem:    +0.08,
    }


class DiabloEnv_ClearAllLevels_v4(DiabloEnv_ClearAllLevels_v3):
    """Like v3 but adds a stuck-urgency scalar to the observation:
      [46] stuck_frac = (steps_cnt - last_steps_cnt) / STUCK_TIMEOUT
    Tells the model how close it is to the stuck penalty, incentivising
    timely exploration. _pad_state_dict handles the 46->47 shape change
    when loading a pre-trained checkpoint."""

    def _get_scalars(self, d):
        base = diablo_state.compute_scalars(d)
        # Fraction of the stuck window elapsed: near 1.0 the agent must
        # make progress (explore/kill) or face the stuck penalty.
        stuck_frac = np.float32((self.steps_cnt - self.last_steps_cnt) / self.STUCK_TIMEOUT)
        return np.append(base, [stuck_frac])


from gymnasium.envs.registration import register

DIABLO_ENVS = [
    { 'id': 'Diablo-FindNextLevel-v0',
      'entry_point': DiabloEnv_FindNextLevel_v0 },
    { 'id': 'Diablo-FindNextLevel-v1',
      'entry_point': DiabloEnv_FindNextLevel_v1 },
    { 'id': 'Diablo-FindRandomGoal-v0',
      'entry_point': DiabloEnv_FindRandomGoal_v0 },
    { 'id': 'Diablo-ClearTheLevel-v0',
      'entry_point': DiabloEnv_ClearTheLevel_v0 },

    { 'id': 'Diablo-ClearAllLevels-v0',
      'entry_point': DiabloEnv_ClearAllLevels_v0 },
    { 'id': 'Diablo-ClearAllLevels-v1',
      'entry_point': DiabloEnv_ClearAllLevels_v1 },
    { 'id': 'Diablo-ClearAllLevels-v2',
      'entry_point': DiabloEnv_ClearAllLevels_v2 },
    { 'id': 'Diablo-ClearAllLevels-v3',
      'entry_point': DiabloEnv_ClearAllLevels_v3 },
    { 'id': 'Diablo-ClearAllLevels-v4',
      'entry_point': DiabloEnv_ClearAllLevels_v4 },

    # HRL Environment Classes

    { 'id': 'Diablo-HRL-ClearTheLevel-v0',
      'entry_point': DiabloEnvHRL_ClearTheLevel_v0 },
]

def register_diablo_envs():
    for env in DIABLO_ENVS:
        register(
            id=env['id'],
            entry_point=env['entry_point'],
        )

register_diablo_envs()
