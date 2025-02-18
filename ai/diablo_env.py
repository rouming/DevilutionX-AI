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

class DiabloEnv(gym.Env):
    class ActionMask(enum.Enum):
        MASK_TRIGGERS      = 1<<0
        MASK_CLOSED_DOORS  = 1<<1
        MASK_WALLS         = 1<<2
        MASK_OTHER_SOLIDS  = 1<<3

    MASK_EVERYTHING = (ActionMask.MASK_TRIGGERS.value |
                       ActionMask.MASK_CLOSED_DOORS.value |
                       ActionMask.MASK_WALLS.value |
                       ActionMask.MASK_OTHER_SOLIDS.value)

    class ActionEnum(enum.Enum):
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

    @staticmethod
    def action_to_key(action):
        match DiabloEnv.ActionEnum(action):
            case DiabloEnv.ActionEnum.Walk_N:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case DiabloEnv.ActionEnum.Walk_NE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case DiabloEnv.ActionEnum.Walk_E:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_RIGHT)
            case DiabloEnv.ActionEnum.Walk_SE:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN)
            case DiabloEnv.ActionEnum.Walk_S:
                key = (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case DiabloEnv.ActionEnum.Walk_SW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case DiabloEnv.ActionEnum.Walk_W:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP |
                       ring.RingEntryType.RING_ENTRY_KEY_LEFT)
            case DiabloEnv.ActionEnum.Walk_NW:
                key = (ring.RingEntryType.RING_ENTRY_KEY_UP)
            case DiabloEnv.ActionEnum.Stand:
                # No key - NOP
                key = 0
            case DiabloEnv.ActionEnum.PrimaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_A)
            case DiabloEnv.ActionEnum.SecondaryAction:
                key = (ring.RingEntryType.RING_ENTRY_KEY_X)
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
        """Computes an action mask for the action space using the state information."""
        mask = np.full(len(DiabloEnv.ActionEnum), 1, dtype=np.int8)
        player_pos = diablo_state.player_position(d)

        neighbors = DiabloEnv.get_matrix_neighbors(env, *player_pos)
        for i, tile in enumerate(neighbors):
            if tile < 0:
                # Block in the direction beyond map
                mask[i] = 0
                continue

            if what & DiabloEnv.ActionMask.MASK_TRIGGERS.value and \
               tile & diablo_state.EnvironmentFlag.Trigger.value:
                # Block in the direction of any trigger, so forbid
                # escaping to the town
                # TODO: this blocks all the trigger including stairs to the
                # next level
                mask[i] = 0
            elif what & DiabloEnv.ActionMask.MASK_WALLS.value and \
                 tile & diablo_state.EnvironmentFlag.Wall.value:
                mask[i] = 0
            elif what & DiabloEnv.ActionMask.MASK_CLOSED_DOORS.value and \
                 tile & diablo_state.EnvironmentFlag.DoorClosed.value:
                mask[i] = 0
            elif what & DiabloEnv.ActionMask.MASK_OTHER_SOLIDS.value and \
                 (tile & diablo_state.EnvironmentFlag.Barrel.value or \
                  tile & diablo_state.EnvironmentFlag.Chest.value or \
                  tile & diablo_state.EnvironmentFlag.Sarcophagus.value or \
                  tile & diablo_state.EnvironmentFlag.Crucifix.value):
                mask[i] = 0

        return mask

    def __init__(self, env_config):
        self.config = env_config
        self.seed = self.config['seed']
        if self.config['train-ai'] and not self.config['same-seed']:
            self.seed += env_config.worker_index - 1
            # Update seed for further diablo run call
            env_config["seed"] = self.seed
        self.paused = False
        self.env_radius = None

        self.log_to_stdout = self.config['log-to-stdout'] \
            if 'log-to-stdout' in self.config else 0
        self.no_actions = self.config['no-actions'] \
            if 'no-actions' in self.config else 0

        # Run or attach to Diablo (devilutionX) instance
        self.game = diablo_state.DiabloGame.run_or_attach(env_config)

        if self.log_to_stdout:
            self.log = sys.stdout
        else:
            logfile = os.path.join(os.path.dirname(self.game.mshared_path), "env.log")
            self.log = open(logfile, "w", buffering=1)

        # Submit SAVE, so we LOAD on reset()
        saves_cnt = self.game.state.game_saves.value
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.game.submit_key(key)

        # Wait for the SAVE to take effect
        while saves_cnt == self.game.state.game_saves.value:
            time.sleep(0.01)

        print("INSTANCE seed=%d" % self.seed, file=self.log)

        d = copy.deepcopy(self.game.state)

        nxtlvl_trig = diablo_state.find_trigger(d, dx.interface_mode.WM_DIABNEXTLVL)
        assert nxtlvl_trig is not None
        # Our goal is to reach the stairs (trigger) to the next level
        start = diablo_state.player_position(d)
        goal = (nxtlvl_trig.position.x, nxtlvl_trig.position.y)

        # Get dungeon graph, doors and path from start to goal
        graph_and_path = diablo_state.get_dungeon_graph_and_path(d, start, goal)
        regions_doors, labeled_regions, regions_path, _ = graph_and_path

        self.goal = goal
        self.regions_doors = regions_doors
        self.labeled_regions = labeled_regions
        self.regions_path = regions_path

        # Starting dungeon level
        self.start_dungeon_level = d.player.plrlevel

        env_status = self.get_env_status(d)
        env = diablo_state.get_environment(d, radius=self.env_radius)

        self.action_space = gym.spaces.Discrete(len(DiabloEnv.ActionEnum))

        env_high = (1 << len(diablo_state.EnvironmentFlag)) - 1
        # Don't forget to change dtype if does not fit max number of bits
        assert env_high <= np.iinfo(env.dtype).max
        self.observation_space = gym.spaces.Dict({
                "env-status":  gym.spaces.Box(low=0,
                                              high=0xffffff,
                                              shape=env_status.shape,
                                              dtype=env_status.dtype),
                "environment": gym.spaces.Box(low=0,
                                              high=env_high,
                                              shape=env.shape,
                                              dtype=env.dtype),
            }
        )

        # XXX This is a nasty kludge to prevent RLlib from spawning
        # XXX an extra runner for training, even though we do evaluation.
        if not env_config['train-ai'] and 'self-evaluation' not in env_config:
            self.game.stop_or_detach()

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
        self.game.stop_or_detach()

    def get_env_status(self, d):
        monsters_cnt = diablo_state.count_active_monsters(d)
        hp = d.player._pHitPoints
        mode = d.player._pmode
        player_pos = diablo_state.player_position(d)
        return np.array([monsters_cnt, hp, mode, *player_pos])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed)

        if self.paused:
            # Resume first
            self.pause_game(False)

        print("RESET", file=self.log)

        # Submit LOAD
        loads_cnt = self.game.state.game_loads.value
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD | \
              ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.game.submit_key(key)

        # Wait for the LOAD to take effect
        while loads_cnt == self.game.state.game_loads:
            time.sleep(0.01)

        d = copy.deepcopy(self.game.state)
        env_status = self.get_env_status(d)
        env = diablo_state.get_environment(d, radius=self.env_radius)

        obj_cnt = diablo_state.count_active_objects(d)
        closed_doors_ids = diablo_state.get_closed_doors_ids(d)
        items_cnt = diablo_state.count_active_items(d)
        monsters_cnt = diablo_state.count_active_monsters(d)
        total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_state.count_explored_tiles(d)
        hp = d.player._pHitPoints
        pos = diablo_state.player_position(d)

        self.prev_obj_cnt = obj_cnt
        self.prev_closed_doors_ids = closed_doors_ids
        self.opened_doors_ids = []
        self.prev_items_cnt = items_cnt
        self.prev_monsters_cnt = monsters_cnt
        self.prev_total_hp = total_hp
        self.prev_explored_cnt = explored_cnt
        self.prev_hp = hp
        self.total_reward = 0.0
        self.exploration_reward = 0.0
        self.hist_player_pos = collections.deque([pos], maxlen=3)
        self.last_player_pos = pos
        self.last_steps_cnt = 0
        self.steps_cnt = 0

        init_obs = {
            "env-status":  env_status,
            "environment": env,
        }

        init_info = {
            "action_mask": DiabloEnv.action_mask(d, env,
                                   DiabloEnv.ActionMask.MASK_TRIGGERS.value)
        }

        return init_obs, init_info

    def is_agent_stuck(self, d, was_exploring):
        p = diablo_state.player_position(d)

        # Update counter and position if the agent was exploring or moved
        # a significant distance away
        if was_exploring or np.any(np.abs(self.last_player_pos - np.asarray(p)) > 10):
            self.last_steps_cnt = self.steps_cnt
            self.last_player_pos = p
            return False

        # Check if the agent was "doing nothing" for some time. We
        # don't use the wall clock here, because different machines have
        # different step rates, so in 10 seconds on one machine, 100 steps
        # can occur, while on another machine, the same 10 seconds can
        # cover 1k steps.
        return self.steps_cnt - self.last_steps_cnt >= 300

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
            doors_here[self.goal] = True
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
        # Mask triggers only, which are stairs to next level or to the town
        action_mask = DiabloEnv.action_mask(d, env,
                               DiabloEnv.ActionMask.MASK_TRIGGERS.value)

        obj_cnt = diablo_state.count_active_objects(d)
        closed_doors_ids = diablo_state.get_closed_doors_ids(d)
        items_cnt = diablo_state.count_active_items(d)
        monsters_cnt = diablo_state.count_active_monsters(d)
        total_hp = diablo_state.count_active_monsters_total_hp(d)
        explored_cnt = diablo_state.count_explored_tiles(d)
        hp = d.player._pHitPoints

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
            # XXX: That's a cludge: I do not know how to make an
            # XXX: action_mask for RLlib work, so we just done with this
            # XXX: episode with negative reward if agent has stepped
            # XXX: into a trigger to escape to the town
            reward = -10.0
            done = True
            print("Escape, R %.1f" % reward, file=self.log)
        elif d.player.plrlevel > self.start_dungeon_level:
            reward = 50.0
            done = True
            print("Next level, R %.1f" % reward, file=self.log)
        else:
            if hp < self.prev_hp:
                # More damage to health - more penalty
                # from 0 to -100
                change = (hp - self.prev_hp) / self.prev_hp
                reward += change * 100.0
                print("Damage, R %.1f" % reward, file=self.log)
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
                    print("Collecting item, R %.1f" % reward, file=self.log)
                self.prev_items_cnt = items_cnt
            if total_hp < self.prev_total_hp:
                # Monster took damage
                reward += 10.0
                self.prev_total_hp = total_hp
                print("Attack monster, R %.1f" % reward, file=self.log)
            if monsters_cnt < self.prev_monsters_cnt:
                # Monsters killed
                reward += (self.prev_monsters_cnt - monsters_cnt) * 20.0
                self.prev_monsters_cnt = monsters_cnt
                print("Kill monster, R %.1f" % reward, file=self.log)\

        # See the definition of @reward: initially, it is set to
        # the integer zero, so we can safely check for type changes
        # if the agent was exploring and @reward has changed to float.
        was_exploring = (type(reward) != int)

        if self.is_agent_stuck(d, was_exploring):
            # Cut this episode, agent is stuck
            truncated = True
            reward = -5.0
            print("Stuck, R %.1f" % reward, file=self.log)
        elif not was_exploring:
            # Penalty for NOP
            reward = -0.1

        return reward, done, truncated, action_mask

    def step(self, action):
        self.steps_cnt += 1

        if self.paused:
            # Resume first
            self.pause_game(False)

        if self.no_actions:
            # We still submit NOP and synchronize with the diablo
            # instance game ticks by waiting for key acceptance
            key = 0
        else:
            key = DiabloEnv.action_to_key(action)

        key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        self.game.submit_key(key)

        d = copy.deepcopy(self.game.state)

        # Maintain history of positions
        pos = diablo_state.player_position(d)
        if self.hist_player_pos[0] != pos:
            self.hist_player_pos.appendleft(pos)

        env_status = self.get_env_status(d)
        env = diablo_state.get_environment(d, radius=self.env_radius)

        reward, done, truncated, action_mask = self.evaluate_step(d, env, action)
        self.total_reward += reward

        if done:
            print("EPISODE DONE, total R %.1f" % self.total_reward, file=self.log)

        next_obs = {
            "env-status":  env_status,
            "environment": env,
        }
        next_info = {
            "action_mask": action_mask
        }

        return next_obs, reward, done, truncated, next_info
