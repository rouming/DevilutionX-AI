"""
diablo_bot.py - Provides various of bot classes for Imitation Learning

Author: Roman Penyaev <r.peniaev@gmail.com>
"""
from types import SimpleNamespace
import hashlib
import numpy as np
import os
import pickle
import sys
import time

import devilutionx as dx
import diablo_state
import diablo_env
import maze
import ring


def checksum(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(data).hexdigest()


def get_bot_constructor(bot_name):
    bot_constructor = getattr(sys.modules[__name__], bot_name, None)
    if bot_constructor is None:
        raise ValueError(f"Bot class '{bot_name}' is not found")
    return bot_constructor


class FindRandomGoal_Bot:
    """
    An autonomous bot that explores a dungeon environment until
    it locates a random goal tile.

    The `FindRandomGoal_Bot` uses visibility-based frontier
    exploration to incrementally uncover unexplored areas of the
    dungeon. It maintains an internal map of explored and unexplored
    regions, computes shortest paths to frontier points using the
    `maze.shortest_path` function, and interacts with the game by
    issuing directional and action key inputs.

    Notes
    -----
    - The agent treats doors and barrels as non-blocking tiles
      but may need to open or destroy them during navigation.
    - Frontier exploration ensures coverage of the entire accessible
      dungeon before concluding the goal is unreachable.
    - Debug output and intermediate visualizations are optionally
      written to disk when certain debugging blocks are enabled.
    - The bot assumes the player is always located on a valid
      traversable tile and that the environment data is consistent
      across frames.
    """
    def __init__(self, game, args, view_radius, controlled_by_env=False):
        if not args.no_monsters:
            raise ValueError(f"Bot '{self.__class__.__name__}' can't be used with monsters, please provide '--no-monsters' option")
        if not args.harmless_barrels:
            raise ValueError(f"Bot '{self.__class__.__name__}' can't be used with explosive barrels, please provide '--harmless-barrels' option")

        assert type(view_radius) == int
        self.game = game
        self.view_radius = view_radius
        self.controlled_by_env = controlled_by_env
        self.rng = np.random.default_rng()
        self.seed = None
        self.attempt = 1

    @property
    def state_dir(self):
        seed_str = str(self.seed) if self.seed is not None else "UNKNOWN"
        return f"STATE-{seed_str}-{self.attempt}"

    def prepare_state_dir(self):
        while os.path.exists(self.state_dir):
            self.attempt += 1

    def reset(self, *, seed=None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.prepare_state_dir()

        # Do not expect a seed in the case of external control
        assert not self.controlled_by_env or seed is None

        if self.controlled_by_env:
            # Send the NOOP to ensure the game is fully initialized
            # before accessing the dungeon state
            key = ring.RingEntryType.RING_ENTRY_KEY_NOOP | \
                  ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
        else:
            seed_data = (0, 0)
            if seed is not None:
                seed_data = (1, seed)
            # Start new game
            key = ring.RingEntryType.RING_ENTRY_KEY_NEW | \
                  ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS

        self.game.submit_key(key, data=seed_data)

        d = self.game.safe_state
        env_whole = diablo_state.get_environment(d, show_invisible=True,
                                                 show_unexplored=True)
        env_empty = (
            (env_whole == 0) |
            (env_whole == diablo_state.EnvironmentFlag.Explored.value) |
            (env_whole == diablo_state.EnvironmentFlag.Visible.value) |
            (env_whole == (diablo_state.EnvironmentFlag.Explored.value |
                           diablo_state.EnvironmentFlag.Visible.value)) |
            # Consider a tile with a door, barrel, and item as
            # unoccupied. The door can be opened, the barrel can be
            # broken, and the item can be picked up (even though we
            # can pass through it)
            ((env_whole & (diablo_state.EnvironmentFlag.Door.value |
                           diablo_state.EnvironmentFlag.Barrel.value |
                           diablo_state.EnvironmentFlag.Item.value) != 0)))
        self.env_whole = env_whole
        self.env_empty = env_empty

        if self.controlled_by_env:
            self.goal_pos = self.game.goal_pos
            assert self.goal_pos and all(self.goal_pos)
        else:
            self.goal_pos = diablo_state.pick_random_empty_tile_pos(env_whole, self.rng)
        self.explored = np.zeros(env_whole.shape, dtype=bool)
        self.unexplored_points = set()
        self.current_path = []
        self.goal_found = False
        self.goal_reached = False
        self.steps_cnt = 0
        self.routes_rebuilds = 0


    def step(self) -> tuple[bool, int]:
        """Executes one reasoning-and-action cycle of the agent.

        The method performs the following operations:
          - Extracts the local environment around the player using
            `diablo_state.get_environment()`.
          - Updates the explored map and frontier tiles based on
            current visibility.
          - Selects a new exploration target (either a frontier tile
            or the goal itself).
          - Recomputes the shortest path if the path is lost or target
            changes.
          - Determines the next movement direction and corresponding
            action code.
          - Handles obstacles like barrels, doors, and items by
            issuing interaction actions (`X`) when necessary, ensuring
            the player faces the target object first.
          - Treats doors, barrels, and items as non-blocking tiles but
            may need to open or destroy them during navigation.
          - Returns `(True, ActionEnum.Stand.value)` if the goal
            tile has been reached, otherwise `(False, action)`
            where `action` encodes the next movement or action (see
            `diablo_env.ActionEnum`).
        """
        if self.goal_reached:
            # Nothing to do
            return True, diablo_env.ActionEnum.Stand.value

        self.steps_cnt += 1

        d = self.game.safe_state

        # Local player position right in the center of the view
        player_pos_local = (self.view_radius, self.view_radius)
        player_pos = diablo_state.player_position(d)
        offset = np.maximum(np.array(player_pos) - np.array(player_pos_local), (0,0))
        env_local = diablo_state.get_environment(d, radius=self.view_radius)

        def visibility_fn(p):
            shape = env_local.shape
            # Boundaires check
            if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
                return 0
            # Visibility check
            return 1 if (env_local[p] & diablo_state.EnvironmentFlag.Visible.value) else 0

        # Get cloud of visible tiles and frontier
        visible, frontier = \
            maze.visibility_frontier(player_pos_local, visibility_fn)

        unexplored_changed = False

        # Mark visible cloud of tiles as explored
        visible_arr = np.asarray(list(visible), dtype=np.int32)
        visible_global = visible_arr + offset
        self.explored[visible_global[:, 0], visible_global[:, 1]] = 1
        vis_tuples = { (int(pr), int(pc)) for pr, pc in visible_global.tolist() }
        if vis_tuples:
            unexplored_points_len = len(self.unexplored_points)
            self.unexplored_points -= vis_tuples
            unexplored_changed = (len(self.unexplored_points) != unexplored_points_len)

        # Check frontiers
        frontier_arr = np.asarray(list(frontier), dtype=np.int32)
        frontier_global = frontier_arr + offset
        for lr, lc, gr, gc in zip(frontier_arr[:,0], frontier_arr[:,1],
                                  frontier_global[:,0], frontier_global[:,1]):
            gpt = (gr, gc)
            # if already explored skip quickly
            if self.explored[gpt]:
                continue
            tile = env_local[(lr, lc)]
            # The closed door has a `wall` flag, so check the door
            # flag explicitly
            if (tile & diablo_state.EnvironmentFlag.Door.value or
                not (tile & diablo_state.EnvironmentFlag.Wall.value)):
                if gpt not in self.unexplored_points:
                    self.unexplored_points.add(gpt)
                    unexplored_changed = True

        def in_local_bounds(pos, shape):
            pos_arr = np.asarray(pos)
            return np.all((pos_arr >= 0) & (pos_arr < shape))

        goal_pos_local = tuple(self.goal_pos - offset)

        # Select next unexplored destination - either none, or the
        # last point in the current path
        next_unexplored = None if not self.current_path else self.current_path[-1]

        if (self.goal_found or
            # Goal pos within reach
            (in_local_bounds(goal_pos_local, env_local.shape) and
             env_local[goal_pos_local] & diablo_state.EnvironmentFlag.Visible.value)):
            # The target may momentarily disappear from the agent's view,
            # causing it to return to unexplored points. This can create
            # an endless loop: the agent sees the target, moves towards
            # the target, loses sight of the target, moves to an
            # unexplored point, sees the target, moves towards the
            # target... To prevent the loop, mark the goal as found.
            self.goal_found = True
            next_unexplored = self.goal_pos
        elif unexplored_changed:
            if not self.unexplored_points:
                # Hey, nothing to explore? No goal found?
                self.save_state(d, player_pos, next_unexplored, offset,
                                env_local, visible, frontier)
                assert 0, "Nothing to explore"

            manhattan_dist = lambda p: np.abs(p[0] - player_pos[0]) + np.abs(p[1] - player_pos[1])
            next_unexplored = min(self.unexplored_points, key=manhattan_dist)

        # We build a shortest path in two cases:
        #
        # - The player lost its path: the player's position is no
        #   longer on the path or the current path is empty.
        #
        # - The destination has changed: the next selected point
        #   does not match the last point in the current path.
        #
        if (player_pos not in self.current_path or
            next_unexplored not in self.current_path):
            def cost_fn(p):
                shape = self.env_empty.shape
                # Boundaires check
                if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
                    return -1
                # Cost. This is tricky - the goal point is always
                # reachable because the frontier tiles can contain
                # objects, which can't be considered unoccupied tiles
                # (e.g. solid objects), but we need to build a path to
                # these objects anyway.
                return 1 if (self.env_empty[p] or p == next_unexplored) else 0

            path = maze.shortest_path(player_pos, next_unexplored, cost_fn)
            if not path:
                self.save_state(d, player_pos, next_unexplored, offset,
                                env_local, visible, frontier)
                assert 0, "Path is empty"

            self.current_path = path
            self.routes_rebuilds += 1

        # Expect the player to be on the path, so ensure the current
        # path starts with a player position
        player_pos_index = self.current_path.index(player_pos)
        self.current_path = self.current_path[player_pos_index:]

        if self.current_path[0] == self.goal_pos:
            # Goal reached
            self.goal_reached = True
            return True, diablo_env.ActionEnum.Stand.value

        # Select the next point
        step_point = self.current_path[1]
        step_point_local = tuple(step_point - offset)
        move_dir = tuple(np.array(step_point_local) - np.array(player_pos_local))
        r, c = player_pos_local
        nr, nc = step_point_local

        def direction_to_action(direction):
            """Direction is (x,y), the (0,0) is in the upper left corner"""
            if direction == (-1, 1):
                return diablo_env.ActionEnum.Walk_SW.value
            elif direction == (0, 1):
                return diablo_env.ActionEnum.Walk_S.value
            elif direction == (1, 1):
                return diablo_env.ActionEnum.Walk_SE.value
            elif direction == (-1, 0):
                return diablo_env.ActionEnum.Walk_W.value
            elif direction == (1, 0):
                return diablo_env.ActionEnum.Walk_E.value
            elif direction == (-1, -1):
                return diablo_env.ActionEnum.Walk_NW.value
            elif direction == (0, -1):
                return diablo_env.ActionEnum.Walk_N.value
            elif direction == (1, -1):
                return diablo_env.ActionEnum.Walk_NE.value
            assert 0, f"Incorrect direction {direction}"

        diag_move = (sum(np.abs(move_dir)) == 2)
        barrel_or_item = (diablo_state.EnvironmentFlag.Barrel.value |
                          diablo_state.EnvironmentFlag.Item.value)

        barrel_or_item = lambda tile: (tile & (diablo_state.EnvironmentFlag.Barrel.value |
                                               diablo_state.EnvironmentFlag.Item.value))
        closed_door = lambda tile: (tile & diablo_state.EnvironmentFlag.Door.value and
                                    not (tile & diablo_state.EnvironmentFlag.Open.value))

        next_tile = env_local[step_point_local]
        action = 0

        if ((diag_move and (barrel_or_item(env_local[r][nc]) or
                            barrel_or_item(env_local[nr][c]))) or
            closed_door(next_tile) or barrel_or_item(next_tile)):

            # If diagonal move - find nearby objects like barrels
            # (which should be broken to pass through) or
            # items (just pick them up)
            #
            # If there is an object on the way - perform an action.
            #
            # But this is tricky - we need to open a door, but first
            # we need to face the door to ensure we open the correct
            # one and not another. Consider this interesting case:
            #
            #     ↘ D
            #     D #
            #
            # If the secondary action is pressed ('X'), we don't
            # actually know which specific door to open, so we need to
            # face it first. Otherwise, we continuously send actions,
            # but our door never opens.
            #
            # The same applies to a barrel and a door combination: we
            # want to break a barrel, and if we are not facing it we
            # can accidentally open a neighboring door instead.
            #
            match diablo_state.player_direction(d):
                case dx.Direction.North.value:
                    player_dir = (0, -1)
                case dx.Direction.NorthEast.value:
                    player_dir = (1, -1)
                case dx.Direction.East.value:
                    player_dir = (1, 0)
                case dx.Direction.SouthEast.value:
                    player_dir = (1, 1)
                case dx.Direction.South.value:
                    player_dir = (0, 1)
                case dx.Direction.SouthWest.value:
                    player_dir = (-1, 1)
                case dx.Direction.West.value:
                    player_dir = (-1, 0)
                case dx.Direction.NorthWest.value:
                    player_dir = (-1, -1)

            if move_dir != player_dir:
                action = direction_to_action(move_dir)
            else:
                action = diablo_env.ActionEnum.SecondaryAction.value
        else:
            action = direction_to_action(move_dir)

        # Perform an environment step if the bot has control
        if not self.controlled_by_env:
            key = diablo_env.DiabloEnv.action_to_key(action) | \
                  ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            self.game.submit_key(key)

        return False, action


    def save_state(self, d, player_pos, next_unexplored, offset,
                   env_local, visible, frontier):
        os.makedirs(self.state_dir, exist_ok=True)

        def copy_bot(bot, *attrs):
            new_bot = bot.__class__.__new__(bot.__class__)
            for k, v in bot.__dict__.items():
                if k == 'game':
                    state = SimpleNamespace(**dict(v.state.__dict__.items()))
                    setattr(new_bot, 'game_state', state)
                else:
                    setattr(new_bot, k, v)
            return new_bot

        # Unfortunately, `game` is not serializable by pickle, so do
        # some tricks
        this = copy_bot(self)

        # Save the whole state
        whole_state = {
            "bot": this,
            "player_pos": player_pos,
            "next_unexplored": next_unexplored,
            "offset": offset,
            "env_local": env_local,
            "visible": visible,
            "frontier": frontier
        }
        with open(f"{self.state_dir}/{self.steps_cnt}-whole_state.pkl", "wb") as f:
            pickle.dump(whole_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save empty environment, current position and a distanation if exists
        ee = self.env_empty.astype(int)
        ee[player_pos] = 2
        if next_unexplored:
            if ee[next_unexplored] > 0:
                ee[next_unexplored] = 3
            else:
                ee[next_unexplored] = 4
        np.savetxt(f"{self.state_dir}/{self.steps_cnt}-env_empty.txt", ee.T, fmt="%d")

        # Save vision
        ee = diablo_state.get_surroundings_by_env(d, env_local)

        up = []
        fr = list(frontier)
        for p_local in list(frontier):
            p = tuple(p_local + offset)
            if p in self.unexplored_points:
                up.append(p)
            else:
                fr.remove(p_local)

        for p_local in fr:
            ee[p_local] = "?"

        index = 0
        if player_pos in self.current_path:
            index = self.current_path.index(player_pos) + 1
        for p in self.current_path[index:]:
            p_local = tuple(p - offset)
            if not (0 <= p_local[0] < ee.shape[0] and 0 <= p_local[1] < ee.shape[1]):
                # The path may be beyond the current vision boundaries
                continue
            ee[p_local] = '*'

        with open(f"{self.state_dir}/{self.steps_cnt}-vision.txt", "w") as f:
            np.savetxt(f, ee.T, fmt="%s")
            print(f"env_whole sha256 '{checksum(self.env_whole)}'", file=f)
            print(f"frontier {[(int(t[0]),int(t[1])) for t in frontier]}", file=f)

            vis = []
            vis_explored = []

            for p_local in list(visible):
                p = tuple(p_local + offset)
                vis.append(p)
                if self.explored[p]:
                    vis_explored.append(p)
            print(f"visible {[(int(t[0]),int(t[1])) for t in vis]}", file=f)
            print(f"visible_explored {[(int(t[0]),int(t[1])) for t in vis_explored]}", file=f)

            up = []
            fr = list(frontier)
            for p_local in fr:
                p = tuple(p_local + offset)
                if p in self.unexplored_points:
                    up.append(p)
            print(f"unexplored {[(int(t[0]),int(t[1])) for t in self.unexplored_points]}", file=f)
            print(f"unexplored in frontier {[(int(t[0]),int(t[1])) for t in up]}", file=f)
            print(f"player_pos {player_pos}", file=f)
            print(f"next_unexplored {next_unexplored}", file=f)
            print(f"path {[(int(t[0]),int(t[1])) for t in self.current_path]}", file=f)
