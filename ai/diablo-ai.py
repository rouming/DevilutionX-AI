#!/usr/bin/env python3

"""Diablo AI tool

   Tool which trains AI for playing Diablo and helps to evalulate and
   play as human

Usage:
  diablo-ai.py play     [--attach=MEM_PATH_OR_PID] [--no-monsters] [--no-env-log] [--seed=SEED]
  diablo-ai.py play-ai  [--attach=MEM_PATH_OR_PID] [--no-monsters] [--seed=SEED]
  diablo-ai.py train-ai [--attach=MEM_PATH_OR_PID] [--no-monsters] [--seed=SEED] [--same-seed] [--train-batch-size=SIZE] [--train-iters=ITERS] [--gpus=NR] [--env-runners=NR] [--tune] [--log-to-stdout] [--no-actions] [--restore-from-checkpoint] [--save-to-checkpoint] [--exploration-door-attraction] [--exploration-door-backtrack-penalty]
  diablo-ai.py list
  diablo-ai.py (-h | --help)
  diablo-ai.py --version

Modes:
  play           Let the human play Diablo or attach to an existing Diablo instance (devilutionX process) by providing the --attach option.
  play-ai        Let AI play Diablo.
  train-ai       Train the AI by creating new workers and Diablo instances (devilutionX processes), or attach to a single existing instance by providing the --attach option (convenient for debug purposes).
  list           List all Diablo instances (devilutionX processes).

Options:
  -h --help
      Show this screen.

  --version
      Show version.

  --attach=MEM_PATH_OR_PID
      Attach to existing Diablo instance by path, pid or  an index of an instance from the `diablo-ai.py list` output. For example:
        Attach by PID:
           diablo-ai.py play --attach 112342

         Attach by path:
           diablo-ai.py play --attach /tmp/diablo-tj3bxyvy/shared.mem

         Attach by index:
           diablo-ai.py play --attach 0

  --no-monsters
    Disable all monsters on the level.

  --no-env-log
      No environment log is shown on the TUI screen, which can speed up refresh rate on slow terminals (default: enabled, meaning that if a log is produced by the Gymnasium environment, it will be displayed).

  --seed=SEED
      Initial seed (default: 0).

  --same-seed
      Set same seed for all runners (default: seed += worker_index - 1)

  --train-batch-size=SIZE
      Size of a train batch (default: 200)

  --train-iters=ITERS
      Number of train iterations (default: 3)

  --gpus=NR
      Number of GPUs (default: 0)

  --env-runners=NR
      Number of environment runners (default: 1)

  --tune
      Run hyperparameters tuner first

  --log-to-stdout
      Write logs to stdout (default: env.log file in Diablo state folder)

  --no-actions
      Disable agent from generating actions. Very handy mode to attach and play manually to simulate reward (default: actions enabled)

  --restore-from-checkpoint
      Restores state from a checkpoint prior training (default: no restore from a checkpoint)

  --save-to-checkpoint
      Saves state to a checkpoint on each train loop (default: no save to a checkpoint)

  --exploration-door-attraction
      Reward increases when the agent moves closer to unexplored doors. Helps guide the agent toward new areas faster (default: disabled).

  --exploration-door-backtrack-penalty
      Penalizes the agent for moving away from unexplored doors after approaching them. Encourages committing to exploration paths instead of turning back. If set, enables the `--exploration-door-attraction`.
"""
VERSION='Diablo AI Tool v1.1'

from docopt import docopt
from pathlib import Path
import collections
import configparser
import copy
import curses
import datetime
import enum
import json
import numpy as np
import os
import psutil
import re
import shlex
import subprocess
import sys
import tempfile
import time

import procutils

def delayed_import(binary_path):
    import devilutionx_generator
    devilutionx_generator.generate(binary_path)

    global dx
    global diablo_env
    global diablo_state
    global ring

    # First goes generated devilutionx
    import devilutionx as dx

    # Then others in any order
    import diablo_env
    import diablo_state
    import ring

# Deactivate the new API stack and switch back to the old one.
# New stack creates several environments (why???), which means we have
# one Diablo process idling. Also train results for old stack include
# reward.
NEW_API_STACK=False

# Flag to control the main loop
running = True
# Global variable to track the last key pressed
last_key = 0
# Global last tick
last_tick = 0

class EventsQueue:
    queue = None
    # Use Braille patterns for representing progress,
    # see here: https://www.unicode.org/charts/nameslist/c_2800.html
    progress = [0x2826, 0x2816, 0x2832, 0x2834]
    progress_cnt = 0
    def __init__(self):
        self.queue = collections.deque(maxlen=10)

# This is weird, but if you place a character in the last column,
# curses fills that position, yet still raises an error.
# These two wrappers attempt to ignore an error if it occurs
# when filling in the last position.
def _addstr(o, y, x, text):
    try:
        return o.addstr(y, x, text)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise
        pass

# See the comment for the _addstr
def _addch(o, y, x, ch):
    try:
        return o.addch(y, x, ch)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise
        pass

def truncate_line(line, N, extra='...'):
    if N <= len(extra):
        return ""
    return line[:N-len(extra)] + extra if len(line) > N else line

class EnvLog:
    fd = None
    queue = None

    def __init__(self, fd):
        self.fd = fd

def open_envlog(game):
    path = os.path.join(os.path.dirname(game.mshared_path), "env.log")
    fd = None
    try:
        fd = open(path, "r")
        return EnvLog(fd)
    except:
        pass
    return None

def display_env_log(game, stdscr, envlog):
    if envlog is None:
        return

    height, width = stdscr.getmaxyx()
    logwin_h = height // 2
    logwin_w = width // 4

    h = max(0, logwin_h - 2)
    w = max(0, logwin_w - 2)

    # Sane limitation
    if h < 10 or w < 20:
        return

    logwin = stdscr.subwin(logwin_h, logwin_w, 4, 1)

    if envlog.queue is None:
        queue = collections.deque(maxlen=h)
    elif envlog.queue.maxlen != logwin_h:
        queue = collections.deque(maxlen=h)
        for line in envlog.queue:
            queue.append(line)
    else:
        queue = envlog.queue

    while True:
        line = envlog.fd.readline()
        if not line:
            break
        queue.append(line)

    logwin.clear()
    logwin.border()
    msg = " Environment log "
    _addstr(logwin, 0, w//2 - len(msg)//2, msg)
    for i, line in enumerate(queue):
        line = truncate_line(line.strip(), w)
        _addstr(logwin, i+1, 1, line)
    logwin.refresh()

    envlog.queue = queue

def dump_cmd_output_to_file(dt, cmd, file_path):
    # Run the command
    result = subprocess.run(
        re.split(r"\s+", cmd),
        capture_output=True,
        text=True,
        check=True
    )

    # Write the output to a file
    with open(file_path, "a") as f:
        f.write(dt + "\n\n")
        f.write(result.stdout)
        f.write("\n")

def dump_dict_to_file(dt, dic, file_path):
    # PosixPath is not serializable, so give a hand to JSON
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(i) for i in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    # Dump dic
    with open(file_path, "a") as f:
        f.write(dt + "\n\n")
        json.dump(convert_paths(dic), f, indent=4)
        f.write("\n\n")

def dump_self_to_file(dt, begin, end, file_path):
    # Self-read
    script = Path(__file__).read_text()
    inside = False
    collected = []

    # Find special markers
    for line in script.splitlines():
        if line.strip() == begin:
            inside = True
        elif line.strip() == end:
            break
        elif inside:
            collected.append(line)

    # Save to file
    with open(file_path, "a") as f:
        f.write(dt + "\n\n")
        f.write("\n".join(collected) + "\n")
        f.write("\n")

def list_devilution_processes(binary_path, mshared_filename):
    result = procutils.find_processes_with_mapped_file(binary_path, mshared_filename)
    if result:
        for proc in result:
            print("%s\t%s" % (proc['pid'], proc['mshared_path']))

def handle_keyboard(stdscr):
    global last_key
    global running

    k = stdscr.getch()
    if k == -1:
        return False

    key = 0

    if k == 259:
        key = ring.RingEntryType.RING_ENTRY_KEY_UP
    elif k == 258:
        key = ring.RingEntryType.RING_ENTRY_KEY_DOWN
    elif k == 260:
        key = ring.RingEntryType.RING_ENTRY_KEY_LEFT
    elif k == 261:
        key = ring.RingEntryType.RING_ENTRY_KEY_RIGHT
    elif k == ord('a'):
        key = ring.RingEntryType.RING_ENTRY_KEY_A
    elif k == ord('b'):
        key = ring.RingEntryType.RING_ENTRY_KEY_B
    elif k == ord('x'):
        key = ring.RingEntryType.RING_ENTRY_KEY_X
    elif k == ord('y'):
        key = ring.RingEntryType.RING_ENTRY_KEY_Y
    elif k == ord('l'):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
    elif k == ord('s'):
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE
    elif k == ord('p'):
        key = ring.RingEntryType.RING_ENTRY_KEY_PAUSE
    elif k == ord('q'):
        running = False  # Stop the main loop

    last_key |= key

    return True

def remap_movement_keys(keys):
    # The dungeon in Diablo is rotated 45 degrees clockwise. To
    # compensate for this rotation and make UP true North, rather than
    # a diagonal movement, we should send two keys for each direction.

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

    return reskeys

def get_radius(d, dunwin):
    height, width = dunwin.getmaxyx()
    dundim = diablo_state.dungeon_dim(d)
    width = min(width, dundim[0])
    height = min(height, dundim[1])

    # Reduce the width by half to make the dungeon visually appear as
    # an accurate square when displayed in a terminal
    return np.array([width // 4, height // 2])

def get_events_as_string(game, events):
    advance_progress = False
    while (event := game.retrieve_event()) != None:
        keys = event.type
        k = "?"

        if keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                    ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # N
            k = "\u2191"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # NE
            k = "\u2197"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                      ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # E
            k = "\u2192"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN):
            # SE
            k = "\u2198"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                      ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # S
            k = "\u2193"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # SW
            k = "\u2199"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                      ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # W
            k = "\u2190"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_UP):
            # NW
            k = "\u2196"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_X):
            k = "X"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_Y):
            k = "Y"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_A):
            k = "A"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_B):
            k = "B"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_SAVE):
            k = "S"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_LOAD):
            k = "L"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_PAUSE):
            k = "P"

        events.queue.append(k)
        advance_progress = True

    if advance_progress:
        events.progress_cnt += 1

    cnt = 0
    s = ""
    for k in events.queue:
        s += " " + k
        cnt += 1

    events_str = " ." * (events.queue.maxlen - cnt) + s
    events_progress = chr(events.progress[events.progress_cnt % len(events.progress)])

    return events_str, events_progress

def display_matrix(dunwin, m):
    cols, rows = m.shape

    # The width is reduced by half (see `get_radius()`), so in order
    # to stretch the dungeon number of columns is multiplied by two
    cols *= 2

    # Get the screen size
    height, width = dunwin.getmaxyx()

    x_off = width // 2 - cols // 2
    y_off = height // 2 - rows // 2

    assert(x_off >= 0)
    assert(y_off >= 0)

    for row in range(rows):
        for col in range(0, cols, 2):
            _addch(dunwin, row + y_off, col + x_off, m[col//2, row])
            # "Stretch" the width by adding a space. With this simple
            # trick the dungeon should visually appear as an accurate
            # square in a terminal
            _addch(dunwin, row + y_off, col + x_off + 1, ' ')

def display_dungeon(d, stdscr):
    height, width = stdscr.getmaxyx()
    dunwin = stdscr.subwin(height - (4 + 1), width, 4, 0)
    radius = get_radius(d, dunwin)
    surroundings = diablo_state.get_surroundings(d, radius)

    display_matrix(dunwin, surroundings)

def display_diablo_state(game, stdscr, events, envlog, missed_ticks):
    # Unfortunately (performance-wise) we have to make a deep copy to
    # prevent partial or complete changes of the state in the middle
    # of this routine
    d = copy.deepcopy(game.state)
    pos = diablo_state.player_position(d)

    # Get the screen size
    height, width = stdscr.getmaxyx()

    msg = "Diablo ticks: %d (missed: %d); Kills: %003d; HP: %d; Pos: %d:%d; State: %-18s" % \
        (game.ticks(d),
         # Always 1 ticks behind
         missed_ticks - 1,
         np.sum(d.MonsterKillCounts_np),
         d.player._pHitPoints,
         pos[0], pos[1],
         dx.PLR_MODE(d.player._pmode).name)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 0, width // 2 - len(msg) // 2, msg)

    msg = "Press 'q' to quit"
    _addstr(stdscr, height - 1, width // 2 - len(msg) // 2, msg)

    msg = "Animation: ticksPerFrame %2d; tickCntOfFrame %2d; frames %2d; frame %2d" % \
        (d.player.AnimInfo.ticksPerFrame,
         d.player.AnimInfo.tickCounterOfCurrentFrame,
         d.player.AnimInfo.numberOfFrames,
         d.player.AnimInfo.currentFrame)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 1, width // 2 - len(msg) // 2, msg)

    obj_cnt = diablo_state.count_active_objects(d)
    items_cnt = diablo_state.count_active_items(d)
    total_hp = diablo_state.count_active_monsters_total_hp(d)
    events_str, events_progress = get_events_as_string(game, events)

    msg = "Total: mons HP %d, items %d, objs %d, lvl %d %c %s" % \
        (total_hp, items_cnt, obj_cnt, d.player.plrlevel,
         events_progress, events_str)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 2, width // 2 - len(msg) // 2, msg)

    display_dungeon(d, stdscr)
    display_env_log(game, stdscr, envlog)

    if diablo_state.is_game_paused(d):
        msgs = ["            ",
                " ┌────────┐ ",
                " │ Paused │ ",
                " └────────┘ ",
                "            "]
        h = height // 2
        for i, msg in enumerate(msgs):
            _addstr(stdscr, h + i, width // 2 - len(msg) // 2, msg)


def run_tui(stdscr, gameconfig):
    global running
    global last_key

    # Run or attach to Diablo
    game = diablo_state.DiabloGame.run_or_attach(gameconfig)

    # Disable cursor and enable keypad input
    curses.curs_set(0)
    stdscr.nodelay(True)

    missed_ticks = 0
    events = EventsQueue()
    envlog = None

    # Main loop
    while running:
        stdscr.clear()

        if not gameconfig['no-env-log'] and envlog is None:
            # Try to open a environment log, can be created later
            envlog = open_envlog(game)

        display_diablo_state(game, stdscr, events, envlog, missed_ticks)

        if last_key:
            # Compensate dungeon 45CW rotation
            key = remap_movement_keys(last_key)
            key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            game.submit_key(key)
            last_key = 0

        # Refresh the screen to show the content
        stdscr.refresh()

        # Handle keys and introduce delay until next tick
        while running and (handle_keyboard(stdscr) or game.same_ticks()):
            pass

        missed_ticks = game.update_ticks()

def ai(args, gameconfig):
    import torch

    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import FlattenObservations
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.utils.metrics import (
        ENV_RUNNER_RESULTS,
        EPISODE_RETURN_MEAN,
        EPISODE_LEN_MEAN,
    )
    import ray
    from ray import air, tune, train
    from ray.air.constants import TRAINING_ITERATION
    from ray.rllib.algorithms.ppo import PPO
    from ray.tune.registry import register_env
    from ray.tune.schedulers import create_scheduler
    from ray.rllib.utils.test_utils import check_learning_achieved

    class DiabloEnvCallback(RLlibCallback):
        # New API
        #def on_sample_end(self, *args, env_runner, metrics_logger, **kwargs):
        def on_sample_end(self, worker, samples):
            """Pause Diablo environment."""
            worker.env.pause_game(True)

    # Register Diablo Gym environment
    register_env("diablo", lambda cfg: diablo_env.DiabloEnv(cfg))

    # BEGIN CONFIG DUMP
    # Prepare config
    config = (
        PPOConfig()
        .environment("diablo", env_config=gameconfig)
        .resources(
            num_gpus=args['--gpus'],
#            num_cpus_per_worker=1,
#            num_gpus_per_worker=0,
        )
        .debugging(log_level='ERROR') # INFO, DEBUG, ERROR, WARN
        .framework('torch')
        .env_runners(
            num_envs_per_env_runner=1,
            num_env_runners=args['--env-runners'] if args['train-ai'] else 0,
            # Observations are discrete (ints) -> We need to flatten (one-hot) them.
            env_to_module_connector=lambda env: FlattenObservations(),
        )
        .callbacks(DiabloEnvCallback)
        .training(
            train_batch_size=args['--train-batch-size'],
            # Reduce learning rate
            lr=0.0001 if not args['--tune'] else tune.grid_search([0.01, 0.005, 0.003, 0.001, 0.0001]),
            # Reduce batch size, large batch sizes causes instability.
            # If batch is too large, the policy updates may be too slow to adapt,
            # or bad updates amplified.
#            train_batch_size_per_learner=4000,
#            minibatch_size=512,
            # Encourage exploration
            entropy_coeff=0.025,
        )
        .evaluation(evaluation_num_env_runners=0)
        .api_stack(enable_rl_module_and_learner=NEW_API_STACK,
                   enable_env_runner_and_connector_v2=NEW_API_STACK)
    )
    # END CONFIG DUMP

    if args['train-ai'] and args['--tune']:
        print("RUN TUNE")

        # ensure that checkpointing works.
        pbt = create_scheduler(
            "pbt",
            perturbation_interval=1,  # To make perturb more often.
            hyperparam_mutations={
                "train_loop_config": {
                    "lr": config.lr
                },
            },
        )

        # Get the best checkpoints from the trial, based on different metrics.
        # Checkpoint with the lowest policy loss value:
        if NEW_API_STACK:
            policy_loss_key = f"{LEARNER_RESULTS}/{DEFAULT_MODULE_ID}/policy_loss"
        else:
            policy_loss_key = "info/learner/default_policy/learner_stats/policy_loss"

        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=train.RunConfig(
                #verbose=1,
                stop={TRAINING_ITERATION: 1},
            ),
            tune_config=tune.TuneConfig(
                num_samples=1,
                metric=policy_loss_key,
                mode="min",
                scheduler=pbt,
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()
        best_config = best_result.config

        pl = best_result.metrics["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]

        print("TUNE FINISHED: lr=%f, policy_loss=%f" % (best_config['lr'], pl))

        config.lr = best_config['lr']

    algo = config.build()

    CHECKPOINT_PATH = os.path.abspath("./diablo.ppo.checkpoint")

    if not args['train-ai'] or \
       args['--restore-from-checkpoint'] and os.path.exists(CHECKPOINT_PATH):
        # Restore for play-ai mode or if told explicitly
        algo.restore(CHECKPOINT_PATH)

    if args['train-ai']:
        if args['--save-to-checkpoint']:
            os.makedirs(CHECKPOINT_PATH + "/state", exist_ok=True)

            now = datetime.datetime.now()
            dt = now.strftime("#\n# %Y-%m-%d %H-%M-%S\n#")

            # Dump exact command line for reproduction
            with open(CHECKPOINT_PATH + "/state/cmdline.txt", "a") as f:
                f.write(dt + "\n\n")
                f.write(shlex.join(sys.argv) + "\n")
                f.write("\n")

            # Dump git diff
            dump_cmd_output_to_file(dt, "git diff --no-color",
                                    CHECKPOINT_PATH + "/state/gitdiff.txt")

            # Dump git log
            dump_cmd_output_to_file(dt, "git log -n 20 --no-color",
                                    CHECKPOINT_PATH + "/state/gitlog.txt")

            # Dump gameconfig
            dump_dict_to_file(dt, gameconfig,
                              CHECKPOINT_PATH + "/state/gameconfig.json")

            # Dump algorithm config
            dump_self_to_file(dt,
                              "# BEGIN CONFIG DUMP",
                              "# END CONFIG DUMP",
                              CHECKPOINT_PATH + "/state/algoconfig.txt")

        for i in range(args['--train-iters']):
            results = algo.train()
            if args['--save-to-checkpoint']:
                algo.save(CHECKPOINT_PATH)

            reward = results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
            len_mean = results[ENV_RUNNER_RESULTS][EPISODE_LEN_MEAN]

            print("Finished iter %d/%d; avg. reward %.2f, episode_len_mean=%.2f" %
                  (i+1, args['--train-iters'], reward, len_mean))

        # Stop the algorithm. Note, this is important for when
        # defining `output_max_rows_per_file`. Otherwise,
        # remaining episodes in the `EnvRunner`s buffer isn't written to disk.
        algo.stop()
    else:
        # XXX This is a nasty kludge to prevent RLlib from spawning an extra runner
        # XXX for training, even though we do evaluation. The `self-evaluation`
        # XXX flag will be checked in the diablo_env.
        gameconfig['self-evaluation'] = True

        env = diablo_env.DiabloEnv(gameconfig)
        # Get the initial observation
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print("Finished 1 episode, total-reward=%.3f" % (total_reward))

def main():
    args = docopt(__doc__, version=VERSION)

    args["--seed"] = 0 if args["--seed"] is None \
        else int(args["--seed"])
    args["--train-batch-size"] = 200 if args["--train-batch-size"] is None \
        else int(args["--train-batch-size"])
    args["--train-iters"] = 3 if args["--train-iters"] is None \
        else int(args["--train-iters"])
    args["--env-runners"] = 1 if args["--env-runners"] is None \
        else int(args["--env-runners"])
    args["--gpus"] = 0 if args["--gpus"] is None \
        else int(args["--gpus"])
    args["--exploration-door-attraction"] = True if args["--exploration-door-backtrack-penalty"] \
        else args["--exploration-door-attraction"]

    config = configparser.ConfigParser()
    config.read('diablo-ai.ini')

    # Absolute path
    diablo_build_path  = Path(config['default']['diablo-build-path']).resolve()
    diablo_mshared_filename = config['default']['diablo-mshared-filename']

    if not diablo_build_path.is_dir() or len(diablo_mshared_filename) == 0:
        print("Error: initial configuration is invalid. Please check your 'diablo-ai.ini' file and provide valid paths for 'diablo-build-path' and 'diablo-mshared-filename' configuration options.")
        sys.exit(1)

    if not (diablo_build_path / "spawn.mpq").exists():
        print(f"Error: Shareware file \"spawn.mpq\" for Diablo content does not exist. Please download and place the file alongside the `devilutionx` binary with the following command:\n\twget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq -P {diablo_build_path}")
        sys.exit(1)

    diablo_bin_path = str(diablo_build_path / "devilutionx")
    delayed_import(diablo_bin_path)

    gameconfig = {
        "mshared-filename": diablo_mshared_filename,
        "diablo-bin-path": diablo_bin_path,

        "train-ai": args["train-ai"],
        "seed": args["--seed"],
        "same-seed": args["--same-seed"],
        "no-monsters": args["--no-monsters"],
        "log-to-stdout": args["--log-to-stdout"],
        "no-actions": args["--no-actions"],
        "no-env-log": args["--no-env-log"],
        "exploration-door-attraction": args["--exploration-door-attraction"],
        "exploration-door-backtrack-penalty": args["--exploration-door-backtrack-penalty"],
    }

    if args['--attach']:
        path_or_pid = args['--attach']

        if re.match(r'^\d+$', path_or_pid):
            pid_or_index = int(path_or_pid)
            procs = procutils.find_processes_with_mapped_file(
                diablo_bin_path, diablo_mshared_filename)
            if pid_or_index < len(procs):
                # Expect index to be a smaller number compared to PID
                index = pid_or_index
                proc = procs[index]
                gameconfig['attach-path'] = proc['mshared_path']
                gameconfig['attach-offset'] = proc['offset']
            else:
                pid = pid_or_index
                mshared_path, offset = procutils.get_mapped_file_and_offset_of_pid(
                    pid, diablo_mshared_filename)
                if mshared_path:
                    gameconfig['attach-path'] = mshared_path
                    gameconfig['attach-offset'] = offset
        elif os.path.exists(path_or_pid):
            mshared_path = path_or_pid
            procs = procutils.find_processes_with_mapped_file(
                diablo_bin_path, mshared_path)
            if len(procs) == 1:
                gameconfig['attach-path'] = mshared_path
                gameconfig['attach-offset'] = procs[0]['offset']

        if 'attach-path' not in gameconfig or 'attach-offset' not in gameconfig:
            print("Error: --attach=%s is not a valid path, PID or index of a Diablo instance" %
                  path_or_pid)
            sys.exit(1)

    if args['play']:
        # Run the curses application
        curses.wrapper(lambda stdscr: run_tui(stdscr, gameconfig))
    elif args['train-ai']:
        ai(args, gameconfig)
    elif args['play-ai']:
        ai(args, gameconfig)
    elif args['list']:
        list_devilution_processes(str(diablo_bin_path),
                                  diablo_mshared_filename)
    else:
        print("Not supported yet")

if __name__ == "__main__":
    main()
