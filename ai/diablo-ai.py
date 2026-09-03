#!/usr/bin/env python3

"""Diablo AI tool

   Tool which trains AI for playing Diablo and helps to evalulate and
   play as human

Author: Roman Penyaev <r.peniaev@gmail.com>
"""
from pathlib import Path
import argparse
import collections
import configparser
import copy
import curses
import json
import os
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import procutils
import sprout
from rl import utils
from rl.constants import KL_GOOD_HI, CLIP_FRAC_GOOD_HI, GRAD_NORM_GOOD_HI
from diablo_state import DungeonLevelSpec, DUNGEON_LEVEL_DEFAULT, CURRICULUM_WEIGHT_STEP

VERSION='Diablo AI Tool v2.0'

def set_sighandlers():
    # Silently terminate on Ctrl-C
    def do_exit(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, do_exit)

def set_rlimits():
    # Get current limits
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set new limits
    new_soft = min(65535, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

def parse_dungeon_level(s):
    """Parse dungeon level spec into DungeonLevelSpec with [(level, weight), ...].

    '8'                  -> level 8 only
    '1-8'                -> levels 1-8 uniform weight 1
    '1=5,2=15,3-16=80'  -> weighted
    '1-16=auto'          -> all levels auto-weighted from env-stats at runtime
    '1-3=auto,4-16=20'  -> L1-3 auto, L4-16 fixed weight 20
    """
    result = []
    auto_levels = set()
    for entry in s.split(','):
        if '=' in entry:
            lvl_part, w_part = entry.split('=', 1)
            if w_part == 'auto':
                weight = CURRICULUM_WEIGHT_STEP
                is_auto = True
            else:
                weight = int(w_part)
                is_auto = False
        else:
            lvl_part = entry
            weight = 1
            is_auto = False
        if '-' in lvl_part:
            lo_s, hi_s = lvl_part.split('-', 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(lvl_part)
        if not (0 <= lo <= hi <= 16):
            raise argparse.ArgumentTypeError(
                "dungeon level must be within 0-16, got %r" % entry)
        for lvl in range(lo, hi + 1):
            result.append((lvl, weight))
            if is_auto:
                auto_levels.add(lvl)
    if not result:
        raise argparse.ArgumentTypeError("empty dungeon level spec: %r" % s)
    return DungeonLevelSpec(sorted(result), auto_levels)

def _parse_stat_strategy_arg(s):
    from diablo_agent import AgentAI
    try:
        return AgentAI.validate_stat_strategy(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def _parse_model_spec_arg(s):
    from diablo_agent import AgentAI
    try:
        return AgentAI.validate_model_spec(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def _compress_level_ranges(levels):
    """Compress sorted list of ints into a single range string: [1,2,3,5] -> '1-3,5'."""
    if not levels:
        return ""
    parts = []
    start = end = levels[0]
    for lvl in levels[1:]:
        if lvl == end + 1:
            end = lvl
        else:
            parts.append(str(start) if start == end else f"{start}-{end}")
            start = end = lvl
    parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)

def _sprout_params_diff(key, old_val, new_val):
    """Diff callback for sprout tree. Returns list of diff lines or None to fall back."""
    if key not in ('dungeon_level', 'eval_dungeon_level'):
        return None
    try:
        new_spec = parse_dungeon_level(new_val)
        old_spec = parse_dungeon_level(old_val) if old_val is not None else DUNGEON_LEVEL_DEFAULT
    except Exception:
        return None

    def _w(lvl, spec):
        w = dict(spec).get(lvl, 0)
        return "auto" if lvl in spec.auto_levels else w

    change_groups = {}
    for lvl in range(1, 17):
        ov, nv = _w(lvl, old_spec), _w(lvl, new_spec)
        if ov != nv:
            change_groups.setdefault((ov, nv), []).append(lvl)
    if not change_groups:
        return None
    entries = []
    for (ov, nv), lvls in sorted(change_groups.items(), key=lambda x: x[1][0]):
        rng = _compress_level_ranges(sorted(lvls))
        entries.append((rng, ov, nv))
    width = max(len(rng) for rng, _, _ in entries)
    lines = [f"⇾ {key}:"]
    for rng, ov, nv in entries:
        pad = " " * (width - len(rng))
        if ov == 0:
            lines.append(f"    {pad}{rng}: {nv}")
        else:
            lines.append(f"    {pad}{rng}: {ov} -> {nv}")
    return lines

def parse_int_with_suffix(value: str) -> int:
    """Parse integer with optional k, M, G suffix or scientific notation."""
    value = value.strip().upper()
    if value.endswith('K'):
        return int(float(value[:-1]) * 1_000)
    if value.endswith('M'):
        return int(float(value[:-1]) * 1_000_000)
    if value.endswith('G'):
        return int(float(value[:-1]) * 1_000_000_000)
    return int(float(value))  # handles 50e6, 1e3, etc.

def fmt_int_with_suffix(value: int) -> str:
    """Format integer with suffix (G/M/K). Exact divisibility wins; otherwise floor to M or K."""
    for div, sfx in [(1_000_000_000, 'G'), (1_000_000, 'M'), (1_000, 'K')]:
        if value % div == 0:
            return f"{value // div}{sfx}"
    if value >= 1_000_000:
        return f"{value // 1_000_000}M"
    if value >= 1_000:
        return f"{value // 1_000}K"
    return str(value)

def resolve_frames(frames_str, current_frames):
    """Resolve --frames value against current frame count.

    Plain N: absolute target.
    +N: snap to the next multiple of N above current_frames.
    Returns the resolved frame count as formatted string.
    """
    if frames_str.startswith('+'):
        n = parse_int_with_suffix(frames_str[1:])
        return fmt_int_with_suffix((current_frames // n + 1) * n)
    return frames_str


class FloatRangeSpec(tuple):
    """(min_pct, max_pct) int tuple; str() produces a re-parseable 'lo-hi' fraction string."""
    def __str__(self):
        lo, hi = self[0], self[1]
        return f"{lo/100:g}" if lo == hi else f"{lo/100:g}-{hi/100:g}"
    def __repr__(self):
        return str(self)

class IntRangeSpec(tuple):
    """(min, max) int tuple; str() produces a re-parseable 'lo-hi' string."""
    def __str__(self):
        lo, hi = self[0], self[1]
        return f"{lo}" if lo == hi else f"{lo}-{hi}"
    def __repr__(self):
        return str(self)

def parse_float_range(s):
    """Parse 'min-max' or single float as a FloatRangeSpec (0-100 int percentages).

    '0.4-1'  -> FloatRangeSpec((40, 100))
    '1'      -> FloatRangeSpec((100, 100))
    '0.5'    -> FloatRangeSpec((50, 50))
    """
    parts = s.split('-')
    if len(parts) == 2:
        lo, hi = float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        lo = hi = float(parts[0])
    else:
        raise argparse.ArgumentTypeError(f"invalid range '{s}': expected 'min-max' or single value")
    if not (0.0 <= lo <= hi <= 1.0):
        raise argparse.ArgumentTypeError(f"invalid range '{s}': values must be in [0, 1] with min <= max")
    return FloatRangeSpec((round(lo * 100), round(hi * 100)))

def parse_int_range(s):
    """Parse 'min-max' or single int as an IntRangeSpec.

    '2-20'  -> IntRangeSpec((2, 20))
    '10'    -> IntRangeSpec((10, 10))
    """
    parts = s.split('-')
    if len(parts) == 2:
        lo, hi = int(parts[0]), int(parts[1])
    elif len(parts) == 1:
        lo = hi = int(parts[0])
    else:
        raise argparse.ArgumentTypeError(f"invalid range '{s}': expected 'min-max' or single value")
    if lo > hi:
        raise argparse.ArgumentTypeError(f"invalid range '{s}': min must be <= max")
    return IntRangeSpec((lo, hi))

# Params excluded from sprout storage and post-run-defaults display.
# These are operational flags (attach, cont) or internal bookkeeping (model,
# demos, no_drop_best) that are not training hyperparameters.
SPROUT_SKIP_PARAMS = {"model", "demos", "cont", "no_drop_best", "best_drop", "load_best", "attach", "help"}

class DiabloParserNamespace(argparse.Namespace):
    @property
    def frames_int(self):
        return parse_int_with_suffix(self.frames)

    @property
    def episodes_int(self):
        return parse_int_with_suffix(self.episodes)


def make_diablo_parser():
    class IndentedHelpFormatter(argparse.RawTextHelpFormatter):
        def __init__(self, *args, **kwargs):
            # Width controls line wrapping; max_help_position controls indent
            kwargs['max_help_position'] = 8
            super().__init__(*args, **kwargs)

    # Define incompatible options
    incompatible_options = {
        '--char-tables': ['--stats-scale', '--eval-stats-scale'],
        '--attach': ['--game-ticks-per-step',
                     '--step-mode',
                     '--invincible-player',
                     '--no-monsters',
                     '--blind-monsters',
                     '--harmless-barrels',
                     '--no-butcher',
                     '--enable-quests',
                     '--seed-base',
                     '--fixed-seed']
    }

    parser = argparse.ArgumentParser(
        prog="diablo-ai.py",
        description=(
            VERSION + "\n\n"
            "Tool which trains AI for playing Diablo and helps to evalulate and play as human.\n\n"
        ),
        epilog="For more details, see https://github.com/rouming/DevilutionX-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=VERSION)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common reusable options
    common_parser = argparse.ArgumentParser(add_help=False)
    # See also `incompatible_options`
    common_parser.add_argument(
        "--attach", metavar="MEM_PATH_OR_PID",
        help=("Attach to existing Diablo instance by path, pid or an index of an instance from the `diablo-ai.py list` output. For example:\n"
              "  Attach by PID:\n"
              "    diablo-ai.py play --attach 112342\n"
              "\n"
              "  Attach by path:\n"
              "    diablo-ai.py play --attach /tmp/diablo-tj3bxyvy/shared.mem\n"
              "\n"
              "  Attach by index:\n"
              "    diablo-ai.py play --attach 0"
              )
    )
    common_parser.add_argument(
        "--view-radius", type=int, default=10,
        help="Number of environment cells surrounding the AI agent. Set to 0 if want the whole dungeon (default: 10)")
    common_parser.add_argument(
        "--game-ticks-per-step", type=int, default=10,
        help="Number of game ticks per a single step (default: 10)")
    common_parser.add_argument(
        "--real-time", action="store_true", default=False,
        help="Run the game loop in real-time, as opposed to step mode (default: false)")
    common_parser.add_argument(
        "--gui", action="store_true",
        help="Start Diablo in GUI mode only")
    # See also `incompatible_options`
    common_parser.add_argument(
        "--invincible-player", action="store_true",
        help="Enable invincible player mode")
    # See also `incompatible_options`
    common_parser.add_argument(
        "--no-monsters", action="store_true",
        help="Disable all monsters on the level")
    # See also `incompatible_options`
    common_parser.add_argument(
        "--blind-monsters", action="store_true",
        help="Monsters stand still and don't react to the player")
    # See also `incompatible_options`
    common_parser.add_argument(
        "--harmless-barrels", action="store_true",
        help="Disable explosive barrels, urns, or pods")
    common_parser.add_argument(
        "--no-butcher", action="store_true",
        help="Skip placing The Butcher on level 2")
    common_parser.add_argument(
        "--enable-quests", action="store_true",
        help="Enable quests (default: quests are disabled; use this to re-enable them)")
    common_parser.add_argument(
        "--spell-potency", type=float, default=0.0, metavar="PROB",
        help="Spell potency multiplier [0.0, 1.0]: 0=normal engine rules, 1=spells one-shot any monster (default: 0.0)")
    common_parser.add_argument(
        "--no-spells", action="store_true",
        help="Disable hero spells at episode start (spell slots stay empty, mana set to 0)")
    common_parser.add_argument(
        "--char-tables", action="store_true", default=True, dest="char_tables",
        help="Use char tables from [devilutionx-gameplay] in diablo-ai.ini (default: on)")
    common_parser.add_argument(
        "--no-char-tables", action="store_false", dest="char_tables",
        help="Disable char tables; required when using --stats-scale or --eval-stats-scale")
    common_parser.add_argument(
        "--stat-strategy", type=_parse_stat_strategy_arg, default='dex-rush',
        help="Stat allocation strategy when --char-tables is used and char level up attrs is not set in the ini: "
             "named preset ('dex-rush', 'str-vit') or per-level spec '1-7=2s2v1d,8-*=5s' (s/m/d/v, N-* open-ended, sum=5)")
    common_parser.add_argument(
        "--stats-scale", type=float, default=1.0, metavar="SCALE",
        help="Scale hero stat midpoints by this factor (1.0=full stats, 0.7=70%% midpoints, noise unchanged; default: 1.0)")
    common_parser.add_argument(
        "--hero-hp-at-start", type=parse_float_range, default=FloatRangeSpec((100, 100)),
        metavar="MIN-MAX",
        help="Hero HP fraction at episode start as a range in [0.0, 1.0] "
             "(e.g. '0.4-1' draws uniformly; single value sets a fixed fraction; default: 1)")
    common_parser.add_argument(
        "--hero-mana-at-start", type=parse_float_range, default=FloatRangeSpec((100, 100)),
        metavar="MIN-MAX",
        help="Hero mana fraction at episode start as a range in [0.0, 1.0] "
             "(e.g. '0.4-1' draws uniformly; single value sets a fixed fraction; default: 1)")
    common_parser.add_argument(
        "--hero-potions-at-start", type=parse_int_range, default=IntRangeSpec((2, 20)),
        metavar="MIN-MAX",
        help="Number of potions at episode start as an integer range "
             "(e.g. '2-20'; single value sets a fixed count; default: 2-20)")
    common_parser.add_argument(
        "--seed", type=int, default=0,
        help="Initial global experiment seed (controls PyTorch, numpy, RNGs, etc) (default: 0)")
    # See also `incompatible_options`
    common_parser.add_argument(
        "--seed-base", type=int, default=0,
        help="Base value used to generate deterministic seeds for each episode or environment runner, so the i-th episode/runner uses `seed_base + i` (default: 0)")
    common_parser.add_argument(
        "--dungeon-level", type=parse_dungeon_level, default=DUNGEON_LEVEL_DEFAULT,
        help="Starting dungeon level: '8', '1-8' (uniform range), "
             "'1=5,2=15,3-16=80' (weighted; weight proportional to frequency), "
             "or '1-16=auto' / '1-3=auto,4-16=20' (auto weights from env-stats.txt "
             "for marked levels; missing levels floored at CURRICULUM_WEIGHT_STEP). "
             "Default: 1")

    #
    # sprout: reuse sprout's parser
    #
    sprout_parser = sprout.build_parser(
        prog="diablo-ai.py sprout",
        suppress_working_dir=True, add_help=False)
    sprout_parser = subparsers.add_parser(
        "sprout", parents=[sprout_parser],
        help="Access AI models through Sprout snapshot manager")

    #
    # play
    #
    play_parser = subparsers.add_parser(
        "play", parents=[common_parser],
        help="Let the human play Diablo or attach to an existing Diablo instance (devilutionX process) by providing the `--attach` option.",
        formatter_class=IndentedHelpFormatter)
    play_parser.add_argument(
        "--no-env-log", action="store_true",
        help="Disable environment log on TUI screen.")
    # See also `incompatible_options`
    play_parser.add_argument(
        "--fixed-seed", action="store_true",
        help="Every new game starts with the same initial seed, so the game world (dungeon) is identical each environment reset")

    #
    # common_ai
    #
    common_ai_parser = argparse.ArgumentParser(add_help=False)
    common_ai_parser.add_argument(
        "--cnn-arch", required=True,
        choices=["cnn1", "cnn2", "cnn3", "cnn31", "cnn32", "cnn32expert", "cnn35", "cnn4"],
        help="Architecture of the CNN to use: cnn1 | cnn2 | cnn3 | cnn31 | cnn32 | cnn32expert | cnn35 | cnn4")
    common_ai_parser.add_argument(
        "--embedding-dim", type=int, default=256,
        help="dimension of embeddings (default: 256)")
    common_ai_parser.add_argument(
        "--no-actions", action="store_true",
        help="Disable agent actions (manual play mode).")


    #
    # play-ai
    #
    play_ai_parser = subparsers.add_parser(
        "play-ai", parents=[common_parser, common_ai_parser],
        help="Let AI play Diablo.",
        formatter_class=IndentedHelpFormatter)
    play_ai_parser.add_argument(
        "--env-runners", type=int, default=1,
        help="Number of environment runners or processes (default: 1)")
    play_ai_parser.add_argument(
        "--env", required=True,
        help="Name of the environment to be run (REQUIRED)")
    play_ai_parser.add_argument(
        "--model", required=True,
        type=_parse_model_spec_arg,
        help="Model name, or per-level spec: 'N-M=model,N-*=model' (REQUIRED)\n"
             "Example: --model 1-4=EarlyModel,5-16=FullModel")
    play_ai_parser.add_argument(
        "--best", action="store_true", default=False,
        help="Loads best model from the folder")
    play_ai_parser.add_argument(
        "--argmax", action="store_true", default=False,
        help="Select the action with highest probability (default: False)")
    play_ai_parser.add_argument(
        "--pause", type=float, default=0,
        help="Pause duration in seconds between two consequent actions of the agent (default: 0)")
    play_ai_parser.add_argument(
        "--episodes", type=str, default='1',
        help="Number of episodes to evaluate (default: 1)")

    #
    # play-bot
    #
    play_bot_parser = subparsers.add_parser(
        "play-bot", parents=[common_parser],
        help="Let Bot play Diablo.",
        formatter_class=IndentedHelpFormatter)
    play_bot_parser.add_argument(
        "--env-runners", type=int, default=1,
        help="Number of environment runners or processes (default: 1)")
    play_bot_parser.add_argument(
        "--bot", type=str, default="FindRandomGoal_Bot",
        help="Name of the bot to be run (default: FindRandomGoal_Bot)")
    play_bot_parser.add_argument(
        "--pause", type=float, default=0,
        help="Pause duration in seconds between two consequent actions of the bot (default: 0)")
    play_bot_parser.add_argument(
        "--episodes", type=str, default='1',
        help="Number of episodes to evaluate (default: 1)")

    #
    # agent-ai
    #
    agent_ai_parser = subparsers.add_parser(
        "agent-ai", parents=[common_parser, common_ai_parser],
        help="Run the algorithmic agent supervisor with a trained RL model "
             "through the full game (town + all 16 dungeon levels).",
        formatter_class=IndentedHelpFormatter)
    agent_ai_parser.add_argument(
        "--env", required=True,
        help="Name of the environment the model was trained on (REQUIRED)")
    agent_ai_parser.add_argument(
        "--model", required=True,
        help="Name of the trained model (REQUIRED)")
    agent_ai_parser.add_argument(
        "--best", action="store_true", default=False,
        help="Load best checkpoint instead of the latest")
    agent_ai_parser.add_argument(
        "--argmax", action="store_true", default=False,
        help="Select the action with highest probability instead of sampling (default: False)")
    agent_ai_parser.add_argument(
        "--kill-threshold", type=float, default=0.5,
        help="Fraction of monsters to kill before pathfinding to stairs (default: 0.5)")
    agent_ai_parser.add_argument(
        "--repair-threshold", type=float, default=0.25,
        help="Repair equipped gear when durability falls below this fraction (default: 0.25)")
    agent_ai_parser.add_argument(
        "--max-steps-per-level", type=int, default=3000,
        help="Step budget per level before forcing pathfind to stairs (default: 3000)")
    agent_ai_parser.add_argument(
        "--no-gear-management", action="store_true", default=False,
        help="Disable all gear management (equip and repair)")
    agent_ai_parser.add_argument(
        "--safe-radius", type=int, default=2,
        help="Monster-free radius required for gear management: 0=always, N=only when no monster within N tiles (default: 2)")
    agent_ai_parser.add_argument(
        "--pause", type=float, default=0,
        help="Seconds to pause between agent steps, useful in GUI mode (default: 0)")

    #
    # train-ai
    #
    train_ai_parser = subparsers.add_parser(
        "train-ai", parents=[common_parser, common_ai_parser],
        help="Train the RL model by creating new workers and Diablo instances (devilutionX processes), or attach to a single existing instance by providing the `--attach` option (convenient for debug purposes).",
        formatter_class=IndentedHelpFormatter)
    # General game env parameters
    train_ai_parser.add_argument(
        "--log-to-stdout", action="store_true",
        help="Write logs to stdout instead of env.log.")
    train_ai_parser.add_argument(
        "--exploration-door-attraction", action="store_true",
        help="Reward for approaching unexplored doors.")
    train_ai_parser.add_argument(
        "--exploration-door-backtrack-penalty", action="store_true",
        help="Penalty for moving away from unexplored doors.")
    train_ai_parser.add_argument(
        "--goal-far-bias", action="store_true",
        help="Bias random goal placement toward far rooms, avoiding trivial near-goal episodes.")

    # General RL parameters
    train_ai_parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not apply gradient updates if set")
    train_ai_parser.add_argument(
        "--algo",
        choices=["a2c", "ppo"], default="ppo",
        help="Algorithm to use: a2c | ppo")
    train_ai_parser.add_argument(
        "--env", required=True,
        help="name of the environment to train on (REQUIRED)")
    train_ai_parser.add_argument(
        "--model", required=True,
        help="Name of the model (REQUIRED)")
    train_ai_parser.add_argument(
        "--continue", action="store_true", dest="cont",
        help="Continue training without taking a snapshot of the model before training begins")
    train_ai_parser.add_argument(
        "--best", action="store_true", dest="load_best",
        help="Load best-status.pt (highest eval score) instead of the latest status.pt")
    best_grp = train_ai_parser.add_mutually_exclusive_group()
    best_grp.add_argument(
        "--no-drop-best", "--no-best-drop", action="store_true", dest="no_drop_best",
        help="Keep the best-status snapshot (by default it is dropped on each run that creates a new snapshot, i.e. without --continue)")
    best_grp.add_argument(
        "--drop-best", "--best-drop", action="store_true", dest="best_drop",
        help="Drop the best-status snapshot even when using --continue")
    train_ai_parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs to use for DDP training (default: 1, no DDP)")
    train_ai_parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Number of updates between two logs; 0 means no logs (default: 1)")
    train_ai_parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Number of updates between two saves; 0 means no saving (default: 10)")
    train_ai_parser.add_argument(
        "--env-runners", type=int, default=1,
        help="Number of environment runners or processes (default: 1)")
    train_ai_parser.add_argument(
        "--frames", type=str, default='10M',
        help="Number of frames of training; prefix with + to snap to next multiple of N\n"
             "above current (e.g. +50M from 1178M -> 1200M, re-run stays at 1200M) (default: 10M)")

    # Parameters for main RL algorithm
    train_ai_parser.add_argument(
        "--epochs", type=int, default=4,
        help="Number of epochs for PPO (default: 4)")
    train_ai_parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for PPO (default: 256)")
    train_ai_parser.add_argument(
        "--frames-per-env-runner", type=int, default=None,
        help="Number of frames per environment runner (process) before update (default: 8 for A2C and 128 for PPO)")
    train_ai_parser.add_argument(
        "--discount", type=float, default=0.99,
        help="Discount factor (default: 0.99)")
    train_ai_parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)")
    train_ai_parser.add_argument(
        "--lr-steps", type=float, default=100,
        help="Number of steps (period) of LR decay (default: 100)")
    train_ai_parser.add_argument(
        "--lr-gamma", type=float, default=1,
        help="Factor of LR decay; 1 means no decay (default: 1)")
    train_ai_parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="Lambda coefficient in GAE formula; 1 means no gae (default: 0.95)")
    train_ai_parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="Entropy term coefficient (default: 0.01)")
    train_ai_parser.add_argument(
        "--value-loss-coef", type=float, default=0.5,
        help="Value loss term coefficient (default: 0.5)")
    train_ai_parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="Maximum norm of gradient (default: 0.5)")
    train_ai_parser.add_argument(
        "--optim-eps", type=float, default=1e-8,
        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    train_ai_parser.add_argument(
        "--optim-alpha", type=float, default=0.99,
        help="RMSprop optimizer alpha (default: 0.99)")
    train_ai_parser.add_argument(
        "--clip-eps", type=float, default=0.2,
        help="Clipping epsilon for PPO (default: 0.2)")
    train_ai_parser.add_argument(
        "--recurrence", type=int, default=1,
        help="Number of time-steps gradient is backpropagated; If > 1, a LSTM is added to the model to have memory (default: 1)")

    # Evaluation parameters
    train_ai_parser.add_argument(
        "--eval-seed", type=int, default=int(1e9),
        help="Seed for environment used for evaluation (default: 1e9)")
    train_ai_parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of episodes used to evaluate the agent (default: 10)")
    train_ai_parser.add_argument(
        "--eval-env-runners", type=int, default=64,
        help="Number of environment runners dedicated to evaluation (default: 64)")
    train_ai_parser.add_argument(
        "--eval-dungeon-level", type=parse_dungeon_level, default=DUNGEON_LEVEL_DEFAULT,
        help="Dungeon level spec for eval environments (default: 1)")
    train_ai_parser.add_argument(
        "--eval-no-spells", action="store_true",
        help="Disable hero spells for eval environments (default: spells enabled)")
    train_ai_parser.add_argument(
        "--no-eval-char-tables", "--eval-no-char-tables", action="store_true",
        dest="no_eval_char_tables",
        help="Disable char tables for eval even when --char-tables is set (e.g. to compare against legacy eval)")
    train_ai_parser.add_argument(
        "--eval-stats-scale", type=float, default=1.0, metavar="SCALE",
        help="Scale hero stat midpoints for eval environments (default: 1.0)")
    train_ai_parser.add_argument(
        "--stats-episodes", type=int, default=1000,
        help="Number of recent training episodes for env-stats file (default: 1000)")
    train_ai_parser.add_argument(
        "--eval-stats-episodes", type=int, default=1000,
        help="Number of recent eval episodes for eval env-stats file (default: 1000)")
    train_ai_parser.add_argument(
        "--eval-hero-hp-at-start", type=parse_float_range, default=FloatRangeSpec((100, 100)),
        metavar="MIN-MAX",
        help="Hero HP fraction for eval environments (default: 1)")
    train_ai_parser.add_argument(
        "--eval-hero-mana-at-start", type=parse_float_range, default=FloatRangeSpec((100, 100)),
        metavar="MIN-MAX",
        help="Hero mana fraction for eval environments (default: 1)")
    train_ai_parser.add_argument(
        "--eval-hero-potions-at-start", type=parse_int_range, default=IntRangeSpec((2, 20)),
        metavar="MIN-MAX",
        help="Potion count for eval environments (default: 2-20)")
    train_ai_parser.add_argument(
        "--eval-kill-threshold", type=float, default=0.0, metavar="RATIO",
        help="Eval success when killed/total >= RATIO (0..1]; 0 disables (default: 0)")
    train_ai_parser.add_argument(
        "--eval-max-steps-per-level", type=int, default=0, metavar="STEPS",
        help="Eval success when agent survives this many steps (0 disables, default: 0)")
    train_ai_parser.add_argument(
        "--eval-no-stuck-timeout", action="store_true",
        help="Disable stuck-timeout during eval episodes")

    #
    # demos-il
    #
    demos_il_parser = subparsers.add_parser(
        "demos-il", parents=[common_parser],
        help="Generate demo episodes for IL (imitation learning) training",
        formatter_class=IndentedHelpFormatter)

    demos_il_parser.add_argument(
        "--bot", type=str, default="FindRandomGoal_Bot",
        help="Name of the bot to be run (default: FindRandomGoal_Bot)")
    demos_il_parser.add_argument(
        "--env", required=True,
        help="Name of the environment to be run (REQUIRED)")
    demos_il_parser.add_argument(
        "--model", required=True,
        help="name of the trained model (REQUIRED)")
    demos_il_parser.add_argument(
        "--continue", action="store_true", dest="cont",
        help="Continue training without taking a snapshot of the model before training begins")
    demos_il_parser.add_argument(
        "--env-runners", type=int, default=1,
        help="Number of environment runners or processes (default: 1)")
    group = demos_il_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--for-training", action="store_true",
        help="Generate demo episodes for training, big set (REQUIRED)")
    group.add_argument(
        "--for-validation", action="store_true",
        help="Generate demo episodes for validation, small set (REQUIRED)")
    demos_il_parser.add_argument(
        "--episodes", type=str, default='1K',
        help="Number of demo episodes to generate (default: 1K)")
    demos_il_parser.add_argument(
        "--skip-seeds", type=str,
        help="Comma-separated list of seeds to be skipped during demo episode generation")
    demos_il_parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Interval between progress reports; 0 means no logs (default: 1)")
    demos_il_parser.add_argument(
        "--save-interval", type=int, default=1,
        help="Interval between demonstrations saving; 0 means no saving (default: 1)")
    demos_il_parser.add_argument(
        "--no-drop-best", "--no-best-drop", action="store_true", dest="no_drop_best",
        help="Keep the best-status snapshot (by default it is dropped on each run that creates a new snapshot, i.e. without --continue)")


    #
    # train-il
    #
    train_il_parser = subparsers.add_parser(
        "train-il", parents=[common_parser, common_ai_parser],
        help="Train the RL model using imitation learning (IL) by creating new workers and Diablo instances (devilutionX processes), or attach to a single existing instance by using the `--attach` option (convenient for debugging purposes).",
        formatter_class=IndentedHelpFormatter)

    train_il_parser.add_argument(
        "--bot", type=str, default="FindRandomGoal_Bot",
        help="Name of the bot to be run (default: FindRandomGoal_Bot)")
    train_il_parser.add_argument(
        "--episodes", type=str, default='0',
        help="Number of episodes of demonstrations to use (default: 0, meaning all demos)")
    train_il_parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size of demo episodes for training (default: 256)")
    train_il_parser.add_argument(
        "--epoch-length", type=int, default=0,
        help="Number of demo episodes per epoch; the batch size is used if 0 (default: 0)")

    group = train_il_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--phase1", action="store_true",
        help="Run phase 1: train the policy (actor)"
    )
    group.add_argument(
        "--phase2", action="store_true",
        help="Run phase 2: warm-up the critic only"
    )
    group.add_argument(
        "--phase3", action="store_true",
        help="Run phase 3: train the policy (actor) and critic together"
    )
    # BEGIN common with `train-ai`

    train_il_parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="Entropy term coefficient (default: 0.01)")
    train_il_parser.add_argument(
        "--value-loss-coef", type=float, default=1,
        help="Value loss term coefficient; used only for phase 3 (default: 1)")
    train_il_parser.add_argument(
        "--recurrence", type=int, default=1,
        help="Number of time-steps gradient is backpropagated; If > 1, a LSTM is added to the model to have memory (default: 1)")
    train_il_parser.add_argument(
        "--env-runners", type=int, default=1,
        help="Number of environment runners or processes (default: 1)")
    train_il_parser.add_argument(
        "--frames", type=str, default='10M',
        help="Number of frames of training (default: 10M)")

    train_il_parser.add_argument(
        "--env", required=True,
        help="name of the environment to train on (REQUIRED)")
    train_il_parser.add_argument(
        "--model", required=True,
        help="Name of the model (REQUIRED)")
    train_il_parser.add_argument(
        "--continue", action="store_true", dest="cont",
        help="Continue training without taking a snapshot of the model before training begins")
    train_il_parser.add_argument(
        "--no-drop-best", "--no-best-drop", action="store_true", dest="no_drop_best",
        help="Keep the best-status snapshot (by default it is dropped on each run that creates a new snapshot, i.e. without --continue)")
    train_il_parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Number of updates between two logs; 0 means no logs (default: 1)")
    train_il_parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)")
    train_il_parser.add_argument(
        "--lr-delicate", type=float, default=0.0001,
        help="Learning rate for delicate group; used only for phase 3 (default: 0.0001)")
    train_il_parser.add_argument(
        "--lr-steps", type=float, default=100,
        help="Number of steps (period) of LR decay (default: 100)")
    train_il_parser.add_argument(
        "--lr-gamma", type=float, default=1,
        help="Factor of LR decay; 1 means no decay (default: 1)")
    train_il_parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="Maximum norm of gradient (default: 0.5)")
    train_il_parser.add_argument(
        "--optim-eps", type=float, default=1e-8,
        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")

    # END common with `train-ai`

    # Validation parameters
    train_il_parser.add_argument(
        "--val-seed", type=int, default=int(1e9),
        help="Seed for environment used for validation (default: 1e9)")
    train_il_parser.add_argument(
        "--val-interval", type=int, default=1,
        help="Number of epochs between two validation checks; 0 means no validation (default: 1)")
    train_il_parser.add_argument(
        "--val-episodes", type=int, default=10,
        help="Number of episodes used to evaluate the agent, and to evaluate validation accuracy (default: 10)")

    #
    # list
    #
    subparsers.add_parser(
        "list",
        help="List all Diablo instances grouped by parent ID of the test runner."
    )

    #
    # model-drop-optimizer
    #
    model_drop_optimizer_parser = subparsers.add_parser(
        "model-drop-optimizer",
        help="Drop the optimizer state from a model's status file. Useful when\n"
             "switching training context (e.g. single-GPU to DDP) where Adam's\n"
             "accumulated moments are stale and cause gradient instability.",
        formatter_class=IndentedHelpFormatter)
    model_drop_optimizer_parser.add_argument(
        "--model", required=True,
        help="Name of the model (REQUIRED)")

    # model-drop-best
    #
    model_drop_best_parser = subparsers.add_parser(
        "model-drop-best",
        help="Drop a model's 'best' marker - delete best-status.pt and clear\n"
             "the 'best' entry from the sprout run's custom metadata. Useful\n"
             "when retraining a cloned checkpoint on a different env, where\n"
             "the previous best is no longer meaningful for the new training.",
        formatter_class=IndentedHelpFormatter)
    model_drop_best_parser.add_argument(
        "--model", required=True,
        help="Name of the model (REQUIRED)")

    #
    # env-stats
    #
    log_stats_parser = subparsers.add_parser(
        "env-stats",
        help="Aggregate event statistics from /tmp/diablo-*/env.log files.",
        formatter_class=IndentedHelpFormatter)
    log_stats_parser.add_argument(
        "--sort", choices=["count", "sum", "label"], default="count",
        help="Sort output by count, sum_R, or label (default: count)")
    log_stats_parser.add_argument(
        "--last-episodes", type=int, default=50000,
        help="Process only the last N episodes; 0 = all (default: 50000)")
    log_stats_parser.add_argument(
        "--eval-runners", action="store_true", default=False,
        help="Process eval runner logs (diablo-eval-*) instead of training runner logs (diablo-run-*)")
    log_stats_parser.add_argument(
        "--dungeon-level", default=None, dest="level_filter",
        help="Show per-level stats only for these levels: '1', '1,10', '1-10,16'")

    return incompatible_options, parser

def delayed_import(binary_path):
    import devilutionx_generator
    devilutionx_generator.generate(binary_path)

    global dx
    global diablo_env
    global diablo_state
    global diablo_bot
    global ring

    # First goes generated devilutionx
    import devilutionx as dx

    # Then others in any order
    import diablo_env
    import diablo_state
    import diablo_bot
    import ring


# Flag to control the main loop
RUNNING = True
LAST_KEY = 0
SHOW_CHARS = False
SHOW_INV = False
SHOW_MAP = False
INV_CMD = ''

class EventsQueue:
    queue = None
    # Use Braille patterns for representing progress,
    # see here: https://www.unicode.org/charts/nameslist/c_2800.html
    progress = [0x2826, 0x2816, 0x2832, 0x2834]
    progress_cnt = 0
    def __init__(self):
        self.queue = collections.deque(maxlen=16)

# This is weird, but if you place a character in the last column,
# curses fills that position, yet still raises an error.
# These two wrappers attempt to ignore an error if it occurs
# when filling in the last position.
def _addstr(o, y, x, text):
    try:
        o.addstr(y, x, text)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise

# See the comment for the _addstr
def _addch(o, y, x, ch):
    try:
        o.addch(y, x, ch)
    except curses.error:
        h, w = o.getmaxyx()
        if y >= h or x >= w:
            raise

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
    path = os.path.join(game.state_path, "env.log")
    fd = None
    try:
        fd = open(path, "r")
        return EnvLog(fd)
    except:
        pass
    return None

def display_env_log(stdscr, envlog):
    if envlog is None:
        return

    height, width = stdscr.getmaxyx()
    # from row 4 down to footer - 2
    logwin_h = height - 6
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
        if isinstance(obj, list):
            return [convert_paths(i) for i in obj]
        if isinstance(obj, Path):
            return str(obj)
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


def _train_ai_ddp_worker(rank, world_size, args, gameconfig, model_dir, run_id, status):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        _train_ai_loop(args, gameconfig, model_dir, run_id, status,
                       rank=rank, world_size=world_size)
    finally:
        dist.destroy_process_group()


def list_devilution_processes(binary_path, mshared_filename):
    result = procutils.find_processes_with_mapped_file(binary_path, mshared_filename)
    if result:
        for i, proc in enumerate(result):
            print("%2d\t%s\t%s" % (i, proc['pid'], proc['mshared_path']))

def load_optimizer_state(optimizer, status, device, current_ebs, logger=None):
    import torch
    """Load optimizer state from status, moving tensors to device.

    Drops the state if effective_batch_size changed since it was saved -
    stale Adam moments cause gradient instability when the gradient scale shifts.
    """
    if "optimizer_state" not in status:
        return
    saved_ebs = status.get("effective_batch_size")
    if saved_ebs is not None and saved_ebs != current_ebs:
        if logger:
            logger.info(
                f"Effective batch size changed ({saved_ebs} -> {current_ebs}): "
                f"dropping optimizer state to avoid stale Adam moments")
        return
    optimizer.load_state_dict(status["optimizer_state"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    if logger:
        logger.info("Optimizer loaded from the state")


def model_drop_optimizer(args):
    """Drop the optimizer state from a model's status file.

    Loads status.pt, removes 'optimizer_state', saves it back. The model
    weights and all other fields are preserved. On the next training run
    the optimizer starts fresh, letting Adam's moments adapt to the current
    gradient distribution without instability from a stale state.
    """
    model_dir = utils.get_run_dir(args.model)
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}")
        return 1

    status_path = utils.get_status_path(model_dir, best=False)
    if not os.path.exists(status_path):
        print(f"No status.pt at {status_path}")
        return 1

    status = utils.get_status(model_dir)
    if "optimizer_state" not in status:
        print("No optimizer_state in status.pt (nothing to remove)")
        return 0

    del status["optimizer_state"]
    utils.save_status(status, model_dir)
    print(f"Removed optimizer_state from {status_path}")
    return 0


def model_drop_best(args):
    """Drop the 'best' marker for a model:
      - delete <model_dir>/best-status.pt (success-rate snapshot of weights),
      - clear the 'best' key from the sprout run's custom metadata.

    After this the next evaluation that improves on a freshly-tracked
    baseline becomes the new best - nothing carries over from the
    previous training context (different env, different action space,
    etc.) where the recorded best is no longer meaningful.
    """
    model_dir = utils.get_run_dir(args.model)
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}")
        return 1

    # 1. best-status.pt on disk
    best_status_path = utils.get_status_path(model_dir, best=True)
    if os.path.exists(best_status_path):
        os.remove(best_status_path)
        print(f"Removed {best_status_path}")
    else:
        print(f"No best-status.pt at {best_status_path} (nothing to remove)")

    # 2. sprout run's custom['best']
    spr = sprout.Sprout(utils.get_models_dir())
    try:
        run, _ = spr.get_run(head=args.model)
    except Exception as e:
        print(f"No sprout run for head '{args.model}' ({e}); skipping sprout step")
        return 0

    custom = dict(run.get("custom", {}) or {})
    eval_custom = custom.get("eval", {})
    if "best" in eval_custom:
        eval_custom.pop("best")
        custom["eval"] = eval_custom
        # custom_update=False replaces the whole custom dict with the popped version.
        spr.edit(head=args.model, custom_dict=custom, custom_update=False)
        print(f"Cleared 'eval/best' from sprout custom metadata for head '{args.model}'")
    else:
        print(f"No 'eval/best' in sprout custom metadata for head '{args.model}' (nothing to clear)")

    return 0

def _slot_for_section(section, idx):
    """Map (section, idx) to inv_item slot, or None if out of range."""
    if section == 'i':
        if 0 <= idx < 40:
            return diablo_state.inv_slot(idx)
    elif section == 'b':
        if 0 <= idx < 8:
            return diablo_state.belt_slot(idx)
    elif section == 'e':
        if 0 <= idx <= 6:
            return idx
    return None

def _parse_target(buf, player_for_inv_bound=None):
    """Parse '<section?><digits>' into inv_item slot, or None.
    If player_for_inv_bound is provided, inventory idx is bounded by _pNumInv (source semantics)."""
    if not buf:
        return None
    if buf[0].isalpha():
        section, digits = buf[0], buf[1:]
    else:
        section, digits = 'i', buf
    if not digits or not digits.isdigit():
        return None
    idx = int(digits)
    if player_for_inv_bound is not None and section == 'i':
        if not (0 <= idx < int(player_for_inv_bound._pNumInv)):
            return None
    return _slot_for_section(section, idx)

def _split_move(buf):
    """If buf contains 'm', return (src, dst). Otherwise (None, None)."""
    if 'm' not in buf:
        return None, None
    src, dst = buf.split('m', 1)
    return src, dst

def _submit_move(game, src_slot, dst_slot):
    key = (ring.RingEntryType.RING_ENTRY_KEY_INV_MOVE_ITEM |
           ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS)
    game.submit_key(key, data=(src_slot, dst_slot))

def _auto_route_dst(game, src_slot):
    """Pick a destination slot based on the source item, or None if no obvious choice."""
    p = game.state.player
    II = dx.inv_item
    IE = dx.item_equip_type
    EMPTY = dx.ItemType.None_.value

    # Body or belt source -> any inventory slot (engine auto-places).
    if src_slot < II.INVITEM_INV_FIRST.value or src_slot > II.INVITEM_INV_LAST.value:
        return II.INVITEM_INV_FIRST.value

    item = p.InvList[src_slot - II.INVITEM_INV_FIRST.value]
    iloc = int(item._iLoc)

    # Equipment -> body slot (prefer empty for paired slots).
    if iloc == IE.ILOC_HELM.value:
        return II.INVITEM_HEAD.value
    if iloc == IE.ILOC_ARMOR.value:
        return II.INVITEM_CHEST.value
    if iloc == IE.ILOC_AMULET.value:
        return II.INVITEM_AMULET.value
    if iloc == IE.ILOC_RING.value:
        return (II.INVITEM_RING_LEFT.value
                if int(p.InvBody[II.INVITEM_RING_LEFT.value]._itype) == EMPTY
                else II.INVITEM_RING_RIGHT.value)
    if iloc in (IE.ILOC_ONEHAND.value, IE.ILOC_TWOHAND.value):
        return (II.INVITEM_HAND_LEFT.value
                if int(p.InvBody[II.INVITEM_HAND_LEFT.value]._itype) == EMPTY
                else II.INVITEM_HAND_RIGHT.value)

    # Non-equipment (potions/scrolls) -> first empty belt slot.
    for i in range(8):
        if int(p.SpdList[i]._itype) == EMPTY:
            return II.INVITEM_BELT_FIRST.value + i
    return II.INVITEM_BELT_FIRST.value

def _handle_inv_key(game, k):
    global INV_CMD, SHOW_INV

    if k == 27:  # Esc: clear buffer and close window
        INV_CMD = ''
        SHOW_INV = False
        return

    if k in (curses.KEY_BACKSPACE, 127, 8):
        INV_CMD = INV_CMD[:-1]
        return

    if k < 0 or k > 127:
        return
    ch = chr(k)

    src_part, dst_part = _split_move(INV_CMD)
    in_move = src_part is not None

    if ch.isdigit():
        INV_CMD += ch
        # Auto-execute move once dst has section + first digit. All dst
        # indices are effectively single-digit (belt 0-7, body 0-6, and
        # inv ignores the specific index since engine auto-places).
        if in_move and dst_part and len(dst_part) >= 1 and dst_part[0].isalpha():
            src_slot = _parse_target(src_part, game.state.player)
            dst_slot = _parse_target(INV_CMD.split('m', 1)[1])
            if src_slot is not None and dst_slot is not None:
                _submit_move(game, src_slot, dst_slot)
            INV_CMD = ''
    elif ch in 'ibe':
        if not in_move:
            # Source section letter (only at start, no digits yet).
            if not INV_CMD or (len(INV_CMD) == 1 and INV_CMD[0].isalpha()):
                INV_CMD = ch
        else:
            # Destination section letter.
            if not dst_part:
                INV_CMD += ch
            elif len(dst_part) == 1 and dst_part.isalpha():
                INV_CMD = src_part + 'm' + ch
            # else: dst has digits already, ignore (use Backspace to change)
    elif ch == 'm':
        if not in_move:
            # Enter move state if source is parseable.
            if _parse_target(INV_CMD, game.state.player) is not None:
                INV_CMD += 'm'
        elif not dst_part:
            # Second 'm' with no dst specified -> auto-route based on src item.
            src_slot = _parse_target(src_part, game.state.player)
            if src_slot is not None:
                dst_slot = _auto_route_dst(game, src_slot)
                if dst_slot is not None:
                    _submit_move(game, src_slot, dst_slot)
            INV_CMD = ''
        # else: dst already started, ignore extra m
    elif ch == 'r':
        key = (ring.RingEntryType.RING_ENTRY_KEY_INV_REORGANIZE |
               ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS)
        game.submit_key(key)
        INV_CMD = ''
    elif ch in 'du':
        if not in_move:
            slot = _parse_target(INV_CMD, game.state.player)
            if slot is not None:
                key_type = (ring.RingEntryType.RING_ENTRY_KEY_INV_DROP_ITEM if ch == 'd'
                            else ring.RingEntryType.RING_ENTRY_KEY_INV_USE_ITEM)
                key = key_type | ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
                game.submit_key(key, data=(slot, 0))
            INV_CMD = ''

def display_automap_window(d, stdscr):
    """Render AutomapView[40][40] as a 40x40 ASCII grid centered on screen.
    '@' = player, '.' = explored, '?' = frontier, ' ' = nothingness."""
    if not hasattr(d, 'AutomapView'):
        return

    pos = diablo_state.player_position(d)
    px = (int(pos[0]) - 16) // 2
    py = (int(pos[1]) - 16) // 2

    e, frontier = diablo_state.automap_frontier(d)

    DMAXX, DMAXY = 40, 40
    lines = []
    for y in range(DMAXY):
        row = []
        for x in range(DMAXX):
            if x == px and y == py:
                row.append('@')
            elif frontier[x, y]:
                row.append('?')
            elif e[x, y]:
                row.append('.')
            else:
                row.append(' ')
        lines.append("".join(row))

    am_cnt = int(np.sum(e))
    fr_cnt = int(np.sum(frontier))
    title = "Automap  explored:%d  frontier:%d  ('m' close)" % (am_cnt, fr_cnt)
    w = max(DMAXX, len(title)) + 2
    h = DMAXY + 3  # title + border top/bot + 1 blank
    scr_h, scr_w = stdscr.getmaxyx()
    y0 = max(0, scr_h // 2 - h // 2)
    x0 = max(0, scr_w // 2 - w // 2)

    _addstr(stdscr, y0, x0, '┌' + '─' * w + '┐')
    _addstr(stdscr, y0 + 1, x0, '│' + title.center(w) + '│')
    _addstr(stdscr, y0 + 2, x0, '│' + '─' * w + '│')
    for i, line in enumerate(lines):
        row = y0 + 3 + i
        if row >= scr_h - 1:
            break
        _addstr(stdscr, row, x0, '│' + line.ljust(w) + '│')
    bot_row = y0 + 3 + DMAXY
    if bot_row < scr_h:
        _addstr(stdscr, bot_row, x0, '└' + '─' * w + '┘')


def handle_keyboard(stdscr, game):
    global LAST_KEY, RUNNING, SHOW_CHARS, SHOW_INV, SHOW_MAP

    k = stdscr.getch()
    if k == -1:
        return False

    # Modal: when inventory window is open, all keys feed the inv parser.
    if SHOW_INV:
        _handle_inv_key(game, k)
        return True

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
    elif k == ord('n'):
        key = ring.RingEntryType.RING_ENTRY_KEY_NEW
    elif k == ord('l'):
        key = ring.RingEntryType.RING_ENTRY_KEY_LOAD
    elif k == ord('s'):
        key = ring.RingEntryType.RING_ENTRY_KEY_SAVE
    elif k == ord('p'):
        key = ring.RingEntryType.RING_ENTRY_KEY_PAUSE
    elif k == 27:  # Esc: close any open overlay
        SHOW_CHARS = False
        SHOW_MAP = False
    elif k == ord('c'):
        SHOW_CHARS = not SHOW_CHARS
    elif k == ord('i'):
        SHOW_INV = not SHOW_INV
    elif k == ord('m'):
        SHOW_MAP = not SHOW_MAP
    elif k == ord('q'):
        RUNNING = False  # Stop the main loop

    LAST_KEY |= key

    return True

def get_radius(d, dunwin):
    height, width = dunwin.getmaxyx()
    dundim = diablo_state.dungeon_dim(d)

    # Compensate for `R * 2 + _1_` (see `EnvRect`).
    # We use `- 2` because of quarter
    width = min(width, dundim[0]) - 2
    height = min(height, dundim[1]) - 1

    # Reduce the horizontal radius by half to make the dungeon
    # visually appear as an accurate square when displayed in a
    # terminal
    return min(width // 4, height // 2)

def get_events_as_string(game, events):
    advance_progress = False
    RE = ring.RingEntryType
    while (event := game.retrieve_event()) is not None:
        keys = event.en_type
        data1 = int(event.en_data1)
        data2 = int(event.en_data2)
        k = None

        if keys == 0:
            # Stand, "◦" - white bullet
            k = "\u25e6"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                      ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # N, "↑" - upwards arrow
            k = "\u2191"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # NE, "↗" - north east arrow
            k = "\u2197"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                      ring.RingEntryType.RING_ENTRY_KEY_RIGHT):
            # E, "→" - rightwards arrow
            k = "\u2192"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN):
            # SE, "↘" - south east arrow
            k = "\u2198"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_DOWN |
                      ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # S, "↓" - downwards arrow
            k = "\u2193"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # SW, "↙" - southwest arrow
            k = "\u2199"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_UP |
                      ring.RingEntryType.RING_ENTRY_KEY_LEFT):
            # W, "←" - leftwards arrow
            k = "\u2190"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_UP):
            # NW, "↖" - north west arrow
            k = "\u2196"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_X):
            k = "X"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_Y):
            k = "Y"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_A):
            k = "A"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_B):
            k = "B"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_NEW):
            k = "N"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_SAVE):
            k = "S"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_LOAD):
            k = "L"
        elif keys == (ring.RingEntryType.RING_ENTRY_KEY_PAUSE):
            k = "P"
        elif keys == RE.RING_ENTRY_KEY_NOOP:
            # Explicit NOOP - same glyph as Stand
            k = "◦"
        elif keys == RE.RING_ENTRY_KEY_SET_GOAL:
            # data1=x, data2=y target tile
            k = f"g{data1},{data2}"
        elif keys == RE.RING_ENTRY_KEY_INV_USE_ITEM:
            # data1=cii: 0-6 body, 7-46 inv, 47-54 belt
            k = f"u{data1}"
        elif keys == RE.RING_ENTRY_KEY_INV_MOVE_ITEM:
            # data1=src_cii, data2=dst_cii; "→" rightwards arrow
            k = f"m{data1}→{data2}"
        elif keys == RE.RING_ENTRY_KEY_INV_DROP_ITEM:
            # data1=cii
            k = f"d{data1}"
        elif keys == RE.RING_ENTRY_KEY_INV_REORGANIZE:
            k = "R"
        elif keys == RE.RING_ENTRY_KEY_CAST_SPELL:
            # data1=SpellID (numeric)
            k = f"~{data1}"

        if k is not None:
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

    # The horizontal radius is reduced by half (see `get_radius()`),
    # so in order to stretch the dungeon number of columns is
    # multiplied by two
    cols *= 2

    # Get the screen size
    height, width = dunwin.getmaxyx()

    x_off = width // 2 - cols // 2
    y_off = height // 2 - rows // 2

    assert x_off >= 0
    assert y_off >= 0

    for row in range(rows):
        for col in range(0, cols, 2):
            _addch(dunwin, row + y_off, col + x_off, m[col//2, row])
            # "Stretch" the width by adding a space. With this simple
            # trick the dungeon should visually appear as an accurate
            # square in a terminal
            _addch(dunwin, row + y_off, col + x_off + 1, ' ')

def _res_char(r, immune_bit, resist_bit):
    if r & immune_bit: return '+'
    if r & resist_bit: return '~'
    return '-'

def _pct(cur, max_val):
    return max(0, min(100, int(100 * cur / max_val))) if max_val > 0 else 0

def _active_flags(p):
    return '[MS]' if p.pManaShield else ''

def _spell_name(p):
    sid = int(p._pRSpell)
    if sid <= 0:
        return '-', 0
    try:
        name = dx.SpellID(sid).name
    except Exception:
        name = f'#{sid}'
    lvl = int(p._pSplLvl[sid]) if 0 <= sid < len(p._pSplLvl) else 0
    return name, lvl

def _spell_summary(d):
    """Available episode spells with levels; active spell prefixed with '*'."""
    import diablo_state as _ds
    episode_spells = [
        (dx.SpellID.Firebolt,    'Fbt'),
        (dx.SpellID.ChargedBolt, 'Cbt'),
        (dx.SpellID.FireWall,    'Fwl'),
        (dx.SpellID.ManaShield,  'Msh'),
        (dx.SpellID.StoneCurse,  'Stc'),
        (dx.SpellID.Phasing,     'Phz'),
        (dx.SpellID.Fireball,    'Fbl'),
    ]
    spell_bits = _ds.player_spell_bits(d)
    p = d.player
    active = int(p._pRSpell)
    parts = []
    for sid, abbrev in episode_spells:
        sv = sid.value
        if not (spell_bits & (1 << sv)):
            continue
        lvl = int(p._pSplLvl[sv]) if 0 <= sv < len(p._pSplLvl) else 0
        parts.append(('*' if sv == active else '') + f'{abbrev}/{lvl}')
    return ' '.join(parts) if parts else '-'

def _speed_strs(iflags):
    f = int(iflags)
    ISE = dx.ItemSpecialEffect
    atk = ('Faster' if f & ISE.FasterAttack.value
           else 'Fast' if f & ISE.FastAttack.value
           else 'Quick' if f & ISE.QuickAttack.value
           else '-')
    rec = ('Fastest' if f & ISE.FastestHitRecovery.value
           else 'Faster' if f & ISE.FasterHitRecovery.value
           else 'Fast' if f & ISE.FastHitRecovery.value
           else '-')
    return atk, rec

def nearest_monster_info(d):
    pos = diablo_state.player_position(d)
    best, best_dist = None, float('inf')
    cnt = int(d.ActiveMonsterCount.value)
    ranged_ids = diablo_state.ranged_ai_ids_array()
    resist = dx.monster_resistance
    unique_none = dx.UniqueMonsterType.None_.value
    for i in range(cnt):
        mid = int(d.ActiveMonsters[i])
        m = d.Monsters[mid]
        if int(m.hitPoints) <= 0:
            continue
        mx, my = int(m.position.tile.x), int(m.position.tile.y)
        if not (int(d.dFlags[mx, my]) & dx.DungeonFlag.Lit.value):
            continue
        dist = abs(mx - pos[0]) + abs(my - pos[1])
        if dist < best_dist:
            best_dist = dist
            hp_pct = _pct(int(m.hitPoints), int(m.maxHitPoints))
            r = int(m.resistance)
            try:
                ti  = d.monster_type_info[int(m.levelType)]
                lvl = int(ti.level)
                mv  = int(ti.walk_frames)
                atk = int(ti.attack_frames)
            except Exception:
                lvl = mv = atk = 0
            try:
                name = dx.MonsterAIID(int(m.ai)).name
            except Exception:
                name = '?'
            max_mv  = max(1, int(d.max_walk_frames.value))
            max_atk = max(1, int(d.max_attack_frames.value))
            best = {
                'dist': dist, 'hp_pct': hp_pct, 'lvl': lvl, 'name': name,
                'ranged': int(m.ai) in ranged_ids,
                'boss':   int(m.uniqueType) != unique_none,
                'fire':   _res_char(r, resist.IMMUNE_FIRE.value,      resist.RESIST_FIRE.value),
                'light':  _res_char(r, resist.IMMUNE_LIGHTNING.value, resist.RESIST_LIGHTNING.value),
                'magic':  _res_char(r, resist.IMMUNE_MAGIC.value,     resist.RESIST_MAGIC.value),
                'spd':  int((1.0 - mv  / max_mv)  * 100),
                'aspd': int((1.0 - atk / max_atk) * 100),
            }
    return best

def display_chars_window(d, stdscr):
    p = d.player
    hp_cur = int(p._pHitPoints) >> 6
    hp_max = int(p._pMaxHP) >> 6
    mp_cur = int(p._pMana) >> 6
    mp_max = int(p._pMaxMana) >> 6
    atk_spd, rec_spd = _speed_strs(int(p._pIFlags))

    lines = [
        f" Name:     {''.join(chr(c) for c in p._pName if c)} (lvl {int(p._pLevel)})",
        f" HP:       {hp_cur} / {hp_max}",
        f" Mana:     {mp_cur} / {mp_max}",
        f" Strength: {int(p._pStrength)}",
        f" Magic:    {int(p._pMagic)}",
        f" Dexterity:{int(p._pDexterity)}",
        f" Vitality: {int(p._pVitality)}",
        f" AC:       {int(p._pIAC)}",
        f" Damage:   {int(p._pIMinDam)}-{int(p._pIMaxDam)}",
        f" To Hit:   {int(p._pIBonusToHit)}%",
        f" Resist:   Fire {int(p._pFireResist)}%  "
        f"Lgth {int(p._pLghtResist)}%  Mag {int(p._pMagResist)}%",
        f" Spells:   {_spell_summary(d)}",
        f" Atk spd:  {atk_spd}",
        f" Rec spd:  {rec_spd}",
    ]

    w = max(len(l) for l in lines) + 2
    h = len(lines) + 2
    scr_h, scr_w = stdscr.getmaxyx()
    y = max(0, scr_h // 2 - h // 2)
    x = max(0, scr_w // 2 - w // 2)

    for i, line in enumerate(lines):
        row = y + 1 + i
        if row >= scr_h:
            break
        padded = line.ljust(w)
        _addstr(stdscr, row, x, '│' + padded + '│')
    top = '┌' + '─' * w + '┐'
    bot = '└' + '─' * w + '┘'
    _addstr(stdscr, y, x, top)
    if y + h - 1 < scr_h:
        _addstr(stdscr, y + h - 1, x, bot)

INV_BODY_LABELS = ["Head", "L.Ring", "R.Ring", "Amul", "L.Hand", "R.Hand", "Chest"]
INV_SECTION_NAMES = {'i': 'inventory', 'b': 'belt', 'e': 'equipment'}

def _item_name(item):
    # Item::clear() only resets _itype to None; _iName and other fields stay stale.
    if int(item._itype) == dx.ItemType.None_.value:
        return ''
    return bytes(item._iName).split(b'\x00')[0].decode('ascii', errors='replace')

def _item_glyph(item):
    """Return a 2-character glyph for the item, or '..' if empty.

    Potions and the healing scroll all share '_iName' starting with 'P' or 'S',
    so a first-letter glyph collapses them to a single 'P' or 'S' and loses
    differentiation. Use the misc-id-aware mapping below for those, fall back
    to the first 2 chars of _iName for everything else (weapons, armor, etc.).
    Lowercase first letter = regular potion, uppercase = Full variant.
    """
    if int(item._itype) == dx.ItemType.None_.value:
        return '..'
    name = _item_name(item)
    is_full = 'Full' in name
    if 'Healing' in name:
        if 'Scroll' in name:
            return 'sh'
        return 'Hp' if is_full else 'hp'
    if 'Mana' in name:
        return 'Mp' if is_full else 'mp'
    if 'Rejuv' in name:
        return 'Rj' if is_full else 'rj'
    if len(name) >= 2:
        return name[:2]
    return (name + '?')[:2]

def _describe_inv_src(buf):
    if not buf:
        return "?"
    if buf[0].isalpha():
        section_name = INV_SECTION_NAMES.get(buf[0], '?')
        digits = buf[1:]
        return f"{section_name} {digits}" if digits else section_name
    return f"inventory {buf}"

def _inv_prompt():
    buf = INV_CMD
    if not buf:
        return "Choose section: i=inv  b=belt  e=equip  (or digit for inv)"
    src, dst = _split_move(buf)
    if src is not None:
        src_desc = _describe_inv_src(src)
        if not dst:
            return f"Move {src_desc} -> m=auto, or section: i=inv  b=belt  e=equip"
        if len(dst) == 1 and dst.isalpha():
            return f"Move {src_desc} -> {INV_SECTION_NAMES[dst]} - choose dst index"
        return f"Move {src_desc} -> {_describe_inv_src(dst)}"
    if len(buf) == 1 and buf[0].isalpha():
        return f"Section: {INV_SECTION_NAMES[buf]} - choose item index"
    return f"{_describe_inv_src(buf).capitalize()} - action: d=drop  u=use  m=move"

def display_inventory_window(d, stdscr):
    p = d.player

    body_lines = ["Body:"]
    for i, label in enumerate(INV_BODY_LABELS):
        name = _item_name(p.InvBody[i]) or "-"
        body_lines.append(f"  {i} {label:7}: {name}")

    # Each grid cell is 2 chars wide so glyphs like 'hp', 'Hp', 'sh' fit.
    # Multi-cell items repeat the same glyph in every cell they occupy, as
    # before - the InvGrid encoding gives the same InvList_index for all of
    # them. 10 cols * 2 chars = 20-char grid interior.
    grid_chars = [['..' for _ in range(10)] for _ in range(4)]
    legend = {}
    for y in range(4):
        for x in range(10):
            cell = int(p.InvGrid[y * 10 + x])
            if cell == 0:
                continue
            inv_idx = abs(cell) - 1
            item = p.InvList[inv_idx]
            glyph = _item_glyph(item)
            grid_chars[y][x] = glyph
            if inv_idx not in legend:
                legend[inv_idx] = (_item_name(item), glyph)

    grid_lines = ["Inventory:"]
    grid_lines.append("  ┌" + "─" * 20 + "┐")
    for row in grid_chars:
        grid_lines.append("  │" + "".join(row) + "│")
    grid_lines.append("  └" + "─" * 20 + "┘")

    legend_lines = ["Items:"]
    for idx in sorted(legend):
        name, glyph = legend[idx]
        legend_lines.append(f"  {idx:2d} [{glyph}]: {name}")

    left_width = max(len(l) for l in grid_lines)
    rows = max(len(grid_lines), len(legend_lines))
    middle_lines = []
    for i in range(rows):
        left = grid_lines[i] if i < len(grid_lines) else ""
        right = legend_lines[i] if i < len(legend_lines) else ""
        middle_lines.append(left.ljust(left_width) + "   " + right)

    belt_parts = []
    for i in range(8):
        belt_parts.append(f"{i}:{_item_glyph(p.SpdList[i])}")
    belt_line = "Belt: [" + "][".join(belt_parts) + "]"

    free_cells = int(np.count_nonzero(np.asarray(p.InvGrid) == 0))
    stats_line = f"Gold: {int(p._pGold)}  Free cells: {free_cells}/40"

    # Pad bottom lines to a stable width so the window doesn't resize as the
    # user types and the prompt text changes length.
    INV_FOOTER_W = 65
    prompt_line  = ("> " + _inv_prompt()).ljust(INV_FOOTER_W)
    cmd_line     = ("  " + INV_CMD + "_").ljust(INV_FOOTER_W)
    globals_line = "  (r=reorganize  Esc=close  Backspace=undo)".ljust(INV_FOOTER_W)

    lines = (body_lines + [""] + middle_lines + [""] + [belt_line] +
             [""] + [stats_line] + [""] + [prompt_line, cmd_line, globals_line])

    w = max(len(l) for l in lines) + 2
    h = len(lines) + 2
    scr_h, scr_w = stdscr.getmaxyx()
    y = max(0, scr_h // 2 - h // 2)
    x = max(0, scr_w // 2 - w // 2)

    for i, line in enumerate(lines):
        row = y + 1 + i
        if row >= scr_h:
            break
        padded = line.ljust(w)
        _addstr(stdscr, row, x, '│' + padded + '│')
    top = '┌' + '─' * w + '┐'
    bot = '└' + '─' * w + '┘'
    _addstr(stdscr, y, x, top)
    if y + h - 1 < scr_h:
        _addstr(stdscr, y + h - 1, x, bot)

def display_dungeon(d, stdscr, view_radius, goal_pos):
    height, width = stdscr.getmaxyx()
    dunwin = stdscr.subwin(height - (4 + 1), width, 4, 0)
    radius = get_radius(d, dunwin)
    if view_radius:
        radius = min(radius, view_radius)
    surroundings = diablo_state.get_surroundings(d, radius, goal_pos)

    display_matrix(dunwin, surroundings)

def display_diablo_state(game, stdscr, events, envlog, view_radius):
    d = game.safe_state
    p = d.player
    pos = diablo_state.player_position(d)
    height, width = stdscr.getmaxyx()

    hp_pct = _pct(int(p._pHitPoints), int(p._pMaxHP))
    mp_pct = _pct(int(p._pMana), int(p._pMaxMana))
    spell_sum = _spell_summary(d)
    active_flags = _active_flags(p)

    ep, stk = int(game.agent_state.episode_steps), int(game.agent_state.stuck_steps)
    env_str = "%d/%d" % (stk, ep) if ep or stk else "-/-"
    msg = "Ticks: %4d  Env: %s  Pos: %d:%d  HP: %3d%%  MP: %3d%%  Spl: %s  State: %s%s" % (
        game.ticks(d),
        env_str,
        pos[0], pos[1],
        hp_pct, mp_pct,
        spell_sum,
        dx.PLR_MODE(d.player._pmode).name,
        (' ' + active_flags) if active_flags else '')
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 0, width // 2 - len(msg) // 2, msg)

    obj_cnt = diablo_state.count_active_objects(d)
    items_cnt = diablo_state.count_active_items(d)
    mon_cnt = diablo_state.count_active_monsters(d)
    events_str, events_progress = get_events_as_string(game, events)

    msg = "Mons: %d  Items: %d  Objs: %d  Lvl: %d  %c %s" % (
        mon_cnt, items_cnt, obj_cnt, d.player.plrlevel,
        events_progress, events_str)
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 1, width // 2 - len(msg) // 2, msg)

    mons = nearest_monster_info(d)
    if mons:
        flags = ''
        if mons['ranged']: flags += ' Rng'
        if mons['boss']:   flags += ' Boss'
        msg = "Nearest: %s  d%d  lv%d  HP%d%%%s  F%sL%sM%s  Rate: mov%d%% atk%d%%" % (
            mons['name'], mons['dist'], mons['lvl'], mons['hp_pct'], flags,
            mons['fire'], mons['light'], mons['magic'],
            mons['spd'], mons['aspd'])
    else:
        msg = "Nearest: none"
    msg = truncate_line(msg, width - 1)
    _addstr(stdscr, 2, width // 2 - len(msg) // 2, msg)

    msg = "'q' quit │ 'c' chars │ 'i' inv │ 'm' map │ 'y' cast spell"
    _addstr(stdscr, height - 1, width // 2 - len(msg) // 2, msg)

    display_dungeon(d, stdscr, view_radius, game.goal_pos)
    display_env_log(stdscr, envlog)

    if SHOW_CHARS:
        display_chars_window(d, stdscr)

    if SHOW_INV:
        display_inventory_window(d, stdscr)

    if SHOW_MAP:
        display_automap_window(d, stdscr)

    if diablo_state.is_game_paused(d):
        msgs = ["            ",
                " ┌────────┐ ",
                " │ Paused │ ",
                " └────────┘ ",
                "            "]
        h = height // 2
        for i, msg in enumerate(msgs):
            _addstr(stdscr, h + i, width // 2 - len(msg) // 2, msg)

def new_game_data(gameconfig, n_counter):
    seed = diablo_state.make_episode_seed(gameconfig['seed'], 0, n_counter)
    spec = gameconfig.get('dungeon-level', DUNGEON_LEVEL_DEFAULT)
    dungeon_level = diablo_state.sample_dungeon_level(spec, seed)
    return (dungeon_level << 1) | 1, seed

def run_tui(stdscr, args, gameconfig):
    global RUNNING
    global LAST_KEY

    # Run or attach to Diablo
    game = diablo_state.DiabloGame.run_or_attach(gameconfig)

    # Disable cursor and enable keypad input
    curses.curs_set(0)
    stdscr.nodelay(True)
    # Reduce Esc delay (default ~1000ms waits for escape sequences).
    curses.set_escdelay(25)

    events = EventsQueue()
    envlog = None
    n_counter = 0

    # Main loop
    while RUNNING:
        stdscr.clear()

        if not args.no_env_log and envlog is None:
            # Try to open a environment log, can be created later
            envlog = open_envlog(game)
        view_radius = args.view_radius

        display_diablo_state(game, stdscr, events, envlog, view_radius)

        if LAST_KEY:
            key = LAST_KEY
            key |= ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
            data = (0, 0)
            if LAST_KEY & ring.RingEntryType.RING_ENTRY_KEY_NEW:
                n_counter += 1
                data = new_game_data(gameconfig, n_counter)
            game.submit_key(key, data=data)
            LAST_KEY = 0

        # Refresh the screen to show the content
        stdscr.refresh()

        # Handle keys
        while handle_keyboard(stdscr, game):
            pass

        game.update_ticks()
        time.sleep(0.01)


def prepare_directory_for_run(args, dir_name):
    # Create Sprout instance
    spr = sprout.Sprout(utils.get_models_dir())

    # The model/demos are essentially just folders, and it depends on how
    # you open a snapshot in Sprout. Therefore, skip the model to
    # avoid long diffs in Sprout's output. Also `continue` flag
    # just controls model states, so should be skipped.
    run_dir = utils.get_run_dir(dir_name)
    # shlex.quote() the value so tuples / strings containing spaces (e.g.
    # dungeon_level=(1, 16)) round-trip through sprout's shlex-based
    # parse_params_string without splitting mid-value.
    params_str = " ".join(f"{k}={shlex.quote(str(v))}"
                          for k, v in vars(args).items()
                          if k not in SPROUT_SKIP_PARAMS)

    creating_snapshot = False
    if not os.path.isdir(run_dir):
        # Create model state
        spr.create(group=args.env, head=dir_name, params_str=params_str)
        creating_snapshot = True
    elif not args.cont:
        creating_snapshot = True
        spr.create_or_edit(
            from_head=dir_name, params_str=params_str,
            alias_str="", description_str="",
            check=sprout.SNAP_CHECK_PARAMS)
    else:
        # Continue in the current head, but be careful; firstly, check
        # if the environment has changed
        run, _ = spr.get_run(head=dir_name)
        params = run.get("params", {})
        env = params.get("env", "")
        if env != args.env:
            print("\nEnvironment mismatch detected!")
            print(f"   Old: {env}")
            print(f"   New: {args.env}\n")

            msg = "Proceed with training? [y/N]: "
            if input(msg).strip().lower() != "y":
                print("Training aborted.")
                sys.exit(1)

        # Change parameters for the existing model and continue
        # training without creating a snapshot
        spr.edit(head=dir_name, params_str=params_str)

    if args.best_drop or (not args.no_drop_best and creating_snapshot):
        model_drop_best(args)

    _, run_id = spr.get_run(head=dir_name)
    return spr, run_dir, run_id


def _fmt_frames(n):
    return f"{int(n):,}".replace(",", "'")

def _fmt_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"

def _sprout_duration(spr, model_dir):
    # spr may be None if sprout is not configured
    try:
        run, _ = spr.get_run(head=os.path.basename(model_dir))
        return run.get("custom", {}).get("eval", {}).get("last", {}).get("duration", 0)
    except Exception:
        return 0

def _scale_bar(value, good_hi, good_lo=0.0, width=4):
    """Bar over the good zone [good_lo, good_hi].
    [^---] in range (near low), [---^] in range (near high),
    [<---] below, [--->] above."""
    if value < good_lo:
        return '[<' + '-' * (width - 1) + ']'
    if value > good_hi:
        return '[' + '-' * (width - 1) + '>]'
    pos = round((value - good_lo) / (good_hi - good_lo) * (width - 1))
    bar = ['-'] * width
    bar[pos] = '^'
    return '[' + ''.join(bar) + ']'


def train_ai(args, gameconfig):
    import torch

    # Load training status; fail fast if already at the frame limit so we
    # do not spawn runners or create a sprout snapshot needlessly.
    model_dir = utils.get_run_dir(args.model)
    load_best = getattr(args, 'load_best', False)
    try:
        status = utils.get_status(model_dir, best=load_best)
    except FileNotFoundError as e:
        if load_best:
            print(f"Error: {e}")
            sys.exit(1)
        status = {"num_frames": 0, "update": 0}
    except OSError:
        status = {"num_frames": 0, "update": 0}
    args.frames = resolve_frames(args.frames, status["num_frames"])
    if status["num_frames"] >= args.frames_int:
        print(f"Already at {status['num_frames']} frames, "
              f"limit is {args.frames_int}. Increase --frames and retry.")
        sys.exit(1)

    # Prepare model dir
    spr, model_dir, run_id = prepare_directory_for_run(args, args.model)

    n_gpus = min(args.gpus, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    if n_gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(_train_ai_ddp_worker,
                 args=(n_gpus, args, gameconfig, model_dir, run_id, status),
                 nprocs=n_gpus, join=True)
        return 0
    return _train_ai_loop(args, gameconfig, model_dir, run_id, status)


def _train_ai_loop(args, gameconfig, model_dir, run_id, status,
                   rank=None, world_size=1):
    delayed_import(gameconfig['diablo-bin-path'])

    from rl import torch_ac
    from rl.evaluate import batch_evaluate
    from rl.flat_model import FlatACModel
    from rl.hrl_model import HRLACModel
    from rl.torch_ac.utils import ParallelEnvPool
    from rl.utils import device
    import tensorboardX
    import torch

    ddp = rank is not None
    is_main = not ddp or rank == 0
    local_device = torch.device(f'cuda:{rank}') if ddp else device

    # Recreate spr in each spawned process - file descriptors from the parent
    # process are not valid after mp.spawn (uses 'spawn', not 'fork').
    spr = sprout.Sprout(utils.get_models_dir()) if is_main else None

    # Load loggers and Tensorboard writer (main rank only)
    if is_main:
        txt_logger = utils.get_txt_logger(model_dir)
        csv_file, csv_logger = utils.get_csv_logger(model_dir)
        tb_writer = tensorboardX.SummaryWriter(model_dir)
        import signal as _signal
        _orig = _signal.getsignal(_signal.SIGINT)
        def _sigint(sig, frame):
            tb_writer.close()
            _signal.signal(_signal.SIGINT, _orig)
            _signal.raise_signal(_signal.SIGINT)
        _signal.signal(_signal.SIGINT, _sigint)
        txt_logger.info("{}\n".format(" ".join(sys.argv)))
        txt_logger.info("{}\n".format(args))
        txt_logger.info(f"Run: {run_id}\n")
        txt_logger.info(f"Device: {local_device}" +
                        (f" (DDP world_size={world_size})\n" if ddp else "\n"))

    # Load environments
    runner_offset = rank * args.env_runners if ddp else 0
    envs = []
    ts = 0
    auto_levels = args.dungeon_level.auto_levels
    if auto_levels:
        gameconfig['dungeon-level'].stats_path = os.path.join(model_dir, "env-stats.txt")
    for i in range(args.env_runners):
        env_config = copy.deepcopy(gameconfig)
        env_config['index'] = runner_offset + i

        # Some old environments have specific configurations that need
        # to be adjusted before starting a game instance
        EnvClass = utils.get_env_class(args.env)
        EnvClass.tune_config(env_config)

        # Run or attach to Diablo (devilutionX) instance
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        env = utils.make_env(args.env, env_config, game)
        envs.append(env)

        if time.time() - ts >= 3.0 or i == args.env_runners - 1:
            ts = time.time()
            print(f"{i+1}/{args.env_runners} environment instances are created")

    penv_pool = ParallelEnvPool(envs)

    eval_envs = []
    ts = 0
    # Eval runners only on main rank; use offset beyond all training runners
    eval_penv_pool = None
    if is_main:
        eval_envs = []
        eval_gameconfig = copy.deepcopy(gameconfig)
        eval_gameconfig['dungeon-level'] = args.eval_dungeon_level
        eval_gameconfig['no-spells']     = args.eval_no_spells
        eval_gameconfig['stats-scale']   = args.eval_stats_scale
        if args.no_eval_char_tables:
            eval_gameconfig['char-tables'] = None
        eval_gameconfig['hero-hp-min-pct']  = args.eval_hero_hp_at_start[0]
        eval_gameconfig['hero-hp-max-pct']  = args.eval_hero_hp_at_start[1]
        eval_gameconfig['hero-mana-min-pct'] = args.eval_hero_mana_at_start[0]
        eval_gameconfig['hero-mana-max-pct'] = args.eval_hero_mana_at_start[1]
        eval_gameconfig['hero-potions-min'] = args.eval_hero_potions_at_start[0]
        eval_gameconfig['hero-potions-max'] = args.eval_hero_potions_at_start[1]
        eval_gameconfig['goal-far-bias']          = False
        eval_gameconfig['eval-kill-threshold']    = args.eval_kill_threshold
        eval_gameconfig['eval-max-steps-per-level'] = args.eval_max_steps_per_level
        eval_gameconfig['eval-no-stuck-timeout']  = args.eval_no_stuck_timeout
        eval_offset = world_size * args.env_runners
        for i in range(args.eval_env_runners):
            env_config = copy.deepcopy(eval_gameconfig)
            env_config['index'] = eval_offset + i
            env_config['eval'] = True
            EnvClass = utils.get_env_class(args.env)
            EnvClass.tune_config(env_config)
            game = diablo_state.DiabloGame.run_or_attach(env_config)
            eval_envs.append(utils.make_env(args.env, env_config, game))

            if time.time() - ts >= 3.0 or i == args.eval_env_runners - 1:
                ts = time.time()
                print(f"{i+1}/{args.eval_env_runners} eval environment instances are created")
        eval_penv_pool = ParallelEnvPool(eval_envs)

    if is_main:
        txt_logger.info("Environments loaded\n")

    # Load best training status if exists
    best_success_rate = 0.0
    if is_main:
        try:
            best_status = utils.get_status(model_dir, best=True)
            best_success_rate = best_status.get("success_rate", 0.0)
            # Old statuses can contain an array
            best_success_rate = np.mean(best_success_rate)
        except OSError:
            pass
        txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    if is_main:
        txt_logger.info("Observations preprocessor loaded")

    num_hierarchy_levels = envs[0].unwrapped.num_hierarchy_levels

    # Load model
    if num_hierarchy_levels == 1:
        acmodel = FlatACModel(obs_space, envs[0].action_space, args.cnn_arch,
                              embedding_dim=args.embedding_dim,
                              use_memory=True, use_text=False)
    else:
        acmodel = HRLACModel(obs_space, envs[0].action_space, args.cnn_arch,
                             embedding_dim=args.embedding_dim,
                             use_memory=True, use_text=False)

    if "model_state" in status:
        acmodel.load_from_status(status, txt_logger if is_main else None)
    acmodel.to(local_device)
    acmodel_raw = acmodel  # keep reference for save/load
    if ddp:
        # CuDNN's BatchNorm backward modifies its workspace in-place, which
        # conflicts with DDP's autograd hooks and causes version mismatch errors.
        # SyncBatchNorm uses a different backward path that avoids this, and also
        # synchronizes batch statistics across ranks for better training quality.
        acmodel_raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(acmodel_raw)
        from torch.nn.parallel import DistributedDataParallel as DDP_cls
        acmodel = DDP_cls(acmodel_raw, device_ids=[rank])
    if is_main:
        txt_logger.info("Model loaded\n")
        txt_logger.info("{}\n".format(acmodel_raw))

    # Load algo
    seeds = range(args.seed_base + runner_offset,
                  args.seed_base + runner_offset + len(penv_pool.envs))
    reshape_reward = None

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(penv_pool, args.seed, seeds, acmodel,
                                local_device, args.frames_per_env_runner,
                                args.discount, args.lr,
                                args.gae_lambda, args.entropy_coef,
                                args.value_loss_coef,
                                args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps,
                                preprocess_obss, reshape_reward)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(penv_pool, args.seed, seeds, acmodel,
                                local_device, args.frames_per_env_runner,
                                args.discount, args.lr,
                                args.gae_lambda, args.entropy_coef,
                                args.value_loss_coef,
                                args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps,
                                args.epochs, args.batch_size,
                                preprocess_obss, reshape_reward)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    load_optimizer_state(algo.optimizer, status, local_device,
                         current_ebs=args.batch_size * world_size,
                         logger=txt_logger if is_main else None)

    # Create exponential decay LR scheduler, so every N steps LR
    # reduced by gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        algo.optimizer,
        step_size=args.lr_steps,
        gamma=args.lr_gamma)

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    duration_offset = status.get("duration", 0) or \
        (is_main and _sprout_duration(spr, model_dir) or 0)
    start_time = time.time()
    start_time -= min(duration_offset, start_time)

    if is_main:
        txt_logger.info(f"Start training from {_fmt_frames(num_frames)} frames | run {run_id}\n")

    train_success_sum = 0.0
    train_success_count = 0

    while num_frames < args.frames_int:
        # Update model parameters
        update_start_time = time.time()
        if ddp:
            # SyncBatchNorm AllReduces during rollout are expensive; eval mode
            # uses stored running stats instead, eliminating per-step syncs.
            acmodel_raw.eval()
        exps, logs1 = algo.collect_experiences()
        if ddp:
            acmodel_raw.train()
        logs2 = algo.update_parameters(exps, apply_update=not args.dry_run)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        if not args.dry_run:
            scheduler.step()

        total_frames = logs["num_frames"] * world_size
        num_frames += total_frames
        update += 1

        success_per_episode = utils.synthesize(
            [1 if s else 0 for s in logs["success_per_episode"]])
        success_rate = success_per_episode['mean']
        train_success_sum += success_rate
        train_success_count += 1
        duration = int(time.time() - start_time)

        # Print logs (main rank only)
        if is_main and args.log_interval > 0 and (update % args.log_interval == 0 or
                                                   num_frames >= args.frames_int):
            fps = total_frames / (update_end_time - update_start_time)
            returns_arr = np.array(logs["return_per_episode"])           # (N, L)
            rreturns_arr = np.array(logs["reshaped_return_per_episode"]) # (N, L)
            L = returns_arr.shape[1]
            return_per_episode = [utils.synthesize(returns_arr[:, i]) for i in range(L)]
            rreturn_per_episode = [utils.synthesize(rreturns_arr[:, i]) for i in range(L)]
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            for i, rr in enumerate(rreturn_per_episode):
                header += [f"rreturn{i}_" + key for key in rr.keys()]
                data += rr.values()
            header += ["success_rate"]
            data += [success_rate]
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()

            metrics = [("entropy", "H"),
                       ("value", "V"),
                       ("policy_loss", "pL"),
                       ("value_loss", "vL"),
                       ("kl", "KL"),
                       ("clip_frac", "cF")]
            for m in metrics:
                v = logs[m[0]]
                header += [f"{m[0]}_lvl{i}" for i in range(len(v))]
                data += v.tolist()

            header += ["grad_norm"]
            data += [logs["grad_norm"]]

            grad = logs["grad_norm"]
            grad_bar = _scale_bar(grad, GRAD_NORM_GOOD_HI)
            bar_metrics = {"kl", "clip_frac"}
            txt_logger.info(
                f"U {update} | F {_fmt_frames(num_frames)} | FPS {fps:04.0f} | D {_fmt_duration(duration)}"
                f" | S {success_rate:.2f} | ∇ {grad:.3f}{grad_bar}")
            for i in range(L):
                rr = list(rreturn_per_episode[i].values())
                mv = " | ".join(f"{m[1]} {logs[m[0]][i]:.3f}"
                               for m in metrics if m[0] not in bar_metrics)
                kl = logs["kl"][i]; kl_bar = _scale_bar(kl, KL_GOOD_HI)
                cf = logs["clip_frac"][i]; cf_bar = _scale_bar(cf, CLIP_FRAC_GOOD_HI)
                e_term = args.entropy_coef       * logs["entropy"][i]
                v_term = args.value_loss_coef    * logs["value_loss"][i]
                p_term = abs(logs["policy_loss"][i])
                total  = e_term + v_term + p_term
                if total > 0:
                    lp = (f"e:{e_term/total*100:.0f}%"
                          f" v:{v_term/total*100:.0f}%"
                          f" p:{p_term/total*100:.0f}%")
                else:
                    lp = "e:0% v:0% p:0%"
                prefix = f"  L{i} | " if L > 1 else "  "
                txt_logger.info(
                    f"{prefix}rR:μσmM {rr[0]:.2f} {rr[1]:.2f} {rr[2]:.2f} {rr[3]:.2f}"
                    f" | {mv}"
                    f" | L {lp} | KL {kl:.3f}{kl_bar} | cF {cf:.3f}{cf_bar}")

            for i, rp in enumerate(return_per_episode):
                header += [f"return{i}_" + key for key in rp.keys()]
                data += rp.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status (main rank only)
        if is_main and args.save_interval > 0 and (update % args.save_interval == 0 or
                                                    num_frames >= args.frames_int):
            num_eval_envs = min(len(eval_penv_pool.envs), args.eval_episodes)
            txt_logger.info("Evaluating the model's {} episodes with {} environments".format(
                args.eval_episodes, num_eval_envs))

            acmodel_raw.eval()
            eval_start_time = time.time()
            vlogs = batch_evaluate(acmodel_raw, preprocess_obss, eval_penv_pool,
                                   argmax=True, global_seed=args.seed,
                                   seed_base=args.eval_seed,
                                   episodes=args.eval_episodes)
            elapsed_time = time.time() - eval_start_time
            acmodel_raw.train()

            returns = vlogs['return_per_episode']
            success_rate = np.mean([1 if s else 0 for s in vlogs["success_per_episode"]])
            returns_arr = np.mean(np.array(returns), axis=0)  # (L,)

            header = ["evaluation_success_rate"] + [f"evaluation_return{i}" for i in range(len(returns_arr))]
            data = [success_rate] + returns_arr.tolist()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

            status = {"num_frames": num_frames,
                      "update": update,
                      "duration": duration,
                      "success_rate": success_rate,
                      "effective_batch_size": args.batch_size * world_size,
                      "optimizer_state": algo.optimizer.state_dict()}
            acmodel_raw.save_to_status(status)
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            if len(returns_arr) == 1:
                R_str = f"R {returns_arr[0]:.3f}"
            else:
                R_str = " | ".join(f"R{i} {r:.3f}" for i, r in enumerate(returns_arr))
            txt_logger.info(f"Evaluation: D {_fmt_duration(elapsed_time)} | {R_str} | S {success_rate:.3f} | bS {best_success_rate:.3f}")
            txt_logger.info("Status saved")

            snap = {"duration": duration,
                    "frames": num_frames,
                    "success_rate": success_rate}
            train_mean_sr = train_success_sum / train_success_count
            eval_dict = {"last": snap}

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                src_path = utils.get_status_path(model_dir, best=False)
                dst_path = utils.get_status_path(model_dir, best=True)
                shutil.copyfile(src_path, dst_path)
                with open(dst_path, 'rb') as f:
                    os.fsync(f.fileno())
                txt_logger.info("Success rate {: .3f}; best model is saved".format(success_rate))
                eval_dict["best"] = snap

            spr.edit(head=args.model,
                     custom_dict={"eval": eval_dict,
                                  "train": {"mean": {"success_rate": train_mean_sr}}},
                     custom_update=True)

            txt_logger.info(f"Collecting env-stats (last {args.stats_episodes}/{args.eval_stats_episodes} training/eval episodes)")
            diablo_state.write_env_stats(os.path.join(model_dir, "env-stats.txt"),
                             eval_runners=False,
                             last_episodes=args.stats_episodes)
            diablo_state.write_env_stats(os.path.join(model_dir, "eval-env-stats.txt"),
                             eval_runners=True,
                             last_episodes=args.eval_stats_episodes)
            if num_frames >= args.frames_int:
                txt_logger.info(f"Collecting env-stats (full training and eval periods)")
                diablo_state.write_env_stats(os.path.join(model_dir, "env-stats-all.txt"),
                                 eval_runners=False,
                                 last_episodes=0)
                diablo_state.write_env_stats(os.path.join(model_dir, "eval-env-stats-all.txt"),
                                 eval_runners=True,
                                 last_episodes=0)

    if ddp:
        import torch.distributed as dist
        # All non-main ranks wait here until rank 0 finishes writing all-episodes
        # stats so their game instances (temp dirs) stay alive during collection.
        dist.barrier()
    if is_main:
        tb_writer.close()
    return 0


def demos_il(args, gameconfig):
    from rl import utils
    from rl.imitation import ImitationLearning, BotEnv
    from rl.torch_ac.utils import ParallelEnvPool

    bot_constructor = diablo_bot.get_bot_constructor(args.bot)

    # Prepare demos dir
    spr, demos_dir, run_id = prepare_directory_for_run(args, args.model)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(demos_dir)
    csv_file, csv_logger = utils.get_csv_logger(demos_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))
    txt_logger.info(f"Run: {run_id}\n")

    num_envs = min(args.env_runners, args.episodes_int)

    # Load bots
    bots_envs = []
    ts = 0
    for i in range(num_envs):
        env_config = copy.deepcopy(gameconfig)
        env_config['index'] = i

        # Run or attach to Diablo (devilutionX) instance
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        bot = bot_constructor(game, args, view_radius=env_config['view-radius'])
        bots_envs.append(BotEnv(bot))

        if time.time() - ts >= 3.0 or i == num_envs - 1:
            ts = time.time()
            print(f"{i+1}/{num_envs} bot instances are created")

    pbot_pool = ParallelEnvPool(bots_envs)

    txt_logger.info("Bots are loaded\n")

    seed = args.seed_base

    validation = hasattr(args, 'for-validation')
    demos_path = utils.get_demos_path(demos_dir, args.env, valid=validation)
    all_demos = utils.load_demos(demos_path)
    all_steps_cnt = 0
    if all_demos:
        txt_logger.info(f"Loaded {len(all_demos)} demo episodes")
        seeds_set = set()
        new_demos = []
        for demo in all_demos:
            seed, steps = demo
            all_steps_cnt += len(steps)
            if seed in seeds_set:
                txt_logger.info(f"Drop duplicate seed {seed}")
            else:
                seeds_set.add(seed)
                new_demos.append(demo)
        seed = max(seeds_set) + 1
        all_demos = new_demos
        txt_logger.info(f"Continue populating demo episodes starting from {seed} seed")

    ts = time.time()
    skip_seeds = {int(s) for s in args.skip_seeds.split(",")} if args.skip_seeds else {}
    steps_cnt = 0
    demos_cnt = 0
    i = 0

    txt_logger.info(f"Start collecting {args.episodes_int:,} demo episodes starting from seed {seed}" +
                    f", skipping seeds {skip_seeds}" if skip_seeds else "")

    while len(all_demos) < args.episodes_int:
        left = args.episodes_int - len(all_demos)
        num_envs = min(left, num_envs)
        seeds = range(seed, seed + num_envs)
        if skip_seeds:
            seeds_set = set(seeds)
            seeds_to_skip = seeds_set.intersection(skip_seeds)
            if seeds_to_skip:
                skip_seeds -= seeds_to_skip
                seeds_set -= seeds_to_skip
                seeds = list(seeds_set)

        seed += num_envs
        demos, _, steps = ImitationLearning.generate_demos(pbot_pool, seeds)
        all_steps_cnt += steps
        steps_cnt += steps
        all_demos += demos
        demos_cnt += len(demos)

        last_iter = (args.episodes_int == len(all_demos))

        # Log some metrics
        if args.log_interval and i % args.log_interval == 0 or last_iter:
            diff = time.time() - ts
            txt_logger.info(f"{i+1} | {len(all_demos):,} / {args.episodes_int:,} demo episodes (seeds {seeds[0]}-{seeds[-1]}) and {steps_cnt} steps generated | {steps_cnt / diff:.0f} FPS | {diff:.0f}s")
            ts = time.time()
            steps_cnt = 0

        # Save demos
        if args.save_interval and i % args.save_interval == 0 or last_iter:
            utils.save_demos(all_demos, demos_path)
            size = os.path.getsize(demos_path)
            if size >> 30:
                size_str = f"{size / (1<<30):.2f}GB"
            else:
                size_str = f"{size / (1<<20):.2f}MB"
            txt_logger.info(f"{i+1} | {len(all_demos):,} demo episodes and {all_steps_cnt:,} steps saved, {size_str}")

        i += 1


def train_il(args, gameconfig):
    import tensorboardX
    from rl.utils import device
    from rl.imitation import ImitationLearning, BotEnv
    from rl.torch_ac.utils import ParallelEnvPool

    bot_constructor = diablo_bot.get_bot_constructor(args.bot)

    # Prepare model dir
    spr, model_dir, run_id = prepare_directory_for_run(args, args.model)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))
    txt_logger.info(f"Run: {run_id}\n")

    # Used device
    txt_logger.info(f"Device: {device}\n")

    # Load environments and bots
    envs = []
    bots_envs = []
    ts = 0
    for i in range(args.env_runners):
        env_config = copy.deepcopy(gameconfig)
        env_config['index'] = i

        # Some old environments have specific configurations that need
        # to be adjusted before starting a game instance
        EnvClass = utils.get_env_class(args.env)
        EnvClass.tune_config(env_config)

        # Run or attach to Diablo (devilutionX) instance
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        env = utils.make_env(args.env, env_config, game)
        bot = bot_constructor(game, args, view_radius=env_config['view-radius'])
        bots_envs.append(BotEnv(bot))
        envs.append(env)

        if time.time() - ts >= 3.0 or i == args.env_runners - 1:
            ts = time.time()
            print(f"{i+1}/{args.env_runners} environment and bot instances are created")

    pbot_pool = ParallelEnvPool(bots_envs)
    penv_pool = ParallelEnvPool(envs)

    txt_logger.info("Environments and bots are loaded\n")

    il_learn = ImitationLearning(args, spr, penv_pool, pbot_pool, model_dir,
                                 None, tb_writer, txt_logger, csv_logger,
                                 train_policy=args.phase1 or args.phase3,
                                 train_critic=args.phase2 or args.phase3)

    # Define logger and Tensorboard writer
    L = envs[0].unwrapped.num_hierarchy_levels
    _lk = lambda name: ([name] if L == 1 else [f"{name}{i}" for i in range(L)])
    header = (["update", "frames", "FPS", "duration"]
              + _lk("entropy") + _lk("policy_loss") + _lk("value_loss")
              + _lk("policy_accuracy") + _lk("value_accuracy") + ["grad_norm"]
              + ["validation_policy_accuracy", "validation_value_accuracy",
                 "validation_return", "validation_success_rate"])

    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(il_learn.acmodel))

    if args.phase1:
        txt_logger.info("Start phase 1: train the policy (actor)")
    elif args.phase2:
        txt_logger.info("Start phase 2: warm-up the critic only")
    elif args.phase3:
        txt_logger.info("Start phase 3: train the policy (actor) and critic together")
    else:
        assert 0, "Uknown training phase"

    # Train the imitation learning agent
    il_learn.train(header)

    return 0


def play_ai(args, gameconfig):
    from rl.evaluate import batch_evaluate
    from rl.imitation import BotEnv
    from rl.flat_model import FlatACModel
    from rl.hrl_model import HRLACModel
    from rl.torch_ac.utils import ParallelEnvPool
    from rl.utils import device

    # Load agent
    model_dir = utils.get_run_dir(args.model)
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"model folder '{model_dir}' does not exist")

    # Used device
    print(f"Device: {device}\n")

    # Load environments
    num_envs = min(args.env_runners, args.episodes_int)
    envs = []
    ts = 0
    for i in range(num_envs):
        env_config = copy.deepcopy(gameconfig)
        env_config['index'] = i

        # Some old environments have specific configurations that need
        # to be adjusted before starting a game instance
        EnvClass = utils.get_env_class(args.env)
        EnvClass.tune_config(env_config)

        # Run or attach to Diablo (devilutionX) instance
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        env = utils.make_env(args.env, env_config, game)
        envs.append(env)

        if time.time() - ts >= 3.0 or i == num_envs - 1:
            ts = time.time()
            print(f"{i+1}/{num_envs} environment instances are created")

    penv_pool = ParallelEnvPool(envs)

    print(f"Environments are loaded\n")

    obs_space = penv_pool.envs[0].observation_space
    action_space = penv_pool.envs[0].action_space
    num_hierarchy_levels = penv_pool.envs[0].unwrapped.num_hierarchy_levels

    obs_space, preprocess_obss = utils.get_obss_preprocessor(obs_space)

    if num_hierarchy_levels == 1:
        acmodel = FlatACModel(obs_space, action_space, args.cnn_arch,
                              embedding_dim=args.embedding_dim,
                              use_memory=True, use_text=False)
    else:
        acmodel = HRLACModel(obs_space, action_space, args.cnn_arch,
                             embedding_dim=args.embedding_dim,
                             use_memory=True, use_text=False)

    acmodel.load_from_status(utils.get_status(model_dir, best=args.best))
    acmodel.to(device)
    acmodel.eval()
    if hasattr(preprocess_obss, "vocab"):
        preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    ts = time.time()
    logs = batch_evaluate(acmodel, preprocess_obss, penv_pool, args.argmax,
                          args.seed, args.seed_base, args.episodes_int,
                          pause=args.pause)
    duration = time.time() - ts

    returns = logs['return_per_episode']
    frames = logs['num_frames_per_episode']
    durations = logs['duration_per_episode']
    seeds = logs['seed_per_episode']

    for f, d, r, s in zip(frames, durations, returns, seeds):
        success = np.all(np.asarray(r) > 0.0)
        print(f"seed {s:2d} | {'success' if success else 'failure'} | steps {f:4d} | {f / d:3.0f} FPS | took {d:.2f}s")

    successes = sum(1 if np.all(np.asarray(r) > 0.0) else 0 for r in returns)
    success_rate = successes / len(returns)
    print(f"S {success_rate:.2f} | {successes} ok / {len(returns) - successes} fail of {len(returns)} | "
          f"steps {_fmt_frames(np.sum(frames))} | time {duration:.2f}s")

    return 0


def agent_ai(args, gameconfig):
    from diablo_agent import AgentAI, ModelRunner
    from rl.flat_model import FlatACModel
    from rl.hrl_model import HRLACModel
    from rl.utils import device

    print(f"Device: {device}\n")

    # Create one env instance to obtain obs/action space metadata and a game
    env_config = copy.deepcopy(gameconfig)
    env_config['index'] = 0
    EnvClass = utils.get_env_class(args.env)
    EnvClass.tune_config(env_config)
    game = diablo_state.DiabloGame.run_or_attach(env_config)
    env  = utils.make_env(args.env, env_config, game)

    obs_space            = env.unwrapped.observation_space
    action_space         = env.unwrapped.action_space
    num_hierarchy_levels = env.unwrapped.num_hierarchy_levels

    obs_space, preprocess_obss = utils.get_obss_preprocessor(obs_space)

    # Build {dungeon_level: ModelRunner}.  Multiple levels may share one runner
    # object when they map to the same model name.
    level_to_name   = AgentAI.parse_model_spec(args.model)
    unique_names    = dict.fromkeys(level_to_name.values())  # preserves order, dedupes
    name_to_runner  = {}
    for model_name in unique_names:
        model_dir = utils.get_run_dir(model_name)
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"model folder '{model_dir}' does not exist")
        if hasattr(preprocess_obss, "vocab"):
            preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))
        if num_hierarchy_levels == 1:
            acmodel = FlatACModel(obs_space, action_space, args.cnn_arch,
                                  embedding_dim=args.embedding_dim,
                                  use_memory=True, use_text=False)
        else:
            acmodel = HRLACModel(obs_space, action_space, args.cnn_arch,
                                 embedding_dim=args.embedding_dim,
                                 use_memory=True, use_text=False)
        acmodel.load_from_status(utils.get_status(model_dir, best=args.best))
        acmodel.to(device)
        acmodel.eval()
        name_to_runner[model_name] = ModelRunner(acmodel, preprocess_obss, device,
                                                 argmax=args.argmax)
        print(f"Loaded model: {model_name}")

    model_runners = {lvl: name_to_runner[name] for lvl, name in level_to_name.items()}

    supervisor = AgentAI(
        game, model_runners,
        view_radius=gameconfig['view-radius'],
        kill_threshold=args.kill_threshold,
        repair_threshold=args.repair_threshold,
        max_steps_per_level=args.max_steps_per_level,
        no_gear_management=args.no_gear_management,
        safe_radius=args.safe_radius,
        pause=args.pause,
        stat_strategy=args.stat_strategy)
    supervisor.run()

    env.close()
    return 0


def play_bot(args, gameconfig):
    from rl.imitation import BotEnv, ImitationLearning
    from rl.torch_ac.utils import ParallelEnvPool

    bot_constructor = diablo_bot.get_bot_constructor(args.bot)

    # Load environments or bots
    num_envs = min(args.env_runners, args.episodes_int)
    envs = []
    ts = 0
    for i in range(num_envs):
        env_config = copy.deepcopy(gameconfig)
        env_config['index'] = i

        # Run or attach to Diablo (devilutionX) instance
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        bot = bot_constructor(game, args, view_radius=env_config['view-radius'])
        env = BotEnv(bot)
        envs.append(env)

        if time.time() - ts >= 3.0 or i == num_envs - 1:
            ts = time.time()
            print(f"{i+1}/{num_envs} bot instances are created")

    pbot_pool = ParallelEnvPool(envs)

    print(f"Bots are loaded\n")

    ts = time.time()
    seeds = range(args.seed_base, args.seed_base + args.episodes_int)
    demos, durations, num_frames = \
        ImitationLearning.generate_demos(pbot_pool, seeds, pause=args.pause)
    duration = time.time() - ts

    for demo, d in zip(demos, durations):
        s, actions = demo
        f = len(actions)
        print(f"seed {s:2d} | steps {f:4d} | {f / d:3.0f} FPS | took {d:.2f}s")
    print(f"{len(demos)} demos | steps {_fmt_frames(sum(num_frames))} | time {duration:.2f}s")

    return 0


def main():
    # Set default signal handlers
    set_sighandlers()
    # Set big enough limits
    set_rlimits()

    incompatible_options, parser = make_diablo_parser()
    args = parser.parse_args(namespace=DiabloParserNamespace())

    # Check if some options are incompatible
    argv_set = set(sys.argv)
    for opt, incompatibles in incompatible_options.items():
        attr = opt.lstrip('-').replace('-', '_')
        if getattr(args, attr, False) and (set(incompatibles) & argv_set):
            parser.error(f"{opt} cannot be used together with {' or '.join(incompatibles)}")

    config = configparser.ConfigParser()
    config.read('diablo-ai.ini')

    # Absolute path
    diablo_build_path  = Path(config['default']['diablo-build-path']).resolve()
    diablo_mshared_filename = config['default']['diablo-mshared-filename']

    if not diablo_build_path.is_dir() or len(diablo_mshared_filename) == 0:
        print("Error: initial configuration is invalid. Please check your 'diablo-ai.ini' file and provide valid paths for 'diablo-build-path' and 'diablo-mshared-filename' configuration options.")
        sys.exit(1)

    def _build_char_tables(a):
        from diablo_agent import AgentAI
        section = 'devilutionx-gameplay'
        if section not in config:
            parser.error(f"[{section}] section missing from diablo-ai.ini")
        tables = dict(config[section])
        if not tables.get('char level up attrs', '').strip():
            tables['char level up attrs'] = AgentAI.stat_strategy_to_attrs(a.stat_strategy)
        return tables

    if not (diablo_build_path / "spawn.mpq").exists():
        print(f"Error: Shareware file \"spawn.mpq\" for Diablo content does not exist. Please download and place the file alongside the `devilutionx` binary with the following command:\n\twget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq -P {diablo_build_path}")
        sys.exit(1)

    diablo_bin_path = str(diablo_build_path / "devilutionx")
    delayed_import(diablo_bin_path)

    if args.command == "sprout":
        # re-run through sprout.main(), but pass sys.argv after "sprout"
        sprout_args = ['--working', utils.get_models_dir()]
        sprout_args += sys.argv[sys.argv.index("sprout")+1:]
        head_idx = next((i for i, a in enumerate(sprout_args) if a == '--head'), -1)
        show_head = sprout_args[head_idx + 1] if head_idx >= 0 else None
        return sprout.main(argv=sprout_args, default_parser=parser,
                           params_diff=_sprout_params_diff,
                           skip_params=SPROUT_SKIP_PARAMS,
                           params_overrides_fn=lambda p: show_head if p == 'model' else None)
    if args.command == 'list':
        list_devilution_processes(str(diablo_bin_path),
                                  diablo_mshared_filename)
        return 0
    if args.command == 'model-drop-optimizer':
        return model_drop_optimizer(args)
    if args.command == 'model-drop-best':
        return model_drop_best(args)
    if args.command == 'env-stats':
        return diablo_state.env_stats(args)

    # Set seed for all randomness sources
    utils.seed(args.seed)

    gameconfig = {
        "mshared-filename": diablo_mshared_filename,
        "diablo-bin-path": diablo_bin_path,

        # Common
        "index": 0, # Just a sequential number, will be overridden for each instance
        "seed": args.seed_base, # Will be overridden by a subsequent env reset with a valid seed
        "fixed-seed": args.fixed_seed \
            if hasattr(args, "fixed_seed") else False,
        "invincible-player": args.invincible_player,
        "no-monsters": args.no_monsters,
        "blind-monsters": args.blind_monsters,
        "harmless-barrels": args.harmless_barrels,
        "no_butcher": args.no_butcher,
        "no-quests": not args.enable_quests,
        "spell-potency": args.spell_potency,
        "no-spells": args.no_spells,
        "stats-scale": args.stats_scale,
        "char-tables": _build_char_tables(args) if args.char_tables else None,
        "hero-hp-min-pct":    args.hero_hp_at_start[0],
        "hero-hp-max-pct":    args.hero_hp_at_start[1],
        "hero-mana-min-pct":  args.hero_mana_at_start[0],
        "hero-mana-max-pct":  args.hero_mana_at_start[1],
        "hero-potions-min":   args.hero_potions_at_start[0],
        "hero-potions-max":   args.hero_potions_at_start[1],
        "no-auto-walk-on-seconday-action": True, # Changed by old environments
        "view-radius": args.view_radius,
        "game-ticks-per-step": args.game_ticks_per_step,
        "step-mode": not args.real_time,
        "gui": args.gui,
        "dungeon-level": args.dungeon_level,

        # AI
        "log-to-stdout": args.log_to_stdout \
            if hasattr(args, "log_to_stdout") else False,
        "no-actions": args.no_actions \
            if hasattr(args, "no_actions") else False,
        "exploration-door-attraction": args.exploration_door_attraction \
            if hasattr(args, "exploration_door_attraction") else False,
        "exploration-door-backtrack-penalty": args.exploration_door_backtrack_penalty \
            if hasattr(args, "exploration_door_backtrack_penalty") else False,
        "goal-far-bias": args.goal_far_bias \
            if hasattr(args, "goal_far_bias") else False,
    }

    if args.attach:
        path_or_pid = args.attach

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

    if args.command == 'play':
        return curses.wrapper(lambda stdscr: run_tui(stdscr, args, gameconfig))
    if args.command == 'train-ai':
        return train_ai(args, gameconfig)
    if args.command == 'demos-il':
        return demos_il(args, gameconfig)
    if args.command == 'train-il':
        return train_il(args, gameconfig)
    if args.command == 'play-ai':
        return play_ai(args, gameconfig)
    if args.command == 'agent-ai':
        return agent_ai(args, gameconfig)
    if args.command == 'play-bot':
        return play_bot(args, gameconfig)

    print("Not supported yet")
    return 1

if __name__ == "__main__":
    main()
