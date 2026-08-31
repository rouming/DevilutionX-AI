#!/usr/bin/env python3
"""
count-monsters.py - Measure actual monster count per dungeon floor by driving the engine.

For each floor 1-16, sends N new-game resets via the ring buffer and reads
ActiveMonsterCount immediately after each dungeon generation. Prints mean/std/min/max
and the per-floor table suitable for DEFAULT_MONSTERS_PER_FLOOR in diablo-sim.py.

Usage (run from ai/ directory):
    python scripts/count-monsters.py [--count N] [--floors FLOOR_SPEC]

Examples:
    python scripts/count-monsters.py
    python scripts/count-monsters.py --count 200 --floors 9-12
"""

import argparse
import configparser
import os
import sys
import statistics
import time
import tempfile
from collections import defaultdict
from pathlib import Path

# Add ai/ to path so imports work when script is run from anywhere
AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR))

import diablo_state
import ring
from diablo_state import DiabloGame, DUNGEON_LEVEL_DEFAULT

DEFAULT_COUNT  = 200
MAX_FLOOR      = 16


def parse_floors(s):
    result = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    invalid = [f for f in result if not 1 <= f <= MAX_FLOOR]
    if invalid:
        raise argparse.ArgumentTypeError(f"Floors out of range 1-{MAX_FLOOR}: {invalid}")
    return sorted(result)


def build_config(count):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(AI_DIR, 'diablo-ai.ini'))

    build_path = Path(cfg['default']['diablo-build-path']).resolve()
    if not build_path.is_absolute():
        build_path = (AI_DIR / build_path).resolve()
    diablo_bin  = str(build_path / 'devilutionx')
    mshared     = cfg['default']['diablo-mshared-filename']

    return {
        "diablo-bin-path":    diablo_bin,
        "mshared-filename":   mshared,
        "seed":               1,
        "fixed-seed":         False,
        "dungeon-level":      DUNGEON_LEVEL_DEFAULT,
        "gui":                False,
        "game-ticks-per-step": 10,
        "step-mode":          True,
        "invincible-player":  True,
        "no-monsters":        False,
        "blind-monsters":     False,
        "harmless-barrels":   False,
        "no_butcher":         True,
        "no-quests":          True,
        "spell-potency":      0.0,
        "no-spells":          False,
        "stats-scale":        1.0,
        "no-auto-walk-on-seconday-action": True,
        "hero-hp-min-pct":    100,
        "hero-hp-max-pct":    100,
        "hero-mana-min-pct":  100,
        "hero-mana-max-pct":  100,
        "hero-potions-min":   0,
        "hero-potions-max":   0,
        "index":              0,
    }


def count_floor(game, floor, n_seeds):
    counts = []
    key = ring.RingEntryType.RING_ENTRY_KEY_NEW | \
          ring.RingEntryType.RING_ENTRY_F_SINGLE_TICK_PRESS
    for seed in range(1, n_seeds + 1):
        data = ((floor << 1) | 1, seed)
        game.submit_key(key, data=data)
        cnt = int(game.state.ActiveMonsterCount.value)
        counts.append(cnt)
    return counts


def main():
    ap = argparse.ArgumentParser(
        description="Measure monster count per dungeon floor using the actual engine.")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT, metavar="N",
                    help=f"Number of random seeds to sample per floor (default {DEFAULT_COUNT})")
    ap.add_argument("--floors", type=parse_floors, default=list(range(1, MAX_FLOOR + 1)),
                    metavar="SPEC",
                    help="Floors to measure, e.g. '9-12' or '1,5,9-12' (default: 1-16)")
    args = ap.parse_args()

    os.chdir(AI_DIR)

    config = build_config(args.count)
    print(f"Starting engine: {config['diablo-bin-path']}")
    print(f"Floors: {args.floors}  Samples per floor: {args.count}")
    print()

    game = DiabloGame.run(config)
    # Give the engine a moment to init before the first reset
    time.sleep(2.0)

    results = {}
    for floor in args.floors:
        sys.stdout.write(f"  d={floor:2d} ... ")
        sys.stdout.flush()
        counts = count_floor(game, floor, args.count)
        results[floor] = counts
        m   = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 0.0
        print(f"mean={m:6.1f}  std={std:5.1f}  min={min(counts):4d}  max={max(counts):4d}")

    game.stop_or_detach()

    print()
    print("Per-floor summary:")
    print(f"  {'d':>2}  {'mean':>6}  {'std':>5}  {'min':>4}  {'max':>4}")
    print("  " + "-" * 30)
    for floor, counts in sorted(results.items()):
        m   = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 0.0
        print(f"  {floor:2d}  {m:6.1f}  {std:5.1f}  {min(counts):4d}  {max(counts):4d}")

    print()
    print("DEFAULT_MONSTERS_PER_FLOOR = {")
    items = sorted(results.items())
    for i, (floor, counts) in enumerate(items):
        m = statistics.mean(counts)
        comma = "," if i < len(items) - 1 else ""
        print(f"    {floor}: {round(m)}{comma}")
    print("}")


if __name__ == "__main__":
    main()
