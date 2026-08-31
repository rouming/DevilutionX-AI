#!/usr/bin/env python3
"""
diablo-sim.py - Combined XP + gear simulation for Diablo 1 Warrior.

Simulates a Warrior running floors 1-16 in a single Markov loop:
  - XP from monster kills -> char level at floor entry
  - Gear carried floor to floor (Markov state)
  - Item requirements checked against effective stats (base level-up + gear)
  - Stat-boosting items applied first to unlock requirement-gated items

Outputs Char level up table, Char gear stats, Char gear combat.

Usage:
    python diablo-sim.py [--clear LO:HI] [--items SPEC] [--survivor-scale MAX:TAU]

Examples:
    python diablo-sim.py
    python diablo-sim.py --clear 60:80
    python diablo-sim.py --items 1=1,2-8=2,9-16=3
    python diablo-sim.py --survivor-scale 1.65:2.5   # match log data AC (survivor bias correction)
"""

import argparse
import csv
import math
import re
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
MONSTDAT_TSV = REPO_ROOT / "assets/txtdata/monsters/monstdat.tsv"
EXP_TSV      = REPO_ROOT / "assets/txtdata/Experience.tsv"
PREFIXES_TSV = REPO_ROOT / "build/assets/txtdata/items/item_prefixes.tsv"
SUFFIXES_TSV = REPO_ROOT / "build/assets/txtdata/items/item_suffixes.tsv"
ITEMDAT_TSV  = REPO_ROOT / "build/assets/txtdata/items/itemdat.tsv"

MAX_FLOOR = 16
N_SIMS    = 10_000

# Monster count statistics per dungeon floor, measured by driving the actual engine
# via count-monsters.py (200 seeds per floor, single-player, no multiplayer bonus).
# Diablo 1 is a fixed game - these values will not change.
# Each entry: (mean, std, min, max).
FLOOR_MONSTER_STATS = {
     1: (122.5,  17.0,  90, 174),  # Cathedral
     2: (153.0,  16.9, 112, 200),
     3: (164.7,  16.0, 131, 200),
     4: (164.0,  17.7, 128, 200),
     5: (136.0,  16.3, 103, 181),  # Catacombs
     6: (132.5,  14.6, 104, 177),
     7: (132.0,  16.0,  96, 184),
     8: (132.3,  16.2,  98, 181),
     9: ( 90.5,   7.5,  78, 113),  # Caves
    10: ( 92.1,   8.3,  78, 132),
    11: ( 89.3,   7.1,  78, 117),
    12: ( 93.1,   8.7,  77, 121),
    13: (119.9,   6.2, 107, 144),  # Hell
    14: (121.1,   7.7, 109, 151),
    15: (121.0,   7.5, 108, 144),
    16: (172.1,   5.0, 163, 188),
}

# Warrior base stats in Diablo 1: level-1 start and per-level gain.
# STR/VIT gain +2/level, DEX +1/level (conservative avg of alternating +1/+2), MAG fixed.
_WAR_BASE = {'str': 30, 'mag': 10, 'dex': 20, 'vit': 25}
_WAR_GAIN = {'str':  2, 'mag':  0, 'dex':  1, 'vit':  2}

def _warrior_stats(level):
    g = level - 1
    return {k: _WAR_BASE[k] + _WAR_GAIN[k] * g for k in _WAR_BASE}

# Engine affix probability constants (items.cpp GetItemPowerPrefixAndSuffix).
P_PREFIX = 0.25 + (0.75 / 3.0 * 0.5)   # ~0.375
P_SUFFIX = 2.0 / 3.0 + (0.75 / 3.0 * 0.5)  # ~0.792

STAT_POWERS = {'STR', 'DEX', 'MAG', 'VIT'}

# Warrior gear slots: (name, affix_itype, score_method, drop_weight).
# Weights derived from itemdat.tsv dropRate sums per slot class.
_SLOT_DEFS = [
    ('weapon', 'Weapon', 'damage', 46),
    ('shield', 'Shield', 'ac',      6),
    ('armor',  'Armor',  'ac',     17),
    ('helm',   'Armor',  'ac',      6),
    ('ring1',  'Misc',   'misc',    2),
    ('ring2',  'Misc',   'misc',    1),
    ('amulet', 'Misc',   'misc',    2),
]
_SLOT_POP  = [s for s, _, _, w in _SLOT_DEFS for _ in range(w)]
_SLOT_INFO = {s: (itype, score_by) for s, itype, score_by, _ in _SLOT_DEFS}

# Shield AC weight in weapon slot score (from agent _is_better_for_warrior).
_AC_WEIGHT = 0.3


# --- data loading ---

def _load_affixes(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            v1, v2 = row['power.value1'].strip(), row['power.value2'].strip()
            rows.append({
                'power':  row['power'],
                'v1':     int(v1) if v1 else 0,
                'v2':     int(v2) if v2 else 0,
                'minlvl': int(row['minLevel']),
                'types':  set(row['itemTypes'].split(',')),
            })
    return rows


def _load_items(path):
    weapons, armors, shields, helms, rings, amulets = [], [], [], [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            dr = row['dropRate'].strip()
            if not dr or int(dr) == 0:
                continue
            cls, equip = row['class'], row['equipType']
            if equip == 'Unequippable':
                continue
            try:
                it = {
                    'mlvl':    int(row['minMonsterLevel'] or 0),
                    'min_dam': int(row['minDamage'] or 0),
                    'max_dam': int(row['maxDamage'] or 0),
                    'min_ac':  int(row['minArmor'] or 0),
                    'max_ac':  int(row['maxArmor'] or 0),
                    'req_str': int(row['minStrength'] or 0),
                    'req_mag': int(row['minMagic'] or 0),
                    'req_dex': int(row['minDexterity'] or 0),
                }
            except (ValueError, KeyError):
                continue
            if cls == 'Weapon' and it['max_dam'] > 0:
                weapons.append(it)
            elif cls == 'Armor':
                if equip == 'Armor' and it['max_ac'] > 0:
                    armors.append(it)
                elif equip == 'One-handed' and it['max_ac'] > 0:
                    shields.append(it)
                elif equip == 'Helm' and it['max_ac'] > 0:
                    helms.append(it)
            elif cls == 'Misc':
                if equip == 'Ring':
                    rings.append(it)
                elif equip == 'Amulet':
                    amulets.append(it)
    return weapons, armors, shields, helms, rings, amulets


def _load_monsters():
    rows = []
    with open(MONSTDAT_TSV, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['availability'] == 'Never':
                continue
            try:
                rows.append({
                    'minLvl': int(row['minDunLvl']),
                    'maxLvl': int(row['maxDunLvl']),
                    'mlevel': int(row['level']),
                    'exp':    int(row['exp']),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _load_xp():
    thresholds = []
    with open(EXP_TSV, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            thresholds.append(int(row['Experience']))
    return thresholds


# --- XP / level helpers ---

def _xp_to_level(xp, thresholds):
    level = 1
    for needed in thresholds:
        if xp >= needed:
            level += 1
        else:
            break
    return level


def _xp_for_kill(player_level, monster):
    delta = monster['mlevel'] - player_level
    return max(0, int(monster['exp'] * (1.0 + delta / 10.0)))


# --- item / gear helpers ---

def _affix_pool(affixes, itype, lvl_lo, lvl_hi):
    return [a for a in affixes
            if itype in a['types'] and lvl_lo <= a['minlvl'] <= lvl_hi]


def _build_caches(prefixes, suffixes, slot_pools):
    """Precompute per-(slot,floor) item pools and per-(itype,lvl_lo,lvl_hi) affix pools."""
    item_cache  = {}
    affix_cache = {}
    for slot, itype, _, _ in _SLOT_DEFS:
        for floor in range(1, MAX_FLOOR + 1):
            item_cache[(slot, floor)] = [it for it in slot_pools[slot] if it['mlvl'] <= floor]
            lvl_lo = max(1, floor // 2)
            lvl_hi = floor
            key = (itype, lvl_lo, lvl_hi)
            if key not in affix_cache:
                affix_cache[key] = {
                    'pre': _affix_pool(prefixes, itype, lvl_lo, lvl_hi),
                    'suf': _affix_pool(suffixes, itype, lvl_lo, lvl_hi),
                }
    return item_cache, affix_cache


def _roll_affixes(rng, affix_cache, itype, lvl_lo, lvl_hi):
    bonuses = defaultdict(int)
    pools = affix_cache[(itype, lvl_lo, lvl_hi)]
    for pool, prob in ((pools['pre'], P_PREFIX), (pools['suf'], P_SUFFIX)):
        if pool and rng.random() < prob:
            a = rng.choice(pool)
            bonuses[a['power']] = rng.randint(a['v1'], a['v2'])
    return bonuses


def _meets_req(it, eff):
    return (eff['str'] >= it['req_str'] and
            eff['mag'] >= it['req_mag'] and
            eff['dex'] >= it['req_dex'])


def _score_item(it, bonuses, score_by, shield_ac=0):
    """Score for equip comparison (all affixes known - sim treats items as identified)."""
    stat_bonus = bonuses.get('STR', 0) + bonuses.get('VIT', 0)
    if score_by == 'damage':
        avg_dam = (it['min_dam'] + it['max_dam']) / 2.0
        return avg_dam * (1 + bonuses.get('DAMP', 0) / 100.0) + shield_ac * _AC_WEIGHT + stat_bonus
    if score_by == 'ac':
        avg_ac = (it['min_ac'] + it['max_ac']) / 2.0
        return avg_ac * (1 + bonuses.get('ACP', 0) / 100.0) + stat_bonus
    # misc (jewelry): warrior values STR + VIT
    return float(stat_bonus)


def _empty_slot():
    return {'score': -1.0, 'stats': {}, 'ac': 0,
            'min_dam': 0, 'max_dam': 0, 'tohit': 0,
            'dam_pct': 0, 'resist': 0, 'atk_tier': 0, 'rec_tier': 0}


def _gear_stat_totals(slots):
    totals = {'str': 0, 'dex': 0, 'mag': 0, 'vit': 0}
    for s in slots.values():
        for k in totals:
            totals[k] += s['stats'].get(k, 0)
    return totals


# --- simulation ---

def simulate(monsters, xp_thresholds, item_cache, affix_cache,
             items_per_floor, clear_lo, clear_hi, seed=42):
    """
    Combined XP + gear Markov simulation.
    Gear state carries floor to floor. Item requirements checked against
    effective stats (warrior base + accumulated gear bonuses).
    Stat-boosting items sorted first to unlock requirement-gated items.
    Stats recorded at floor ENTRY (before new items on that floor).
    """
    rng = random.Random(seed)

    floor_pools = {
        d: [m for m in monsters if m['minLvl'] <= d <= m['maxLvl']]
        for d in range(1, MAX_FLOOR + 1)
    }

    level_res  = defaultdict(list)
    gear_res   = defaultdict(lambda: defaultdict(list))
    combat_res = defaultdict(lambda: defaultdict(list))
    req_rej = req_tot = 0

    for _ in range(N_SIMS):
        total_xp = 0
        slots = {s: _empty_slot() for s, *_ in _SLOT_DEFS}
        # Warrior starting gear: Short Sword (2-6 dam) + Buckler (3 AC)
        # Short Sword avg_dam=4.0; score includes current shield_ac*_AC_WEIGHT (Buckler=3)
        slots['weapon'] = dict(_empty_slot(), score=4.0 + 3.0 * _AC_WEIGHT, min_dam=2, max_dam=6)
        # Buckler avg_ac=3.0; use ac-slot score formula so replacements are apples-to-apples
        slots['shield'] = dict(_empty_slot(), score=3.0, ac=3)

        for floor in range(1, MAX_FLOOR + 1):
            player_level = _xp_to_level(total_xp, xp_thresholds)
            level_res[floor].append(player_level)

            # Record gear at floor ENTRY
            gear_bonus = _gear_stat_totals(slots)
            for k in ('str', 'dex', 'mag', 'vit'):
                gear_res[floor][k].append(gear_bonus[k])

            total_ac = sum(slots[s]['ac']
                           for s in ('armor', 'helm', 'shield', 'ring1', 'ring2', 'amulet'))
            wpn = slots['weapon']
            combat_res[floor]['weapon_min'].append(wpn['min_dam'])
            combat_res[floor]['weapon_max'].append(wpn['max_dam'])
            combat_res[floor]['armor_ac'].append(total_ac)
            combat_res[floor]['to_hit_pct'].append(sum(slots[s]['tohit'] for s in slots))
            combat_res[floor]['dam_bonus_pct'].append(wpn['dam_pct'])
            combat_res[floor]['resist'].append(max(slots[s]['resist'] for s in slots))
            combat_res[floor]['atk_tier'].append(wpn['atk_tier'])
            combat_res[floor]['rec_tier'].append(max(slots[s]['rec_tier'] for s in slots))

            # Effective stats for requirement checking
            base = _warrior_stats(player_level)
            eff  = {k: base[k] + gear_bonus[k] for k in base}

            # Draw items for this floor
            lvl_lo = max(1, floor // 2)
            lvl_hi = floor
            n = round(items_per_floor.get(floor, 2.0))
            candidates = []
            for _ in range(n):
                slot = _SLOT_POP[rng.randrange(len(_SLOT_POP))]
                itype, score_by = _SLOT_INFO[slot]
                pool = item_cache[(slot, floor)]
                if not pool:
                    continue
                it = rng.choice(pool)
                bonuses   = _roll_affixes(rng, affix_cache, itype, lvl_lo, lvl_hi)
                shield_ac = slots['shield']['ac'] if slot == 'weapon' else 0
                score     = _score_item(it, bonuses, score_by, shield_ac=shield_ac)
                candidates.append((slot, it, bonuses, score))

            # Stat-boosting items first so their bonuses may unlock requirement-gated items
            candidates.sort(
                key=lambda c: any(c[2].get(p, 0) > 0 for p in STAT_POWERS),
                reverse=True,
            )

            for slot, it, bonuses, new_score in candidates:
                itype, _ = _SLOT_INFO[slot]
                req_tot += 1
                if not _meets_req(it, eff):
                    req_rej += 1
                    continue
                if new_score <= slots[slot]['score']:
                    continue
                # Roll actual AC (random within item range)
                base_ac = rng.randint(it['min_ac'], it['max_ac']) if it['max_ac'] > 0 else 0
                ac_pct  = bonuses.get('ACP', 0)
                ac_b    = base_ac * ac_pct // 100
                if ac_b == 0 and ac_pct > 0 and base_ac > 0:
                    ac_b = 1
                stat_map = {p.lower(): bonuses[p] for p in STAT_POWERS if bonuses.get(p, 0) != 0}
                slots[slot] = {
                    'score':    new_score,
                    'stats':    stat_map,
                    'ac':       base_ac + ac_b,
                    'min_dam':  it['min_dam'],
                    'max_dam':  it['max_dam'],
                    'tohit':    bonuses.get('TOHIT', 0),
                    'dam_pct':  bonuses.get('DAMP', 0),
                    'resist':   max(bonuses.get('FIRERES', 0), bonuses.get('MAGICRES', 0),
                                    bonuses.get('LIGHTRES', 0), bonuses.get('ALLRES', 0)),
                    'atk_tier': bonuses.get('FASTATTACK', 0) if itype == 'Weapon' else 0,
                    'rec_tier': bonuses.get('FASTRECOVER', 0),
                }
                # Refresh eff stats if this item added stat bonuses
                if stat_map:
                    gear_bonus = _gear_stat_totals(slots)
                    eff = {k: base[k] + gear_bonus[k] for k in base}

            # Kill monsters, earn XP
            mpool = floor_pools[floor]
            if mpool:
                mean, std, lo_c, hi_c = FLOOR_MONSTER_STATS[floor]
                n_total = max(lo_c, min(hi_c, round(rng.gauss(mean, std))))
                n_kills = round(n_total * rng.uniform(clear_lo, clear_hi))
                for _ in range(n_kills):
                    monster = rng.choice(mpool)
                    total_xp += _xp_for_kill(player_level, monster)
                    player_level = _xp_to_level(total_xp, xp_thresholds)

    return level_res, gear_res, combat_res, req_rej, req_tot


# --- output helpers ---

def _mean(vals):
    return sum(vals) / len(vals)

def _std(vals):
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

def _fmt(vals):
    return f"{_mean(vals):6.1f} +- {_std(vals):5.1f}"

def _r(vals, clamp=0):
    m, s = _mean(vals), _std(vals)
    lo = int(max(clamp, math.floor(m - s)))
    hi = int(math.ceil(m + s))
    return f"{lo}:{hi}"


def _r_scaled(vals, scale=1.0, clamp=0):
    m, s = _mean(vals) * scale, _std(vals) * scale
    lo = int(max(clamp, math.floor(m - s)))
    hi = int(math.ceil(m + s))
    return f"{lo}:{hi}"


# --- arg parsing ---

def _parse_survivor_scale(s):
    m = re.match(r'^(\d+(?:\.\d+)?)[:\-](\d+(?:\.\d+)?)$', s)
    if not m:
        sys.exit(f"Bad --survivor-scale: {s!r}  expected MAX:TAU, e.g. 1.65:2.5")
    return float(m.group(1)), float(m.group(2))


def _surv_scale(d, surv_max, surv_tau):
    """Per-floor survivor-bias multiplier: 1 + (surv_max-1)*(1-exp(-(d-1)/surv_tau)).
    Corrects for the fact that log data reflects survivors, which self-select for better gear.
    Calibrated to log data: surv_max=1.65, surv_tau=2.5.
    """
    if surv_max <= 1.0:
        return 1.0
    return 1.0 + (surv_max - 1.0) * (1.0 - math.exp(-(d - 1) / surv_tau))


def _parse_clear(s):
    m = re.match(r'^(\d+(?:\.\d+)?)[:\-](\d+(?:\.\d+)?)$', s)
    if not m:
        sys.exit(f"Bad --clear value: {s!r}  (expected LO:HI, e.g. 70:90)")
    lo, hi = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    if not (0 < lo <= hi <= 1.0):
        sys.exit("--clear values must be in 0-100 range with lo <= hi")
    return lo, hi


def _parse_items(spec):
    spec = spec.strip()
    # bare integer: apply to all floors
    if re.match(r'^\d+(?:\.\d+)?$', spec):
        return {d: float(spec) for d in range(1, MAX_FLOOR + 1)}
    result = {}
    for entry in spec.split(','):
        entry = entry.strip()
        m = re.match(r'^(\d+)-(\d+)=(\d+(?:\.\d+)?)$', entry)
        if m:
            for fl in range(int(m.group(1)), int(m.group(2)) + 1):
                result[fl] = float(m.group(3))
        else:
            m = re.match(r'^(\d+)=(\d+(?:\.\d+)?)$', entry)
            if m:
                result[int(m.group(1))] = float(m.group(2))
            else:
                sys.exit(f"Bad --items entry: {entry!r}  (expected integer, N=V, or N-M=V)")
    return result


# --- main ---

def main():
    ap = argparse.ArgumentParser(
        description="Simulate Warrior XP + gear progression, output Char level up table, "
                    "Char gear stats, Char gear combat.")
    ap.add_argument("--clear", default="70:90", metavar="LO:HI",
                    help="Floor clear %% range (default 70:90)")
    ap.add_argument("--items", metavar="SPEC", default=None,
                    help='Items drawn per floor: bare int (all floors), N=V, or N-M=V (default: 12)')
    ap.add_argument("--survivor-scale", metavar="MAX:TAU", default="1.0:1.0",
                    help="Survivor-bias AC scale for charGearCombat: "
                         "1+(MAX-1)*(1-exp(-(d-1)/TAU)). "
                         "Calibrated to log data: 1.65:2.5 (default: 1.0 = disabled)")
    args = ap.parse_args()

    clear_lo, clear_hi = _parse_clear(args.clear)
    surv_max, surv_tau = _parse_survivor_scale(args.survivor_scale)

    if args.items:
        items_per_floor = _parse_items(args.items)
        last = 12.0
        for fl in range(1, MAX_FLOOR + 1):
            if fl in items_per_floor:
                last = items_per_floor[fl]
            else:
                items_per_floor[fl] = last
        print(f"Items per floor  : {args.items}")
    else:
        items_per_floor = {d: 12 for d in range(1, MAX_FLOOR + 1)}
        print("Items per floor  : 12 (default, calibrated against log floor-2 data)")

    print(f"Clear range      : {clear_lo*100:.0f}%-{clear_hi*100:.0f}%")
    if surv_max > 1.0:
        print(f"Survivor AC scale: max={surv_max:.2f}  tau={surv_tau:.2f}  (applied to Char gear combat AC only)")
    else:
        print(f"Survivor AC scale: disabled (use --survivor-scale 1.65:2.5 to match log data)")
    print(f"Simulations      : {N_SIMS}")
    print(f"XP formula       : kill_xp * (1 + (monster_lvl - player_lvl) / 10)  [engine exact]")
    print()

    monsters   = _load_monsters()
    thresholds = _load_xp()
    prefixes   = _load_affixes(PREFIXES_TSV)
    suffixes   = _load_affixes(SUFFIXES_TSV)
    weapons, armors, shields, helms, rings, amulets = _load_items(ITEMDAT_TSV)

    slot_pools = {
        'weapon': weapons, 'shield': shields, 'armor': armors,
        'helm': helms, 'ring1': rings, 'ring2': rings, 'amulet': amulets,
    }
    item_cache, affix_cache = _build_caches(prefixes, suffixes, slot_pools)

    # Monster pool info
    print("Eligible monster pool per floor:")
    hdr = f"  {'d':>2}  {'types':>5}  {'avg_exp':>7}  {'avg_mlvl':>8}  {'n_mean':>8}  {'n_std':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for d in range(1, MAX_FLOOR + 1):
        pool = [m for m in monsters if m['minLvl'] <= d <= m['maxLvl']]
        avg_exp = _mean([m['exp']    for m in pool]) if pool else 0.0
        avg_lvl = _mean([m['mlevel'] for m in pool]) if pool else 0.0
        mean, std = FLOOR_MONSTER_STATS[d][:2]
        print(f"  {d:2d}  {len(pool):5d}  {avg_exp:7.0f}  {avg_lvl:8.1f}  {mean:8.1f}  {std:6.1f}")
    print()

    print(f"Running Monte Carlo ({N_SIMS} sims)...")
    level_res, gear_res, combat_res, req_rej, req_tot = simulate(
        monsters, thresholds, item_cache, affix_cache,
        items_per_floor, clear_lo, clear_hi)

    pct_rej = 100.0 * req_rej / req_tot if req_tot else 0.0
    print(f"  Item requirement rejections: {req_rej}/{req_tot} ({pct_rej:.1f}%)")
    print()

    # --- Char level table ---
    print("Char level at floor entry (simulated):")
    hdr2 = f"  {'d':>2}  {'mean':>6}  {'std':>5}  {'min':>4}  {'max':>4}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for d in range(1, MAX_FLOOR + 1):
        vals = level_res[d]
        m, s = _mean(vals), _std(vals)
        print(f"  {d:2d}  {m:6.2f}  {s:5.2f}  {min(vals):4d}  {max(vals):4d}")
    print()

    # --- Gear stat table ---
    COL  = 15
    GKEYS = ('str', 'dex', 'mag', 'vit')
    GLBLS = ('STR', 'DEX', 'MAG', 'VIT')
    print("Gear stat bonuses at floor entry:")
    hdr3 = f"  {'d':>2}  " + "  ".join(f"{lbl:^{COL}}" for lbl in GLBLS)
    print(hdr3)
    print("-" * len(hdr3))
    for d in range(1, MAX_FLOOR + 1):
        cols = [_fmt(gear_res[d][k]) for k in GKEYS]
        print(f"  {d:2d}  {'  '.join(cols)}")
    print()

    print("Suggested gear bonus to add on top of level-up attrs (mean values):")
    for k, lbl in zip(GKEYS, GLBLS):
        vals_by_d = [round(_mean(gear_res[d][k])) for d in range(1, MAX_FLOOR + 1)]
        print(f"  {lbl}: {','.join(str(v) for v in vals_by_d)}")
    print()

    # --- Combat stats table ---
    CSTATS = [
        ("weapon_min",    "wpn_min_dam"),
        ("weapon_max",    "wpn_max_dam"),
        ("armor_ac",      "total_ac"),
        ("to_hit_pct",    "to_hit_%"),
        ("dam_bonus_pct", "dam_bonus_%"),
        ("resist",        "resistance"),
        ("atk_tier",      "atk_speed"),
        ("rec_tier",      "rec_speed"),
    ]
    print("Combat stats from gear at floor entry:")
    hdr4 = f"  {'d':>2}  " + "  ".join(f"{lbl:^{COL}}" for _, lbl in CSTATS)
    print(hdr4)
    print("-" * len(hdr4))
    for d in range(1, MAX_FLOOR + 1):
        cols = [_fmt(combat_res[d][k]) for k, _ in CSTATS]
        print(f"  {d:2d}  {'  '.join(cols)}")
    print()

    print("vs. legacy linear estimation (scale=0.7):")
    scale = 0.7
    hdr5 = f"  {'d':>3}  {'total_ac':>10}  {'wpn_min':>10}  {'wpn_max':>10}  {'to_hit_%':>10}  {'resistance':>10}"
    print(hdr5)
    print("  " + "-" * (len(hdr5) - 2))
    for d in range(1, MAX_FLOOR + 1):
        ac    = round((5 + 6.5 * d) * scale)
        dmin  = round(((1 + 0.10*d*d) + (2 + 0.18*d*d)) / 2 * scale)
        dmax  = round(((2 + 0.28*d*d) + (4 + 0.42*d*d)) / 2 * scale)
        tohit = round((10 + 3 * d) * scale)
        res   = max(0, (d - 4) * 8)
        print(f"  {d:3d}  {ac:>10}  {dmin:>10}  {dmax:>10}  {tohit:>10}  {res:>10}")
    print()

    # --- Config strings ---
    entries = []
    for d in range(1, MAX_FLOOR + 1):
        vals = level_res[d]
        m, s = _mean(vals), _std(vals)
        lo = max(1, round(m - s))
        hi = max(lo, round(m + s))
        entries.append(f"{lo}:{hi}")
    print("# Char level up table: 16 lo:hi pairs, one per dungeon floor 1-16.")
    print("#   Char level is drawn ri(lo, hi) each episode. Controls hero power scaling with depth.")
    print("Char level up table = " + ", ".join(entries))
    print()

    entries = []
    for d in range(1, MAX_FLOOR + 1):
        fields = [_r(gear_res[d][k]) for k in GKEYS]
        entries.append("/".join(fields))
    print("# Char gear stats: 16 entries, one per floor. Each: str_lo:str_hi/dex_lo:dex_hi/mag_lo:mag_hi/vit_lo:vit_hi")
    print("#   Gear stat bonus drawn ri(lo, hi) added on top of base level-up stats before HP is calculated.")
    print("#   Empty = no gear stat bonuses.")
    print("Char gear stats    = " + ", ".join(entries))
    print()

    entries = []
    for d in range(1, MAX_FLOOR + 1):
        dmin_m = round(_mean(combat_res[d]['weapon_min']))
        dmax_m = round(_mean(combat_res[d]['weapon_max']))
        ac_scale = _surv_scale(d, surv_max, surv_tau)
        fields = [
            f"{dmin_m}:{dmax_m}",
            _r_scaled(combat_res[d]['armor_ac'], scale=ac_scale),
            _r(combat_res[d]['to_hit_pct']),
            _r(combat_res[d]['dam_bonus_pct']),
            _r(combat_res[d]['resist']),
            str(round(_mean(combat_res[d]['atk_tier']))),
            str(round(_mean(combat_res[d]['rec_tier']))),
        ]
        entries.append("/".join(fields))
    print("# Char gear combat: 16 entries, one per floor. Each field:")
    print("#   dmin:dmax           - weapon damage range (absolute values)")
    print("#   ac_lo:ac_hi         - armor class drawn ri(lo, hi)")
    print("#   hit_lo:hit_hi       - to-hit bonus % drawn ri(lo, hi)")
    print("#   bdam_lo:bdam_hi     - bonus damage % (_pIBonusDam) drawn ri(lo, hi)")
    print("#   resist_lo:resist_hi - all resistances drawn ri(lo, hi), capped at 75")
    print("#   atk                 - attack speed tier (0=none 1=Quick 2=Fast 3=Faster)")
    print("#   rec                 - recovery speed tier (0=none 1=Fast 2=Faster 3=Fastest)")
    print("#   Empty = built-in depth-scaling formulas.")
    print("Char gear combat   = " + ", ".join(entries))


if __name__ == "__main__":
    main()
