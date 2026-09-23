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
    python diablo-sim.py --optimize [--floor SPEC] [--opt-sims N] [--opt-maxiter N] [--opt-popsize N]

Examples:
    python diablo-sim.py
    python diablo-sim.py --clear 60:80
    python diablo-sim.py --items 1=1,2-8=2,9-16=3
    python diablo-sim.py --survivor-scale 1.65:2.5   # match log data AC (survivor bias correction)
    python diablo-sim.py --optimize --floor 1-4       # optimize for Cathedral floors only

Optimizer presets (approximate wall-clock on a modern CPU):
    quick directional (~30 min) : --opt-sims 200  --opt-maxiter 20 --opt-popsize 7
    reliable    (~1.5 hours)    : --opt-sims 500  --opt-maxiter 25 --opt-popsize 8
    thorough    (~3 hours)      : --opt-sims 1000 --opt-maxiter 30 --opt-popsize 10
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

# Warrior base stats from assets/txtdata/classes/warrior/attributes.tsv
_WAR_BASE = {'str': 30, 'mag': 10, 'dex': 20, 'vit': 25}
_WAR_MAX  = {'str': 250, 'mag': 50, 'dex': 60, 'vit': 100}

# Named stat strategy presets (same spec as charLevelUpAttrs engine option).
_STAT_STRATEGY_ATTRS = {
    'dex-rush': '2-9=5d,10-*=3s2v',
    'str-vit':  '1-*=3s2v',
    'str-dump': '1-*=5s',
}

def _parse_stat_strategy(spec):
    # Source: Source/player.cpp:2739-2781 charLevelUpAttrs parser (mirrored in Python).
    # Parses "2-9=5d,10-*=3s2v" into list of (lo, hi, {str,mag,dex,vit}) ranges.
    # hi=-1 means open-ended (*).  Returns None for 'free' (handled by optimizer).
    if spec == 'free':
        return None
    spec = _STAT_STRATEGY_ATTRS.get(spec, spec)
    ranges = []
    for part in spec.split(','):
        part = part.strip()
        m = re.match(r'^(\d+)(?:-(\d+|\*))?=([0-9smdv]+)$', part)
        if not m:
            sys.exit(f"Bad stat strategy part: {part!r}")
        lo = int(m.group(1))
        hi = -1 if (m.group(2) is None or m.group(2) == '*') else int(m.group(2))
        alloc = {'str': 0, 'mag': 0, 'dex': 0, 'vit': 0}
        tail = m.group(3)
        for n, k in re.findall(r'(\d+)([smdv])', tail):
            key = {'s': 'str', 'm': 'mag', 'd': 'dex', 'v': 'vit'}[k]
            alloc[key] += int(n)
        ranges.append((lo, hi, alloc))
    return ranges

def _warrior_stats(level, alloc_ranges):
    # Simulate bonus point allocation from level 1 up to `level`.
    # Source: Source/player.cpp:2788-2806 (logic mirrored).
    s = dict(_WAR_BASE)
    for i in range(level - 1):
        lvl = i + 2
        for lo, hi, alloc in alloc_ranges:
            if lvl >= lo and (hi == -1 or lvl <= hi):
                for k in ('str', 'mag', 'dex', 'vit'):
                    s[k] = min(_WAR_MAX[k], s[k] + alloc[k])
                break
    return s

# Engine affix probability constants (items.cpp GetItemPowerPrefixAndSuffix).
P_PREFIX = 0.25 + (0.75 / 3.0 * 0.5)   # ~0.375
P_SUFFIX = 2.0 / 3.0 + (0.75 / 3.0 * 0.5)  # ~0.792

STAT_POWERS = {'STR', 'DEX', 'MAG', 'VIT'}

IGNORE_BOW = True   # warrior cannot use bows
IGNORE_2HW = True   # warrior uses 1H+shield; never equip 2H melee/staves

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


# --- scoring coefficient vector ---
# Each index corresponds to one scoring component used in _score_item_c().
# C_DEFAULT matches the old _score_item() behavior so existing output is unchanged.
# C_AGENT reflects diablo_agent.py's current hardcoded constants.
CI_STR          = 0   # score per STR affix point
CI_VIT          = 1   # score per VIT affix point
CI_DEX          = 2   # score per DEX affix point
CI_TOHIT        = 3   # score per TOHIT affix point
CI_LIFE         = 4   # score per LIFE (HP) affix point (raw HP, not *64 engine unit)
CI_GETHIT       = 5   # score per GETHIT tier (fast hit recovery)
CI_RES          = 6   # score per resistance affix point (fire/magic/light/all summed)
CI_ARMOR_AC     = 7   # multiplier on avg_ac for armor and helm slots
CI_SHIELD_AC    = 8   # multiplier on avg_ac for shield slot
CI_DAM_PCT      = 9   # score per damage-% point on jewelry (maps to agent dam_pct)

C_DEFAULT = [
    1.0,   # CI_STR    - old sim used STR+VIT only
    1.0,   # CI_VIT
    0.0,   # CI_DEX    - old sim did not use DEX
    0.0,   # CI_TOHIT  - old sim did not use TOHIT
    0.0,   # CI_LIFE   - old sim did not use LIFE
    0.0,   # CI_GETHIT - old sim did not use GETHIT
    0.0,   # CI_RES    - old sim did not use resistances
    1.0,   # CI_ARMOR_AC - old sim: avg_ac * 1.0
    1.0,   # CI_SHIELD_AC
    0.0,   # CI_DAM_PCT  - old sim did not score jewelry damage %
]

# Agent's current hardcoded constants from diablo_agent.py.
# LIFE coefficient: agent computes _iPLHP * 0.005 where engine stores HP*64, so per raw HP = 0.32.
# GETHIT coefficient: agent computes -_iPLGetHit * 1.0 (engine value is negative for beneficial
# affixes). TSV 'GETHIT' tier 1-6 maps to engine -1..-6, so per TSV tier = 1.0.
C_AGENT = [
    1.0,   # CI_STR
    1.0,   # CI_VIT
    1.0,   # CI_DEX
    0.5,   # CI_TOHIT
    0.32,  # CI_LIFE  (0.005 * 64)
    1.0,   # CI_GETHIT
    0.03,  # CI_RES   (_res_sum_score coefficient)
    0.5,   # CI_ARMOR_AC  (_ARMOR_AC_WEIGHT)
    1.0,   # CI_SHIELD_AC (no discount for shield)
    0.2,   # CI_DAM_PCT   (agent dam_pct)
]


def _apply_acp(ac, acp):
    """Apply ACP% bonus with engine integer arithmetic (Source/items.cpp GetBonusAC)."""
    b = ac * acp // 100
    if b == 0 and acp != 0 and ac > 0:
        b = 1 if acp > 0 else -1
    return ac + b


def _score_item_c(it, bonuses, slot, C, rolled_ac=0):
    """Parameterized item score used for gear selection in the optimizer."""
    # ALLRES sets _iPLFR + _iPLLR + _iPLMR all to the same value, so counts 3x.
    res_sum = (bonuses.get('FIRERES', 0) + bonuses.get('LIGHTRES', 0) +
               bonuses.get('MAGICRES', 0) + 3 * bonuses.get('ALLRES', 0))
    # TARGAC (enemy AC reduction) adds to attack rating just like TOHIT.
    bonus = (bonuses.get('STR',    0) * C[CI_STR]    +
             bonuses.get('VIT',    0) * C[CI_VIT]    +
             bonuses.get('DEX',    0) * C[CI_DEX]    +
             (bonuses.get('TOHIT', 0) + bonuses.get('TARGAC', 0)) * C[CI_TOHIT] +
             bonuses.get('LIFE',   0) * C[CI_LIFE]   +
             bonuses.get('GETHIT', 0) * C[CI_GETHIT] +
             res_sum                  * C[CI_RES])
    if slot == 'weapon':
        avg_dam = (it['min_dam'] + it['max_dam']) / 2.0
        # DAMMOD is flat damage added before the % multiplier (mirrors _identified_weapon_dmg).
        return (avg_dam + bonuses.get('DAMMOD', 0)) * (1 + bonuses.get('DAMP', 0) / 100.0) + bonus
    if slot in ('armor', 'helm'):
        return _apply_acp(rolled_ac, bonuses.get('ACP', 0)) * C[CI_ARMOR_AC] + bonus
    if slot == 'shield':
        return _apply_acp(rolled_ac, bonuses.get('ACP', 0)) * C[CI_SHIELD_AC] + bonus
    # misc (jewelry): stat, bonus, and damage-%
    return bonus + bonuses.get('DAMP', 0) * C[CI_DAM_PCT]


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
                # Source: Source/items.cpp:1082-1085 GetItemPowerPrefixAndSuffix()
                # Prefixes with PLDouble=true are added twice to the eligible list,
                # giving them 2x probability. Suffixes never have this flag.
                'double': row.get('doubleChance', '').strip().lower() == 'true',
                # PLOk: affix is non-cursed (useful=true in TSV).
                # Source: Source/items.cpp:1179  if (!onlygood && !FlipCoin(3)) onlygood = true;
                # 2/3 of drops allow only PLOk=true affixes; 1/3 allow cursed too.
                'ok':     row.get('useful', '').strip().lower() == 'true',
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
                    'mlvl':      int(row['minMonsterLevel'] or 0),
                    'min_dam':   int(row['minDamage'] or 0),
                    'max_dam':   int(row['maxDamage'] or 0),
                    'min_ac':    int(row['minArmor'] or 0),
                    'max_ac':    int(row['maxArmor'] or 0),
                    'req_str':   int(row['minStrength'] or 0),
                    'req_mag':   int(row['minMagic'] or 0),
                    'req_dex':   int(row['minDexterity'] or 0),
                    'drop_rate': int(dr),
                }
            except (ValueError, KeyError):
                continue
            if cls == 'Weapon' and it['max_dam'] > 0:
                itype_col = row.get('itemType', '')
                it['is_bow'] = (itype_col == 'Bow')
                it['is_2hw'] = (row['equipType'] == 'Two-handed' and not it['is_bow'])
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
                    'toHit':  int(row['toHit']),
                    'minDam': int(row['minDamage']),
                    'maxDam': int(row['maxDamage']),
                    'ac':     int(row['armorClass']),
                    'minHP':  int(row['hitPointsMinimum']),
                    'maxHP':  int(row['hitPointsMaximum']),
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

def _affix_pool(affixes, itype, lvl_lo, lvl_hi, is_prefix):
    # Source: Source/items.cpp:1182-1203 GetItemPowerPrefixAndSuffix()
    # Both prefix and suffix require lvl_lo <= minlvl <= lvl_hi.
    # Prefixes with PLDouble=true are appended twice (2x probability).
    result = []
    for a in affixes:
        if itype not in a['types']:
            continue
        if not (lvl_lo <= a['minlvl'] <= lvl_hi):
            continue
        result.append(a)
        if is_prefix and a['double']:
            result.append(a)
    return result


def _build_caches(prefixes, suffixes, slot_pools):
    """Precompute per-(slot,floor) item pools and per-(itype,lvl_lo,lvl_hi) affix pools."""
    item_cache  = {}
    affix_cache = {}
    for slot, itype, _, _ in _SLOT_DEFS:
        for floor in range(1, MAX_FLOOR + 1):
            pool = [it for it in slot_pools[slot] if it['mlvl'] <= floor]
            weights = [it['drop_rate'] for it in pool]
            item_cache[(slot, floor)] = (pool, weights)
            lvl_lo = floor
            lvl_hi = 2 * floor
            key = (itype, lvl_lo, lvl_hi)
            if key not in affix_cache:
                pre_all  = _affix_pool(prefixes, itype, lvl_lo, lvl_hi, is_prefix=True)
                suf_all  = _affix_pool(suffixes, itype, lvl_lo, lvl_hi, is_prefix=False)
                affix_cache[key] = {
                    'pre':      pre_all,
                    'pre_good': [a for a in pre_all if a['ok']],
                    'suf':      suf_all,
                    'suf_good': [a for a in suf_all if a['ok']],
                }
    return item_cache, affix_cache


# Source: Source/items.cpp:670-698 CalculateToHitBonus()
# Maps TOHIT_DAMP prefix param1 (v1) to the (lo, hi) range for the TOHIT roll.
_TOHIT_DAMP_TOHIT = {
    20:  (1,   5),
    36:  (6,   10),
    51:  (11,  15),
    66:  (16,  20),
    81:  (21,  30),
    96:  (31,  40),
    111: (41,  50),
    126: (51,  75),
    151: (76,  100),
}


def _p_magic(floor, always_magic):
    # Source: Source/items.cpp:1501-1504 GetItemBLevel()
    # P(magic) = 1 - P(first rnd > 10) * P(second rnd > floor)
    # Rings, amulets, staves are always magic (IMISC_RING/AMULET path).
    if always_magic:
        return 1.0
    return 1.0 - (89.0 / 100.0) * ((100.0 - 2 * floor) / 100.0)


def _roll_affixes(rng, affix_cache, itype, lvl_lo, lvl_hi, floor):
    # Source: Source/items.cpp:1169-1179 GetItemPowerPrefixAndSuffix()
    # Rings and amulets are always magic; weapons/armor/shields/helms are not.
    always_magic = itype == 'Misc'
    if rng.random() >= _p_magic(floor, always_magic):
        return defaultdict(int)   # plain item - no affixes

    # Source: Source/items.cpp:1179  if (!onlygood && !FlipCoin(3)) onlygood = true;
    # FlipCoin(3) = 1/3 chance true, so 2/3 of drops restrict to non-cursed affixes only.
    only_good = rng.randrange(3) != 0   # 2/3 chance
    pools = affix_cache[(itype, lvl_lo, lvl_hi)]
    pre_pool = pools['pre_good'] if only_good else pools['pre']
    suf_pool = pools['suf_good'] if only_good else pools['suf']

    bonuses = defaultdict(int)
    for pool, prob in ((pre_pool, P_PREFIX), (suf_pool, P_SUFFIX)):
        if pool and rng.random() < prob:
            a = rng.choice(pool)   # pre_pool has doubles duplicated for prefixes
            r = rng.randint(a['v1'], a['v2'])
            if a['power'] == 'TOHIT_DAMP':
                # Source: Source/items.cpp:727-731 IPL_TOHIT_DAMP
                # _iPLDam += roll; _iPLToHit += CalculateToHitBonus(param1)
                bonuses['DAMP'] += r
                bonuses['TOHIT'] += rng.randint(*_TOHIT_DAMP_TOHIT[a['v1']])
            elif a['power'] == 'ATTRIBS':
                # Source: Source/items.cpp:814-818 IPL_ATTRIBS
                # _iPLStr += r; _iPLMag += r; _iPLDex += r; _iPLVit += r
                bonuses['STR'] += r
                bonuses['DEX'] += r
                bonuses['VIT'] += r
            else:
                bonuses[a['power']] += r
    return bonuses


def _meets_req(it, eff):
    return (eff['str'] >= it['req_str'] and
            eff['mag'] >= it['req_mag'] and
            eff['dex'] >= it['req_dex'])


def _score_item(it, bonuses, score_by):
    """Legacy score - kept so C_DEFAULT produces identical output to the old sim."""
    return _score_item_c(it, bonuses,
                         'weapon' if score_by == 'damage' else
                         ('armor'  if score_by == 'ac'     else 'misc'),
                         C_DEFAULT)


def _empty_slot():
    return {'score': -1.0, 'stats': {}, 'ac': 0, 'life': 0,
            'min_dam': 0, 'max_dam': 0, 'tohit': 0,
            'dam_pct': 0, 'dam_mod': 0, 'resist': 0, 'atk_tier': 0, 'rec_tier': 0}


def _gear_stat_totals(slots):
    totals = {'str': 0, 'dex': 0, 'mag': 0, 'vit': 0, 'life': 0}
    for s in slots.values():
        for k in ('str', 'dex', 'mag', 'vit'):
            totals[k] += s['stats'].get(k, 0)
        totals['life'] += s.get('life', 0)
    return totals


# --- engine-mirrored combat formulas ---
# Each function cites the source file and line range it mirrors.

# Warrior class attributes from assets/txtdata/classes/warrior/attributes.tsv
_WAR_BASE_MELEE_TOHIT = 70   # baseMeleeToHit
_WAR_BLOCK_BONUS      = 30   # blockBonus


def _floor_min_hit(floor):
    # Source: Source/monster.cpp:1113-1123 GetMinHit()
    if floor == 16: return 30
    if floor == 15: return 25
    if floor == 14: return 20
    return 15


def _warrior_player_ac(gear_ac, gear_bonus_ac, dex):
    # Source: Source/player.h:567-570 Player::GetArmor()
    #   return _pIBonusAC + _pIAC + _pDexterity / 5
    return gear_bonus_ac + gear_ac + dex // 5


def _warrior_melee_tohit_pct(clvl, dex, item_tohit, item_en_ac, monster_ac):
    # Source: Source/player.h:575-578 Player::GetMeleePiercingToHit()
    #         Source/player.cpp:542   PlrHitMonst() - hper clamp [5,95]
    #   hper = clvl + DEX/2 + item_tohit + item_en_ac + baseMeleeToHit - monster_ac
    hper = clvl + dex // 2 + item_tohit + item_en_ac + _WAR_BASE_MELEE_TOHIT - monster_ac
    return max(5, min(95, hper))


def _monster_tohit_player_pct(monster_toHit, monster_mlvl, player_clvl, player_ac, floor):
    # Source: Source/monster.cpp:1143-1148 MonsterAttackPlayer()
    #   hit = monster_toHit + 2*(mlvl - clvl) + 30 - player_ac
    #   hit = max(hit, GetMinHit())
    hit = monster_toHit + 2 * (monster_mlvl - player_clvl) + 30 - player_ac
    return max(_floor_min_hit(floor), min(95, hit))


def _warrior_block_pct(dex, clvl, monster_mlvl, has_shield):
    # Source: Source/player.h:621-628  Player::GetBlockChance()
    #         Source/monster.cpp:1155-1160 MonsterAttackPlayer() block check
    #   blk = DEX + blockBonus + clvl*2 - mlvl*2;  clamp [0,100]
    #   Block only applies when _pBlockFlag is set (shield in off-hand).
    if not has_shield:
        return 0
    blk = dex + _WAR_BLOCK_BONUS + clvl * 2 - monster_mlvl * 2
    return max(0, min(100, blk))


def _warrior_hp(clvl, total_vit, base_vit, life_bonus=0):
    # Source: assets/txtdata/classes/warrior/attributes.tsv
    #   adjLife=18  lvlLife=2  chrLife=2  itmLife=2
    # HP = start(70) + (clvl-1)*lvlLife + (base_vit-WAR_BASE_VIT)*chrLife + gear_vit*itmLife
    #      + life_bonus  (from LIFE affix, direct raw-HP addition)
    # base_vit is the VIT accumulated by the stat strategy (from _warrior_stats).
    # VIT from strategy contributes chrLife(2) HP per point above the starting 25.
    # Gear VIT (eff_vit - base_vit) contributes itmLife(2) HP per point.
    vit_from_alloc = base_vit - _WAR_BASE['vit']
    gear_vit       = max(0, total_vit - base_vit)
    return 70 + (clvl - 1) * 2 + vit_from_alloc * 2 + gear_vit * 2 + life_bonus


def _warrior_expected_damage(clvl, eff_str, wpn_min, wpn_max, dam_pct, flat_bonus):
    # Source: Source/player.cpp:560-568 PlrHitMonst() damage block
    #   dam = randint(min, max)
    #   dam += dam * pIBonusDam / 100
    #   dam += pIBonusDamMod + pDamageMod          (pDamageMod = clvl*STR/100 for warrior)
    # Source: Source/items.cpp:2584   CalcPlrDamageMod() warrior default case
    #   pDamageMod = clvl * STR / 100
    # Warrior crit: GenerateRnd(100) < clvl doubles damage.
    # Source: Source/player.cpp:571-573 PlrHitMonst()
    avg_base    = (wpn_min + wpn_max) / 2.0
    avg_after   = avg_base * (1 + dam_pct / 100.0) + flat_bonus + (clvl * eff_str / 100)
    crit_factor = 1.0 + clvl / 100.0   # E[dam] = base*(1 + crit_chance) since crit doubles
    return avg_after * crit_factor


def _floor_combat_efficiency(clvl, eff, base_vit, slots, floor_monsters, floor):
    """Expected kills-per-death ratio averaged over floor monster pool.

    efficiency = warrior_hp * warrior_dps / (monster_avg_hp * monster_dps_effective)

    A higher value means the warrior can kill more monsters before dying.
    Coefficients are depth-dependent: deep floors (Hell) penalise low AC/resist
    heavily while shallow floors reward raw damage.
    """
    if not floor_monsters:
        return 0.0

    gear_ac      = sum(slots[s]['ac'] for s in ('armor', 'helm', 'shield',
                                                 'ring1', 'ring2', 'amulet'))
    gear_bonus_ac = 0   # sim does not track _pIBonusAC separately; folded into ac
    player_ac    = _warrior_player_ac(gear_ac, gear_bonus_ac, eff['dex'])
    has_shield   = slots['shield']['ac'] > 0
    wpn          = slots['weapon']
    item_tohit   = sum(slots[s]['tohit'] for s in slots)

    life_bonus = sum(slots[s].get('life', 0) for s in slots)
    warrior_hp = _warrior_hp(clvl, eff['vit'], base_vit, life_bonus)

    total = 0.0
    for m in floor_monsters:
        hit_pct   = _warrior_melee_tohit_pct(clvl, eff['dex'], item_tohit, 0, m['ac'])
        exp_dam   = _warrior_expected_damage(clvl, eff['str'],
                                             wpn['min_dam'], wpn['max_dam'],
                                             wpn['dam_pct'], wpn['dam_mod'])
        warrior_dps = exp_dam * hit_pct / 100.0

        mon_hit_pct  = _monster_tohit_player_pct(m['toHit'], m['mlevel'], clvl, player_ac, floor)
        blk_pct      = _warrior_block_pct(eff['dex'], clvl, m['mlevel'], has_shield)
        mon_exp_dam  = (m['minDam'] + m['maxDam']) / 2.0
        mon_dps_eff  = mon_exp_dam * mon_hit_pct / 100.0 * (1.0 - blk_pct / 100.0)
        if mon_dps_eff <= 0:
            mon_dps_eff = 0.01   # avoid division by zero against trivial monsters

        mon_avg_hp = (m['minHP'] + m['maxHP']) / 2.0
        if mon_avg_hp <= 0 or warrior_dps <= 0:
            total += 0.0
        else:
            total += (warrior_hp * warrior_dps) / (mon_avg_hp * mon_dps_eff)

    return total / len(floor_monsters)


# --- simulation ---

def simulate(monsters, xp_thresholds, item_cache, affix_cache,
             items_per_floor, clear_lo, clear_hi, alloc_ranges,
             seed=42, C=None, n_sims=None, free_strat=None):
    """
    Combined XP + gear Markov simulation.
    Gear state carries floor to floor. Item requirements checked against
    effective stats (warrior base + accumulated gear bonuses).
    Stat-boosting items sorted first to unlock requirement-gated items.
    Stats recorded at floor ENTRY (before new items on that floor).
    """
    if C is None:
        C = C_DEFAULT
    if n_sims is None:
        n_sims = N_SIMS
    rng = random.Random(seed)

    floor_pools = {
        d: [m for m in monsters if m['minLvl'] <= d <= m['maxLvl']]
        for d in range(1, MAX_FLOOR + 1)
    }

    level_res    = defaultdict(list)
    gear_res     = defaultdict(lambda: defaultdict(list))
    combat_res   = defaultdict(lambda: defaultdict(list))
    efficiency_res = defaultdict(list)
    req_rej = req_tot = 0

    for _sim in range(n_sims):
        total_xp = 0
        slots = {s: _empty_slot() for s, *_ in _SLOT_DEFS}
        # Warrior starting gear: Short Sword (2-6 dam) + Buckler (3 AC)
        # Short Sword avg_dam=4.0; initial score uses starting C weights.
        slots['weapon'] = dict(_empty_slot(), score=4.0, min_dam=2, max_dam=6)
        # Buckler avg_ac=3.0; use ac-slot score so replacements are apples-to-apples
        slots['shield'] = dict(_empty_slot(), score=3.0 * C[CI_SHIELD_AC], ac=3)
        # Free strategy: track base stats cumulatively across floors.
        fs_base  = dict(_WAR_BASE) if free_strat is not None else None
        fs_prev  = 1

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
            if free_strat is not None:
                # Apply previous floor's strategy for level-ups gained since last floor entry.
                prev_fl = floor - 1
                if prev_fl >= 1:
                    x_dex, x_str = free_strat[prev_fl - 1]
                    for _lu in range(player_level - fs_prev):
                        hdex = _WAR_MAX['dex'] - fs_base['dex']
                        bd   = max(0, min(int(round(x_dex)), hdex, 5))
                        bs   = max(0, min(int(round(x_str)), 5 - bd,
                                          _WAR_MAX['str'] - fs_base['str']))
                        bv   = max(0, min(5 - bd - bs, _WAR_MAX['vit'] - fs_base['vit']))
                        fs_base['dex'] += bd
                        fs_base['str'] += bs
                        fs_base['vit'] += bv
                fs_prev = player_level
                base = dict(fs_base)
            else:
                base = _warrior_stats(player_level, alloc_ranges)
            eff  = {k: base[k] + gear_bonus[k] for k in base}

            # Combat efficiency vs floor monster pool (recorded at floor ENTRY, before new gear)
            efficiency_res[floor].append(
                _floor_combat_efficiency(player_level, eff, base['vit'], slots, floor_pools[floor], floor)
            )

            # Draw items for this floor
            lvl_lo = floor
            lvl_hi = 2 * floor
            n = round(items_per_floor.get(floor, 2.0))
            candidates = []
            for _ in range(n):
                slot = _SLOT_POP[rng.randrange(len(_SLOT_POP))]
                itype, score_by = _SLOT_INFO[slot]
                pool, weights = item_cache[(slot, floor)]
                if not pool:
                    continue
                it        = rng.choices(pool, weights=weights, k=1)[0]
                bonuses   = _roll_affixes(rng, affix_cache, itype, lvl_lo, lvl_hi, floor)
                rolled_ac = rng.randint(it['min_ac'], it['max_ac']) if it['max_ac'] > 0 else 0
                if (IGNORE_BOW and it.get('is_bow')) or (IGNORE_2HW and it.get('is_2hw')):
                    continue  # consume RNG but never equip
                score     = _score_item_c(it, bonuses, slot, C, rolled_ac=rolled_ac)
                candidates.append((slot, it, bonuses, score, rolled_ac))

            # Stat-boosting items first so their bonuses may unlock requirement-gated items
            candidates.sort(
                key=lambda c: any(c[2].get(p, 0) > 0 for p in STAT_POWERS),
                reverse=True,
            )

            for slot, it, bonuses, new_score, rolled_ac in candidates:
                itype, _ = _SLOT_INFO[slot]
                req_tot += 1
                if not _meets_req(it, eff):
                    req_rej += 1
                    continue
                if new_score <= slots[slot]['score']:
                    continue
                stat_map = {p.lower(): bonuses[p] for p in STAT_POWERS if bonuses.get(p, 0) != 0}
                slots[slot] = {
                    'score':    new_score,
                    'stats':    stat_map,
                    'ac':       _apply_acp(rolled_ac, bonuses.get('ACP', 0)),
                    'life':     bonuses.get('LIFE', 0),
                    'min_dam':  it['min_dam'],
                    'max_dam':  it['max_dam'],
                    'tohit':    bonuses.get('TOHIT', 0) + bonuses.get('TARGAC', 0),
                    'dam_pct':  bonuses.get('DAMP', 0),
                    'dam_mod':  bonuses.get('DAMMOD', 0),
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

    return level_res, gear_res, combat_res, efficiency_res, req_rej, req_tot


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


# --- coefficient optimizer ---

_C_NAMES = ['str', 'vit', 'dex', 'tohit', 'life', 'gethit', 'res',
            'armor_ac', 'shield_ac', 'dam_pct']

# Bounds for differential_evolution: (lo, hi) per coefficient.
# Upper bound ~4x the agent values; lower bound 0.
_C_BOUNDS = [
    (0.0, 4.0),   # CI_STR
    (0.0, 4.0),   # CI_VIT
    (0.0, 4.0),   # CI_DEX
    (0.0, 2.0),   # CI_TOHIT
    (0.0, 1.0),   # CI_LIFE
    (0.0, 4.0),   # CI_GETHIT
    (0.0, 0.2),   # CI_RES
    (0.0, 2.0),   # CI_ARMOR_AC
    (0.0, 2.0),   # CI_SHIELD_AC
    (0.0, 1.0),   # CI_DAM_PCT
]


def _parse_floors(spec):
    """Parse floor spec: '4', '1-4', '9-16', '1-4,13-16' -> sorted list of floor ints."""
    floors = set()
    for part in spec.split(','):
        part = part.strip()
        m = re.match(r'^(\d+)-(\d+)$', part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if not (1 <= lo <= hi <= MAX_FLOOR):
                sys.exit(f"Bad --floor range: {part!r}  (floors 1-{MAX_FLOOR})")
            floors.update(range(lo, hi + 1))
        elif re.match(r'^\d+$', part):
            f = int(part)
            if not (1 <= f <= MAX_FLOOR):
                sys.exit(f"Bad --floor value: {part!r}  (floors 1-{MAX_FLOOR})")
            floors.add(f)
        else:
            sys.exit(f"Bad --floor spec: {part!r}  (expected N or N-M)")
    return sorted(floors)


def _strat_from_C(C):
    """Extract free strategy from C[10:] as list of 16 (x_dex, x_str) pairs."""
    return [(C[10 + 2 * d], C[10 + 2 * d + 1]) for d in range(MAX_FLOOR)]


def _strat_effective_allocs(strat, mean_clvls):
    """Simulate per-clvl allocation respecting WAR_MAX caps cumulatively.
    Returns list of (clvl, bd, bs, bv) for each level-up clvl 2..50.
    mean_clvls: list of 16 mean clvl values at dlvl entry (from calibration).
    """
    base = dict(_WAR_BASE)
    allocs = []
    for d in range(1, MAX_FLOOR + 1):
        lo_clvl = (mean_clvls[d - 2] + 1) if d > 1 else 2
        hi_clvl = mean_clvls[d - 1] if d < MAX_FLOOR else 50
        x_dex, x_str = strat[d - 1]
        for clvl in range(lo_clvl, hi_clvl + 1):
            bd = max(0, min(int(round(x_dex)), _WAR_MAX['dex'] - base['dex'], 5))
            bs = max(0, min(int(round(x_str)), 5 - bd, _WAR_MAX['str'] - base['str']))
            bv = max(0, min(5 - bd - bs, _WAR_MAX['vit'] - base['vit']))
            base['dex'] += bd
            base['str'] += bs
            base['vit'] += bv
            allocs.append((clvl, bd, bs, bv))
    return allocs


def _strat_to_charspec(strat, mean_clvls):
    """Convert per-dlvl strategy to charLevelUpAttrs spec string.
    mean_clvls: list of 16 mean clvl values at each dlvl entry (from calibration).
    Returns a spec like '2-9=5d,10-*=3s2v' directly usable as --stat-strategy.
    Accounts for WAR_MAX caps cumulatively, so DEX stops at 60 automatically.
    """
    allocs = _strat_effective_allocs(strat, mean_clvls)
    if not allocs:
        return ''
    groups = []
    for clvl, bd, bs, bv in allocs:
        alloc = (bd, bs, bv)
        if not groups or alloc != groups[-1][2] or clvl != groups[-1][1] + 1:
            groups.append((clvl, clvl, alloc))
        else:
            lo, _, a = groups[-1]
            groups[-1] = (lo, clvl, a)
    parts = []
    for i, (lo, hi, (bd, bs, bv)) in enumerate(groups):
        rng = f"{lo}-*" if i == len(groups) - 1 else (str(lo) if lo == hi else f"{lo}-{hi}")
        alloc_str = ''.join(f"{n}{c}" for n, c in [(bs,'s'),(bv,'v'),(bd,'d')] if n > 0) or '0s'
        parts.append(f"{rng}={alloc_str}")
    return ','.join(parts)


def optimize_coefficients(monsters, thresholds, item_cache, affix_cache,
                          items_per_floor, clear_lo, clear_hi, alloc_ranges,
                          n_opt_sims=200, maxiter=14, popsize=5, seed=42,
                          floors=None, free_strategy=False):
    """Find scoring coefficients that maximize mean combat efficiency.

    Uses scipy.optimize.differential_evolution with a fixed-seed simulate() so the
    fitness landscape is deterministic (no MC noise between evaluations).
    Default parameters target ~30 min runtime (200 sims/eval, 14 generations, pop=5*10=50).
    floors: list of floor numbers to include in fitness (default: all 1-16).
    free_strategy: also optimize per-floor stat allocation (32 extra dims).
    Returns the optimal C vector (and strategy dims appended if free_strategy).
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        sys.exit("scipy is required for --optimize. Install with: pip install scipy")

    if floors is None:
        floors = list(range(1, MAX_FLOOR + 1))

    # Calibrate clvl per dlvl entry for charspec conversion (fast, 200 sims).
    mean_clvls = None
    if free_strategy:
        _cal_alloc = _parse_stat_strategy('dex-rush')
        _cal_lr, _, _, _, _, _ = simulate(
            monsters, thresholds, item_cache, affix_cache,
            items_per_floor, clear_lo, clear_hi, _cal_alloc,
            seed=seed, n_sims=200)
        mean_clvls = [round(_mean(_cal_lr[d])) if _cal_lr[d] else d
                      for d in range(1, MAX_FLOOR + 1)]

    # Strategy dims: 2 per floor (x_dex, x_str), each in [0, 5].
    # Per level-up on floor d: bonus_dex=round(x_dex), bonus_str=round(x_str),
    # bonus_vit=5-both, all clamped inside simulate() to enforce feasibility.
    strat_bounds = [(0.0, 5.0), (0.0, 5.0)] * MAX_FLOOR if free_strategy else []
    bounds = _C_BOUNDS + strat_bounds

    ndim       = len(bounds)
    pop_size   = popsize * ndim
    total_gen  = maxiter
    total_eval = pop_size + total_gen * pop_size   # initial + generations

    best = {'eff': -1.0, 'C': None}
    gen  = [0]

    def fitness(C):
        fs = _strat_from_C(C) if free_strategy else None
        _, _, _, eff_res, _, _ = simulate(
            monsters, thresholds, item_cache, affix_cache,
            items_per_floor, clear_lo, clear_hi, alloc_ranges,
            seed=seed, C=list(C[:10]), n_sims=n_opt_sims, free_strat=fs)
        vals  = [_mean(eff_res[d]) for d in floors if eff_res[d]]
        score = _mean(vals) if vals else 0.0
        if score > best['eff']:
            best['eff'] = score
            best['C']   = list(C)
        return -score

    def callback(xk, convergence=None):
        gen[0] += 1
        coeffs = ' '.join(f'{v:.2f}' for v in best['C'][:10])
        conv   = f'{convergence:.3f}' if convergence is not None else '  n/a'
        print(f"  gen {gen[0]:3d}/{total_gen}  best_eff={best['eff']:8.2f}"
              f"  conv={conv}  [{coeffs}]", flush=True)
        if free_strategy and best['C'] is not None:
            spec = _strat_to_charspec(_strat_from_C(best['C']), mean_clvls)
            print(f"  stat-strategy: {spec}", flush=True)

    floor_desc = (f"floors {floors[0]}-{floors[-1]}" if floors == list(range(floors[0], floors[-1]+1))
                  else f"floors {floors}")
    strat_desc = " + free strategy (32 dims)" if free_strategy else ""
    print(f"Optimizing {ndim} coefficients{strat_desc}"
          f"  sims/eval={n_opt_sims}  generations={total_gen}"
          f"  pop={pop_size}  total_evals~{total_eval}  seed={seed}"
          f"  {floor_desc}")
    print(f"  {'name':<12}  {'lo':>5}  {'hi':>5}  {'agent':>7}")
    for name, (lo, hi), va in zip(_C_NAMES, _C_BOUNDS, C_AGENT):
        print(f"  {name:<12}  {lo:5.2f}  {hi:5.2f}  {va:7.3f}")
    print()

    result = differential_evolution(
        fitness, bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.7, 1.9),   # higher mutation keeps population diverse longer
        recombination=0.5,     # lower recombination slows premature convergence
        polish=True,
        disp=False,
        callback=callback,
    )
    print(f"  done  total_evals~{total_eval}  best_eff={best['eff']:.2f}")
    return best['C'], mean_clvls


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
    ap.add_argument("--optimize", action="store_true",
                    help="Run scipy differential_evolution to find optimal score coefficients "
                         "(requires scipy). Prints per-generation progress and final C vs agent.")
    ap.add_argument("--opt-sims", type=int, default=200, metavar="N",
                    help="Sims per fitness evaluation (default 200, ~30 min total)")
    ap.add_argument("--opt-maxiter", type=int, default=14, metavar="N",
                    help="Max generations for differential_evolution (default 14)")
    ap.add_argument("--opt-popsize", type=int, default=5, metavar="N",
                    help="Population size multiplier; actual pop = N * n_coefficients (default 5)")
    ap.add_argument("--floor", metavar="SPEC", default=None,
                    help="Floors to optimize against: '4', '1-4', '9-16', '1-4,13-16' "
                         "(default: all 1-16)")
    ap.add_argument("--stat-strategy", metavar="NAME_OR_SPEC", default="dex-rush",
                    help="Stat point allocation strategy per level-up "
                         "(default: dex-rush). Named presets: dex-rush, str-vit, str-dump. "
                         "Or raw spec: '2-9=5d,10-*=3s2v'")
    args = ap.parse_args()

    clear_lo, clear_hi = _parse_clear(args.clear)
    surv_max, surv_tau = _parse_survivor_scale(args.survivor_scale)
    if args.stat_strategy == 'free' and not args.optimize:
        sys.exit("--stat-strategy free requires --optimize (strategy is determined by the optimizer)")
    alloc_ranges = _parse_stat_strategy(args.stat_strategy)

    if args.items:
        items_per_floor = _parse_items(args.items)
        last = 12.0
        for fl in range(1, MAX_FLOOR + 1):
            if fl in items_per_floor:
                last = items_per_floor[fl]
            else:
                items_per_floor[fl] = last
        if not args.optimize:
            print(f"Items per floor  : {args.items}")
    else:
        items_per_floor = {d: 12 for d in range(1, MAX_FLOOR + 1)}
        if not args.optimize:
            print("Items per floor  : 12 (default, calibrated against log floor-2 data)")

    if not args.optimize:
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

    if not args.optimize:
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

    if args.optimize:
        print()
        floors        = _parse_floors(args.floor) if args.floor else None
        free_strategy = (args.stat_strategy == 'free')
        C_opt, mean_clvls = optimize_coefficients(
            monsters, thresholds, item_cache, affix_cache,
            items_per_floor, clear_lo, clear_hi, alloc_ranges,
            n_opt_sims=args.opt_sims,
            maxiter=args.opt_maxiter,
            popsize=args.opt_popsize,
            floors=floors,
            free_strategy=free_strategy)

        print()
        print("Optimal coefficients found:")
        print(f"  {'name':<14s}  {'default':>8}  {'agent':>8}  {'opt':>8}")
        print("  " + "-" * 44)
        for name, v_def, v_agt, v_opt in zip(_C_NAMES, C_DEFAULT, C_AGENT, C_opt):
            print(f"  {name:<14s}  {v_def:8.4f}  {v_agt:8.4f}  {v_opt:8.4f}")

        # Unit conversions: sim uses real-HP units; agent stores HP*64 in _iPLHP.
        # All other sim coefficients map directly to agent units.
        life_idx = _C_NAMES.index('life')
        coeff_agent = list(C_opt)
        coeff_agent[life_idx] = C_opt[life_idx] / 64.0

        print()
        print("# Paste into diablo_agent.py _SCORE_COEFF:")
        print("_SCORE_COEFF = {")
        for name, v in zip(_C_NAMES, coeff_agent):
            print(f"    '{name}':{' ' * (10 - len(name))}{v:.4f},")
        print("}")

        # Print optimal stat strategy if free mode
        if free_strategy:
            strat = _strat_from_C(C_opt)
            eff_allocs = _strat_effective_allocs(strat, mean_clvls)
            # Build per-dlvl effective allocation (first level-up of each floor after caps).
            clvl_alloc_map = {clvl: (bd, bs, bv) for clvl, bd, bs, bv in eff_allocs}
            print()
            print("Effective stat allocation per level-up per floor (after caps):")
            print(f"  {'d':>2}  {'dex':>4}  {'str':>4}  {'vit':>4}")
            print("  " + "-" * 22)
            for d in range(1, MAX_FLOOR + 1):
                lo_clvl = (mean_clvls[d - 2] + 1) if d > 1 else 2
                bd, bs, bv = clvl_alloc_map.get(lo_clvl, (0, 0, 5))
                print(f"  {d:2d}  {bd:4d}  {bs:4d}  {bv:4d}")
            print()
            td = sum(bd for _, bd, _, _ in eff_allocs)
            ts = sum(bs for _, _, bs, _ in eff_allocs)
            tv = sum(bv for _, _, _, bv in eff_allocs)
            print(f"Approximate total stat allocation across full run:")
            print(f"  dex={td:3d}  str={ts:3d}  vit={tv:3d}  (sum={td+ts+tv})")
            print()
            spec = _strat_to_charspec(strat, mean_clvls)
            print(f"# Paste as --stat-strategy or charLevelUpAttrs ini option:")
            print(f"--stat-strategy {spec}")

        # Compute efficiency under each C for comparison
        print()
        print("Mean combat efficiency by C (1000-sim verification, fixed seed):")
        print(f"  {'d':>2}  {'C_DEFAULT':>10}  {'C_AGENT':>10}  {'C_OPT':>10}")
        print("  " + "-" * 38)
        eff_sets = {}
        opt_fs      = _strat_from_C(C_opt) if free_strategy else None
        verify_alloc = _parse_stat_strategy('dex-rush') if free_strategy else alloc_ranges
        for label, Cv, fs in [('C_DEFAULT', C_DEFAULT, None),
                               ('C_AGENT',   C_AGENT,   None),
                               ('C_OPT',     C_opt,     opt_fs)]:
            _, _, _, er, _, _ = simulate(
                monsters, thresholds, item_cache, affix_cache,
                items_per_floor, clear_lo, clear_hi, verify_alloc,
                seed=42, C=list(Cv[:10]), n_sims=1000, free_strat=fs)
            eff_sets[label] = er
        for d in range(1, MAX_FLOOR + 1):
            row = f"  {d:2d}"
            for label in ('C_DEFAULT', 'C_AGENT', 'C_OPT'):
                vals = eff_sets[label][d]
                row += f"  {_mean(vals):10.2f}" if vals else f"  {'n/a':>10}"
            print(row)
        print()

    if args.optimize:
        return

    print(f"Running Monte Carlo ({N_SIMS} sims)...")
    level_res, gear_res, combat_res, efficiency_res, req_rej, req_tot = simulate(
        monsters, thresholds, item_cache, affix_cache,
        items_per_floor, clear_lo, clear_hi, alloc_ranges)

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
    print()

    # --- Combat efficiency table ---
    # efficiency = avg kills-per-death vs floor monster pool at floor ENTRY.
    # Computed with exact engine formulas; depth-dependent because monster AC,
    # damage, and HP all grow while warrior AC and block chance matter more on
    # deeper floors.
    print("Combat efficiency at floor entry (kills-per-death, higher=better):")
    hdr6 = f"  {'d':>2}  {'mean':>7}  {'std':>6}  {'min':>6}  {'max':>6}"
    print(hdr6)
    print("  " + "-" * (len(hdr6) - 2))
    for d in range(1, MAX_FLOOR + 1):
        vals = efficiency_res[d]
        if not vals:
            print(f"  {d:2d}  {'n/a':>7}")
            continue
        m, s = _mean(vals), _std(vals)
        print(f"  {d:2d}  {m:7.2f}  {s:6.2f}  {min(vals):6.2f}  {max(vals):6.2f}")
    print()


if __name__ == "__main__":
    main()
