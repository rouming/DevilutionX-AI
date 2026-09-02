#!/usr/bin/env python3
import re
import sys
import math
import statistics
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    print("usage: agent-log-analysis.py <log>")
    sys.exit(1)

raw = open(sys.argv[1], 'rb').read().replace(b'\x00', b'').decode('utf-8', errors='replace')

# --- death level distribution (collected, printed last) ---

levels = [int(m) for m in re.findall(r"hero died at level (\d+)", raw)]
_death_lines = []
if levels:
    n = len(levels)
    _death_lines.append("runs=%d  min=%d  max=%d  mean=%.2f  std=%.2f" % (
        n, min(levels), max(levels),
        statistics.mean(levels), statistics.stdev(levels) if n > 1 else 0.0))
    _death_lines.append("")
    for lvl, cnt in sorted(Counter(levels).items()):
        bar = "#" * cnt
        _death_lines.append("  level %2d: %3d (%5.1f%%)  %s" % (lvl, cnt, 100 * cnt / n, bar))
    _death_lines.append("")
    _death_lines.append("survival function (fraction of runs that reached at least level N):")
    for lvl in range(1, max(levels) + 1):
        survived = sum(1 for l in levels if l >= lvl)
        _death_lines.append("  level %2d: %5.1f%%" % (lvl, 100 * survived / n))
else:
    _death_lines.append("no 'hero died at level N' lines found")

# --- per-floor hero gear stats ---
# Parsed from: "now on level N, monsters M (initial M), hero stats: slots=S str=BASE(+BONUS) ..."
# str/dex/mag/vit capture the gear bonus (delta). New format: BASE(+BONUS); old format: BONUS.
# Regex handles both: (?:\d+\()? skips the base in new format, captures bonus in both.

STAT_RE = re.compile(
    r'now on level (\d+), monsters \d+ \(initial \d+\), hero stats: '
    r'slots=(\d+) str=(?:\d+\()?([+-]?\d+)\)? dex=(?:\d+\()?([+-]?\d+)\)? '
    r'mag=(?:\d+\()?([+-]?\d+)\)? vit=(?:\d+\()?([+-]?\d+)\)? '
    r'ac=(-?\d+) bac=(-?\d+) dmin=(-?\d+) dmax=(-?\d+) '
    r'bdam=(-?\d+) hit=(-?\d+) '
    r'fr=(-?\d+) lr=(-?\d+) mr=(-?\d+) '
    r'atk=(\d+) rec=(\d+)'
)

SEED_RE = re.compile(r'^\s+seed \d+\s*$')
LEVELUP_RE = re.compile(r'agent \d+: level up (\d+)')

by_floor = defaultdict(list)
clvl_by_floor = defaultdict(list)
current_clvl = 1
for line in raw.splitlines():
    if SEED_RE.match(line):
        current_clvl = 1
        continue
    m_lu = LEVELUP_RE.search(line)
    if m_lu:
        current_clvl = int(m_lu.group(1))
        continue
    m = STAT_RE.search(line)
    if m:
        vals = [int(x) for x in m.groups()]
        lvl, slots, str_, dex, mag, vit, ac, bac, dmin, dmax, bdam, hit, fr, lr, mr, atk, rec = vals
        clvl_by_floor[lvl].append(current_clvl)
        by_floor[lvl].append({
            'slots': slots,
            'str': str_, 'dex': dex, 'mag': mag, 'vit': vit,
            'total_ac': ac + bac,
            'dmin': dmin, 'dmax': dmax,
            'bdam': bdam, 'hit': hit,
            'resist': max(fr, lr, mr),
            'atk': atk, 'rec': rec,
        })

if not by_floor:
    print("no 'now on level N ... hero stats' lines found")
    print()
    print("\n".join(_death_lines))
    sys.exit(0)

MAX_FLOOR = 16
# floors with enough samples to trust for fitting
MIN_N_FIT = 5

def _fmt(mean, std):
    return "%5.2f +- %5.2f" % (mean, std)

def _stats(rows, key):
    vals = [r[key] for r in rows]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std

def _wlsq(xs, ys, ws):
    """Weighted least squares fit: y = a*x + b."""
    W = sum(ws)
    Wx = sum(w * x for w, x in zip(ws, xs))
    Wy = sum(w * y for w, y in zip(ws, ys))
    Wxx = sum(w * x * x for w, x in zip(ws, xs))
    Wxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = W * Wxx - Wx * Wx
    if abs(denom) < 1e-10:
        return 0.0, Wy / W
    a = (W * Wxy - Wx * Wy) / denom
    b = (Wy - a * Wx) / W
    return a, b

def _build_extrapolator(key, clamp_min=None):
    """Fit y = a*sqrt(d) + b on floors with n >= MIN_N_FIT for both mean and std."""
    xs, means, stds, ws = [], [], [], []
    for d, rows in by_floor.items():
        if len(rows) < MIN_N_FIT:
            continue
        xs.append(math.sqrt(d))
        means.append(statistics.mean(r[key] for r in rows))
        stds.append(statistics.stdev(r[key] for r in rows) if len(rows) > 1 else 0.0)
        ws.append(len(rows))
    if len(xs) < 2:
        return None
    a_m, b_m = _wlsq(xs, means, ws)
    a_s, b_s = _wlsq(xs, stds,  ws)
    def predict(d):
        sqd = math.sqrt(d)
        m = a_m * sqd + b_m
        s = a_s * sqd + b_s
        if clamp_min is not None:
            m = max(clamp_min, m)
            s = max(0.0, s)
        return m, s
    return predict

# pre-build extrapolators for all stat keys
_extrap = {
    key: _build_extrapolator(key, clamp_min=0.0)
    for key in ('slots', 'str', 'dex', 'mag', 'vit',
                'total_ac', 'dmin', 'dmax', 'bdam', 'hit',
                'resist', 'atk', 'rec')
}

def _build_clvl_extrapolator():
    xs, means, stds, ws = [], [], [], []
    for d, cvlvls in clvl_by_floor.items():
        if len(cvlvls) < MIN_N_FIT:
            continue
        xs.append(math.sqrt(d))
        means.append(statistics.mean(cvlvls))
        stds.append(statistics.stdev(cvlvls) if len(cvlvls) > 1 else 0.0)
        ws.append(len(cvlvls))
    if len(xs) < 2:
        return lambda d: (1.0, 0.0)
    a_m, b_m = _wlsq(xs, means, ws)
    a_s, b_s = _wlsq(xs, stds, ws)
    def predict(d):
        sqd = math.sqrt(d)
        return max(1.0, a_m * sqd + b_m), max(0.0, a_s * sqd + b_s)
    return predict

_extrap_clvl = _build_clvl_extrapolator()

def _floor_rows(d):
    """Return actual rows for floor d if n >= MIN_N_FIT, else None."""
    rows = by_floor.get(d)
    if rows and len(rows) >= MIN_N_FIT:
        return rows
    return None

# Attribute table
print("Gear stat bonuses at floor entry (actual; ~ extrapolated for n < %d):" % MIN_N_FIT)
print()
hdr = "  d      n      slots         STR              DEX              MAG              VIT"
print(hdr)
print("-" * len(hdr))
for d in range(1, MAX_FLOOR + 1):
    rows = _floor_rows(d)
    if rows:
        n = len(rows)
        slots_m, slots_s = _stats(rows, 'slots')
        str_m,   str_s   = _stats(rows, 'str')
        dex_m,   dex_s   = _stats(rows, 'dex')
        mag_m,   mag_s   = _stats(rows, 'mag')
        vit_m,   vit_s   = _stats(rows, 'vit')
        print("  %2d  %5d  %s  %s  %s  %s  %s" % (
            d, n, _fmt(slots_m, slots_s),
            _fmt(str_m, str_s), _fmt(dex_m, dex_s),
            _fmt(mag_m, mag_s), _fmt(vit_m, vit_s)))
    else:
        p = {k: (_extrap[k](d) if _extrap[k] else (0.0, 0.0))
             for k in ('slots', 'str', 'dex', 'mag', 'vit')}
        print("~ %2d         %s  %s  %s  %s  %s" % (
            d, _fmt(*p['slots']), _fmt(*p['str']), _fmt(*p['dex']),
            _fmt(*p['mag']), _fmt(*p['vit'])))

print()

# Combat stats table
print("Combat stats from gear at floor entry (actual; ~ extrapolated for n < %d):" % MIN_N_FIT)
print()
hdr2 = ("  d      n    wpn_min_dam    wpn_max_dam     total_ac      to_hit_%     dam_bonus_%"
        "    resistance     atk_speed     rec_speed")
print(hdr2)
print("-" * len(hdr2))
for d in range(1, MAX_FLOOR + 1):
    rows = _floor_rows(d)
    if rows:
        n = len(rows)
        dmin_m, dmin_s = _stats(rows, 'dmin')
        dmax_m, dmax_s = _stats(rows, 'dmax')
        ac_m,   ac_s   = _stats(rows, 'total_ac')
        hit_m,  hit_s  = _stats(rows, 'hit')
        bdam_m, bdam_s = _stats(rows, 'bdam')
        res_m,  res_s  = _stats(rows, 'resist')
        atk_m,  atk_s = _stats(rows, 'atk')
        rec_m,  rec_s = _stats(rows, 'rec')
        print("  %2d  %5d  %s  %s  %s  %s  %s  %s  %s  %s" % (
            d, n,
            _fmt(dmin_m, dmin_s), _fmt(dmax_m, dmax_s),
            _fmt(ac_m, ac_s), _fmt(hit_m, hit_s),
            _fmt(bdam_m, bdam_s), _fmt(res_m, res_s),
            _fmt(atk_m, atk_s), _fmt(rec_m, rec_s)))
    else:
        p = {k: (_extrap[k](d) if _extrap[k] else (0.0, 0.0))
             for k in ('dmin', 'dmax', 'total_ac', 'hit', 'bdam', 'resist', 'atk', 'rec')}
        print("~ %2d         %s  %s  %s  %s  %s  %s  %s  %s" % (
            d, _fmt(*p['dmin']), _fmt(*p['dmax']), _fmt(*p['total_ac']),
            _fmt(*p['hit']), _fmt(*p['bdam']), _fmt(*p['resist']),
            _fmt(*p['atk']), _fmt(*p['rec'])))

# --- INI copy-paste output ---

def _get(key, d):
    rows = _floor_rows(d)
    if rows:
        return _stats(rows, key)
    fn = _extrap.get(key)
    return fn(d) if fn else (0.0, 0.0)

def _get_clvl(d):
    cvlvls = clvl_by_floor.get(d, [])
    if len(cvlvls) >= MIN_N_FIT:
        m = statistics.mean(cvlvls)
        s = statistics.stdev(cvlvls) if len(cvlvls) > 1 else 0.0
        return m, s
    return _extrap_clvl(d)

def _r(mean, std, clamp=0):
    lo = int(max(clamp, math.floor(mean - std)))
    hi = int(math.ceil(mean + std))
    return "%d:%d" % (lo, hi)

print()

entries = []
for d in range(1, MAX_FLOOR + 1):
    m, s = _get_clvl(d)
    lo = max(1, round(m - s))
    hi = max(lo, round(m + s))
    entries.append("%d:%d" % (lo, hi))
print("# Char level up table: 16 lo:hi pairs, one per dungeon floor 1-16.")
print("#   Char level is drawn ri(lo, hi) each episode. Controls hero power scaling with depth.")
print("Char level up table = " + ", ".join(entries))
print()

entries = []
for d in range(1, MAX_FLOOR + 1):
    fields = [_r(*_get(k, d)) for k in ('str', 'dex', 'mag', 'vit')]
    entries.append("/".join(fields))
print("# Char gear stats: 16 entries, one per floor. Each: str_lo:str_hi/dex_lo:dex_hi/mag_lo:mag_hi/vit_lo:vit_hi")
print("#   Gear stat bonus drawn ri(lo, hi) added on top of base level-up stats before HP is calculated.")
print("#   Empty = no gear stat bonuses.")
print("Char gear stats    = " + ", ".join(entries))
print()

entries = []
for d in range(1, MAX_FLOOR + 1):
    dmin_m, _ = _get('dmin', d)
    dmax_m, _ = _get('dmax', d)
    fields = [
        "%d:%d" % (round(dmin_m), round(dmax_m)),
        _r(*_get('total_ac', d)),
        _r(*_get('hit', d)),
        _r(*_get('bdam', d)),
        _r(*_get('resist', d)),
        str(round(_get('atk', d)[0])),
        str(round(_get('rec', d)[0])),
    ]
    entries.append("/".join(fields))
print("# Char gear combat: 16 entries, one per floor. Each field:")
print("#   dmin:dmax        - weapon damage range (min_damage:max_damage, absolute values)")
print("#   ac_lo:ac_hi      - armor class drawn ri(lo, hi)")
print("#   hit_lo:hit_hi    - to-hit bonus % drawn ri(lo, hi)")
print("#   bdam_lo:bdam_hi  - bonus damage % (_pIBonusDam) drawn ri(lo, hi)")
print("#   resist_lo:resist_hi - all resistances drawn ri(lo, hi), capped at 75")
print("#   atk              - attack speed tier (0=none 1=Quick 2=Fast 3=Faster)")
print("#   rec              - recovery speed tier (0=none 1=Fast 2=Faster 3=Fastest)")
print("#   Empty = built-in depth-scaling formulas.")
print("Char gear combat   = " + ", ".join(entries))

print()
print("\n".join(_death_lines))
