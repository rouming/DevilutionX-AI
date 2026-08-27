#!/usr/bin/env python3
import re
import sys
import statistics
from collections import Counter

if len(sys.argv) < 2:
    print("usage: agent-log-analysis.py <log>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    content = f.read()

levels = [int(m) for m in re.findall(r"hero died at level (\d+)", content)]

if not levels:
    print("no 'hero died at level N' lines found")
    sys.exit(1)

n = len(levels)
print("runs=%d  min=%d  max=%d  mean=%.2f  std=%.2f" % (
    n, min(levels), max(levels),
    statistics.mean(levels), statistics.stdev(levels) if n > 1 else 0.0))
print()
for lvl, cnt in sorted(Counter(levels).items()):
    bar = "#" * cnt
    print("  level %2d: %3d (%5.1f%%)  %s" % (lvl, cnt, 100 * cnt / n, bar))

print()
print("survival function (fraction of runs that reached at least level N):")
all_levels = range(1, max(levels) + 1)
for lvl in all_levels:
    survived = sum(1 for l in levels if l >= lvl)
    print("  level %2d: %5.1f%%" % (lvl, 100 * survived / n))
