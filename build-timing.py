#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
import hashlib
import math
import operator
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import List, Tuple


TRAILING_FRACTION = [
    '',
    '▏',
    '▎',
    '▍',
    '▌',
    '▋',
    '▊',
    '▉',
]
# We could use the trailing_fraction strings with reverse video.
# initial_fraction = [''] + [f'\x1b[7m{c}\x1b[27m' for c in trailing_fraction[7:0:-1]]
INITIAL_FRACTION = [
    '',
    '▕',
    '🮇',
    '🮈',
    '▐',
    '🮉',
    '🮊',
    '🮋',
]
# If the start and end position fall to the same character we would need characters with start and end
# positions in the ⅛th grid.  These do not exist.  So, we just use the characters which have a single
# ⅛th line.
SMALL_FRACTION = [
    '▕',
    '🭵',
    '🭴',
    '🭳',
    '🭲',
    '🭱',
    '🭰',
    '▏',
]
# We do not use reverse text most of the time but instead the full block.
FULL = '█'


assert len(TRAILING_FRACTION) == len(INITIAL_FRACTION)
assert len(TRAILING_FRACTION) == len(SMALL_FRACTION)
NFRAC = len(TRAILING_FRACTION)

COLOR_EMPH = '\x1b[38;5;226m'
COLOR_BG = '\x1b[48;5;234m'
COLOR_OFF = '\x1b[0m'
INVERSE = '\x1b[7m'
INVERSE_OFF = '\x1b[27m'

COLUMNS, _ = os.get_terminal_size()


def id(s: str) -> str:
    "Identity function"
    return s


def obs(s: str) -> str:
    "Obsfucate string"
    return hashlib.sha256(s.encode()).hexdigest()[:COLUMNS //8]


def find_buildfile(path: pathlib.Path, all_dirs: bool) -> List[pathlib.Path]:
    "Find build files (Makefile, build*.ninja) in current or all subdirs."
    return list(path.glob(f'{'*/' if all_dirs else ''}Makefile')) + list(path.glob(f'{'*/' if all_dirs else ''}build*.ninja'))


def determine_path(argv: List[str]) -> pathlib.Path:
    "Determine automatically which subdir (if any) to run the build process in and what generator to use"
    possible = []
    for i, a in enumerate(argv):
        # This test depends on the fact that both make and ninja use the -C argument to select an alternative build directory.
        if a.startswith('-C'):
            if a == '-C':
                if i + 1 >= len(argv):
                    break
                path = pathlib.Path(argv[i + 1])
            else:
                path = pathlib.Path(argv[i][2:])
            possible = find_buildfile(path, False)
            break
    if not possible:
        possible = find_buildfile(pathlib.Path('.'), False) or find_buildfile(pathlib.Path('.'), True)
        if not possible:
            print('*** no build directory found')
            sys.exit(1)
    if any(f for f in possible if f.name not in ('Makefile', 'build.ninja')):
        print('*** Ninja Multi-Config not supported')
        sys.exit(1)
    if len(possible) != 1:
        builddirs = set(f.parent for f in possible)
        if len(builddirs) != 1:
            print(f'*** more than one build directory found: {", ".join([str(f) for f in builddirs])}')
        else:
            print('*** both make and ninja build possible')
        sys.exit(1)

    return possible[0]


def run(argv: List[str]) -> List[Tuple[float,float,str]]:
    "Run the build process, instructing the launchers to record start and end time for each built file"
    genpath = determine_path(argv)

    if genpath.name == 'Makefile':
        generator = os.getenv("MAKE") or 'make'
    elif genpath.name == 'build.ninja':
        generator = os.getenv("NINJA") or 'ninja'
    else:
        print(f'*** cannot determine generator for building in {genpath.parent}')
        sys.exit(1)

    with tempfile.NamedTemporaryFile("w+") as tf:
        c_builddir = [] if genpath.parent == pathlib.Path('') else ['-C', str(genpath.parent)]
        r = subprocess.call(["env", f'MAKE_TIMING_OUTPUT={tf.name}', generator] + c_builddir + argv)
        if r != 0:
            sys.exit(r)

        name_encode = obs if os.getenv("MAKE_TIMING_NAMEOBS") else id
        return [(float(l[0]), float(l[1]), name_encode(l[2])) for l in [l.split() for l in tf.readlines()]]


def get_limits(meas: List[Tuple[float, float, str]]) -> Tuple[float, float, int]:
    """Determine start and end time for the recordings and the longest output file name."""
    start = min(meas, key=operator.itemgetter(0))[0]
    end = max(meas, key=operator.itemgetter(1))[1]
    labelmaxlen = min(len(max(meas, key=lambda e: len(e[2]))[2]), COLUMNS // 3)

    return start, end - start, labelmaxlen


def map_to_step(t: float, start: float, stepsize: float) -> int:
    """Compute index from time value based on precomputed start time and resolution."""
    return round((t - start) / stepsize)


def to_ns(t: float) -> int:
    """Compute integer nanosecond value from floating-point seconds."""
    return round(1_000_000_000 * t)


def getcoord(m: Tuple[float, float, str], start: float, stepsize: float) -> Tuple[int, int, str, int]:
    """Translate the data recorded by the launchers for one tool to indices for the graph and duration times."""
    return map_to_step(m[0], start, stepsize), map_to_step(m[1], start, stepsize), m[2], to_ns(m[1] - m[0])


def compute_utilization(coords: List[Tuple[int, int, str, int]], nsteps: int) -> Tuple[int, float, float]:
    """The returned tuple contains a number of values computed from the individual tool runtimes mapped to the
    graph grid:
    - the number of time steps which have any activity
    - how much parallel activity happened at each time step
    - the median time of the tool runtimes"""
    busy = [False] * nsteps
    efficient = [0] * nsteps
    ts = []

    for c in coords:
        efficient[c[0]:c[1]] += map(int, busy[c[0]:c[1]])
        busy[c[0]:c[1]] = [True] * (c[1] - c[0])
        ts.append(c[1] - c[0])

    ts.sort()
    lts = len(ts) // 2
    median = (ts[lts] + ts[~lts]) / 2

    return sum(busy), sum(efficient) / nsteps, median


def fmttime(t: int) -> str:
    """The parameter is a time duration value in nanoseconds.  Return a string with the appropriately scaled
    time value with three decimal digits of precision."""
    s = ''
    if t >= 60_000_000_000:
        s = f'{t / 60_000_000_000:.0f}m'
        if t >= 6_000_000_000_000:
            return s
        t %= 60_000_000_000
        if t < 1_000_000_000:
            return s
    m = math.log10(t)
    mor = int(m / 3)
    d = 10 ** (mor * 3)
    tf = t / d
    frac = 2 - (int(m) - mor * 3)
    return f'{s}{tf:.{frac}f}{["n","µ","m",""][mor]}s'


def get_bar_string(from_t: int, to_t: int, total_t:int, labelwidth: int, fg: str, bg: str) -> str:
    """Return a string using Unicode block elements representing a bar with the given start and
    end position.  The horizontal resolution is NFRAC (= 8) steps per characters.
    Note that for durations less than a NFRAC steps not all values can be correctly represented"""
    tfmt = fmttime(total_t).strip()

    pos_leadfrac = from_t // NFRAC
    pos_tailfrac = to_t // NFRAC

    leadfrac = ((pos_leadfrac + 1) * NFRAC - from_t) % NFRAC
    nleadfrac = 1 if leadfrac > 0 else 0

    nfull = max(0, (to_t - (from_t + leadfrac)) // NFRAC)

    tailfrac = to_t % NFRAC
    ntailfrac = 1 if pos_leadfrac != pos_tailfrac and tailfrac > 0 else 0

    if len(tfmt) <= nfull:
        res = f'{"":{pos_leadfrac}}'
        res += fg
        res += INITIAL_FRACTION[leadfrac]
        res += INVERSE + f'{tfmt:^{nfull}}' + INVERSE_OFF
        res += TRAILING_FRACTION[tailfrac]
        res += COLOR_OFF + bg
    elif labelwidth + 1 + pos_leadfrac + nleadfrac + nfull + ntailfrac + 1 + len(tfmt) > COLUMNS:
        res = f'{tfmt:>{pos_leadfrac-1}} '
        res += fg
        if pos_leadfrac == pos_tailfrac:
            if leadfrac == 0:
                res += TRAILING_FRACTION[max(1, tailfrac)]
            elif tailfrac == NFRAC - 1:
                # This means we are drawing ⅛th too long, smaller error than other possibilities.
                res += INITIAL_FRACTION[leadfrac]
            else:
                res += SMALL_FRACTION[leadfrac]
        else:
            res += INITIAL_FRACTION[leadfrac]
            res += FULL * nfull
            res += TRAILING_FRACTION[tailfrac]
        res += COLOR_OFF + bg
    else:
        res = f'{"":{pos_leadfrac}}'
        res += fg
        if pos_leadfrac == pos_tailfrac:
            if leadfrac == 0:
                res += TRAILING_FRACTION[max(1, tailfrac)]
            elif tailfrac == NFRAC - 1:
                # This means we are drawing ⅛th too long, smaller error than other possibilities.
                res += INITIAL_FRACTION[leadfrac]
            else:
                res += SMALL_FRACTION[leadfrac]
        else:
            res += INITIAL_FRACTION[leadfrac]
            res += FULL * nfull
            res += TRAILING_FRACTION[tailfrac]
        res += COLOR_OFF + bg
        res += f' {tfmt}'
    return res


def percent(v: float) -> int:
    """Map floating point value to integer percent value"""
    return int(v * 100 + 0.5)


def main(argv: List[str]) -> None:
    meas = run(argv)
    if not meas:
        sys.exit(0)

    start, duration, labelwidth = get_limits(meas)

    barwidth = COLUMNS - 1 - labelwidth

    nsteps = NFRAC * barwidth

    stepsize = duration / nsteps

    coords = list(map(lambda m: getcoord(m, start, stepsize), meas))

    tbusy, efficiency, median = compute_utilization(coords, nsteps)

    title = f" {COLOR_EMPH}Build Report{COLOR_OFF} "
    print(f'{title:🮁^{COLUMNS + len(COLOR_EMPH) + len(COLOR_OFF)}s}')

    for i, c in enumerate(coords):
        bg = COLOR_BG if i % 2 else ''
        if c[1] - c[0] <= median:
            fg = f'\x1b[38;2;{int(255 * (c[1] - c[0]) / median)};255;0m'
        else:
            fg = f'\x1b[38;2;255;{max(0, int(255 * (1 - (c[1] - c[0] - median) / median)))};0m'
        print(f'{bg}{c[2][-labelwidth:]:>{labelwidth}} {get_bar_string(c[0], c[1], c[3], labelwidth, fg, bg)}\x1b[0K\x1b[0m')

    totalfmt = f' {COLOR_EMPH}{fmttime(to_ns(duration))}{COLOR_OFF} '
    print(f'{" "*labelwidth} ◀{totalfmt:─^{barwidth - 2 + len(COLOR_EMPH) + len(COLOR_OFF)}}▶')

    print(f'{"overhead":>{labelwidth}} {COLOR_EMPH}{percent(1 - tbusy / nsteps)}%{COLOR_OFF}')
    print(f'{"efficiency":>{labelwidth}} {COLOR_EMPH}{percent(efficiency)}%{COLOR_OFF}')


if __name__ == '__main__':
    import signal
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)
