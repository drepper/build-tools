#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
import math
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

assert len(TRAILING_FRACTION) == len(INITIAL_FRACTION)
assert len(TRAILING_FRACTION) == len(SMALL_FRACTION)
NFRAC = len(TRAILING_FRACTION)

COLOR_EMPH = '\x1b[38;5;226m'
COLOR_BG = '\x1b[48;5;234m'
COLOR_OFF = '\x1b[0m'

COLUMNS, _ = os.get_terminal_size()


def run(argv: List[str]) -> Tuple[float, float, int, List[Tuple[float,float,str]]]:
    tf = tempfile.NamedTemporaryFile("w+")

    path = None
    builddir = []
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith('-C'):
            if a == '-C':
                if i + 2 < len(sys.argv):
                    path = sys.argv[i + 2]
            else:
                path = sys.argv[i + 1][2:]
            break
    if not path:
        path = pathlib.Path('build')
        if path:
            builddir = ['-C', 'build']
        else:
            path = pathlib.Path('')

    if (path / pathlib.Path("Makefile")).exists():
        generator = os.getenv("MAKE") or 'make'
    elif (path / pathlib.Path("build.ninja")).exists():
        generator = os.getenv("NINJA") or 'ninja'
    else:
        print(f'cannot determine generator for building in {path}')
        sys.exit(1)

    r = subprocess.call(["env", f'MAKE_TIMING_OUTPUT={tf.name}', generator] + builddir + argv)
    if r != 0:
        sys.exit(r)

    meas = [(float(l[0]), float(l[1]), l[2]) for l in [l.split() for l in tf.readlines()]]
    if not meas:
        sys.exit(0)

    start = min(meas, key=lambda e: e[0])[0]
    end = max(meas, key=lambda e: e[1])[1]
    labelmaxlen = min(len(max(meas, key=lambda e: len(e[2]))[2]), COLUMNS // 3)

    return start, end - start, labelmaxlen, sorted(meas, key=lambda e: e[0])


def map_to_step(t: float, start: float, stepsize: float) -> int:
    return round((t - start) / stepsize)


def to_ns(t: float) -> int:
    return round(1_000_000_000 * t)


def getcoord(m, start: float, stepsize: float) -> Tuple[int, int, str, int]:
    return map_to_step(m[0], start, stepsize), map_to_step(m[1], start, stepsize), m[2], to_ns(m[1] - m[0])


def compute_utilization(coords, nsteps: int) -> Tuple[int, float, int]:
    busy = [False] * nsteps
    efficient = [0] * nsteps
    ts = []

    for c in coords:
        efficient[c[0]:c[1]] += map(int, busy[c[0]:c[1]])
        busy[c[0]:c[1]] = [True] * (c[1] - c[0])
        ts.append(c[1] - c[0])

    sorted(ts)

    return sum(busy), sum(efficient) / nsteps, ts[len(ts) // 2]


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
    d = 10**(mor * 3)
    tf = t / d
    frac = 2 - (int(m) - mor * 3)
    return f'{s}{tf:.{frac}f}{["n","µ","m",""][mor]}s'


def bar(from_t: int, to_t: int, total_t:int, labelwidth: int, fg: str, bg: str) -> str:
    tfmt = fmttime(total_t).strip()

    pos_leadfrac = from_t // NFRAC
    pos_tailfrac = to_t // NFRAC

    leadfrac = ((pos_leadfrac + 1) * NFRAC - from_t) % NFRAC
    nleadfrac = 1 if leadfrac > 0 else 0

    nfull = max(0, (to_t - (from_t + leadfrac)) // NFRAC)

    tailfrac = to_t % NFRAC
    ntailfrac = 1 if pos_leadfrac != pos_tailfrac and tailfrac > 0 else 0

    color_off = '\x1b[0m'

    if len(tfmt) <= nfull:
        res = f'{"":{pos_leadfrac}}'
        res += fg
        res += INITIAL_FRACTION[leadfrac]
        res += f'\x1b[7m{tfmt:^{nfull}}\x1b[27m'
        res += TRAILING_FRACTION[tailfrac]
        res += color_off + bg
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
            res += '█' * nfull
            res += TRAILING_FRACTION[tailfrac]
        res += color_off + bg
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
            res += '█' * nfull
            res += TRAILING_FRACTION[tailfrac]
        res += color_off + bg
        res += f' {tfmt}'
    return res


def percent(v: float) -> int:
    return int(v * 100 + 0.5)


def main(argv: List[str]) -> None:
    start, duration, labelwidth, meas = run(sys.argv[1:])

    barwidth = COLUMNS - 1 - labelwidth

    nsteps = NFRAC * barwidth

    stepsize = duration / nsteps

    coords = list(map(lambda m: getcoord(m, start, stepsize), meas))

    tbusy, efficiency, median = compute_utilization(coords, nsteps)

    title = " Build Report "
    nfront = (COLUMNS - len(title)) // 2
    nback = COLUMNS - len(title) - nfront
    print(f'{"🮁"*nfront}{COLOR_EMPH}{title}{COLOR_OFF}{"🮁"*nback}\n')

    for i, c in enumerate(coords):
        bg = COLOR_BG if i % 2 else ''
        if c[1] - c[0] <= median:
            fg = f'\x1b[38;2;{int(255 * (c[1] - c[0]) / median)};255;0m'
        else:
            fg = f'\x1b[38;2;255;{int(255 * (1 - (c[1] - c[0] - median) / median))};0m'
        print(f'{bg}{c[2][-labelwidth:]:>{labelwidth}} {bar(c[0], c[1], c[3], labelwidth, fg, bg)}\x1b[0K\x1b[0m')

    totalfmt = fmttime(to_ns(duration))
    nlinefront = (barwidth - (len(totalfmt) + 2) - 2) // 2
    nlineback = barwidth - (len(totalfmt) + 2) - 2 - nlinefront
    print(f'{" "*labelwidth} ◀{"─"*nlinefront} {COLOR_EMPH}{totalfmt}{COLOR_OFF} {"─"*nlineback}▶')

    print(f'{"overhead":>{labelwidth}} {COLOR_EMPH}{percent(1 - tbusy / nsteps)}%{COLOR_OFF}')
    print(f'{"efficiency":>{labelwidth}} {COLOR_EMPH}{percent(efficiency)}%{COLOR_OFF}')


if __name__ == '__main__':
    main(sys.argv)
