#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
"""Build script with invokes the generator configure for the project.  If the CMakeFiles.txt file
is appropriately modified by using the provided launcher script the script collects build times
and in the end shows a timeline for the build process."""
import enum
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


# Glyphs used for the histogram
use_braille = False
def get_graphdots():
    """Get the glyphs to use for the histogram."""
    if use_braille:
        return [
            ' ', '⡀', '⡄', '⡆', '⡇',
            '⢀', '⣀', '⣄', '⣆', '⣇',
            '⢠', '⣠', '⣤', '⣦', '⣧',
            '⢰', '⣰', '⣴', '⣶', '⣷',
            '⢸', '⣸', '⣼', '⣾', '⣿'
        ]
    return [
        ' ', '🬏', '🬓', '▌',
        '🬞', '🬭', '🬱', '🬲',
        '🬦', '🬵', '🬹', '🬺',
        '▐', '🬷', '🬻', '█'
    ]
# We are using the Sextant blocks or Braille glyphs which have two dots horizontally
NHDOTS = 2
assert NFRAC % NHDOTS == 0


COLOR_EMPH = '\x1b[38;5;226m'
COLOR_BG = '\x1b[48;5;234m'
COLOR_OFF = '\x1b[0m'
INVERSE = '\x1b[7m'
INVERSE_OFF = '\x1b[27m'

COLOR_LOWEST = (255, 255, 0)
COLOR_MEDIAN = (0, 255, 0)
COLOR_HIGHEST = (255, 0, 0)

COLUMNS, _ = os.get_terminal_size()

@enum.unique
class GENERATORS(enum.Enum):
    "The supported generators.  The list is determined by cmake."
    MAKE = enum.auto()
    NINJA = enum.auto()

GENERATOR_BINARIES = {
    GENERATORS.MAKE: os.getenv('MAKE') or 'make',
    GENERATORS.NINJA: os.getenv('NINJA') or 'ninja',
}


def idfct(s: str) -> str:
    "Identity function"
    return s


def obs(s: str) -> str:
    "Obsfucate string"
    return hashlib.sha256(s.encode()).hexdigest()[:COLUMNS //8]


def find_buildfile(path: pathlib.Path, all_dirs: bool) -> List[pathlib.Path]:
    "Find build files (Makefile, build*.ninja) in current or all subdirs."
    return list(path.glob(f'{'*/' if all_dirs else ''}Makefile')) + list(path.glob(f'{'*/' if all_dirs else ''}build.ninja'))


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
            if pathlib.Path('CMakeLists.txt').exists() and not pathlib.Path('build').exists():
                try:
                    subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True)
                except subprocess.CalledProcessError as e:
                    print(f'*** cmake failed with error code {e.returncode}')
                    sys.exit(1)
                path = pathlib.Path('build')
                possible = find_buildfile(path, False)
            else:
                print('*** no build directory found')
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
        generator = GENERATORS.MAKE
    elif genpath.name == 'build.ninja':
        generator = GENERATORS.NINJA
    else:
        print(f'*** cannot determine generator for building in {genpath.parent}')
        sys.exit(1)

    # Maybe it is useful to switch the representation of the histogram.
    if '--braille' in argv:
        global use_braille # pylint: disable=global-statement
        use_braille = True
        argv.remove('--braille')

    # Since it might not be clear to the user when the script is started whether make or ninja is used
    # it is problematic to always require parallel builds.  A build using make requires a -j parameter.
    # Passing this it ninja fails.  This loop removes the -j for builds when ninja is used.  This way
    # -j can be passed whenever parallel builds are wanted.
    if generator == GENERATORS.NINJA:
        for i, arg in enumerate(argv):
            if arg.startswith('-j'):
                del argv[i]
                break

    with tempfile.NamedTemporaryFile("w+") as tf:
        # Construct the command line.
        cmdline = ["env"]
        cmdline.append(f'MAKE_TIMING_OUTPUT={tf.name}')
        # XYZ Make this optional?
        cmdline.append('CTEST_OUTPUT_ON_FAILURE=1')
        cmdline.append(GENERATOR_BINARIES[generator])
        if genpath.parent != pathlib.Path(''):
            cmdline += ['-C', str(genpath.parent)]
        cmdline += argv

        try:
            subprocess.run(cmdline, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)

        name_encode = obs if os.getenv("MAKE_TIMING_NAMEOBS") else idfct
        res = [(float(l[0]), float(l[1]), name_encode(l[2])) for l in [l.split() for l in tf.readlines()]]
        res.sort()
        return res


def get_limits(meas: List[Tuple[float, float, str]]) -> Tuple[float, float, int]:
    """Determine start and end time for the recordings and the longest output file name."""
    start = min(meas, key=operator.itemgetter(0))[0]
    end = max(meas, key=operator.itemgetter(1))[1]
    labelmaxlen = max(min(len(max(meas, key=lambda e: len(e[2]))[2]), COLUMNS // 3), 20)

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


def compute_utilization(coords: List[Tuple[int, int, str, int]], nsteps: int) -> Tuple[int, float, float, List[int], List[int]]:
    """The returned tuple contains a number of values computed from the individual tool runtimes mapped to the
    graph grid:
    - the number of time steps which have any activity
    - how much parallel activity happened at each time step
    - the median time of the tool runtimes"""
    busy = [False] * nsteps
    efficient = [0] * nsteps
    ts = []
    remaining = [len(coords)] * nsteps

    for c in coords:
        for t in range(c[0], c[1]):
            efficient[t] += int(busy[t])
            busy[t] = True
        for t in range(c[1], nsteps):
            remaining[t] -= 1
        ts.append(c[1] - c[0])

    ts.sort()
    lts = len(ts) // 2
    median = (ts[lts] + ts[~lts]) / 2

    return sum(busy), sum(efficient) / nsteps, median, list(map(operator.add, efficient, busy)), remaining


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


def lerp_color(start: Tuple[int, int, int], end: Tuple[int, int, int], t: float) -> str:
    """Linearly interpolate between two colors."""
    assert len(start) == len(end)
    t = max(0, min(1, t))
    return ';'.join(str(int(start[i] + (end[i] - start[i]) * t)) for i in range(len(start)))


def quant_time(t: float) -> float:
    """Determine time quantization for display.  We try to find a reasonable, multiple of 5 or 10, multiple of a unit
    of time so that we have between 2 and two """
    if t >= 3_600_000_000_000:
        # No multi-hour support
        return t
    m = int(math.log10(t))
    ts1 = int(10**m)
    ts2 = 5 * int(10**(m-1))
    ideal = 4
    return (ts1 if abs(int(t / ts1) - ideal) < abs(int(t / ts2) - ideal) else ts2) / t


def scaled_y(y: int, ticks: int, nvdots: int, factor_y: float) -> float:
    """Scale the y coordinate to the count scale."""
    return ((y - 1) * nvdots + ticks) / factor_y


def get_axis_labels(nvert: int, njobs: int, factor_y: float, maxval: float) -> List[str]:
    """Create the y-axis labels"""
    graphdots = get_graphdots()
    nvdots = round(math.sqrt(len(graphdots))) - 1

    ncpus = os.cpu_count() or njobs
    maxuse_percent = 100 * maxval / ncpus
    maxuse_power10 = round(math.log(maxuse_percent, 10)) - 1
    maxuse_normal = int(maxuse_percent / (10 ** maxuse_power10))
    maxuse_step = (1 if maxuse_normal < 5 else 5 if maxuse_normal < 20 else 10) * (10 ** maxuse_power10)

    res = ['']
    for y in range(1, nvert + 1):
        if y == 1:
            label = '0%▁'
        else:
            utilization = [100 * scaled_y(y, i, nvdots, factor_y) / ncpus for i in range(nvdots + 1)]

            label = ''
            for i in range(nvdots):
                if int(utilization[i] / maxuse_step) != int(utilization[i + 1] / maxuse_step):
                    ldiff = -(utilization[i] - int(utilization[i + 1] / maxuse_step) * maxuse_step)
                    hdiff = utilization[i + 1] - int(utilization[i + 1] / maxuse_step) * maxuse_step
                    sep = ('🬭🬋🬂' if nvdots == 3 else '▂🭺🭸▔')[i + (0 if (i == nvdots - 1 or ldiff < hdiff) else 1)]
                    label=f'{int(utilization[i + 1] / maxuse_step) * maxuse_step}%{sep}'
                    break
        res.append(label)
    return res


def plot_histogram(labelwidth: int, histogram: List[int], remaining: List[int], duration: float, njobs: int, overhead: int, efficiency: int) -> None:
    """Plot a histogram of utilization."""
    factor_x = NFRAC // NHDOTS

    reduced = list(map(lambda idx: round(sum(histogram[idx * factor_x:(idx + 1) * factor_x]) / factor_x), range(len(histogram) // factor_x)))
    maxval = max(reduced)
    assert maxval <= njobs

    remaining_reduced = list(map(lambda idx: round(sum(remaining[idx * factor_x:(idx + 1) * factor_x]) / factor_x), range(len(histogram) // factor_x)))

    graphdots = get_graphdots()
    nvdots = round(math.sqrt(len(graphdots))) - 1
    assert (nvdots + 1) ** 2 == len(graphdots)
    nvert = round((maxval + nvdots - 1) / nvdots)
    factor_y = 1
    min_vert = 4
    if nvert < min_vert:
        factor_y = round(2 * min_vert * nvdots / maxval) / 2
        nvert = round((maxval * factor_y) / nvdots)

    labels = get_axis_labels(nvert, njobs, factor_y, maxval)

    dt = round(len(reduced) * quant_time(duration))

    for y in range(nvert, 0, -1):
        min_y = scaled_y(y, 0, nvdots, factor_y)
        ls = [min(max(0, round((v - min_y) * factor_y)), nvdots) for v in reduced]

        oklevel = 0.5

        s = ''
        last_bg_colored = True
        for t in range(0, len(ls), 2):
            if ((t // dt) % 2 == 1) != last_bg_colored:
                s += COLOR_OFF if last_bg_colored else COLOR_BG
                last_bg_colored = not last_bg_colored

            if ls[t] + ls[t + 1] > 0:
                frac = (reduced[t] + reduced[t + 1] + 1) / (remaining_reduced[t] + remaining_reduced[t + 1])
                if frac >= oklevel:
                    s += f'\x1b[38;2;{lerp_color(COLOR_LOWEST, COLOR_MEDIAN, (frac - oklevel) / (1 - oklevel))}m'
                else:
                    s += f'\x1b[38;2;{lerp_color(COLOR_HIGHEST, COLOR_LOWEST, frac / oklevel)}m'
            s += graphdots[ls[t] + (nvdots + 1) * ls[t + 1]]
        s += COLOR_OFF

        label = labels[y]
        if y <= 2:
            if y == 2:
                r = f'  overhead {COLOR_EMPH}{overhead}%{COLOR_OFF}'
            else:
                r = f'efficiency {COLOR_EMPH}{efficiency}%{COLOR_OFF}'
            label = f'{r:<{labelwidth + 1 + (len(COLOR_EMPH) + len(COLOR_OFF)) - len(label)}}{label}'

        print(f'{label:>{labelwidth + 1}}{s}')


def main(argv: List[str]) -> None:
    """Main function of the script."""
    meas = run(argv)
    if not meas:
        sys.exit(0)

    start, duration, labelwidth = get_limits(meas)

    barwidth = COLUMNS - 1 - labelwidth

    nsteps = NFRAC * barwidth

    stepsize = duration / nsteps

    coords = list(map(lambda m: getcoord(m, start, stepsize), meas))

    tbusy, efficiency, median, histogram, remaining = compute_utilization(coords, nsteps)
    try:
        env = os.getenv("XDG_STATE_HOME")
        fname = pathlib.Path(env) if env else (pathlib.Path.home() / ".local" / "state")
        with open(fname / "build-tools.log", "w", encoding="utf8") as fd:
            fd.write(str(histogram) + '\n')
            fd.write(str(remaining) + '\n')
    except FileNotFoundError:
        # Ignore errors when writing the logging data.
        pass

    title = f" {COLOR_EMPH}Build Report{COLOR_OFF} "
    print(f'{title:🮁^{COLUMNS + len(COLOR_EMPH) + len(COLOR_OFF)}s}')

    for i, c in enumerate(coords):
        bg = COLOR_BG if i % 2 else ''
        if c[1] - c[0] <= median:
            fg = f'\x1b[38;2;{lerp_color(COLOR_LOWEST, COLOR_MEDIAN, (c[1] - c[0]) / median)}m'
        else:
            fg = f'\x1b[38;2;{lerp_color(COLOR_MEDIAN, COLOR_HIGHEST, (c[1] - c[0] - median) / median)}m'
        print(f'{bg}{c[2][-labelwidth:]:>{labelwidth}} {get_bar_string(c[0], c[1], c[3], labelwidth, fg, bg)}\x1b[0K\x1b[0m')

    totalfmt = f' {COLOR_EMPH}{fmttime(to_ns(duration))}{COLOR_OFF} '
    print(f'{" "*labelwidth} ◀{totalfmt:─^{barwidth - 2 + len(COLOR_EMPH) + len(COLOR_OFF)}}▶')

    assert len(histogram) == NFRAC * barwidth
    plot_histogram(labelwidth, histogram, remaining, duration, len(coords), percent(1 - tbusy / nsteps), percent(efficiency))


if __name__ == '__main__':
    import signal
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)
