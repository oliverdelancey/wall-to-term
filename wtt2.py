#!/usr/bin/env python3

import argparse
from collections import OrderedDict
from itertools import repeat
import multiprocessing
from operator import mul
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from colorindex import ColorIndex

from themes.alacritty import AlacrittyTheme
from themes.konsole import KonsoleTheme
from themes.xfce4term import XFCE4TerminalTheme
from themes.colorswatch import ColorSwatch
from themes.raw import Raw
from themes.doomemacs import DoomEmacsTheme

theme_registry = {
    "alacritty": AlacrittyTheme,
    "doom-emacs": DoomEmacsTheme,
    "konsole": KonsoleTheme,
    "xfce4": XFCE4TerminalTheme,
    "raw": Raw,
    "swatch": ColorSwatch,
}


def brighten(rgb, values=(30, 30, 30)):
    """
    Brighten an RGB color.

    `values' is added to `rgb'. The resulting triplet is clipped between 0 and 255. This
    is implemented to guarantee a valid (and hopefully different) color is returned.

    Parameters
    ----------
    rgb : tuple[int]
        An RGB tuple.
    values : tuple[int], optional
        Amount of brightening to apply to R, G, B respectively.

    Returns
    -------
    tuple[int]
        A brightened RGB color.
    """

    return np.clip(np.add(rgb, values), 0, 255)


# CURRENTLY UNUSED
def text_color(bg):
    """
    Determine text color based off background color.

    Parameters
    ----------
    bg : tuple[int]
        Background RGB color.

    Returns
    -------
    tuple[int]
        Foreground RGB color.
    """
    luminance = sum(map(mul, (0.299, 0.587, 0.114), bg)) / 255
    if luminance > 0.5:
        fg = (0, 0, 0)
    else:
        fg = (255, 255, 255)
    return fg


def closest_color(rgb, colors):
    """
    Determine the closest color in `colors` to `rgb`.

    Parameters
    ----------
    rgb : tuple[int]
        An RGB color.
    colors : np.array(np.array(_, dtype=uint8))

    Returns
    -------
    tuple[int]
        An RGB color.
    """
    r, g, b = rgb

    def color_diff(x):
        return abs(r - x[0]) ** 2 + abs(g - x[1]) ** 2 + abs(b - x[2]) ** 2

    color_diffs = np.array([color_diff(i) for i in colors])
    if np.min(color_diffs) > args.threshold:
        return rgb
    loc = np.where(color_diffs == np.min(color_diffs))
    result = colors[loc][0]
    return result


def color_diff_par(pair):
    """
    Find the difference between two colors. This is part of `closest_color_parallel`
    below, but multiprocessing cannot pickle nested functions.

    Parameters
    ----------
    pair : (rgb, pixel)
        Check how close `rgb` is to `pixel`.

    Returns
    -------
    int
        The difference between `rgb` and `pixel`.
    """
    rgb, x = pair
    r, g, b = rgb
    return abs(r - x[0]) ** 2 + abs(g - x[1]) ** 2 + abs(b - x[2]) ** 2


def closest_color_parallel(rgb, colors):
    """
    Determine the closest color in `colors` to `rgb`. (parallel version)

    Parameters
    ----------
    rgb : tuple[int]
        An RGB color.
    colors : list[tuple[int]]

    Returns
    -------
    tuple[int]
        An RGB color.
    """
    pool = multiprocessing.Pool(8)
    color_diffs = np.array(pool.map(color_diff_par, zip(repeat(rgb), colors)))
    if np.min(color_diffs) > args.threshold:
        return rgb
    loc = np.where(color_diffs == np.min(color_diffs))
    result = colors[loc][0]
    return result


def unique_colors(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    infoprint("Reshaping and removing duplicate pixels...")
    pool = np.unique(
        np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2])), axis=0
    )
    return pool


def kmeans_colors(image):
    h, w, d = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((h * w, d))
    clt = MiniBatchKMeans(n_clusters=32)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, d))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    quant = quant.reshape((h * w, d))
    return np.unique(quant, axis=0)


def gencolors(goal, colors):
    c = closest_color(goal, colors)
    return (c, brighten(c))


def gencolors_parallel(goal, colors):
    c = closest_color_parallel(goal, colors)
    return (c, brighten(c))


class tcolors:
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def infoprint(s):
    print(f"=> {s}")


def errorprint(s):
    print(f"{tcolors.FAIL}<ERROR> => {s}{tcolors.RESET}")


class ColorPalette:
    def __init__(self, palette_name=None):
        colors = OrderedDict(
            (
                ("black", (0, 0, 0)),
                ("red", (255, 0, 0)),
                ("green", (0, 255, 0)),
                ("yellow", (255, 255, 0)),
                ("blue", (0, 0, 255)),
                ("magenta", (255, 0, 255)),
                ("cyan", (0, 255, 255)),
                ("white", (255, 255, 255)),
            )
        )

        if palette_name:
            with open(palette_name, "r") as f:
                contents = f.read().strip().split("\n")
            if len(contents) != 8:
                errorprint(
                    f"Palette '{palette_name}' has {len(contents)} colors, not 8."
                )
                sys.exit(1)
            for line, color, linno in zip(contents, colors, range(len(contents))):
                infoprint(f"Comparing {line} and {color}")
                if line != color:
                    colors[color] = self._analyze(line, linno)

        self.black = colors["black"]
        self.red = colors["red"]
        self.green = colors["green"]
        self.yellow = colors["yellow"]
        self.blue = colors["blue"]
        self.magenta = colors["magenta"]
        self.cyan = colors["cyan"]
        self.white = colors["white"]

    def _analyze(self, s, line):
        try:
            vals = list(map(int, s.split(",")))
        except ValueError:
            errorprint(f"Bad formatting on line {line} in palette.")
            sys.exit(1)
        if len(vals) != 3:
            errorprint(
                f"Found {len(vals)} values on line {line} in palette. Should be 3."
            )
            sys.exit(1)
        for v in vals:
            if v > 255 or v < 0:
                errorprint(f"Value out of range on line {line} in palette.")
                sys.exit(1)
        return vals



if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Convert pictures to terminal colors.",
        epilog="Contact the developer at <oliverdelancey@gmail.com> for any issues.",
    )
    parser.add_argument("picture", help="The path to the picture.")
    parser.add_argument(
        "term", help="The terminal theme type.", choices=list(theme_registry)
    )
    parser.add_argument(
        "dest", help="The path to the generated theme (sans extention)."
    )
    parser.add_argument("name", help="The name of the generated theme.")
    parser.add_argument(
        "-p", "--palette", action="store", help="Specify a palette file."
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store",
        type=int,
        default=255 ** 2 * 3,
        help="Set a color choice threshold, based on the Euclidean distance between two"
        " colors. Max 195075, min 0.",
    )
    parser.add_argument(
        "-l",
        "--light",
        action="store_true",
        help="Generate a white-base theme instead of a black-base theme.",
    )
    parser.add_argument(
        "-w", "--white", action="store_true", help="Force a pure white background."
    )
    parser.add_argument(
        "-b", "--black", action="store_true", help="Force a black background."
    )
    parser.add_argument(
        "-k",
        "--kmeans",
        action="store_true",
        help="Use K-means to determine color pool.",
    )
    parser.add_argument(
        "-P", "--parallel", action="store_true", help="Use parallel processing."
    )
    parser.add_argument(
        "-C",
        "--clear-cache",
        action="store_true",
        help="Clear cache directory before proceeding.",
    )
    parser.add_argument(
        "-T", "--time-analysis", action="store_true", help="Time the analysis section."
    )
    args = parser.parse_args()

    if args.palette:
        infoprint(f"Reading palette {args.palette}...")
        palette = ColorPalette(args.palette)
    else:
        palette = ColorPalette()

    if (args.light and args.white) or args.black:
        errorprint("--light, --white, and --black are mutually exclusive.")
        sys.exit(1)

    cache_dir = Path("~/.cache/wall-to-term").expanduser()
    if not cache_dir.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=False)
    if args.clear_cache:
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=False)
    current_cacheable_name = "wttcache"
    current_cacheable_name += str(Path(args.picture).expanduser()).replace("/", "-")
    current_cacheable_name += ".km" if args.kmeans else ".un"
    current_cacheable_name += ".npy"

    if (cache_dir / current_cacheable_name).is_file():
        # Load the cached file.
        print(f"Found cached file {current_cacheable_name} ...")
        with open(cache_dir / current_cacheable_name, "rb") as f:
            pool = np.load(f)
    else:
        # Read image, analyze, find colors, generate theme, cache analysis.
        infoprint("Reading image...")
        image = cv2.imread(args.picture)
        if image is False:
            print(f"ERROR: the picture path '{args.picture}' is invalid.")
            sys.exit(1)

        infoprint("Reshaping and removing duplicate pixels...")
        if args.kmeans:
            pool = kmeans_colors(image)
        else:
            pool = unique_colors(image)

    infoprint("Deciding colors...")

    color_index = ColorIndex()

    if args.parallel:
        gencolors = gencolors_parallel
    if args.time_analysis:
        analysis_start = time.time()

    if args.black or args.white:
        black, briblack = (0, 0, 0), (10, 10, 10)
        white, briwhite = (246, 247, 248), (255, 255, 255)
    else:
        infoprint("black")
        black, briblack = gencolors(palette.black, pool)
        infoprint("white")
        white, briwhite = gencolors(palette.white, pool)
    if args.light or args.white:
        black, white = white, black
        briblack, briwhite = briwhite, briblack
    color_index.black = black
    color_index.briblack = briblack
    color_index.white = white
    color_index.briwhite = briwhite

    infoprint("red")
    color_index.red, color_index.brired = gencolors(palette.red, pool)
    infoprint("green")
    color_index.green, color_index.brigreen = gencolors(palette.green, pool)
    infoprint("yellow")
    color_index.yellow, color_index.briyellow = gencolors(palette.yellow, pool)
    infoprint("blue")
    color_index.blue, color_index.briblue = gencolors(palette.blue, pool)
    infoprint("magenta")
    color_index.magenta, color_index.brimagenta = gencolors(palette.magenta, pool)
    infoprint("cyan")
    color_index.cyan, color_index.bricyan = gencolors(palette.cyan, pool)

    if args.time_analysis:
        analysis_end = time.time()
        infoprint(f"Analysis took {analysis_end - analysis_start} seconds.")

    infoprint("Generating theme...")
    theme = theme_registry[args.term](args.name, color_index)
    theme.render()
    infoprint("Saving theme...")
    theme.save(args.dest)

    with open(cache_dir / current_cacheable_name, "wb") as f:
        np.save(f, pool)

    infoprint(f"Done. Took {time.time() - start_time} seconds.")
