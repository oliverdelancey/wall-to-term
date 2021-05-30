#!/usr/bin/env python3

import argparse
from itertools import repeat
import multiprocessing
from operator import mul
import sys
import time

import cv2
import numpy as np

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
WHITE = (255, 255, 255)


def brighten(rgb, values=(30, 30, 30)):
    """
    Brighten an RGB color.

    If the brightened color is over (255, 255, 255), subtract `values` instead of adding
    `values`, effectively dimming the color. If this results in a negative number,
    return the original color. This is implemented to guarantee a valid (and hopefully
    different) color is returned.

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
    # Apply brightness.
    br_color = np.add(rgb, values)
    if any(i > 255 for i in br_color):
        br_color = np.subtract(rgb, values)
    if any(i < 0 for i in br_color):
        br_color = rgb

    return br_color


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
    loc = np.where(color_diffs == np.min(color_diffs))
    result = colors[loc][0]
    return result


def gencolors(goal, colors):
    c = closest_color(goal, colors)
    return (c, brighten(c))


def gencolors_parallel(goal, colors):
    c = closest_color_parallel(goal, colors)
    return (c, brighten(c))


def infoprint(s):
    print(f"=> {s}")


class ColorIndex:
    def __init__(self, vals=None):
        if not vals:
            vals = [i for i in [None] * 16]
        if len(vals) != 16:
            infoprint(f"Bad length {len(vals)}")
            sys.exit(1)
        (
            self.black,
            self.red,
            self.green,
            self.yellow,
            self.blue,
            self.magenta,
            self.cyan,
            self.white,
            self.briblack,
            self.brired,
            self.brigreen,
            self.briyellow,
            self.briblue,
            self.brimagenta,
            self.bricyan,
            self.briwhite,
        ) = vals

    def list(self):
        return [
            self.black,
            self.red,
            self.green,
            self.yellow,
            self.blue,
            self.magenta,
            self.cyan,
            self.white,
            self.briblack,
            self.brired,
            self.brigreen,
            self.briyellow,
            self.briblue,
            self.brimagenta,
            self.bricyan,
            self.briwhite,
        ]


# --------------
# Theme Classes
# --------------


class KonsoleTheme:
    def __init__(self, name, colors):
        self.name = name
        self.extension = "colortheme"
        self.entries = [
            ("Background", colors.black),
            ("BackgroundIntense", colors.black),
            ("Foreground", colors.briwhite),
            ("ForegroundIntense", colors.briwhite, True),
            ("Color0", colors.black),
            ("Color0Intense", colors.briblack),
            ("Color1", colors.red),
            ("Color1Intense", colors.brired),
            ("Color2", colors.green),
            ("Color2Intense", colors.brigreen),
            ("Color3", colors.yellow),
            ("Color3Intense", colors.briyellow),
            ("Color4", colors.blue),
            ("Color4Intense", colors.briblue),
            ("Color5", colors.magenta),
            ("Color5Intense", colors.brimagenta),
            ("Color6", colors.cyan),
            ("Color6Intense", colors.bricyan),
            ("Color7", colors.white),
            ("Color7Intense", colors.briwhite),
        ]

    def _render_color(self, name, color, bold=False):
        retval = f"[{name}]\nColor={color[0]},{color[1]},{color[2]}"
        if bold:
            retval += "\nBold=true"
        return retval + "\n"

    def render(self):
        theme = ""
        for e in self.entries:
            theme += self._render_color(*e)
        theme += f"[General]\nDescription={self.name}\nOpacity=1\nWallpaper="
        return theme


class XFCE4TerminalTheme:
    def __init__(self, name, colors):
        self.name = name
        self.extension = "theme"
        self.colors = ColorIndex(list(map(self._rgb_to_12hex, colors.list())))

    def _rgb_to_12hex(self, rgb):
        return "#" + "".join(list(map(lambda x: hex(x)[2:].zfill(2), rgb)))

    def render(self):
        theme = (
            "[Scheme]\n"
            f"Name={self.name}\n"
            f"ColorCursor={self.colors.yellow}\n"
            "ColorCursorUseDefault=FALSE\n"
            f"ColorForeground={self.colors.briwhite}\n"
            f"ColorBackground={self.colors.black}\n"
            f"ColorPalette={';'.join(self.colors.list())}"
        )
        return theme


if __name__ == "__main__":
    start_time = time.time()

    theme_registry = {
        "konsole": KonsoleTheme,
        "xfce4": XFCE4TerminalTheme,
    }

    parser = argparse.ArgumentParser(
        description="Convert pictures to terminal colors.",
        epilog="Contact the developer at <oliversandli@icloud.com> for any issues.",
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
        "-p", "--parallel", action="store_true", help="Use parallel processing."
    )
    parser.add_argument(
        "-t", "--time-analysis", action="store_true", help="Time the analysis section."
    )
    args = parser.parse_args()

    if (args.light and args.white) or args.black:
        print("ERROR: --light, --white, and --black are mutually exclusive.")
        sys.exit(1)

    infoprint("Reading image...")
    image = cv2.imread(args.picture)
    if image is False:
        print(f"ERROR: the picture path '{args.picture}' is invalid.")
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    infoprint("Reshaping and removing duplicate pixels...")
    pool = np.unique(
        np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2])), axis=0
    )

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
        black, briblack = gencolors(BLACK, pool)
        infoprint("white")
        white, briwhite = gencolors(WHITE, pool)
    if args.light or args.white:
        black, white = white, black
        briblack, briwhite = briwhite, briblack
    color_index.black = black
    color_index.briblack = briblack
    color_index.white = white
    color_index.briwhite = briwhite

    infoprint("red")
    color_index.red, color_index.brired = gencolors(RED, pool)
    infoprint("green")
    color_index.green, color_index.brigreen = gencolors(GREEN, pool)
    infoprint("yellow")
    color_index.yellow, color_index.briyellow = gencolors(YELLOW, pool)
    infoprint("blue")
    color_index.blue, color_index.briblue = gencolors(BLUE, pool)
    infoprint("magenta")
    color_index.magenta, color_index.brimagenta = gencolors(MAGENTA, pool)
    infoprint("cyan")
    color_index.cyan, color_index.bricyan = gencolors(CYAN, pool)

    if args.time_analysis:
        analysis_end = time.time()
        infoprint(f"Analysis took {analysis_end - analysis_start} seconds.")

    infoprint("Generating theme...")
    theme = theme_registry[args.term](args.name, color_index)
    theme_text = theme.render()
    infoprint("Saving theme...")
    try:
        with open(f"{args.dest}.{theme.extension}", "w") as f:
            f.write(theme_text)
    except FileNotFoundError:
        print(f"ERROR: the destination path '{args.dest}' is invalid.")
        sys.exit(1)
    infoprint(f"Done. Took {time.time() - start_time} seconds.")
