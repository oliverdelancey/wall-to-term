#!/usr/bin/env python3

import argparse
from operator import add, sub, mul
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
    br_color = tuple(map(add, rgb, values))
    if any(i > 255 for i in br_color):
        br_color = tuple(map(sub, rgb, values))
    if any(i < 0 for i in br_color):
        br_color = rgb

    # Make sure elements are ints.
    br_color = tuple(map(round, br_color))

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

    WARNING: this function is *destructive*. It removes the result from the input
    `colors` list. This is to prevent overlapping colors in the color assignment below
    (starting line 263). It is recommended to pass a copied list to this function if you
    wish to protect the original list.

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
    r, g, b = rgb
    color_diffs = []
    for color in colors:
        cr, cg, cb = color
        color_diff = (abs(r - cr) ** 2 + abs(g - cg) ** 2 + abs(b - cb) ** 2) ** (1 / 2)
        color_diffs.append((color_diff, color))
    result = min(color_diffs)[1]
    colors.remove(result)
    return result

class KonsoleColor:
    def __init__(self, name, color, bold=False):
        self.name = name
        self.color = color
        self.bold = bold
    def render(self):
        retval = f"[{self.name}]\nColor={self.color[0]},{self.color[1]},{self.color[2]}"
        if self.bold:
            retval += "\nBold=true"
        return retval + "\n"

def gencolors(goal, colors):
    c = closest_color(goal, colors)
    return (c, brighten(c))

def infoprint(s):
    print(f"=> {s}")

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Convert pictures to terminal colors.")
    parser.add_argument("picture", help="The path to the picture.")
    parser.add_argument("dest", help="The path to the generated theme.")
    parser.add_argument("name", help="The name of the generated theme.")
    parser.add_argument("-l", "--light", action="store_true", help="Generate a white-base theme instead of a black-base theme.")
    parser.add_argument("-w", "--white", action="store_true", help="Force a pure white background.")
    parser.add_argument("-b", "--black", action="store_true", help="Force a black background.")
    parser.add_argument("-t", "--time-analysis", action="store_true", help="Time the analysis section.")
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
    infoprint("Converting image to set...")
    pool = set()
    for i in image:
        for j in i:
            pool.add(tuple(j))
    infoprint("Deciding colors...")
    if args.time_analysis:
        analysis_start = time.time()
    if args.black or args.white:
        black, briblack = (0, 0, 0), (10, 10, 10)
        white, briwhite = (246, 247, 248), (255, 255, 255)
    else:
        black, briblack = gencolors(BLACK, pool)
        white, briwhite = gencolors(WHITE, pool)
    if args.light or args.white:
        black, white = white, black
        briblack, briwhite = briwhite, briblack
    red, brired = gencolors(RED, pool)
    green, brigreen = gencolors(GREEN, pool)
    yellow, briyellow = gencolors(YELLOW, pool)
    blue, briblue = gencolors(BLUE, pool)
    magenta, brimagenta = gencolors(MAGENTA, pool)
    cyan, bricyan = gencolors(CYAN, pool)
    if args.time_analysis:
        analysis_end = time.time()
        infoprint(f"Analysis took {analysis_end - analysis_start} seconds.")
    mappings = [
            ("Background", black),
            ("BackgroundIntense", black),
            ("Foreground", briwhite),
            ("ForegroundIntense", briwhite, True),
            ("Color0", black),
            ("Color0Intense", briblack),
            ("Color1", red),
            ("Color1Intense", brired),
            ("Color2", green),
            ("Color2Intense", brigreen),
            ("Color3", yellow),
            ("Color3Intense", briyellow),
            ("Color4", blue),
            ("Color4Intense", briblue),
            ("Color5", magenta),
            ("Color5Intense", brimagenta),
            ("Color6", cyan),
            ("Color6Intense", bricyan),
            ("Color7", white),
            ("Color7Intense", briwhite)
            ]
    infoprint("Generating theme...")
    theme = ""
    for m in mappings:
        k = KonsoleColor(*m)
        theme += k.render()
    theme += f"[General]\nDescription={args.name}\nOpacity=1\nWallpaper="
    infoprint("Saving theme...")
    try:
        with open(args.dest, "w") as f:
            f.write(theme)
    except FileNotFoundError:
        print(f"ERROR: the destination path '{args.dest}' is invalid.")
        sys.exit(1)
    infoprint(f"Done. Took {time.time() - start_time} seconds.")
