#!/usr/bin/env python3

import argparse
from itertools import repeat
import json
import multiprocessing
from operator import add, sub, mul
from pathlib import Path
import sys
import time

import cerberus
import cv2
import numpy as np
import yaml

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
WHITE = (255, 255, 255)

SILENT = False

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
        return (abs(r - x[0]) ** 2 + abs(g - x[1]) ** 2 + abs(b - x[2]) ** 2)
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
    return (abs(r - x[0]) ** 2 + abs(g - x[1]) ** 2 + abs(b - x[2]) ** 2)

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

class KonsoleTheme:
    def __init__(self, name, colorset):
        self.name = name
        self.colorset = colorset
    def rgb_commas(self, rgb):
        return f"{rgb[0]},{rgb[1]},{rgb[2]}"
    def render(self):
        for c in self.colorset:
            self.colorset[c] = rgb_commas(self.colorset[c])
        text = f"""
[Background]
Color={self.colorset["black"]}

[BackgroundIntense]
Color={self.colorset["black"]}

[Foreground]
Color={self.colorset["briwhite"]}

[ForegroundIntense]
Color={self.colorset["briwhite"]}
Bold=true

[Color0]
Color={self.colorset["black"]}

[Color0Intense]
Color={self.colorset["briblack"]}

[Color1]
Color={self.colorset["red"]}

[Color1Intense]
Color={self.colorset["brired"]}

[Color2]
Color={self.colorset["green"]}

[Color2Intense]
Color={self.colorset["brigreen"]}

[Color3]
Color={self.colorset["yellow"]}

[Color3Intense]
Color={self.colorset["briyellow"]}

[Color4]
Color={self.colorset["blue"]}

[Color4Intense]
Color={self.colorset["briblue"]}

[Color5]
Color={self.colorset["magenta"]}

[Color5Intense]
Color={self.colorset["brimagenta"]}

[Color6]
Color={self.colorset["cyan"]}

[Color6Intense]
Color={self.colorset["bricyan"]}

[Color7]
Color={self.colorset["white"]}

[Color7Intense]
Color={self.colorset["briwhite"]}

[General]
Description={self.name}
Opacity=1
Wallpaper=
"""
        return text


# Register theme classes here.
THEME_REGISTRY = {
    "konsole": KonsoleTheme,
}


def gencolors(goal, colors):
    c = closest_color(goal, colors)
    return (c, brighten(c))

def gencolors_parallel(goal, colors):
    c = closest_color_parallel(goal, colors)
    return (c, brighten(c))


def infoprint(s):
    if not SILENT:
        print(f"=> {s}")

def errorprint(s):
    if not SILENT:
        print(f"\033[41m!!\033[0m\033[31m {s}\033[0m")

def mehprint(s):
    if not SILENT:
        print(f"\033[43m**\033[0m\033[33m {s}\033[0m")


def analyze_document(document):
    """
    Analyze a settings document. Find invalid keys/values, and populate optional+missing
    key-value pairs.

    Parameters
    ----------
    document : dict
        A config document.

    Returns
    -------
    dict
        A full-fleshed config document.
    """
    main_schema = {
        "file": {"type": "string", "required": True},
        "outputs": {
            "type": "list",
            "required": True,
            "schema": {
                "file": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "format": {
                    "type": "string",
                    "required": True,
                    "allowed": THEME_REGISTRY.keys(),
                },
                "method": {
                    "type": "string",
                    "required": False,
                    "allowed": ["black", "white", "dark", "light"],
                    "default": "dark",
                },
            }}}
    new_schema = {
        "file": {"type": "string", "required": True},
        "outputs": {
            "type": "list",
            "items": [
                {"type": "dict", "schema": {"file": {"type": "string", "required": True}}},
                {"type": "dict", "schema": {"name": {"type": "string", "required": True}}},
                {"type": "dict", "schema": {"format":
                                            {"type": "string",
                                             "required": True,
                                             "allowed": THEME_REGISTRY.keys()}}},
                {"type": "dict", "schema": {"method":
                                            {"type": "string",
                                             "required": False,
                                             "allowed": ["black", "white", "dark", "light"],
                                             "default": "dark"}}}]}}

    v = cerberus.Validator(new_schema, purge_unknown=False)
    #  document = v.normalized(document)
    #  if not v.validate(document):
        #  for e in v.errors:
            #  print(f"Error for key '{e}': {v.errors[e]}.")
        #  sys.exit(1)
    new_document = {}
    exit_on_finish = False
    for heading in document:
        print("Checking task...")
        new_document[heading] = v.normalized(document[heading])
        print(json.dumps(new_document[heading], indent=4))
        if not v.validate(new_document[heading]):
            exit_on_finish = True
            print(v.errors)
    if exit_on_finish:
        sys.exit(1)
    return new_document


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Convert pictures to terminal colors.",
        epilog="Contact the developer at <oliversandli@icloud.com> for any issues."
    )
    parser.add_argument("taskfile", help="The path to a task file.")
    parser.add_argument("-p", "--parallel", action="store_true", help="Use parallel processing.")
    parser.add_argument("-t", "--time-analysis", action="store_true", help="Time the analysis section.")
    parser.add_argument("-c", "--config-only", action="store_true", help="Only parse the task file, then exit.")
    parser.add_argument("-s", "--silent", action="store_true", help="Silent operation.")
    args = parser.parse_args()

    conf_type = Path(args.taskfile).suffix
    with open(args.taskfile, "r") as f:
        if conf_type == ".yaml":
            config = yaml.load(f, yaml.FullLoader)
        elif conf_type == ".json":
            config = json.load(f)

    if args.config_only:
        analyze_document(config)
        sys.exit(0)

    for task_name in config:
        task = populate_task(config[task_name], task_name)

        infoprint(f"Reading image {task['image']}...")
        image = cv2.imread(task["image"])
        if image is False:
            print(f"ERROR: the image path '{task['image']}' is invalid.")
            sys.exit(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        infoprint("Reshaping and removing duplicate pixels...")
        pool = np.unique(np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2])), axis=0)

        infoprint("Deciding colors...")
        if args.parallel:
            gencolors = gencolors_parallel
        if args.time_analysis:
            analysis_start = time.time()
        if task["mode"] in ("black", "white"):
            black, briblack = (0, 0, 0), (10, 10, 10)
            white, briwhite = (246, 247, 248), (255, 255, 255)
        else:
            infoprint("black")
            black, briblack = gencolors(BLACK, pool)
            infoprint("white")
            white, briwhite = gencolors(WHITE, pool)
        if task["mode"] in ("light", "white"):
            black, white = white, black
            briblack, briwhite = briwhite, briblack
        infoprint("red")
        red, brired = gencolors(RED, pool)
        infoprint("green")
        green, brigreen = gencolors(GREEN, pool)
        infoprint("yellow")
        yellow, briyellow = gencolors(YELLOW, pool)
        infoprint("blue")
        blue, briblue = gencolors(BLUE, pool)
        infoprint("magenta")
        magenta, brimagenta = gencolors(MAGENTA, pool)
        infoprint("cyan")
        cyan, bricyan = gencolors(CYAN, pool)
        if args.time_analysis:
            analysis_end = time.time()
            infoprint(f"Analysis took {analysis_end - analysis_start} seconds.")
        colorset = {
                "black": black,
                "briblack": briblack,
                "white": white,
                "red": red,
                "brired": brired,
                "green": green,
                "brigreen": brigreen,
                "yellow": yellow,
                "briyellow": briyellow,
                "blue": blue,
                "briblue": briblue,
                "magenta": magenta,
                "brimagenta": brimagenta,
                "cyan": cyan,
                "bricyan": bricyan,
                }
        infoprint("Generating theme...")
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
