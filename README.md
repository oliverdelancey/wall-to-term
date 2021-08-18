# wall-to-term

## Contents
* [Supported Terminals](#supported-terminals)
* [Usage](#usage)
* [Requirements](#requirements)
* [Versions](#versions)
* [Roadmap](#roadmap)

`wall-to-term` converts images to terminal themes by finding the closest colors in the
picture to the corresponding ideal colors. For instance, it looks for the closest color to
red for determining the red color for the theme.

## Supported Terminals

* Konsole
* XFCE4 Terminal

I can add support for others. Just raise an issue and I'll see what I can do!

## Usage

For general usage, just run `./wtt2.py --help`. All the options and flags are described
there.

Note that I tend to get the best results with the following settings (of course, this
could and probably will be different for each image and your personal preference):

* Play quite a bit with the `--threshold`. This is especially important for images which
  do not have, for instance, a green color.
* Use `--kmeans`. There is nothing inherently better about it, it just gives slightly
  different results, which look better to me.
* Experiment with different palettes. This changes what colors `wtt2` regards as
  "best-case" colors, and therefore can drastically affect output.
* Use the `swatch` theme type. It will generate a color swatch image. This is extremely
  useful for quickly testing other parameters, such as `--threshold` or `--palette`.

### Palettes

`wall-to-term` works by trying to find the colors in an image that are closest to a set of
ideal colors.  By default, it tries to find the closest color to absolute red (`255,0,0`),
green, blue, etc.  This set of colors is considered a palette. It is essentially a
collection of "goal" colors for the program to try to reach.

With the `--palette` option, you can supply your own palette in the form of a text file.
It must follow the following syntax:
* 8 colors and 8 lines, each line corresponding to a color.
* Color names may be used, such as `black` or `yellow`. Case does not matter.
* For exact colors, write a comma-separated triplet of 0-255 RGB values, such as `255,0,0`.

Each line corresponds to a specific color, in this order:
```txt
black
red
green
yellow
blue
magenta
cyan
white
```

An example palette file:
```txt
black
255,0,0
0,255,0
yellow
BLUE
255,0,255
0,255,255
white
```

## Requirements

### wtt2

* OpenCV `python3 -m pip install opencv-python`
* Numpy `python3 -m pip install numpy`
* scikit-learn `python3 -m pip install scikit-learn`


## Versions

The current working/featureful version is `wtt2`. It has more advanced, more accurate, and
faster image analysis algorithms than `wtt` (and more features).  `wtt3` is a WIP for
advanced scripting options, i.e. batch processing/generation. It is currently unstable to
the point of being unusable.  Same thing goes for `wtt-numba`. It is a slightly
radioactive experiment which cannot be disposed of as it serves as an interesting example
of performance testing. Note that, according to very basic tests, `wtt2` with the
`--parallel` flag tends to be as fast as `wtt-numba`, but without the extra library
dependency.

## Roadmap

* Add Alacritty support.
* Add Gnome Terminal support.
