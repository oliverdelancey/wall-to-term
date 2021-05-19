# wall-to-term

`wall-to-term` converts images to terminal themes by finding the closest colors in the
picture to the corresponding ideal colors. For instance, it looks for the closest color to
red for determining the red color for the theme.

## Supported Terminals

* Konsole
* XFCE4 Terminal

## Requirements

### wtt and wtt2

* OpenCV `python3 -m pip install opencv-python`
* Numpy `python3 -m pip install numpy`

### wtt3 (experimental/broken)

* Cerberus `python3 -m pip install cerberus`
* PyYaml `python3 -m pip install pyyaml`


## Versions

The current working/featureful version is `wtt2`. It has more advanced, more accurate, and
faster image analysis algorithms than `wtt` (and more features).  `wtt3` is a WIP for
advanced scripting options, i.e. batch processing/generation. It is currently unstable to
the point of being unusable.  Same thing goes for `wtt-numba`. It is a slightly
radioactive experiment which cannot be disposed of as it serves as an interesting example
of performance testing. Note that, according to very basic tests, `wtt2` with the
`--parallel` flag tends to be as fast as `wtt-numba`, but without the extra library
dependency.
