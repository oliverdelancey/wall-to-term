#!/usr/bin/env python

from theme import Theme
from colorindex import ColorIndex

class Raw(Theme):
    def __init__(self, name, colors):
        self.name = name
        self.extension = "txt"
        self.color_index = colors

    def _rgb_to_hex(self, rgb):
        return "#" + "".join(list(map(lambda x: hex(x)[2:].zfill(2), rgb)))

    def _decify(self, color):
        return str(color[0]) + "," + str(color[1]) + "," + str(color[2])

    def render(self):
        self.hex_colors = [self._rgb_to_hex(i) for i in self.color_index.list()]
        self.decimal_colors = [self._decify(i) for i in self.color_index.list()]
        self.theme = (f"{self.name.upper()}" + "\n\n" +
                         "#### HEXADECIMAL ####\n\n" +
                         "\n".join(self.hex_colors) +
                         "\n\n\n#### DECIMAL ####\n\n" +
                         "\n".join(self.decimal_colors))
