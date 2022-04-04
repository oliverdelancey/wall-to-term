from theme import Theme
from colorindex import ColorIndex

class XFCE4TerminalTheme(Theme):
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
        self.theme = theme
