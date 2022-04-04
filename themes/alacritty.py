from theme import Theme

class AlacrittyTheme(Theme):
    def __init__(self, name, colors):
        self.name = name
        self.extension = "yml"
        self.colors = colors

    def _rgb_to_hex(self, rgb):
        return "#" + "".join(list(map(lambda x: hex(x)[2:].zfill(2), rgb)))

    def render(self):
        theme = f"{self.name}: &{self.name.lower()}\n"
        theme += "  bright:\n"
        theme += f"    black: '{self._rgb_to_hex(self.colors.briblack)}'\n"
        theme += f"    blue: '{self._rgb_to_hex(self.colors.briblue)}'\n"
        theme += f"    cyan: '{self._rgb_to_hex(self.colors.bricyan)}'\n"
        theme += f"    green: '{self._rgb_to_hex(self.colors.brigreen)}'\n"
        theme += f"    magenta: '{self._rgb_to_hex(self.colors.brimagenta)}'\n"
        theme += f"    red: '{self._rgb_to_hex(self.colors.brired)}'\n"
        theme += f"    white: '{self._rgb_to_hex(self.colors.briwhite)}'\n"
        theme += f"    yellow: '{self._rgb_to_hex(self.colors.briyellow)}'\n"
        theme += "  cursor:\n"
        theme += f"    cursor: '{self._rgb_to_hex(self.colors.briwhite)}'\n"
        theme += f"    text: '{self._rgb_to_hex(self.colors.briblack)}'\n"
        theme += "  normal:\n"
        theme += f"    black: '{self._rgb_to_hex(self.colors.black)}'\n"
        theme += f"    blue: '{self._rgb_to_hex(self.colors.blue)}'\n"
        theme += f"    cyan: '{self._rgb_to_hex(self.colors.cyan)}'\n"
        theme += f"    green: '{self._rgb_to_hex(self.colors.green)}'\n"
        theme += f"    magenta: '{self._rgb_to_hex(self.colors.magenta)}'\n"
        theme += f"    red: '{self._rgb_to_hex(self.colors.red)}'\n"
        theme += f"    white: '{self._rgb_to_hex(self.colors.white)}'\n"
        theme += f"    yellow: '{self._rgb_to_hex(self.colors.yellow)}'\n"
        theme += "  primary:\n"
        theme += f"    background: '{self._rgb_to_hex(self.colors.black)}'\n"
        theme += f"    foreground: '{self._rgb_to_hex(self.colors.white)}'\n"
        theme += "  selection:\n"
        theme += f"    background: '{self._rgb_to_hex(self.colors.white)}'\n"
        theme += f"    text: '{self._rgb_to_hex(self.colors.black)}'\n"
        self.theme = theme
