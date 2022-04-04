from theme import Theme

class KonsoleTheme(Theme):
    def __init__(self, name, colors):
        self.name = name
        self.extension = "colorscheme"
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
        self.theme = theme
