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
