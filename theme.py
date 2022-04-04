class Theme:
    def __init__(self, name, colors):
        self.name = name
        self.extension = "txt"

    def render(self):
        self.theme = ""

    def save(self, dest):
        try:
            with open(f"{dest}.{self.extension}", "w") as f:
                f.write(self.theme)
        except FileNotFoundError:
            print(f"ERROR: the destination path '{dest}' is invalid.")
            sys.exit(1)
