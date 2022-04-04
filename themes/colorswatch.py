from theme import Theme
from colorindex import ColorIndex

class ColorSwatch:
    def __init__(self, name, colors):
        self.name = name
        self.extension = "png"
        self.colors = ColorIndex(
            [list(map(lambda x: x / 255, i)) for i in colors.list()]
        )

    def render(self):
        import matplotlib
        import matplotlib.pyplot as plt

        square_size = 100
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color_list = self.colors.list()
        # Generate top row with standard colors.
        for i, color in enumerate(color_list[:8]):
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (i * square_size, square_size),
                    square_size,
                    square_size,
                    color=color,
                )
            )
        # Generate bottom row with bold/bright colors.
        for i, color in enumerate(color_list[8:]):
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (i * square_size, 0),
                    square_size,
                    square_size,
                    color=color,
                )
            )
        plt.xlim([0, square_size * 8])
        plt.ylim([0, square_size * 2])
        plt.title(self.name)
        plt.axis("off")
        self.plt = plt

    def save(self, dest):
        self.plt.savefig(f"{dest}.{self.extension}")
