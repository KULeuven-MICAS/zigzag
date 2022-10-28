from zigzag.classes.mapping.loop import Loop


class TemporalLoop(Loop):
    """Class that represents a temporal loop.
    """
    def __init__(self, dimension, size) -> None:
        super().__init__(dimension, size)
        self.type = "temporal"

    def __str__(self):
        return f"TemporalLoop({self.dimension},{self.size})"

    def __repr__(self):
        return str(self)