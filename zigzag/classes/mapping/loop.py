from abc import ABCMeta


class Loop(metaclass=ABCMeta):
    """!  Abstract base class that represents a loop. Could be spatial or temporal loop."""

    def __init__(self, dimension, size) -> None:
        """!  The class constructor
        Initialize this Loop object.
        @param dimension (str): The dimension, e.g. "K"
        @param size (float): The loop size, e.g. 16.0
        """
        self.type = None
        self.dimension = dimension
        self.size = size

    # def __str__(self):
    #     return f"Loop({self.dimension},{self.size})"

    # def __repr__(self):
    #     return str(self)
