class Loop:
    """
    Class that stores a single for loop's information.
    """
    def __init__(self, dimension: str, size: int, type: str = "temporal"):
        """
        Initialize the loop with the given dimension string and size.
        """
        self.dimension = dimension
        self.size = size
        self.type = type

    def __str__(self):
        return f"{self.type.capitalize()}Loop({self.dimension},{self.size})"

    def __repr__(self):
        return str(self)
