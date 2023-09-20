## Class that stores a single for loop's information.
class Loop:

    ## The class constructor
    # Initialize the loop with the given dimension string and size
    # @param dimenstion
    # @param size
    # @param type
    def __init__(self, dimension: str, size: int, type: str = "temporal"):
        self.dimension = dimension
        self.size = size
        self.type = type

    def __str__(self):
        return f"{self.type.capitalize()}Loop({self.dimension},{self.size})"

    def __repr__(self):
        return str(self)
