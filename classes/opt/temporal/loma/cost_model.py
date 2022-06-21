


class CostModelCaller:
    """
    Class that calls an underlying cost model for the loma allocated ordering.
    """
    def __init__(self, layer, memory_hierarchy, spatial_mapping):
        """
        Initialize the cost model with the:
        - layer
        - memory hierarchy
        - spatial mapping
        """
        self.layer = layer
        self.memory_hierarchy = memory_hierarchy
        self.spatial_mapping = spatial_mapping

    def run(self, temporal_mapping):
        """
        Run an underlying cost model for a given temporal mapping.
        """

        # TODO


class CostModelOutput:

    def __init__(self):
        self.energy = 123
        self.latency = 456

    def __lt__(self, other):
        return self.energy < other.energy