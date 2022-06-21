import logging

import classes.io.input_config as inputs
from classes.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class WorkloadRunner:
    """
    Class that runs the workload through the underlying optimizations or cost model evaluation.
    """
    def __init__(self):
        """
        Initialize the runner by setting the workload it will run.
        """
        self.workload = inputs.workload

    def run(self):
        """
        Run the WorkloadRunner by going through all the nodes in the workload graph.
        """
        for node in self.workload.nodes():
            self.run_workload_node(node)

    def run_workload_node(self, node: LayerNode):
        """
        Run a single workload node through the zigzag pipelines.
        """
        logger.info(f"Running workload node {node}: {node.loop_dim_size}.")
