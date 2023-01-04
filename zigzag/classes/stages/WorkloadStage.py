import networkx as nx
import logging

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.dnn_workload import DNNWorkload
from zigzag.classes.workload.dummy_node import DummyNode


logger = logging.getLogger(__name__)

class WorkloadStage(Stage):
    """
    Class that iterates through the nodes in a given workload graph.
    """
    def __init__(self, list_of_callables, *, workload, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def run(self):
        for id, layer in enumerate(nx.topological_sort(self.workload)):
            if type(layer) == DummyNode:
                continue  # skip the DummyNodes
            kwargs = self.kwargs.copy()
            kwargs['layer'] = layer
            if layer.name:
                layer_name = layer.name
            else:
                layer_name = id
            logger.info(f"Processing layer {layer_name}...")
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (layer, extra_info)
