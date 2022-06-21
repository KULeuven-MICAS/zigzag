"""
File that contains all user-provided inputs and can be accessed globally.
"import classes.io.input_config as inputs",
then access one of the inputs using inputs.XXX.
"""
from types import ModuleType




class MainInputs:
    def __init__(self, workload : 'DNNWorkload'=None,
                 layer: 'LayerNode'=None,
                 accelerator: 'Accelerator'=None,
                 spatial_mapping: 'SpatialMapping'=None,
                 temporal_mapping: 'TemporalMapping' = None,
                 settings: ModuleType=None):
        self.workload = workload
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.accelerator = accelerator
        self.temporal_mapping = temporal_mapping
        self.settings = settings


global_main_inputs = MainInputs()


def init(input_workload, input_accelerator, input_settings):
    """
    Set the fields of the globally accessible global_main_inputs:
    - workload
    - accelerator
    - settings module
    """
    global_main_inputs.workload = input_workload
    global_main_inputs.accelerator = input_accelerator
    global_main_inputs.settings = input_settings


def set_layer(input_layer: 'LayerNode'):
    """
    sets the layer field of the globally accessible global_main_inputs.
    """
    global_main_inputs.layer = input_layer


def set_spatial_mapping(input_spatial_mapping: 'SpatialMapping'):
    """
    Sets the global spatial mapping.
    """
    global_main_inputs.spatial_mapping = input_spatial_mapping


def set_temporal_mapping(input_temporal_mapping: 'TemporalMapping'):
    """
    Sets the global temporal mapping.
    """
    global_main_inputs.temporal_mapping = input_temporal_mapping