import numpy as np
from typing import Dict
if __name__ == "__main__":
    from dimension import Dimension
    from DimcArrayUnit import DimcArrayUnit
    from AimcArrayUnit import AimcArrayUnit
else:
    from zigzag.classes.hardware.architecture.dimension import Dimension
    from zigzag.classes.hardware.architecture.DimcArrayUnit import DimcArrayUnit
    from zigzag.classes.hardware.architecture.AimcArrayUnit import AimcArrayUnit


class ImcArray:
    def __init__(self, tech_param: Dict[str, float], hd_param: dict, dimensions: Dict[str, int]):
        """
        This class defines the general IMC array (including AIMC and DIMC)
        :param tech_param: definition of technology-related parameters
        :param hd_param: hardware architecture parameters except dimensions
        :param dimensions: dimensions definition
        """
        if hd_param["imc_type"] == "digital":
            self.unit = DimcArrayUnit(tech_param, hd_param, dimensions)
        elif hd_param["imc_type"] == "analog":
            self.unit = AimcArrayUnit(tech_param, hd_param, dimensions)
        self.unit.get_area() # update self.area and self.area_breakdown
        self.unit.get_delay() # update self.delay and self.delay_breakdown
        self.area_breakdown = self.unit.area_breakdown
        self.total_area = self.unit.area
        self.tclk_breakdown = self.unit.delay_breakdown # clock period breakdown
        self.tclk = self.unit.delay # maximum clock period (unit: ns)
        base_dims = [
            Dimension(idx, name, size)
            for idx, (name, size) in enumerate(dimensions.items())
        ]
        self.dimensions = base_dims
        self.dimension_sizes = [dim.size for dim in base_dims]
        self.nb_dimensions = len(base_dims)
        self.total_unit_count = np.prod(list(dimensions.values()))
        self.pe_type = hd_param["pe_type"]
        self.imc_type = hd_param["imc_type"]
        self.tops_peak, self.topsw_peak, self.topsmm2_peak = self.unit.get_macro_level_peak_performance()

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return {"operational_unit": self.unit, "dimensions": self.dimensions}

