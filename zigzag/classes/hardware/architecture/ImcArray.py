import numpy as np
from typing import Dict

if __name__ == "__main__":
    from zigzag.classes.hardware.architecture.Dimension import Dimension
    from DimcArray import DimcArray
    from AimcArray import AimcArray
    from zigzag.classes.hardware.architecture.OperationalArray import OperationalArray
else:
    from zigzag.classes.hardware.architecture.Dimension import Dimension
    from zigzag.classes.hardware.architecture.DimcArray import DimcArray
    from zigzag.classes.hardware.architecture.AimcArray import AimcArray
    from zigzag.classes.hardware.architecture.OperationalArray import OperationalArray


class ImcArray(OperationalArray):
    def __init__(self, tech_param: dict[str, float], hd_param: dict, dimensions: dict[str, int]):
        # This class defines the general IMC array (including AIMC and DIMC)
        # @param tech_param: definition of technology-related parameters
        # @param hd_param: hardware architecture parameters except dimensions
        # @param dimensions: dimensions definition
        if hd_param["imc_type"] == "digital":
            super().__init__(
                operational_unit=DimcArray(tech_param, hd_param, dimensions),
                dimensions=dimensions,
            )
        elif hd_param["imc_type"] == "analog":
            super().__init__(
                operational_unit=AimcArray(tech_param, hd_param, dimensions),
                dimensions=dimensions,
            )

        self.unit.get_area()  # update self.area and self.area_breakdown
        self.unit.get_delay()  # update self.delay and self.delay_breakdown
        self.area_breakdown = self.unit.area_breakdown
        self.total_area = self.unit.area
        self.tclk_breakdown = self.unit.delay_breakdown  # clock period breakdown
        self.tclk = self.unit.delay  # maximum clock period (unit: ns)
        self.pe_type = hd_param["pe_type"]
        self.imc_type = hd_param["imc_type"]
        self.tops_peak, self.topsw_peak, self.topsmm2_peak = self.unit.get_macro_level_peak_performance()

    def __jsonrepr__(self):
        # JSON Representation of this class to save it to a json file.
        return {"operational_unit": self.unit, "dimensions": self.dimensions}
