import os
import sys
import logging as _logging
sys.path.insert(0, os.getcwd())
from zigzag.hardware.architecture.ImcArray import ImcArray
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

hardware = "lab4/imc_macro.yaml"

accelerator = AcceleratorParserStage.parse_accelerator(hardware)
core = accelerator.get_core(1)
imc = core.operational_array
assert isinstance(imc, ImcArray)

logger.info("Total IMC area (mm^2): %s", round(imc.area, 4))
logger.info("area breakdown: %s", imc.area_breakdown)
logger.info("Tclk (ns): %s", round(imc.tclk, 4))
logger.info("Tclk breakdown (ns): %s", imc.tclk_breakdown)
exit()
