import logging as _logging
import os
import sys

sys.path.insert(0, os.getcwd())
from zigzag.hardware.architecture.imc_array import ImcArray
from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

hardware = "lab4/imc_macro.yaml"

accelerator = AcceleratorParserStage.parse_accelerator(hardware)
core = accelerator
imc = core.operational_array
assert isinstance(imc, ImcArray)
peak_energy_breakdown = imc.get_peak_energy_single_cycle()
peak_energy = sum([v for v in imc.get_peak_energy_single_cycle().values()])

logger.info("Total IMC area (mm^2): %s", round(imc.area, 4))
logger.info("area breakdown: %s", imc.area_breakdown)
logger.info("Tclk (ns): %s", round(imc.tclk, 4))
logger.info("Tclk breakdown (ns): %s", imc.tclk_breakdown)
logger.info("Peak energy (pJ/cycle): %s", round(peak_energy, 4))
logger.info("Peak energy breakdown (pJ/cycle): %s", peak_energy_breakdown)
exit()
