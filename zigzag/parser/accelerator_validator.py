import logging
from math import log2
from typing import Any

from cerberus import Validator  # type: ignore

logger = logging.getLogger(__name__)


class AcceleratorValidator:
    """Validates a single Zigzag accelerator from a user-provided yaml file. Checks if the entries of the yaml file
    are valid and replace unspecified values with defaults."""

    OPERAND_REGEX = r"^I[12]$|^O$"
    DIMENSION_REGEX = r"^D\d$"
    PORT_REGEX = r"^[r]?[w]?_port_\d+$"

    # Intermediate output operand. Hard coded, and must be specified by the user as such
    OUTPUT_OPERAND_STR = "O"
    # Final output operand after scaling. Hard coded, and must be specified by the user as such
    FINAL_OUTPUT_OPERAND_STR = "O_final"
    MEM_OP_1_STR = "I1"
    MEM_OP_2_STR = "I2"

    SCHEMA = {
        "name": {"type": "string", "required": True},
        "memories": {
            "type": "dict",
            "required": True,
            "valuesrules": {
                "type": "dict",
                "schema": {
                    "size": {"type": "integer", "required": True},
                    "r_bw": {"type": "integer", "required": True},
                    "w_bw": {"type": "integer", "required": True},
                    "r_cost": {
                        "type": "float",
                        "required": False,
                        "nullable": True,
                        "default": None,
                    },
                    "w_cost": {
                        "type": "float",
                        "required": False,
                        "nullable": True,
                        "default": None,
                    },
                    "area": {
                        "type": "float",
                        "required": False,
                        "nullable": True,
                        "default": None,
                    },
                    "r_port": {"type": "integer", "required": True},
                    "w_port": {"type": "integer", "required": True},
                    "rw_port": {"type": "integer", "required": True},
                    "latency": {"type": "integer", "required": True},
                    "min_r_granularity": {
                        "type": "integer",
                        "required": False,
                        "nullable": True,
                        "default": None,
                    },
                    "min_w_granularity": {
                        "type": "integer",
                        "required": False,
                        "nullable": True,
                        "default": None,
                    },
                    "mem_type": {
                        "type": "string",
                        "required": False,
                        "default": "sram",
                    },
                    "auto_cost_extraction": {"type": "boolean", "default": False},
                    "operands": {
                        "type": "list",
                        "required": True,
                        "schema": {"type": "string", "regex": OPERAND_REGEX},
                    },
                    "ports": {
                        "type": "list",
                        "required": True,
                        "schema": {
                            "type": "dict",
                            "schema": {
                                "fh": {
                                    "type": "string",
                                    "required": False,
                                    "regex": PORT_REGEX,
                                },
                                "tl": {
                                    "type": "string",
                                    "required": False,
                                    "regex": PORT_REGEX,
                                },
                                "fl": {
                                    "type": "string",
                                    "required": False,
                                    "regex": PORT_REGEX,
                                },
                                "th": {
                                    "type": "string",
                                    "required": False,
                                    "regex": PORT_REGEX,
                                },
                            },
                        },
                    },
                    "served_dimensions": {
                        "type": "list",
                        "required": True,
                        "schema": {"type": "string", "regex": DIMENSION_REGEX},
                    },
                },
            },
        },
        "operational_array": {
            "type": "dict",
            "required": True,
            "schema": {
                # Shared for regular and IMC
                "dimensions": {
                    "type": "list",
                    "required": True,
                    "schema": {"type": "string", "regex": DIMENSION_REGEX},
                },
                "is_imc": {"type": "boolean", "default": False},
                "sizes": {
                    "type": "list",
                    "required": True,
                    "schema": {"type": "integer", "min": 0},
                },
                "input_precision": {
                    "type": "list",
                    "required": False,
                    "schema": {"type": "integer"},
                    "minlength": 2,
                    "maxlength": 2,
                },
                # Non-IMC properties
                "multiplier_energy": {"type": "float", "required": False},
                "multiplier_area": {"type": "float", "required": False},
                # IMC properties
                "imc_type": {
                    "type": "string",
                    "allowed": ["analog", "digital"],
                    "nullable": True,
                    "default": None,
                },
                "adc_resolution": {
                    "type": "float",
                    "required": False,
                    "nullable": True,
                    "default": 0,
                },
                "bit_serial_precision": {
                    "type": "float",
                    "required": True,
                    "nullable": True,
                    "default": None,
                },
            },
        },
        "dataflows": {
            "type": "dict",
            "schema": {
                "D1": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"},
                    "required": True,
                },
                "D2": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"},
                    "required": False,
                },
                "D3": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"},
                    "required": False,
                },
                "D4": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"},
                    "required": False,
                },
            },
            "required": False,
            "nullable": False,
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = Validator()
        self.validator.schema = AcceleratorValidator.SCHEMA  # type: ignore
        self.data: dict[str, Any] = self.validator.normalized(data)  # type: ignore
        self.is_valid = True

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined accelerator is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data)  # type: ignore
        errors = self.validator.errors  # type: ignore
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        # Extra validation rules outside of schema
        self.is_imc = self.data["operational_array"]["is_imc"]
        self.validate_operational_array()

        for mem_name in self.data["memories"]:
            self.validate_single_memory(mem_name)

        if self.is_imc:
            self.validate_cells_imc()

        return self.is_valid

    def validate_single_memory(self, mem_name: str) -> None:
        mem_data: dict[str, Any] = self.data["memories"][mem_name]
        expected_oa_dims: list[str] = self.data["operational_array"]["dimensions"]

        # Auto-cost extraction using CACTI
        if mem_data["auto_cost_extraction"]:
            if mem_data["size"] % 8 != 0:
                self.invalidate(
                    f"Memory size of {mem_name} must be a multiple of 8 when automatically extracting "
                    f"costs using CACTI."
                )
        else:
            if mem_data["r_cost"] is None:
                self.invalidate(f"`r_cost` of {mem_name} is missing, and is not automatically extracted using CACTI.")
            if mem_data["w_cost"] is None:
                self.invalidate(f"`w_cost` of {mem_name} is missing, and is not automatically extracted using CACTI.")
            if mem_data["area"] is None:
                self.invalidate(f"`area` of {mem_name} is missing, and is not automatically extracted using CACTI.")

        # Number of port allocations is consistent with memory operands
        nb_operands = len(mem_data["operands"])
        nb_ports = len(mem_data["ports"])
        if nb_ports != nb_operands:
            self.invalidate(
                f"Number of memory ports ({nb_ports}) does not equal number of operands ({nb_operands}) for {mem_name}"
            )

        # No unexpected served dimensions
        for served_dimension in mem_data["served_dimensions"]:
            if served_dimension not in expected_oa_dims:
                self.invalidate(f"Invalid served dimension {served_dimension} in memory {mem_name}")

        # Number of allocated ports per type equals given number of ports
        port_data: list[dict[str, str]] = mem_data["ports"]
        r_ports: set[str] = set()
        w_ports: set[str] = set()
        rw_ports: set[str] = set()
        for port_dict in port_data:
            for port_name in port_dict.values():
                match port_name[0:2]:
                    case "r_":
                        r_ports.add(port_name)
                    case "w_":
                        w_ports.add(port_name)
                    case "rw":
                        rw_ports.add(port_name)
                    case _:
                        raise ValueError("Invalid port name")
        if len(r_ports) != mem_data["r_port"]:
            self.invalidate(
                f"Number of given read ports ({mem_data['r_port']}) does not equal number of allocated read ports "
                f"({len(r_ports)}) for {mem_name}"
            )
        if len(w_ports) != mem_data["w_port"]:
            self.invalidate(
                f"Number of given write ports ({mem_data['w_port']}) does not equal number of allocated write ports "
                f"({len(w_ports)}) for {mem_name}"
            )
        if len(rw_ports) != mem_data["rw_port"]:
            self.invalidate(
                f"Number of given read/write ports ({mem_data['rw_port']}) does not equal number of allocated "
                f"read/write ports ({len(rw_ports)}) for {mem_name}"
            )

        # Direction of ports is valid
        for port_dict in port_data:
            for direction, port_name in port_dict.items():
                if (direction == "fh" or direction == "fl") and (port_name.startswith("r_")):
                    self.invalidate(f"Read port given for write direction in {mem_name}")
                if (direction == "th" or direction == "tl") and (port_name.startswith("w_")):
                    self.invalidate(f"Write port given for read direction in {mem_name}")

    def validate_cells_imc(self):
        if "cells" not in self.data["memories"]:
            self.invalidate("IMC architecture has no memory level called `cells`.")

        cells_data = self.data["memories"]["cells"]
        # Lowest memory level should only serve weight
        if cells_data["operands"] != ["I2"]:
            self.invalidate("IMC cells can only serve weights. Set `operands` to `I2`.")
        # Served dimension should be empty;
        if cells_data["served_dimensions"] != []:
            self.invalidate("IMC cells must be fully unrolled. Set `served_dimensions` to `[]`")
        # Memory size should be a multiply (e.g. 1,2,..) of weight precision.
        if cells_data["size"] % self.data["operational_array"]["input_precision"][1] != 0:
            self.invalidate("IMC cells' size must be a multiply of the weight precision")

    def validate_operational_array(self):
        multiplier_data = self.data["operational_array"]

        # For both IMC and non-IMC:
        oa_dims: list[str] = multiplier_data["dimensions"]
        if len(oa_dims) != len(multiplier_data["sizes"]):
            # TODO Alternatively, you can force the user to not
            self.invalidate("Core dimensions and sizes do not match.")

        if self.is_imc:
            self.validate_operational_array_imc()
        else:
            self.validate_operational_array_non_imc()

    def validate_operational_array_imc(self):
        """Assumes that the multiplier type is IMC"""
        # All previous IMC checks are now part of the schema
        imc_data = self.data["operational_array"]
        if imc_data["bit_serial_precision"] > imc_data["input_precision"][0]:
            self.invalidate("Bit serial input precision for IMC are bigger than activation precision.")
        if log2(imc_data["bit_serial_precision"]) % 1 != 0:
            self.invalidate("Bit serial input precision is not in the power of 2.")
        if log2(imc_data["input_precision"][0]) % 1 != 0:
            self.invalidate("Activation precision is not in the power of 2.")
        if log2(imc_data["input_precision"][1]) % 1 != 0:
            self.invalidate("Weight precision is not in the power of 2.")
        if imc_data["imc_type"] == "digital" and imc_data["adc_resolution"] != 0:
            self.invalidate("Digital IMC core but 'adc_resolution' is defined")
        if "input_precision" not in imc_data:
            self.invalidate("IMC core requires definition of 'input_precision'")

    def validate_operational_array_non_imc(self):
        """Assumes that the multiplier type is not IMC"""
        multiplier_data = self.data["operational_array"]
        # All IMC related properties should be None
        if multiplier_data["imc_type"] is not None or multiplier_data["imc_type"] is True:
            self.invalidate("Multiplier are non-IMC but `imc_type` is defined as True")
        if multiplier_data["adc_resolution"] != 0:
            self.invalidate("Multiplier are non-IMC but `adc_resolution` is defined")
        if multiplier_data["bit_serial_precision"] is not None:
            self.invalidate("Multiplier are non-IMC but `bit_serial_precision` is defined")

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data
