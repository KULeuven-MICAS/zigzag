import logging
from typing import Any
from cerberus import Validator


logger = logging.getLogger(__name__)


class AcceleratorValidator:
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
                    "r_cost": {"type": "float", "required": True},
                    "w_cost": {"type": "float", "required": True},
                    "area": {"type": "float", "required": True},
                    "r_port": {"type": "integer", "required": True},
                    "w_port": {"type": "integer", "required": True},
                    "rw_port": {"type": "integer", "required": True},
                    "latency": {"type": "integer", "required": True},
                    "min_r_granularity": {"type": "integer", "required": False, "nullable": True, "default": None},
                    "min_w_granularity": {"type": "integer", "required": False, "nullable": True, "default": None},
                    "mem_type": {"type": "string", "required": False, "default": "sram"},
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
                                "fh": {"type": "string", "required": False, "regex": PORT_REGEX},
                                "tl": {"type": "string", "required": False, "regex": PORT_REGEX},
                                "fl": {"type": "string", "required": False, "regex": PORT_REGEX},
                                "th": {"type": "string", "required": False, "regex": PORT_REGEX},
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
        "multipliers": {
            "type": "dict",
            "required": True,
            "schema": {
                "input_precision": {
                    "type": "list",
                    "required": True,
                    "schema": {"type": "integer"},
                    "minlength": 2,
                    "maxlength": 2,
                },
                "multiplier_energy": {"type": "float", "required": True},
                "multiplier_area": {"type": "float", "required": True},
                "dimensions": {
                    "type": "list",
                    "required": True,
                    "schema": {"type": "string", "regex": DIMENSION_REGEX},
                },
                "sizes": {"type": "list", "required": True, "schema": {"type": "integer", "min": 0}},
            },
        },
        "dataflows": {
            "type": "dict",
            "schema": {
                "D1": {"type": "list", "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"}, "required": False},
                "D2": {"type": "list", "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"}, "required": False},
                "D3": {"type": "list", "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"}, "required": False},
                "D4": {"type": "list", "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"}, "required": False},
            },
            "required": False,
            "nullable": False,
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = Validator()
        self.validator.schema = AcceleratorValidator.SCHEMA
        self.data: dict[str, Any] = self.validator.normalized(data)
        self.is_valid = True

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined accelerator is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data)
        errors = self.validator.errors
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        # Extra validation rules outside of schema

        # Dimension sizes are consistent
        oa_dims: list[str] = self.data["multipliers"]["dimensions"]
        if len(oa_dims) != len(self.data["multipliers"]["sizes"]):
            self.invalidate("Multiplier dimensions and sizes do not match.")

        for mem_name in self.data["memories"]:
            self.validate_single_memory(mem_name, oa_dims)

        return self.is_valid

    def validate_single_memory(self, mem_name: str, expected_oa_dims: list[str]) -> None:
        mem_data: dict[str, Any] = self.data["memories"][mem_name]

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

        # # Contains output operand - This is not required
        # if AcceleratorValidator.OUTPUT_OPERAND_STR not in mem_data["operands"]:
        #     self.invalidate(f"{mem_name} does not contain output operand `{AcceleratorValidator.OUTPUT_OPERAND_STR}`")

    @property
    def normalized_data(self) -> dict[str, Any]:
        """Returns the user-provided data after normalization by the validator. (Normalization happens during
        initialization)"""
        return self.data
