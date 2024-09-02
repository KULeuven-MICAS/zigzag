import logging
from typing import Any

# from cerberus import Validator
from zigzag.parser.upgraded_validator import UpgradedValidator

logger = logging.getLogger(__name__)


class WorkloadValidator:
    """Class to validate user-defined workloads from yaml files according to the given schema and rules"""

    EQUATION_REGEX = r"^O(\[\w+\])+\+?=\w(\[\w+\])+[*+]\w(\[(?:\w+)?\])+$"
    LAYER_DIM_RELATION_REGEX = r"^(\w+)\s*=\s*(?:(\w+)\s*\*\s*)?(\w+)\s*\+\s*(?:(\w+)\s*\*\s*)?(\w+)$"
    ALLOWED_OPERATORS: list[str] = [
        "Max",
        "Conv",
        "Pooling",
        "Add",
        "Conv_downsample",
        "Gemm",
        "Pool",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
    ]

    # Schema for a single layer, UpgradeValidator extrapolates to list of layers
    LAYER_SCHEMA: dict[str, Any] = {
        "id": {"type": "integer", "required": True},
        "name": {"type": "string", "nullable": True, "default": None},
        "operator_type": {
            "type": "string",
            "allowed": ALLOWED_OPERATORS,
            "required": True,
        },
        "equation": {"type": "string", "required": True, "regex": EQUATION_REGEX},
        "dimension_relations": {
            "type": "list",
            "schema": {"type": "string", "regex": LAYER_DIM_RELATION_REGEX},
            "required": False,
        },
        "loop_dims": {"type": "list", "schema": {"type": "string"}, "required": True},
        "loop_sizes": {"type": "list", "schema": {"type": "integer"}, "required": True},
        "operand_precision": {
            "type": "dict",
            "required": True,
            "schema": {
                "I": {"type": "integer", "required": False},
                "W": {"type": "integer", "required": False},
                "O": {"type": "integer", "required": True},
                "O_final": {"type": "integer", "required": True},
            },
        },
        "operand_source": {
            "type": "dict",
            "required": False,
            "schema": {
                "W": {"type": "integer", "required": False},
                "I": {"type": "integer", "required": False},
            },
        },
        "pr_loop_dims": {
            "type": "list",
            "schema": {"type": "string"},
            "required": False,
            "nullable": True,
            "default": None,
        },
        "pr_loop_sizes": {
            "type": "list",
            "schema": {"type": "integer"},
            "required": False,
            "nullable": True,
            "default": None,
        },
        "padding": {
            "type": "list",
            "schema": {
                "type": "list",
                "schema": {"type": "integer"},
                "default": [0, 0],
                "minlength": 2,
                "maxlength": 2,
            },
            "required": False,
            "nullable": True,
            "default": None,
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalized user-given data"""
        self.validator = UpgradedValidator(is_array=True)
        self.schema = WorkloadValidator.LAYER_SCHEMA
        self.data: list[dict[str, Any]] = self.validator.normalize_list(data, schema=self.schema)

        self.is_valid = True

    @property
    def normalized_data(self):
        """! Return normalized, user-provided data."""
        # Can only be called after __init__, where data is automatically normalized
        return self.data

    def __invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined workload is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data, schema=self.schema)
        errors = self.validator.errors
        if not validate_success:
            self.__invalidate(f"The following restrictions apply: {errors}")

        for layer_data in self.data:
            self.__validate_single_layer(layer_data)

        return self.is_valid

    def __validate_single_layer(self, layer_data: dict[str, Any]) -> None:
        """Run extra checks on a single layer"""

        # Check PR loop dims
        if "padding" in layer_data and layer_data["padding"] is not None:
            if "pr_loop_dims" not in layer_data:
                self.__invalidate("Padding defined, but no corresponding PR loop dimensions")
            elif len(layer_data["padding"]) != len(layer_data["pr_loop_dims"]):
                self.__invalidate("Number of PR loop dimensions not equal to number of corresponding paddings")

        if "pr_loop_sizes" in layer_data and layer_data["pr_loop_sizes"] is not None:
            if "pr_loop_dims" not in layer_data:
                self.__invalidate("PR loop sizes defined, but no corresponding PR loop dimensions")
            elif len(layer_data["pr_loop_sizes"]) != len(layer_data["pr_loop_dims"]):
                self.__invalidate("Number of PR loop dimensions not equal to number of corresponding sizes")
