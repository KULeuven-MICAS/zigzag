import logging
from typing import Any

from zigzag.parser.UpgradedValidator import UpgradedValidator

logger = logging.getLogger(__name__)


class MappingValidator:
    """Class to validate user-given mappings from yaml file"""

    # Schema for a single operation, UpgradeValidator extrapolates to list of operations
    SCHEMA_SINGLE = {
        "name": {"type": "string", "required": True},
        "core_allocation": {
            "type": "list",
            "schema": {"type": "integer"},
            "default": [0],
        },
        "core_allocation_is_fixed": {"type": "boolean", "default": False},
        "spatial_mapping": {
            "type": "dict",
            "schema": {
                "D1": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+, [0-9]+$"},
                    "required": False,
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
            "nullable": True,
        },
        "memory_operand_links": {
            "type": "dict",
            "schema": {
                "O": {"type": "string", "required": True},
                "W": {"type": "string", "required": True},
                "I": {"type": "string", "required": True},
            },
            "default": {"O": "O", "I": "I1", "W": "I2"},
        },
        "spatial_mapping_hint": {
            "type": "dict",
            "schema": {
                "D1": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+$"},
                    "required": False,
                },
                "D2": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+$"},
                    "required": False,
                },
                "D3": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+$"},
                    "required": False,
                },
                "D4": {
                    "type": "list",
                    "schema": {"type": "string", "regex": r"^[A-Z]+$"},
                    "required": False,
                },
            },
            "required": False,
        },
        "temporal_ordering": {
            "type": "list",
            "schema": {
                "type": "list",
                "items": [{"type": "string"}, {"oneof": [{"type": "integer"}, {"type": "string", "allowed": ["*"]}]}],
                "minlength": 2,
                "maxlength": 2,
            },
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = UpgradedValidator(is_array=True)
        self.schema = MappingValidator.SCHEMA_SINGLE
        self.data: list[dict[str, Any]] = self.validator.normalize_list(data, schema=self.schema)
        self.is_valid = True

    @property
    def normalized_data(self):
        """! Return normalized, user-provided data."""
        # Can only be called after __init__, where data is automatically normalized
        return self.data

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined mapping is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data, schema=self.schema)
        errors = self.validator.errors
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        # Extra checks
        if "default" not in map(lambda x: x["name"], self.data):
            self.invalidate("No default mapping defined.")

        for mapping_data in self.data:
            self.validate_single_mapping(mapping_data)

        return self.is_valid

    def validate_single_mapping(self, layer_data: dict[str, Any]) -> None:
        """
        # TODO check that there are no OADimensions that are not defined in the architecture
        """
        pass
