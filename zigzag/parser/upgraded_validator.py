"""
Copyright jdotjdot (https://github.com/pyeve/cerberus/issues/220#issuecomment-205047415)
"""

import copy
from typing import Any

# using Cerberus 0.9.2
import six
from cerberus import Validator  # type: ignore


class UpgradedValidator(Validator):
    """
    Subclass of Cerberus's Validator that adds some custom types and allows for the document to be a top-level array by
    setting is_array=True
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.is_array: bool = kwargs.get("is_array", False)
        super(UpgradedValidator, self).__init__(*args, **kwargs)

    def validate(  # pylint: disable=W0237
        self,
        document: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        update: bool = False,
        context: Any | None = None,
    ) -> bool:
        # This gets confusing because this method seems to be called internally for validation as well
        # and we don't want to add "rows" to sub-schemas as well, only the
        # top-level.

        if self.is_array and not context:  # checking for "context" seems to help with not adding 'rows' to every dict
            schema = schema or self.schema  # type: ignore

            if "rows" not in schema:
                if "type" in schema:  # is a list
                    schema = {"rows": {"type": "list", "required": True, "schema": schema}}
                else:  # is a dict
                    schema = {
                        "rows": {
                            "type": "list",
                            "required": True,
                            "schema": {"type": "dict", "schema": schema},
                        }
                    }

        if "rows" not in document:  # type: ignore
            document_dict = {"rows": document}
        else:
            document_dict = document
        return super(UpgradedValidator, self).validate(document_dict, schema, update, context)  # type: ignore

    @property
    def errors(self) -> dict[str, Any]:
        errors = super(UpgradedValidator, self).errors  # type: ignore
        if self.is_array and "rows" in errors:
            return errors["rows"]  # type: ignore
        else:
            return errors  # type: ignore

    _type_defaults: dict[str, Any] = {
        "integer": 0,
        "list": [],
        "dict": {},
        "string": "",
    }

    def get_type_default(self, type_: str):
        return self._type_defaults.get(type_)

    def get_default(self, field_schema: dict[str, Any]):
        if "default" in field_schema:
            return field_schema.get("default")

        if field_schema.get("nullable", False):
            return None

        return self.get_type_default(field_schema["type"])

    def add_defaults_to_doc(self, document: dict[str, Any], doc_schema: dict[str, Any]) -> dict[str, Any]:
        new_doc: dict[str, Any] = copy.deepcopy(document)
        for field, field_schema in doc_schema.items():
            if field not in document:
                new_doc[six.u(field)] = self.get_default(field_schema)

        return new_doc

    def normalize_list(
        self, document: list[dict[str, Any]], schema: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        # Needed to write this because the .normalized() method doesn't come out until Cerberus 0.10
        #  which has not yet been released

        # This is a bit lazy and assumes a list of dicts, since that's what
        # this whole subclass was written for

        schema = schema or self.schema  # type: ignore
        schema = schema["rows"]["schema"] if "rows" in schema else schema  # type: ignore
        assert isinstance(document, (list, tuple, set))
        return [self.add_defaults_to_doc(doc, schema) for doc in document]  # type: ignore
