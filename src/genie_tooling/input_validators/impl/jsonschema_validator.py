### src/genie_tooling/input_validators/impl/jsonschema_validator.py
import logging
from typing import Any, Dict

from genie_tooling.input_validators.abc import (
    InputValidationException,
    InputValidator,
)

try:
    import jsonschema
    from jsonschema import validators
    from jsonschema.exceptions import SchemaError as JSONSchemaSchemaError
    from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None # type: ignore
    validators = None # type: ignore
    JSONSchemaValidationError = None # type: ignore
    JSONSchemaSchemaError = None # type: ignore
    JSONSCHEMA_AVAILABLE = False

logger = logging.getLogger(__name__)

def _extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS.get("properties")
    if validate_properties is None:
        return validator_class

    def set_defaults_and_validate(validator, properties, instance, schema):
        if isinstance(instance, dict):
            for property_name, subschema in properties.items():
                if isinstance(subschema, dict) and "default" in subschema:
                    instance.setdefault(property_name, subschema["default"])
        yield from validate_properties(validator, properties, instance, schema)

    return validators.extend(validator_class, {"properties": set_defaults_and_validate})


class JSONSchemaInputValidator(InputValidator):
    plugin_id: str = "jsonschema_input_validator_v1"
    description: str = "Validates input parameters against a JSON Schema definition and fills defaults."

    _DefaultFillingValidator: Any = None

    def __init__(self):
        self._jsonschema_available = JSONSCHEMA_AVAILABLE
        if self._jsonschema_available and jsonschema and validators:
            self._DefaultFillingValidator = _extend_with_default(jsonschema.Draft7Validator)
            logger.debug(f"{self.plugin_id}: Initialized with jsonschema library and default-filling validator.")
        else:
            logger.warning(
                f"{self.plugin_id}: 'jsonschema' library not found. Validation and default filling will be skipped."
            )

    def validate(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self._jsonschema_available or not self._DefaultFillingValidator:
            logger.debug(f"{self.plugin_id}: jsonschema or default-filling validator not available, skipping validation.")
            return params

        params_to_validate = params.copy()
        logger.debug(f"{self.plugin_id}: Validating params (and filling defaults): {params_to_validate} against schema: {schema}")

        try:
            self._DefaultFillingValidator.check_schema(schema)
            validator = self._DefaultFillingValidator(schema)

            # ARCHITECTURAL FIX: Use iter_errors to collect all errors, not just the first one.
            errors = list(validator.iter_errors(params_to_validate))

            if errors:
                logger.warning(f"{self.plugin_id}: Validation failed with {len(errors)} errors.")
                detailed_errors = [
                    {
                        "message": error.message, "path": list(error.path),
                        "validator": error.validator, "validator_value": error.validator_value,
                        "instance_failed": error.instance, "schema_path": list(error.schema_path),
                    }
                    for error in errors
                ]
                raise InputValidationException(
                    "Input validation failed.",
                    errors=detailed_errors,
                    params=params
                )

            logger.debug(f"{self.plugin_id}: Validation and default filling successful. Result: {params_to_validate}")
            return params_to_validate

        except JSONSchemaSchemaError as e_schema:
            error_message = f"Invalid JSON Schema provided: {e_schema!s}"
            logger.error(f"{self.plugin_id}: {error_message}", exc_info=False)
            schema_error_details = {"message": str(e_schema)}
            if hasattr(e_schema, "path") and e_schema.path is not None:
                schema_error_details["path"] = list(e_schema.path)
            raise InputValidationException(
                f"Invalid schema configuration: {e_schema!s}",
                errors=[schema_error_details], params=params
            ) from e_schema
        except InputValidationException:
            raise
        except Exception as e_other:
            logger.error(f"{self.plugin_id}: Unexpected error during validation: {e_other}", exc_info=True)
            raise InputValidationException(f"Unexpected validation error: {e_other!s}", params=params) from e_other
