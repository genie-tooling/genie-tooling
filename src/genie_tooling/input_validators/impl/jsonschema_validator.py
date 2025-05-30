import logging
from typing import Any, Dict

from genie_tooling.input_validators.abc import (
    InputValidationException,
    InputValidator,
)

try:
    import jsonschema
    from jsonschema import validators  # Import validators for extend
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
    """
    Helper function to extend a jsonschema validator class to fill in default values.
    """
    validate_properties = validator_class.VALIDATORS.get("properties")
    if validate_properties is None: # Should not happen for standard validators
        return validator_class

    def set_defaults_and_validate(validator, properties, instance, schema):
        # Fill defaults first
        if isinstance(instance, dict): # Ensure instance is a dict before using setdefault
            for property_name, subschema in properties.items():
                if isinstance(subschema, dict) and "default" in subschema:
                    instance.setdefault(property_name, subschema["default"])

        # Then, yield from the original properties validator
        # This ensures that other validations (like type, required) still run
        # on the (potentially) modified instance.
        # The original validate_properties is a generator.
        yield from validate_properties(validator, properties, instance, schema)

    return validators.extend(validator_class, {"properties": set_defaults_and_validate})


class JSONSchemaInputValidator(InputValidator):
    plugin_id: str = "jsonschema_input_validator_v1"
    description: str = "Validates input parameters against a JSON Schema definition and fills defaults."

    _DefaultFillingValidator: Any = None

    def __init__(self):
        self._jsonschema_available = JSONSCHEMA_AVAILABLE
        if self._jsonschema_available and jsonschema and validators:
            # Extend Draft7Validator (or any other preferred base) to include default filling
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

        # The extended validator will modify 'params_to_validate' in-place.
        params_to_validate = params.copy()

        logger.debug(f"{self.plugin_id}: Validating params (and filling defaults): {params_to_validate} against schema: {schema}")

        try:
            # Check schema validity first
            self._DefaultFillingValidator.check_schema(schema)

            # jsonschema.validate will use the cls's iter_errors, which will trigger our custom logic.
            # The instance `params_to_validate` will be modified in-place by the set_defaults function.
            jsonschema.validate(instance=params_to_validate, schema=schema, cls=self._DefaultFillingValidator)

            logger.debug(f"{self.plugin_id}: Validation and default filling successful. Result: {params_to_validate}")
            return params_to_validate # Return the (potentially) modified params

        except JSONSchemaValidationError as e_val: # type: ignore
            logger.warning(f"{self.plugin_id}: Validation failed: {e_val.message}")
            detailed_error = {
                "message": e_val.message, "path": list(e_val.path),
                "validator": e_val.validator, "validator_value": e_val.validator_value,
                "instance_failed": e_val.instance, "schema_path": list(e_val.schema_path),
            }
            raise InputValidationException(
                "Input validation failed.",
                errors=[detailed_error],
                params=params # Original params in exception
            ) from e_val

        except JSONSchemaSchemaError as e_schema: # type: ignore
            error_message = f"Invalid JSON Schema provided: {str(e_schema)}"
            logger.error(f"{self.plugin_id}: {error_message}", exc_info=False)
            schema_error_details = {"message": str(e_schema)}
            if hasattr(e_schema, "path") and e_schema.path is not None:
                schema_error_details["path"] = list(e_schema.path) # type: ignore
            raise InputValidationException(
                f"Invalid schema configuration: {str(e_schema)}",
                errors=[schema_error_details], params=params
            ) from e_schema
        except InputValidationException:
            raise
        except Exception as e_other:
            logger.error(f"{self.plugin_id}: Unexpected error during validation: {e_other}", exc_info=True)
            raise InputValidationException(f"Unexpected validation error: {str(e_other)}", params=params) from e_other
