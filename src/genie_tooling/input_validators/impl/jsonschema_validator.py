"""JSONSchemaInputValidator: Validates input against a JSON Schema."""
import logging
from typing import Any, Dict, List, Optional

# Updated import paths for InputValidator and InputValidationException
from genie_tooling.input_validators.abc import (
    InputValidationException,
    InputValidator,
)

try:
    import jsonschema
    from jsonschema import validators
    from jsonschema.exceptions import SchemaError as JSONSchemaSchemaError
    from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
except ImportError:
    jsonschema = None
    validators = None
    JSONSchemaValidationError = None # type: ignore
    JSONSchemaSchemaError = None # type: ignore

logger = logging.getLogger(__name__)

class JSONSchemaInputValidator(InputValidator):
    """Validates input against a JSON Schema using the jsonschema library."""
    plugin_id: str = "jsonschema_input_validator_v1"
    description: str = "Validates input parameters against a JSON Schema definition."

    def __init__(self):
        self._jsonschema_available = False
        if jsonschema and validators and JSONSchemaValidationError and JSONSchemaSchemaError:
            self._jsonschema_available = True
            self._validator_class = jsonschema.Draft7Validator
            logger.debug("JSONSchemaInputValidator initialized with jsonschema library.")
        else:
            logger.warning(
                "JSONSchemaInputValidator: 'jsonschema' library not found or specific exceptions not available. Validation will be skipped. "
                "Install it with: poetry install --extras validation"
            )

    def validate(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self._jsonschema_available:
            logger.debug("JSONSchemaInputValidator: jsonschema not available, skipping validation, returning params as is.")
            return params
        try:
            validator_instance = self._validator_class(schema)
            validation_errors = list(validator_instance.iter_errors(params))

            if validation_errors:
                error_messages = []
                detailed_errors_for_exception: List[Dict[str, Any]] = []
                for error in validation_errors:
                    path_str = " -> ".join(map(str, error.path)) if error.path else "root"
                    message_for_log = f"Error at '{path_str}': {str(error)} (validator: {error.validator}, schema value: {error.validator_value}, instance: {error.instance})"
                    error_messages.append(message_for_log)
                    detailed_errors_for_exception.append({
                        "message": str(error),
                        "path": list(error.path),
                        "validator": error.validator,
                        "validator_value": error.validator_value,
                        "instance_failed": error.instance,
                        "schema_path": list(error.schema_path),
                    })

                full_error_message_for_log = "Input validation failed with multiple errors:\n" + "\n".join(error_messages)
                logger.warning(f"JSONSchema validation errors for schema {schema.get('title', 'N/A')}:\n{full_error_message_for_log}")
                raise InputValidationException(
                    "Input validation failed.",
                    errors=detailed_errors_for_exception,
                    params=params
                )

            logger.debug("JSONSchema validation successful.")
            return params

        except (jsonschema.exceptions.UnknownType, jsonschema.exceptions.SchemaError) if jsonschema else () as e_schema_problem: # type: ignore
            error_message_from_exception = str(e_schema_problem)
            logger.error(f"Invalid JSON Schema provided (Type: {type(e_schema_problem).__name__}): {error_message_from_exception}", exc_info=False)
            schema_error_details = {"message": error_message_from_exception}
            if hasattr(e_schema_problem, "path") and e_schema_problem.path is not None:
                schema_error_details["path"] = list(e_schema_problem.path)
            raise InputValidationException(
                f"Invalid schema configuration: {error_message_from_exception}",
                errors=[schema_error_details],
                params=params
            ) from e_schema_problem
        except InputValidationException:
            raise
        except Exception as e_other:
            logger.error(f"An unexpected error occurred during JSON schema validation: {str(e_other)} (Type: {type(e_other).__name__})", exc_info=True)
            is_jsonschema_lib_error = False
            if jsonschema:
                if isinstance(e_other, jsonschema.exceptions.JSonschemaException):
                     is_jsonschema_lib_error = True
            if is_jsonschema_lib_error:
                logger.warning(f"A jsonschema library error ({type(e_other).__name__}) was caught by generic Exception block. Check specific except clauses.")
                raise InputValidationException(
                    f"Invalid schema configuration (unhandled jsonschema error type): {str(e_other)}",
                    params=params
                ) from e_other
            else:
                raise InputValidationException(f"Unexpected validation error: {str(e_other)}", params=params) from e_other

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
