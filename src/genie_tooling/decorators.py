import inspect
import re
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Optional,
    Union,
    get_type_hints,
)


# A simple docstring parser (can be made more robust)
def _parse_docstring_for_params(docstring: Optional[str]) -> Dict[str, str]:
    """Parses Google-style docstrings for parameter descriptions."""
    param_descriptions: Dict[str, str] = {}
    if not docstring:
        return param_descriptions

    lines = docstring.splitlines()
    in_args_section = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower().startswith(("args:", "arguments:", "parameters:")):
            in_args_section = True
            continue
        if in_args_section:
            if not line.startswith("    ") and line_stripped: # Heuristic: if line is not indented and not empty, probably end of Args
                in_args_section = False
                continue # Stop processing if we've left the args section

            if ":" in line_stripped:
                # Regex to capture "param_name (param_type): description"
                # It handles optional type information in parentheses.
                param_match = re.match(r"^\s*(\w+)\s*(?:\((.*?)\))?:\s*(.*)", line_stripped)
                if param_match:
                    name, _type_info, desc = param_match.groups()
                    param_descriptions[name.strip()] = desc.strip()
    return param_descriptions

def _resolve_forward_refs(py_type: Any, globalns: Optional[Dict[str, Any]] = None, localns: Optional[Dict[str, Any]] = None) -> Any:
    """Recursively resolves ForwardRef annotations."""
    if isinstance(py_type, ForwardRef):
        return py_type._evaluate(globalns, localns, frozenset()) # type: ignore

    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", None)

    if origin and args:
        resolved_args = tuple(_resolve_forward_refs(arg, globalns, localns) for arg in args)
        if hasattr(py_type, "_subs_tree") and callable(getattr(py_type, "_subs_tree", None)): # For older typing e.g. Python 3.8 List
             # This is a bit of a hack for older Python versions where List[T] etc. might not re-evaluate easily.
             # For modern Python (3.9+), this might not be necessary as types are more robust.
             try:
                 return py_type.copy_with(resolved_args)
             except Exception: # Fallback if copy_with is not available or fails
                 return origin[resolved_args] # type: ignore
        elif hasattr(origin, "__getitem__"): # For types like list, dict, tuple, Union
             try:
                return origin[resolved_args] # type: ignore
             except TypeError: # Handle cases like Union not being subscriptable directly in some contexts
                if origin is Union:
                    return Union[resolved_args] # type: ignore
        return py_type # Fallback if can't reconstruct
    return py_type


def _map_type_to_json_schema(py_type: Any, is_optional: bool = False) -> Dict[str, Any]:
    """Maps Python types to JSON schema type definitions."""
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", None)

    if origin is Union: # Handles Optional[T] which is Union[T, NoneType]
        # Filter out NoneType and check if it makes the type optional
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # This was an Optional[T] or Union[T, None]
            # Recurse with the non-None type and mark as optional
            return _map_type_to_json_schema(non_none_args[0], is_optional=True)
        else:
            # This is a more complex Union, e.g., Union[int, str]
            # JSON schema 'type' can be an array of types
            # Or use 'anyOf' for more complex union structures
            # For simplicity here, we'll map to a generic type or the first one
            # A robust solution would use "anyOf" with schemas for each type in non_none_args
            if non_none_args:
                # Try to map the first non-None type
                first_type_schema = _map_type_to_json_schema(non_none_args[0], is_optional=is_optional)
                # If it was an Optional that became a Union, it's still effectively optional
                # The 'is_optional' flag is more about whether it's required in the parent object
                return first_type_schema
            else: # Should not happen if Union had types other than None
                return {"type": "string"} # Fallback

    schema: Dict[str, Any] = {}
    if py_type == str: schema = {"type": "string"}
    elif py_type == int: schema = {"type": "integer"}
    elif py_type == float: schema = {"type": "number"}
    elif py_type == bool: schema = {"type": "boolean"}
    elif py_type == list or origin == list:
        item_schema = {}
        if args and len(args) == 1: # For List[T]
            item_schema = _map_type_to_json_schema(args[0])
        schema = {"type": "array", "items": item_schema or {}} # Default to empty schema for items if not determinable
    elif py_type == dict or origin == dict:
        # For Dict[K, V], JSON schema usually represents this as an object
        # with properties, or uses patternProperties/additionalProperties.
        # Simple mapping here:
        schema = {"type": "object"}
        # A more complex mapping could inspect Dict args for K,V if needed
    elif py_type is type(None): schema = {"type": "null"}
    elif py_type is Any: schema = {} # Any type, no specific schema constraint
    else: schema = {"type": "string"} # Default for unknown types

    # The 'is_optional' flag from parameter analysis (default value or Optional type hint)
    # primarily influences the 'required' list, not usually the 'type' itself unless
    # we want to add "null" to the type array, e.g. {"type": ["string", "null"]}.
    # For simplicity, we let the 'required' list handle optionality.
    return schema


def tool(func: Callable) -> Callable:
    """
    Decorator to mark a function as a Genie Tool and auto-generate its metadata.
    """
    # Resolve ForwardRefs in type hints
    globalns = getattr(func, "__globals__", {})
    try:
        type_hints = get_type_hints(func, globalns=globalns) # type: ignore
    except NameError as e:
        # This can happen if a type hint refers to a name not yet defined
        # and not resolvable through ForwardRef evaluation in this context.
        # For robust ForwardRef resolution, the module defining the types might need to be fully loaded.
        # Fallback: try to get hints without full resolution, might miss some.
        try:
            type_hints = get_type_hints(func)
        except Exception: # Catch any error during get_type_hints
            type_hints = {} # Fallback to empty if still failing
            print(f"Warning: Could not fully resolve type hints for {func.__name__} due to {e}. Schemas might be incomplete.")


    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    main_description = docstring.split("\n\n")[0].strip()
    if not main_description and func.__name__: # Fallback description
        main_description = f"Executes the '{func.__name__}' tool."

    param_descriptions_from_doc = _parse_docstring_for_params(docstring)

    properties: Dict[str, Any] = {}
    required_params: List[str] = []

    for name, param in sig.parameters.items():
        if name == "self" or name == "cls": continue

        param_py_type = type_hints.get(name, Any)

        # Resolve ForwardRef if param_py_type is a string (common for forward refs)
        if isinstance(param_py_type, str):
            try:
                param_py_type = ForwardRef(param_py_type)._evaluate(globalns, {}, frozenset()) # type: ignore
            except Exception: # Fallback if evaluation fails
                 pass # Keep as string, _map_type_to_json_schema might handle it or default

        is_optional_hint = False
        origin = getattr(param_py_type, "__origin__", None)
        args = getattr(param_py_type, "__args__", None)
        if origin is Union and type(None) in (args or []):
            is_optional_hint = True
            # Get the non-None type from Optional[T]
            if args:
                param_py_type = next((t for t in args if t is not type(None)), Any)


        schema_type_def = _map_type_to_json_schema(param_py_type)

        param_info_schema = schema_type_def
        param_info_schema["description"] = param_descriptions_from_doc.get(name, f"Parameter '{name}'.")

        if param.default is inspect.Parameter.empty:
            if not is_optional_hint: # Only add to required if no default AND not Optional[T]
                required_params.append(name)
        else:
            param_info_schema["default"] = param.default

        properties[name] = param_info_schema

    input_schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required_params:
        input_schema["required"] = required_params

    return_py_type = type_hints.get("return", Any)
    if isinstance(return_py_type, str): # Resolve forward ref for return type
        try:
            return_py_type = ForwardRef(return_py_type)._evaluate(globalns, {}, frozenset()) # type: ignore
        except Exception:
            pass # Keep as string if resolution fails

    output_schema_prop_def = _map_type_to_json_schema(return_py_type)

    # Standardize output to be an object with a 'result' property
    output_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"result": output_schema_prop_def},
    }
    if output_schema_prop_def.get("type") != "null": # If return is not None, result is required
         output_schema["required"] = ["result"]


    tool_metadata = {
        "identifier": func.__name__,
        "name": func.__name__.replace("_", " ").title(),
        "description_human": main_description,
        "description_llm": main_description,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "key_requirements": [],
        "tags": ["decorated_tool"],
        "version": "1.0.0",
        "cacheable": False,
    }

    # Attach metadata to the function object that will be wrapped
    # The FunctionToolWrapper will use this.

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    chosen_wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    chosen_wrapper._tool_metadata_ = tool_metadata
    chosen_wrapper._original_function_ = func # Store original for FunctionToolWrapper

    return chosen_wrapper
