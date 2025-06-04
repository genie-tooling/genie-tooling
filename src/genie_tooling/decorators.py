### src/genie_tooling/decorators.py
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
        # MODIFIED: Pass recursive_guard as a keyword argument
        return py_type._evaluate(globalns, localns, recursive_guard=frozenset()) # type: ignore

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
            if non_none_args:
                first_type_schema = _map_type_to_json_schema(non_none_args[0], is_optional=is_optional)
                return first_type_schema
            else:
                return {"type": "string"}

    schema: Dict[str, Any] = {}
    if py_type == str:
        schema = {"type": "string"}
    elif py_type == int:
        schema = {"type": "integer"}
    elif py_type == float:
        schema = {"type": "number"}
    elif py_type == bool:
        schema = {"type": "boolean"}
    elif py_type == list or origin == list or py_type == set or origin == set or py_type == tuple or origin == tuple:
        item_schema = {}
        if args and len(args) >= 1:
            if origin == tuple and len(args) > 1 and args[1] is not Ellipsis:
                 item_schema = _map_type_to_json_schema(args[0])
            else: # List[T], Set[T], Tuple[T, ...]
                item_schema = _map_type_to_json_schema(args[0])
        schema = {"type": "array", "items": item_schema or {}} # Ensure items is at least {}
    elif py_type == dict or origin == dict:
        schema = {"type": "object"}
    elif py_type is type(None):
        schema = {"type": "null"}
    elif py_type is Any:
        schema = {} # MODIFIED: Any maps to empty schema {}
    else:
        schema = {"type": "string"}
    return schema


def tool(func: Callable) -> Callable:
    """
    Decorator to mark a function as a Genie Tool and auto-generate its metadata.
    """
    globalns = getattr(func, "__globals__", {})
    try:
        type_hints = get_type_hints(func, globalns=globalns)
    except NameError as e:
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
            print(f"Warning: Could not fully resolve type hints for {func.__name__} due to {e}. Schemas might be incomplete.")

    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    main_description = docstring.split("\n\n")[0].strip()
    if not main_description and func.__name__:
        main_description = f"Executes the '{func.__name__}' tool."

    param_descriptions_from_doc = _parse_docstring_for_params(docstring)

    properties: Dict[str, Any] = {}
    required_params: List[str] = []

    for name, param in sig.parameters.items():
        if name == "self" or name == "cls" or \
           param.kind == inspect.Parameter.VAR_POSITIONAL or \
           param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        param_py_type = type_hints.get(name, Any)

        if isinstance(param_py_type, str):
            try:
                # MODIFIED: Pass recursive_guard as a keyword argument
                param_py_type = ForwardRef(param_py_type)._evaluate(globalns, {}, recursive_guard=frozenset())
            except Exception:
                 pass

        is_optional_hint = False
        origin = getattr(param_py_type, "__origin__", None)
        args = getattr(param_py_type, "__args__", None)
        if origin is Union and type(None) in (args or []):
            is_optional_hint = True
            if args:
                param_py_type = next((t for t in args if t is not type(None)), Any)

        schema_type_def = _map_type_to_json_schema(param_py_type)

        # If _map_type_to_json_schema returned {} (for Any), default to string for schema
        if not schema_type_def and param_py_type is Any: # MODIFIED
            schema_type_def = {"type": "string"}

        param_info_schema = schema_type_def
        param_info_schema["description"] = param_descriptions_from_doc.get(name, f"Parameter '{name}'.")

        if param.default is inspect.Parameter.empty:
            if not is_optional_hint:
                required_params.append(name)
        else:
            param_info_schema["default"] = param.default

        properties[name] = param_info_schema

    input_schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required_params:
        input_schema["required"] = required_params

    return_py_type = type_hints.get("return", Any)
    if isinstance(return_py_type, str):
        try:
            # MODIFIED: Pass recursive_guard as a keyword argument
            return_py_type = ForwardRef(return_py_type)._evaluate(globalns, {}, recursive_guard=frozenset())
        except Exception:
            pass

    output_schema_prop_def = _map_type_to_json_schema(return_py_type)

    output_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"result": output_schema_prop_def},
    }
    if output_schema_prop_def.get("type") != "null":
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

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    chosen_wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    chosen_wrapper._tool_metadata_ = tool_metadata
    chosen_wrapper._original_function_ = func

    return chosen_wrapper