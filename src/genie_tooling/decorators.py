# src/genie_tooling/decorators.py
import functools
import inspect
import re
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
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
            if not line.startswith("    ") and line_stripped:
                in_args_section = False
                continue

            if ":" in line_stripped:
                param_match = re.match(r"^\s*(\w+)\s*(?:\((.*?)\))?:\s*(.*)", line_stripped)
                if param_match:
                    name, _type_info, desc = param_match.groups()
                    param_descriptions[name.strip()] = desc.strip()
    return param_descriptions


def _resolve_forward_refs(
    py_type: Any, globalns: Optional[Dict[str, Any]] = None, localns: Optional[Dict[str, Any]] = None
) -> Any:
    """Recursively resolves ForwardRef annotations."""
    if isinstance(py_type, ForwardRef):
        return py_type._evaluate(globalns, localns, recursive_guard=frozenset())  # type: ignore

    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", None)

    if origin and args:
        resolved_args = tuple(_resolve_forward_refs(arg, globalns, localns) for arg in args)
        if hasattr(py_type, "_subs_tree") and callable(getattr(py_type, "_subs_tree", None)):
            try:
                return py_type.copy_with(resolved_args)
            except Exception:
                return origin[resolved_args]  # type: ignore
        elif hasattr(origin, "__getitem__"):
            try:
                return origin[resolved_args]  # type: ignore
            except TypeError:
                if origin is Union:
                    return Union[resolved_args]  # type: ignore
        return py_type
    return py_type


def _map_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """
    Maps Python types to JSON schema type definitions.
    Note: Optionality (Union with None) is handled by the caller, which
    determines if a field is 'required'. This function just maps the core type.
    """
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if not non_none_args:
            return {"type": "null"}

        schemas = [_map_type_to_json_schema(arg) for arg in non_none_args]
        if all("type" in s and len(s) == 1 and isinstance(s["type"], str) for s in schemas):
            all_types = list(set(s["type"] for s in schemas))
            # FIX: If only one type remains after removing None, don't use a list.
            if len(all_types) == 1:
                return {"type": all_types[0]}
            return {"type": all_types}
        # For more complex unions (e.g., Union[str, MyPydanticModel]), use anyOf
        return {"anyOf": schemas}

    if py_type is type(None):
        return {"type": "null"}
    if py_type == str:
        return {"type": "string"}
    if py_type == int:
        return {"type": "integer"}
    if py_type == float:
        return {"type": "number"}
    if py_type == bool:
        return {"type": "boolean"}
    if py_type == dict or origin == dict:
        return {"type": "object"}
    if py_type is Any:
        return {}  # No constraint

    if py_type in (list, List, set, Set, tuple, Tuple) or origin in (list, List, set, Set, tuple, Tuple):
        item_schema = {}
        # For List[T], Set[T], Tuple[T, ...], get the type of T
        if args and len(args) >= 1:
            # For Tuple[T1, T2], we simplify and just take the first element type for the "items" schema
            element_type = args[0]
            item_schema = _map_type_to_json_schema(element_type)
        return {"type": "array", "items": item_schema}

    return {"type": "string"}


FRAMEWORK_INJECTED_PARAMS: Set[str] = {"context", "key_provider"}


def tool(func: Callable) -> Callable:
    """Decorator to transform a Python function into a Genie-compatible Tool."""
    globalns = getattr(func, "__globals__", {})
    try:
        type_hints = get_type_hints(func, globalns=globalns)
    except NameError as e:
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
            print(
                f"Warning: Could not fully resolve type hints for {func.__name__} due to {e}. Schemas might be incomplete."
            )

    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    main_description = docstring.split("\n\n")[0].strip() or f"Executes the '{func.__name__}' tool."
    param_descriptions_from_doc = _parse_docstring_for_params(docstring)
    properties: Dict[str, Any] = {}
    required_params: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        param_py_type_hint = type_hints.get(name, Any)
        if isinstance(param_py_type_hint, str):
            try:
                param_py_type_hint = ForwardRef(param_py_type_hint)._evaluate(  # type: ignore
                    globalns, {}, recursive_guard=frozenset()
                )
            except Exception:
                param_py_type_hint = Any

        is_optional_from_union_type = False
        actual_param_type_for_schema = param_py_type_hint
        origin = get_origin(param_py_type_hint)
        args = get_args(param_py_type_hint)

        if origin is Union and type(None) in (args or []):
            is_optional_from_union_type = True
            if args:
                non_none_args = [t for t in args if t is not type(None)]
                if len(non_none_args) == 1:
                    actual_param_type_for_schema = non_none_args[0]
                else:
                    actual_param_type_for_schema = Union[tuple(non_none_args)]  # Keep as Union of non-None

        param_schema_def = _map_type_to_json_schema(actual_param_type_for_schema)
        # --- FIX: Default to 'string' if schema is empty (from Any type) ---
        if not param_schema_def:
            param_schema_def["type"] = "string"

        param_schema_def["description"] = param_descriptions_from_doc.get(name, f"Parameter '{name}'.")

        # Handle optionality by adding "null" to the type list
        if is_optional_from_union_type:
            if "type" in param_schema_def and isinstance(param_schema_def["type"], str):
                param_schema_def["type"] = [param_schema_def["type"], "null"]
            elif "type" in param_schema_def and isinstance(param_schema_def["type"], list):
                if "null" not in param_schema_def["type"]:
                    param_schema_def["type"].append("null")

        if param.default is inspect.Parameter.empty:
            if not is_optional_from_union_type and name not in FRAMEWORK_INJECTED_PARAMS:
                required_params.append(name)
        else:
            param_schema_def["default"] = param.default

        properties[name] = param_schema_def

    input_schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required_params:
        input_schema["required"] = required_params

    return_py_type_hint = type_hints.get("return", Any)
    if isinstance(return_py_type_hint, str):
        try:
            return_py_type_hint = ForwardRef(return_py_type_hint)._evaluate(  # type: ignore
                globalns, {}, recursive_guard=frozenset()
            )
        except Exception:
            return_py_type_hint = Any

    actual_return_type_for_schema = return_py_type_hint
    ret_origin = get_origin(return_py_type_hint)
    ret_args = get_args(return_py_type_hint)
    if ret_origin is Union and type(None) in (ret_args or []):
        if ret_args:
            actual_return_type_for_schema = next((t for t in ret_args if t is not type(None)), Any)

    output_schema_prop_def = _map_type_to_json_schema(actual_return_type_for_schema)
    if not output_schema_prop_def:
        output_schema_prop_def = {"type": "object"}

    output_schema: Dict[str, Any] = {"type": "object", "properties": {"result": output_schema_prop_def}}
    if (
        output_schema_prop_def.get("type") != "null"
        and not (
            isinstance(output_schema_prop_def.get("type"), list)
            and "null" in output_schema_prop_def["type"]
            and len(output_schema_prop_def["type"]) == 1
        )
        and output_schema_prop_def != {}
    ):
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

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    wrapper._tool_metadata_ = tool_metadata  # type: ignore
    wrapper._original_function_ = func  # type: ignore
    return wrapper
