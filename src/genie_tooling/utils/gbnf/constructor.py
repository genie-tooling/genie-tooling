### src/genie_tooling/utils/gbnf/constructor.py
### src/genie_tooling/utils/gbnf/constructor.py
from __future__ import annotations

import inspect
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin, get_type_hints, Literal

from pydantic import BaseModel, Field as PydanticField # Renamed Field to avoid conflict

from .core import (
    PydanticDataType,
    format_model_and_field_name,
    map_pydantic_type_to_gbnf,
    regex_to_gbnf,
    remove_empty_lines,
)

logger = logging.getLogger(__name__)


def generate_list_rule(element_rule_name: str, list_rule_definition_name: str) -> str:
    """
    Generates a GBNF rule for a list of a given element type.
    """
    list_rule = rf'{list_rule_definition_name} ::= "[" ws ( {element_rule_name} ( ws "," ws {element_rule_name} )* ws )? "]"'
    return list_rule


def generate_gbnf_integer_rules(max_digit: Optional[int] = None, min_digit: Optional[int] = None) -> Tuple[str, List[str]]:
    """Generates GBNF rules for integers with min/max digit constraints."""
    additional_rules: List[str] = []
    rule_name_parts = ["integer-part"]
    constraints_present = False
    if max_digit is not None:
        rule_name_parts.append(f"max{max_digit}")
        constraints_present = True
    if min_digit is not None:
        rule_name_parts.append(f"min{min_digit}")
        constraints_present = True

    integer_rule_name = "-".join(rule_name_parts)
    if not constraints_present:
        return PydanticDataType.INTEGER.value, []

    rule_definition_parts: List[str] = []
    actual_min_digits = min_digit if min_digit is not None else (1 if max_digit != 0 else 0)

    if actual_min_digits > 0:
        rule_definition_parts.extend(["[0-9]"] * actual_min_digits)

    if max_digit is not None:
        num_optional_digits = max_digit - actual_min_digits
        if num_optional_digits > 0:
            rule_definition_parts.extend(["[0-9]?"] * num_optional_digits)
    elif min_digit is not None : # If max_digit is None but min_digit is present, allow more digits
        rule_definition_parts.append("[0-9]*")


    rule_definition_str = " ".join(rule_definition_parts).strip()
    if not rule_definition_str and max_digit == 0: # Handles min_digit=0, max_digit=0
        additional_rules.append(f"{integer_rule_name} ::= ")
    elif rule_definition_str:
        additional_rules.append(f"{integer_rule_name} ::= {rule_definition_str}")
    elif constraints_present:
        if not rule_definition_str:
             additional_rules.append(f"{integer_rule_name} ::= ")
    else:
        return PydanticDataType.INTEGER.value, []


    return integer_rule_name, additional_rules


def generate_gbnf_float_rules(
    max_digit: Optional[int] = None, min_digit: Optional[int] = None,
    max_precision: Optional[int] = None, min_precision: Optional[int] = None
) -> Tuple[str, List[str]]:
    """Generates GBNF rules for floats with digit and precision constraints."""
    additional_rules: List[str] = []
    constraints_parts: List[str] = ["float"]
    constraints_applied = False

    integer_part_ref = PydanticDataType.INTEGER.value
    if max_digit is not None or min_digit is not None:
        int_rule_name, int_rules_defs = generate_gbnf_integer_rules(max_digit, min_digit)
        additional_rules.extend(r for r in int_rules_defs if r not in additional_rules)
        integer_part_ref = int_rule_name
        constraints_parts.append(f"d{(max_digit if max_digit is not None else 'X')}-{(min_digit if min_digit is not None else 'X')}")
        constraints_applied = True

    fractional_part_ref = "[0-9]+"
    if max_precision is not None or min_precision is not None:
        frac_rule_name_parts = ["fractional-part"]
        frac_rule_def_parts: List[str] = []
        constraints_applied = True

        actual_min_precision = min_precision if min_precision is not None else 1

        if actual_min_precision > 0:
            frac_rule_def_parts.extend(["[0-9]"] * actual_min_precision)

        if max_precision is not None:
            frac_rule_name_parts.append(f"max{max_precision}")
            num_optional_precision = max_precision - actual_min_precision
            if num_optional_precision > 0:
                frac_rule_def_parts.extend(["[0-9]?"] * num_optional_precision)
        elif min_precision is not None:
            frac_rule_def_parts.append("[0-9]*")


        if min_precision is not None:
            frac_rule_name_parts.append(f"min{min_precision}")

        if frac_rule_def_parts:
            fractional_part_rule_name_final = "-".join(frac_rule_name_parts)
            if fractional_part_rule_name_final == "fractional-part":
                fractional_part_rule_name_final += "-default"
            rule_def_str = " ".join(frac_rule_def_parts).strip()
            new_frac_rule = f"{fractional_part_rule_name_final} ::= {rule_def_str if rule_def_str else '[0-9]+'}"
            if new_frac_rule not in additional_rules:
                 additional_rules.append(new_frac_rule)
            fractional_part_ref = fractional_part_rule_name_final
            constraints_parts.append(f"p{(max_precision if max_precision is not None else 'X')}-{(min_precision if min_precision is not None else 'X')}")


    if not constraints_applied:
        return PydanticDataType.FLOAT.value, additional_rules

    float_rule_name_final = "-".join(constraints_parts)
    float_def = f'{float_rule_name_final} ::= {integer_part_ref} "." {fractional_part_ref}'
    if float_def not in additional_rules:
        additional_rules.append(float_def)

    return float_rule_name_final, additional_rules


def get_members_structure(cls: Type[Any], rule_name: str) -> str:
    """Generates GBNF for Enum members. Ensures string members are GBNF string literals."""
    if inspect.isclass(cls) and issubclass(cls, Enum):
        enum_rule_name_formatted = format_model_and_field_name(cls.__name__)
        members = []
        for _, member in cls.__members__.items():
            if isinstance(member.value, str):
                gbnf_literal = f'"\\"{json.dumps(member.value)[1:-1]}\\""'
            else:
                gbnf_literal = json.dumps(member.value)
            members.append(gbnf_literal)
        return f"{enum_rule_name_formatted} ::= {' | '.join(members)}"
    else:
        logger.warning(f"get_members_structure called with non-Enum class {cls.__name__}. "
                       f"Its object structure should be handled by generate_gbnf_grammar. Falling back to generic object for {rule_name}.")
        return f'{rule_name} ::= object'


def generate_gbnf_rule_for_type(
    model_name_context: str,
    field_name_orig: str,
    field_type: Any,
    is_field_optional_in_model: bool,
    processed_models: Set[Type[BaseModel]],
    created_rules: Dict[str, List[str]],
    field_info: Optional[Any] = None,
) -> Tuple[str, List[str]]:
    newly_defined_rules_this_call: List[str] = []
    model_name_gbnf = format_model_and_field_name(model_name_context)
    field_name_gbnf = format_model_and_field_name(field_name_orig)

    gbnf_type_name_for_field: Optional[str] = None

    origin = get_origin(field_type)
    args = get_args(field_type)

    def add_rule_def_if_new(rule_name: str, rule_def_str: str):
        if not rule_name.strip() or not rule_def_str.strip(): return
        if rule_name not in created_rules or rule_def_str not in created_rules.get(rule_name, []):
            created_rules.setdefault(rule_name, []).append(rule_def_str)
            if rule_def_str not in newly_defined_rules_this_call:
                newly_defined_rules_this_call.append(rule_def_str)

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        type_contains_none = type(None) in args
        if not non_none_args: gbnf_type_name_for_field = PydanticDataType.NULL.value
        else:
            union_def_rule_name_base = f"{model_name_gbnf}-{field_name_gbnf}"
            member_gbnf_names: List[str] = []
            for i, member_type in enumerate(non_none_args):
                member_type_name, member_rules_new = generate_gbnf_rule_for_type(
                    model_name_context, f"{field_name_orig}_member{i}", member_type,
                    False, processed_models, created_rules, None)
                newly_defined_rules_this_call.extend(r for r in member_rules_new if r not in newly_defined_rules_this_call)
                member_gbnf_names.append(member_type_name)
            rule_definition_parts = sorted(list(set(member_gbnf_names)))
            is_optional_union = type_contains_none or is_field_optional_in_model
            if len(rule_definition_parts) == 1:
                base_member_name = rule_definition_parts[0]
                if is_optional_union:
                    union_def_rule_name = f"{union_def_rule_name_base}-optional-def"
                    rule_definition = f"{union_def_rule_name} ::= {base_member_name} | {PydanticDataType.NULL.value}"
                else:
                    gbnf_type_name_for_field = base_member_name
                    return gbnf_type_name_for_field, newly_defined_rules_this_call
            else:
                union_def_rule_name = f"{union_def_rule_name_base}-union-def"
                if is_optional_union and PydanticDataType.NULL.value not in rule_definition_parts:
                    rule_definition_parts.append(PydanticDataType.NULL.value)
                rule_definition = f"{union_def_rule_name} ::= {' | '.join(sorted(list(set(rule_definition_parts))))}"
            add_rule_def_if_new(union_def_rule_name, rule_definition)
            gbnf_type_name_for_field = union_def_rule_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif origin is list or origin is List or origin is set or origin is Set:
        element_type = args[0] if args else Any
        item_field_name_context = f"{field_name_orig}_item"
        list_rule_definition_name = f"{model_name_gbnf}-{field_name_gbnf}-list-def"

        element_gbnf_name, element_rules_new = generate_gbnf_rule_for_type(
            model_name_context, item_field_name_context, element_type,
            False, processed_models, created_rules, None)
        newly_defined_rules_this_call.extend(r for r in element_rules_new if r not in newly_defined_rules_this_call)
        list_gbnf_rule_str = generate_list_rule(element_gbnf_name, list_rule_definition_name)
        add_rule_def_if_new(list_rule_definition_name, list_gbnf_rule_str)
        gbnf_type_name_for_field = list_rule_definition_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif origin is dict or origin is Dict:
        gbnf_type_name_for_field = PydanticDataType.OBJECT.value
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif origin is Literal:
        literal_values_str = []
        for val in args:
            if isinstance(val, str):
                literal_values_str.append(f'"\\"{json.dumps(val)[1:-1]}\\""')
            else:
                literal_values_str.append(json.dumps(val))
        literal_rule_name = f"{model_name_gbnf}-{field_name_gbnf}-literal-def"
        rule_def = f"{literal_rule_name} ::= {' | '.join(literal_values_str)}"
        add_rule_def_if_new(literal_rule_name, rule_def)
        gbnf_type_name_for_field = literal_rule_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        model_as_field_type_name = format_model_and_field_name(field_type.__name__)
        if field_type not in processed_models:
            nested_model_rules_new, _ = generate_gbnf_grammar(field_type, processed_models, created_rules)
            for r_def in nested_model_rules_new:
                 if r_def not in newly_defined_rules_this_call:
                      newly_defined_rules_this_call.append(r_def)
        gbnf_type_name_for_field = model_as_field_type_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif inspect.isclass(field_type) and issubclass(field_type, Enum):
        enum_rule_name = format_model_and_field_name(field_type.__name__)
        enum_def = get_members_structure(field_type, enum_rule_name)
        add_rule_def_if_new(enum_rule_name, enum_def)
        gbnf_type_name_for_field = enum_rule_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif field_type is str and field_info:
        logger.debug(f"Field: {field_name_orig}, Type: {field_type}, FieldInfo type: {type(field_info)}, FieldInfo repr: {field_info!r}")

        regex_pattern: Optional[str] = None
        
        # Attempt 0: Inspect field_info.metadata (Pydantic V2 standard for constraints)
        _field_info_metadata = getattr(field_info, 'metadata', [])
        if isinstance(_field_info_metadata, list):
            for constraint_obj in _field_info_metadata:
                _constraint_pattern_value = getattr(constraint_obj, 'pattern', None)
                if _constraint_pattern_value is not None:
                    if isinstance(_constraint_pattern_value, str):
                        regex_pattern = _constraint_pattern_value
                    elif isinstance(_constraint_pattern_value, re.Pattern):
                        regex_pattern = _constraint_pattern_value.pattern
                    
                    if regex_pattern:
                        logger.debug(f"Pattern for '{field_name_orig}' found via FieldInfo.metadata constraint object: '{regex_pattern}'")
                        break 
        
        # Attempt 1: Direct 'pattern' attribute (Should work for Pydantic V2 FieldInfo if metadata check missed it)
        if not regex_pattern:
            _raw_pattern_value_direct = getattr(field_info, 'pattern', None)
            if isinstance(_raw_pattern_value_direct, str):
                regex_pattern = _raw_pattern_value_direct
                logger.debug(f"Pattern for '{field_name_orig}' found via direct FieldInfo.pattern: '{regex_pattern}'")
            elif isinstance(_raw_pattern_value_direct, re.Pattern):
                regex_pattern = _raw_pattern_value_direct.pattern
                logger.debug(f"Pattern for '{field_name_orig}' found via compiled FieldInfo.pattern: '{regex_pattern}'")

        # Attempt 2: 'pattern' within 'json_schema_extra' (Pydantic V2)
        if not regex_pattern:
            _json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if isinstance(_json_schema_extra, dict):
                _pattern_val_js_extra = _json_schema_extra.get("pattern")
                if isinstance(_pattern_val_js_extra, str):
                    regex_pattern = _pattern_val_js_extra
                    logger.debug(f"Pattern for '{field_name_orig}' found via json_schema_extra['pattern']: '{regex_pattern}'")
        
        # Attempt 3: Pydantic V1 compatibility
        if not regex_pattern:
            _v1_regex_attr = getattr(field_info, 'regex', None) 
            if isinstance(_v1_regex_attr, str):
                regex_pattern = _v1_regex_attr
                logger.debug(f"Pattern for '{field_name_orig}' found via direct FieldInfo.regex attribute (V1 style): '{regex_pattern}'")
            else:
                _v1_extra_dict = getattr(field_info, 'extra', None) 
                if isinstance(_v1_extra_dict, dict):
                    _pattern_from_v1_extra_pattern_key = _v1_extra_dict.get("pattern")
                    if isinstance(_pattern_from_v1_extra_pattern_key, str):
                        regex_pattern = _pattern_from_v1_extra_pattern_key
                        logger.debug(f"Pattern for '{field_name_orig}' found via extra['pattern'] (V1 style): '{regex_pattern}'")
                    else: 
                        _pattern_from_v1_extra_regex_key = _v1_extra_dict.get("regex")
                        if isinstance(_pattern_from_v1_extra_regex_key, str):
                            regex_pattern = _pattern_from_v1_extra_regex_key
                            logger.debug(f"Pattern for '{field_name_orig}' found via extra['regex'] (V1 style): '{regex_pattern}'")
        
        if not regex_pattern:
            logger.debug(f"No regex pattern ultimately found for field '{field_name_orig}'. FieldInfo: {field_info!r}")
        
        if regex_pattern:
            pattern_content_rule_name = f"{model_name_gbnf}-{field_name_gbnf}-pattern-content"
            gbnf_pattern_for_content = regex_to_gbnf(regex_pattern)
            rule_def_content = f'{pattern_content_rule_name} ::= {gbnf_pattern_for_content}'
            add_rule_def_if_new(pattern_content_rule_name, rule_def_content)

            pattern_field_string_rule_name = f"{model_name_gbnf}-{field_name_gbnf}-string-def"
            str_rule_def = rf'{pattern_field_string_rule_name} ::= "\"" {pattern_content_rule_name} "\"" ws'
            add_rule_def_if_new(pattern_field_string_rule_name, str_rule_def)
            gbnf_type_name_for_field = pattern_field_string_rule_name
            return gbnf_type_name_for_field, newly_defined_rules_this_call

        _json_schema_extra_for_special_strings = getattr(field_info, "json_schema_extra", None)
        if isinstance(_json_schema_extra_for_special_strings, dict):
            if _json_schema_extra_for_special_strings.get("triple_quoted_string"):
                gbnf_type_name_for_field = PydanticDataType.TRIPLE_QUOTED_STRING.value
                return gbnf_type_name_for_field, newly_defined_rules_this_call
            elif _json_schema_extra_for_special_strings.get("markdown_code_block"):
                gbnf_type_name_for_field = PydanticDataType.MARKDOWN_CODE_BLOCK.value
                return gbnf_type_name_for_field, newly_defined_rules_this_call
        
        gbnf_type_name_for_field = PydanticDataType.STRING.value
        return gbnf_type_name_for_field, newly_defined_rules_this_call


    elif field_type is int and field_info and hasattr(field_info, "json_schema_extra") and isinstance(field_info.json_schema_extra, dict):
        rule_name, rules_defs_new = generate_gbnf_integer_rules(
            field_info.json_schema_extra.get("max_digit"), field_info.json_schema_extra.get("min_digit")
        )
        if rules_defs_new:
            for r_def in rules_defs_new: add_rule_def_if_new(rule_name, r_def)
        gbnf_type_name_for_field = rule_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    elif field_type is float and field_info and hasattr(field_info, "json_schema_extra") and isinstance(field_info.json_schema_extra, dict):
        rule_name, rules_defs_new = generate_gbnf_float_rules(
            field_info.json_schema_extra.get("max_digit"), field_info.json_schema_extra.get("min_digit"),
            field_info.json_schema_extra.get("max_precision"), field_info.json_schema_extra.get("min_precision")
        )
        if rules_defs_new:
            for r_def in rules_defs_new:
                 def_name_part = r_def.split(" ::= ")[0].strip()
                 add_rule_def_if_new(def_name_part, r_def)
        gbnf_type_name_for_field = rule_name
        return gbnf_type_name_for_field, newly_defined_rules_this_call

    if gbnf_type_name_for_field is None: # If not set by any specific handler above
        gbnf_type_name_for_field = map_pydantic_type_to_gbnf(field_type)
        if gbnf_type_name_for_field == "unknown" and field_type is not Any:
            logger.warning(f"Unknown type {field_type} mapping to GBNF for field '{field_name_orig}'. Defaulting to 'string'.")
            gbnf_type_name_for_field = PydanticDataType.STRING.value
    return gbnf_type_name_for_field, newly_defined_rules_this_call


def generate_gbnf_grammar(
    model_type: Type[BaseModel],
    processed_models: Set[Type[BaseModel]],
    created_rules: Dict[str, List[str]],
) -> Tuple[List[str], bool]:
    if not (inspect.isclass(model_type) and issubclass(model_type, BaseModel)):
        if not inspect.isclass(model_type):
            logger.debug(f"generate_gbnf_grammar called with non-class type: {model_type}. Skipping.")
            return [], False
        if inspect.isabstract(model_type):
            logger.debug(f"generate_gbnf_grammar called with abstract class {model_type.__name__}. Skipping.")
            return [], False
        logger.warning(f"generate_gbnf_grammar called with non-Pydantic class {model_type.__name__}. GBNF generation for generic classes is basic.")
        model_name_gbnf = format_model_and_field_name(model_type.__name__)
        if model_name_gbnf not in created_rules:
            generic_object_rule = get_members_structure(model_type, model_name_gbnf)
            created_rules.setdefault(model_name_gbnf, []).append(generic_object_rule)
            return [generic_object_rule], False
        return created_rules.get(model_name_gbnf, []), False


    if model_type in processed_models:
        model_name_key = format_model_and_field_name(model_type.__name__)
        main_model_defs = created_rules.get(model_name_key, [])
        return main_model_defs, False

    processed_models.add(model_type)
    model_name_gbnf = format_model_and_field_name(model_type.__name__)
    model_fields = model_type.model_fields
    newly_defined_rules_for_this_model_pass: List[str] = []

    if hasattr(model_type, "root_field") and model_type.root_field is not None: # type: ignore
        root_field_info = model_type.root_field # type: ignore
        root_field_actual_type = get_type_hints(model_type)[root_field_info.name]
        is_optional_root = not root_field_info.is_required()

        root_gbnf_type_name, root_field_specific_new_rules = generate_gbnf_rule_for_type(
            model_type.__name__, "root", root_field_actual_type, is_optional_root,
            processed_models, created_rules, root_field_info
        )
        newly_defined_rules_for_this_model_pass.extend(r for r in root_field_specific_new_rules if r not in newly_defined_rules_for_this_model_pass)

        root_model_definition_str = f"{model_name_gbnf} ::= {root_gbnf_type_name}"
        if model_name_gbnf not in created_rules or root_model_definition_str not in created_rules.get(model_name_gbnf, []):
            created_rules.setdefault(model_name_gbnf, []).append(root_model_definition_str)
            if root_model_definition_str not in newly_defined_rules_for_this_model_pass:
                newly_defined_rules_for_this_model_pass.append(root_model_definition_str)
        has_special_string_for_root = root_gbnf_type_name in [
            PydanticDataType.TRIPLE_QUOTED_STRING.value, PydanticDataType.MARKDOWN_CODE_BLOCK.value
        ]
        return newly_defined_rules_for_this_model_pass, has_special_string_for_root

    if not model_fields:
        empty_object_rule = rf'{model_name_gbnf} ::= "{{" ws "}}"'
        if model_name_gbnf not in created_rules or empty_object_rule not in created_rules.get(model_name_gbnf,[]):
            created_rules.setdefault(model_name_gbnf, []).append(empty_object_rule)
            newly_defined_rules_for_this_model_pass.append(empty_object_rule)
        return newly_defined_rules_for_this_model_pass, False

    model_rule_parts: List[str] = []
    has_special_string_overall = False
    type_hints = get_type_hints(model_type)
    sorted_field_names = sorted(model_fields.keys())

    for field_name in sorted_field_names:
        field_info = model_fields[field_name]
        field_actual_type = type_hints.get(field_name, field_info.annotation)
        is_optional_field = not field_info.is_required()

        field_gbnf_type_name, field_specific_new_rules = generate_gbnf_rule_for_type(
            model_type.__name__, field_name, field_actual_type, is_optional_field,
            processed_models, created_rules, field_info
        )
        newly_defined_rules_for_this_model_pass.extend(r for r in field_specific_new_rules if r not in newly_defined_rules_for_this_model_pass)
        gbnf_json_key = f'"\\"{json.dumps(field_name)[1:-1]}\\""'
        model_rule_parts.append(f' ws {gbnf_json_key} ":" ws {field_gbnf_type_name}')
        if field_gbnf_type_name in [PydanticDataType.TRIPLE_QUOTED_STRING.value, PydanticDataType.MARKDOWN_CODE_BLOCK.value]:
            has_special_string_overall = True

    fields_joined = r' "," "\n" '.join(model_rule_parts)
    model_definition_str = rf'{model_name_gbnf} ::= "{{" "\n" {fields_joined} "\n" ws "}}"'
    if model_name_gbnf not in created_rules or model_definition_str not in created_rules.get(model_name_gbnf,[]):
        created_rules.setdefault(model_name_gbnf, []).append(model_definition_str)
        if model_definition_str not in newly_defined_rules_for_this_model_pass:
            newly_defined_rules_for_this_model_pass.append(model_definition_str)
    return newly_defined_rules_for_this_model_pass, has_special_string_overall


def generate_gbnf_grammar_from_pydantic_models(
    models: List[Type[BaseModel]],
    outer_object_name: Optional[str] = None,
    outer_object_content_key: Optional[str] = None,
    list_of_outputs: bool = False,
) -> str:
    processed_models: Set[Type[BaseModel]] = set()
    all_generated_rules_map: Dict[str, List[str]] = {}
    overall_has_special_strings = False

    for model_class in models:
        _, has_special = generate_gbnf_grammar(model_class, processed_models, all_generated_rules_map)
        if has_special:
            overall_has_special_strings = True

    root_rule_definition: str
    if outer_object_name is None:
        model_names_formatted = sorted([format_model_and_field_name(m.__name__) for m in models])
        grammar_models_def_rhs = " | ".join(model_names_formatted) if model_names_formatted else rf'"{{" ws "}}"'
        grammar_models_rule_name = "grammar-models"
        grammar_models_rule_def = f"{grammar_models_rule_name} ::= {grammar_models_def_rhs}"
        if grammar_models_rule_name not in all_generated_rules_map or grammar_models_rule_def not in all_generated_rules_map.get(grammar_models_rule_name, []):
            all_generated_rules_map.setdefault(grammar_models_rule_name, []).append(grammar_models_rule_def)
        if list_of_outputs:
            root_rule_definition = rf'root ::= (" "| "\n")? "[" ws ( {grammar_models_rule_name} ( ws "," ws {grammar_models_rule_name} )* ws )? "]"'
        else:
            root_rule_definition = rf'root ::= (" "| "\n")? {grammar_models_rule_name}'
    else:
        formatted_outer_name = format_model_and_field_name(outer_object_name)
        model_wrapper_rule_names = []
        for model_class in models:
            model_name_formatted = format_model_and_field_name(model_class.__name__)
            if outer_object_content_key:
                wrapper_rule_name = f"{model_name_formatted}-wrapper-for-{formatted_outer_name}"
                gbnf_model_class_name_literal = f'"\\"{json.dumps(model_class.__name__)[1:-1]}\\""'
                gbnf_outer_content_key_literal = f'"\\"{json.dumps(outer_object_content_key)[1:-1]}\\""'
                wrapper_def = rf'{wrapper_rule_name} ::= {gbnf_model_class_name_literal} ws "," ws {gbnf_outer_content_key_literal} ws ":" ws {model_name_formatted}'
                if wrapper_rule_name not in all_generated_rules_map or wrapper_def not in all_generated_rules_map.get(wrapper_rule_name,[]):
                    all_generated_rules_map.setdefault(wrapper_rule_name, []).append(wrapper_def)
                model_wrapper_rule_names.append(wrapper_rule_name)
            else: model_wrapper_rule_names.append(model_name_formatted)
        grammar_models_for_outer_rhs = " | ".join(sorted(list(set(model_wrapper_rule_names))))
        grammar_models_outer_rule_name = f"grammar-models-for-{formatted_outer_name}"
        grammar_models_outer_def = f"{grammar_models_outer_rule_name} ::= {grammar_models_for_outer_rhs}"
        if grammar_models_outer_rule_name not in all_generated_rules_map or grammar_models_outer_def not in all_generated_rules_map.get(grammar_models_outer_rule_name,[]):
            all_generated_rules_map.setdefault(grammar_models_outer_rule_name, []).append(grammar_models_outer_def)

        gbnf_outer_object_key_literal = f'"\\"{json.dumps(outer_object_name)[1:-1]}\\""'
        outer_obj_def = rf'{formatted_outer_name} ::= (" "| "\n")? "{{" ws {gbnf_outer_object_key_literal} ws ":" ws {grammar_models_outer_rule_name} ws "}}"'
        if formatted_outer_name not in all_generated_rules_map or outer_obj_def not in all_generated_rules_map.get(formatted_outer_name,[]):
            all_generated_rules_map.setdefault(formatted_outer_name, []).append(outer_obj_def)
        root_rule_definition = f"root ::= {formatted_outer_name}"
        if list_of_outputs:
            root_rule_definition = rf'root ::= (" "| "\n")? "[" ws ( {formatted_outer_name} ( ws "," ws {formatted_outer_name} )* ws )? "]"'

    final_grammar_parts: List[str] = [root_rule_definition]
    collected_definitions_for_final_output_set: Set[str] = {root_rule_definition}
    sorted_rule_names_for_output = sorted(all_generated_rules_map.keys())
    for rule_name_key in sorted_rule_names_for_output:
        unique_definitions_for_rule = sorted(list(set(all_generated_rules_map[rule_name_key])))
        for rule_definition_str in unique_definitions_for_rule:
            if rule_definition_str not in collected_definitions_for_final_output_set:
                final_grammar_parts.append(rule_definition_str)
                collected_definitions_for_final_output_set.add(rule_definition_str)
    full_grammar = "\n".join(final_grammar_parts)
    return remove_empty_lines(full_grammar + get_primitive_grammar(full_grammar, grammar_has_special_strings=overall_has_special_strings))


def get_primitive_grammar(grammar_str_to_check: str, grammar_has_special_strings: bool = False) -> str:
    primitives_set: Set[str] = set()
    base_primitives = r"""
ws ::= ([ \t\n] ws)?
boolean ::= "true" | "false"
null ::= "null"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) ws
float ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
number ::= integer | float
"""
    primitives_set.add(base_primitives)
    if "unknown" in grammar_str_to_check or " value" in grammar_str_to_check or \
       PydanticDataType.OBJECT.value in grammar_str_to_check or \
       PydanticDataType.ARRAY.value in grammar_str_to_check or \
       PydanticDataType.ANY.value in grammar_str_to_check:
        any_block = """
value ::= object | array | string | number | boolean | null
object ::= "{" ws ( string ":" ws value ( ws "," ws string ":" ws value)* ws )? "}" ws
array  ::= "[" ws ( value ( ws "," ws value)* ws )? "]" ws
"""
        primitives_set.add(any_block)
    if grammar_has_special_strings or PydanticDataType.MARKDOWN_CODE_BLOCK.value in grammar_str_to_check:
        markdown_block = r'''
markdown-code-block ::= opening-triple-ticks markdown-code-block-content closing-triple-ticks ws
markdown-code-block-content ::= ([^`\\] | "\\" . | "`" [^`] | "`" "`" [^`])*
opening-triple-ticks ::= "```" ([a-zA-Z0-9+#-]*)? "\n"
closing-triple-ticks ::= "\n" "```"
'''
        primitives_set.add(markdown_block)
    if grammar_has_special_strings or PydanticDataType.TRIPLE_QUOTED_STRING.value in grammar_str_to_check:
        triple_block = r"""
triple-quoted-string ::= "'''" triple-quoted-string-content "'''" ws
triple-quoted-string-content ::= ([^'\\] | "\\" . | "'" [^'] | "'" "'" [^'] )*
"""
        primitives_set.add(triple_block)
    return "\n" + "\n".join(sorted(list(primitives_set)))