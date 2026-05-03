import tiktoken
import json
from typing import Sequence, List, Dict, Union
from .config import DEFAULT_MODEL_ENCODING, DEFAULT_MAX_TOKENS


def resolve_output_schema(output_schema) -> Union[dict, None]:
    """Convert a Pydantic BaseModel subclass to a Tavily-compatible JSON schema dict.

    Tavily's API only accepts 'properties' and 'required' at the top level.
    This strips the full Pydantic JSON schema down to those keys and resolves
    any $ref/$defs references inline so nested models are inlined.

    Plain dicts are passed through unchanged. If pydantic is not installed
    and a non-dict is passed, it is returned as-is.
    """
    if output_schema is None or not isinstance(output_schema, type):
        return output_schema
    try:
        from pydantic import BaseModel
    except ImportError:
        return output_schema
    if not issubclass(output_schema, BaseModel):
        return output_schema
    schema = output_schema.model_json_schema()
    defs = schema.get("$defs", {})

    def _resolve(obj, visiting=frozenset()):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in visiting:
                    return {}  # break cycle
                return _resolve(defs[ref_name], visiting | {ref_name})
            result = {}
            for k, v in obj.items():
                if k == "title":
                    continue  # strip Pydantic metadata annotation
                if k == "properties":
                    # resolve each field definition but preserve field names as-is,
                    # so a user field named "title" is not accidentally dropped
                    result[k] = {pk: _resolve(pv, visiting) for pk, pv in v.items()}
                else:
                    result[k] = _resolve(v, visiting)
            return result
        if isinstance(obj, list):
            return [_resolve(i, visiting) for i in obj]
        return obj

    resolved = _resolve(schema)
    return {k: resolved[k] for k in ("properties", "required") if k in resolved}


def get_total_tokens_from_string(string: str, encoding_name: str = DEFAULT_MODEL_ENCODING) -> int:
    """
        Get total amount of tokens from string using the specified encoding (based on openai compute)
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(string)
    return len(tokens)

def get_max_items_from_list(data: Sequence[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> List[Dict[str,str]]:
    """
        Get max items from list of items based on defined max tokens (based on openai compute)
    """
    result = []
    current_tokens = 0
    for item in data:
        item_str = json.dumps(item)
        new_total_tokens = current_tokens + get_total_tokens_from_string(item_str)
        if new_total_tokens > max_tokens:
            break
        else:
            result.append(item)
            current_tokens = new_total_tokens
    return result
