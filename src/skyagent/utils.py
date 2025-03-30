from __future__ import annotations

from typing import Any


TEXT_CHARS = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


def is_binary_string(bytes: bytes) -> bool:
    # based on https://github.com/file/file/blob/f2a6e7cb7db9b5fd86100403df6b2f830c7f22ba/src/encoding.c#L151-L228
    return bool(bytes.translate(None, TEXT_CHARS))


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    return any(i > n for i, _ in enumerate(obj))


def _resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert isinstance(
            value, dict
        ), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value

    return resolved


def to_strict_json_schema(
    schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """
    Algorithm from https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
    """
    if not isinstance(schema, dict):
        raise TypeError(f"Expected {schema} to be a dictionary; path={path}")

    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for def_name, def_schema in defs.items():
            to_strict_json_schema(
                def_schema, path=(*path, "$defs", def_name), root=root
            )

    definitions = schema.get("definitions")
    if isinstance(definitions, dict):
        for definition_name, definition_schema in definitions.items():
            to_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    schema_type = schema.get("type")
    if schema_type == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)
        schema["properties"] = {
            key: to_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    items = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = to_strict_json_schema(items, path=(*path, "items"), root=root)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        schema["anyOf"] = [
            to_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1:
            schema.update(
                to_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root)
            )
            schema.pop("allOf")
        else:
            schema["allOf"] = [
                to_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    if schema.get("default", False) is None:
        schema.pop("default")

    ref = schema.get("$ref")
    if ref and _has_more_than_n_keys(schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = _resolve_ref(root=root, ref=ref)
        if not isinstance(resolved, dict):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        schema.update({**resolved, **schema})
        schema.pop("$ref")
        return to_strict_json_schema(schema, path=path, root=root)

    return schema
