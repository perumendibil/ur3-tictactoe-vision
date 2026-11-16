"""Save values into a JSON settings file without losing its comments ("_" keys)."""

import json


def dumps_one_key_per_line(data):
    """JSON text with one top-level key per line (values on a single line), easy to edit."""
    lines = [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in data.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def update_config_file(path, updates):
    """Change some keys of the JSON file and keep every other key (and its order)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps_one_key_per_line(raw))
    return raw
