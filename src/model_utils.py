from __future__ import annotations

import ast
import json
import os
import re
import string
from datetime import datetime
from typing import Any, Iterable, MutableMapping

import yaml
from joblib import dump


ConfigDict = MutableMapping[str, Any]


def load_config(path: str) -> ConfigDict:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader)


def parse_config(config: ConfigDict) -> ConfigDict:
    """Convert YAML-loaded strings like '(1, 2)' into tuples."""

    def convert_value(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        if isinstance(value, list):
            return [convert_value(item) for item in value]
        if isinstance(value, dict):
            return {key: convert_value(item) for key, item in value.items()}
        return value

    return convert_value(config)


def save_json(data: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def save_pipeline_obj(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)


def log_experiment(results: dict[str, Any], config: ConfigDict, output_file: str) -> None:
    """
    Unisce risultati e metadati di configurazione in un singolo file JSON
    """
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
    }
    save_json(log, output_file)
    print(f"Experiment log saved to {output_file}")


def clean_texts(texts: Iterable[str]) -> list[str]:
    """
    Pulisce una lista di testi:
    - lowercase
    - rimozione punteggiatura
    - rimozione spazi extra
    """
    return [
        re.sub(r"\s+", " ", text.lower().translate(str.maketrans("", "", string.punctuation))).strip()
        for text in texts
    ]
