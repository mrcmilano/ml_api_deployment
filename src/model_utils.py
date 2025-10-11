import yaml
import json
import os
import re
import string
from datetime import datetime
from joblib import dump

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_pipeline_obj(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)

def log_experiment(results: dict, config: dict, output_file: str):
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

def clean_texts(texts):
    """
    Pulisce una lista di testi:
    - lowercase
    - rimozione punteggiatura
    - rimozione spazi extra
    """
    return [
        re.sub(r"\s+", " ", t.lower().translate(str.maketrans("", "", string.punctuation))).strip()
        for t in texts
    ]