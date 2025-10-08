import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import get_scorer
# from app.utils import clean_texts  # funzione di pulizia testo sta qui per essere caricata in produzione
from src.model_utils import load_config, clean_texts, save_pipeline_obj, log_experiment
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import os

def main(config_path: str):
    # Carica configurazione
    cfg = load_config(config_path)
    print(f"Loaded configuration: {config_path}")

    # Carica e prepara dataset di allenamento
    print(f"Loading dataset from {cfg['dataset']['path']}")
    df = pd.read_csv(cfg["dataset"]["path"])
    X = df[cfg["dataset"]["text_column"]]
    y = df[cfg["dataset"]["label_column"]]

    # Pulizia e encoding label
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Costruzione pipeline
    vectorizer_params = cfg["vectorizer"]["params"]
    if "ngram_range" in vectorizer_params:   # Converte ngram_range in tupla per evitare errori
        vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    vectorizer = TfidfVectorizer(**vectorizer_params)
    print(f"Vectorizer params: {vectorizer_params}")
    
    # modello
    model_params = cfg["model"]["params"]
    model = MultinomialNB(**model_params)
    print(f"Model params: {model_params}")

    pipeline = Pipeline([
        ("cleaner", FunctionTransformer(clean_texts)),
        ("vectorizer", vectorizer),
        ("clf", model)
    ])

    # Cross-validation
    scorer = get_scorer(cfg["evaluation"]["metric"])
    scores = cross_val_score(pipeline, X, y_enc, cv=cfg["evaluation"]["cv_folds"], scoring=scorer)

    mean_score = np.mean(scores)
    print(f"{cfg['evaluation']['metric']} (CV mean): {mean_score:.4f}")

    # Fit finale su tutto il dataset
    pipeline.fit(X, y_enc)

    # Salvataggio modello e encoder
    output_dir = cfg["output"]["model_dir"]
    save_pipeline_obj(pipeline, os.path.join(output_dir, cfg["output"]["model_filename"]))
    save_pipeline_obj(le, os.path.join(output_dir, cfg["output"]["label_enc_filename"]))

    # Log dei risultati
    results = {
        "cv_scores": scores.tolist(),
        "mean_score": mean_score,
    }
    log_experiment(results, cfg, cfg["output"]["results_file"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
