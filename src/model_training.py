import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from src.model_utils import (
    load_config, parse_config, clean_texts,\
    save_pipeline_obj, log_experiment,
)     


def prepare_data(cfg):
    df = pd.read_csv(cfg["dataset"]["path"])
    if "keep_languages" in cfg["dataset"]:
        df = df[df[cfg["dataset"]["label_column"]].isin(cfg["dataset"]["keep_languages"])]
    X = df[cfg["dataset"]["text_column"]]
    y = df[cfg["dataset"]["label_column"]]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y, y_enc, le


def build_pipeline(cfg):
    vectorizer = TfidfVectorizer(**cfg["vectorizer"]["params"])
    clf = MultinomialNB(**cfg["model"]["params"])
    return Pipeline([
        ("cleaner", FunctionTransformer(clean_texts)),
        ("vectorizer", vectorizer),
        ("clf", clf),
    ])


def train_model(cfg):
    cfg = parse_config(cfg)
    X, y, y_enc, le = prepare_data(cfg)
    pipeline = build_pipeline(cfg)

    print("Pipeline configuration:")
    for name, step in pipeline.steps:
        print(f" - {name}: {step}")

    scores, mean_score = None, None

    if cfg["grid_search"]["enabled"]:
        param_grid = cfg["grid_search"]["param_grid"]
        if not param_grid:
            raise ValueError("Grid search enabled but no param_grid provided")

        param_grid = parse_config(param_grid)
        print(f"Starting GridSearchCV with params:\n{param_grid}")

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cfg["grid_search"]["cv_folds"],
            scoring=cfg["grid_search"]["metric"],
            verbose=2,
            n_jobs=cfg["grid_search"].get("n_jobs", -1),
        )
        grid.fit(X, y_enc)
        pipeline = grid.best_estimator_
        scores = grid.cv_results_["mean_test_score"]
        mean_score = float(np.mean(scores))
        best_params = grid.best_params_

        print(f"Best params: {best_params}")
        print(f"Mean CV score: {mean_score:.4f}")

        y_pred_enc = pipeline.predict(X)
        y_pred = le.inverse_transform(y_pred_enc)
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=le.classes_, digits=4))


    else:
        print("GridSearchCV disabled. Training default pipeline...")
        pipeline.fit(X, y_enc)

    return pipeline, le, scores, mean_score, best_params


def main(config_path: str):
    cfg = load_config(config_path)
    pipeline, le, scores, mean_score, best_params = train_model(cfg)

    output_dir = cfg["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    save_pipeline_obj(pipeline, os.path.join(output_dir, cfg["output"]["model_filename"]))
    save_pipeline_obj(le, os.path.join(output_dir, cfg["output"]["label_enc_filename"]))

    results = {
        "cv_scores": scores.tolist() if scores is not None else None,
        "mean_score": mean_score,
        "best_estimator_params": best_params
    }

    # Costruisce il path con timestamp per il file dei risultati
    results_file_with_ts = os.path.join(
        os.path.dirname(cfg["output"]["results_file"]),
        f"{os.path.splitext(os.path.basename(cfg['output']['results_file']))[0]}_" \
        f"{datetime.now():%Y%m%d_%H%M%S}.json"
    )

    log_experiment(results, cfg, results_file_with_ts)
    print("Training complete and model saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
