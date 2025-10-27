from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from src.model_utils import (
    ConfigDict,
    clean_texts,
    load_config,
    log_experiment,
    parse_config,
    save_pipeline_obj,
)


def prepare_data(cfg: ConfigDict) -> tuple[pd.Series, pd.Series, np.ndarray, LabelEncoder]:
    dataset_cfg = cfg["dataset"]
    df = pd.read_csv(dataset_cfg["path"])
    if "keep_languages" in dataset_cfg:
        df = df[df[dataset_cfg["label_column"]].isin(dataset_cfg["keep_languages"])]
    X = df[dataset_cfg["text_column"]]
    y = df[dataset_cfg["label_column"]]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y, y_encoded, label_encoder


def build_pipeline(cfg: ConfigDict) -> Pipeline:
    vectorizer = TfidfVectorizer(**cfg["vectorizer"]["params"])
    classifier = MultinomialNB(**cfg["model"]["params"])
    return Pipeline(
        steps=[
            ("cleaner", FunctionTransformer(clean_texts)),
            ("vectorizer", vectorizer),
            ("clf", classifier),
        ]
    )


def train_model(cfg: ConfigDict) -> tuple[Pipeline, LabelEncoder, np.ndarray | None, float | None, dict[str, Any] | None]:
    parsed_cfg = parse_config(cfg)
    X, y, y_encoded, label_encoder = prepare_data(parsed_cfg)
    pipeline = build_pipeline(parsed_cfg)

    print("Pipeline configuration:")
    for name, step in pipeline.steps:
        print(f" - {name}: {step}")

    scores: np.ndarray | None = None
    mean_score: float | None = None
    best_params: dict[str, Any] | None = None

    grid_cfg = parsed_cfg["grid_search"]
    if grid_cfg["enabled"]:
        param_grid = grid_cfg["param_grid"]
        if not param_grid:
            raise ValueError("Grid search enabled but no param_grid provided")

        parsed_param_grid = parse_config(param_grid)
        print(f"Starting GridSearchCV with params:\n{parsed_param_grid}")

        grid = GridSearchCV(
            pipeline,
            param_grid=parsed_param_grid,
            cv=grid_cfg["cv_folds"],
            scoring=grid_cfg["metric"],
            verbose=2,
            n_jobs=grid_cfg.get("n_jobs", -1),
        )
        grid.fit(X, y_encoded)
        pipeline = grid.best_estimator_
        scores = grid.cv_results_["mean_test_score"]
        mean_score = float(np.mean(scores))
        best_params = grid.best_params_

        print(f"Best params: {best_params}")
        print(f"Mean CV score: {mean_score:.4f}")

        y_pred_encoded = pipeline.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=label_encoder.classes_, digits=4))
    else:
        print("GridSearchCV disabled. Training default pipeline...")
        pipeline.fit(X, y_encoded)

    return pipeline, label_encoder, scores, mean_score, best_params


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    pipeline, label_encoder, scores, mean_score, best_params = train_model(cfg)

    output_dir = cfg["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    save_pipeline_obj(pipeline, os.path.join(output_dir, cfg["output"]["model_filename"]))
    save_pipeline_obj(label_encoder, os.path.join(output_dir, cfg["output"]["label_enc_filename"]))

    results = {
        "cv_scores": scores.tolist() if scores is not None else None,
        "mean_score": mean_score,
        "best_estimator_params": best_params,
    }

    # Costruisce il path con timestamp per il file dei risultati
    results_file_with_ts = os.path.join(
        os.path.dirname(cfg["output"]["results_file"]),
        f"{os.path.splitext(os.path.basename(cfg['output']['results_file']))[0]}_{datetime.now():%Y%m%d_%H%M%S}.json",
    )

    log_experiment(results, cfg, results_file_with_ts)
    print("Training complete and model saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
