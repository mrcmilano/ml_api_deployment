from __future__ import annotations

from typing import Iterable, List, Protocol, Sequence
import re
import string

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder


class ProbabilisticTextClassifier(Protocol):
    """Protocol describing the scikit-learn interface we rely on."""

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float_]:
        ...

    def predict(self, texts: Sequence[str]) -> NDArray[np.int_]:
        ...


def clean_texts(texts: Iterable[str]) -> List[str]:
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


def predict_language_safe(
    model: ProbabilisticTextClassifier,
    le: LabelEncoder,
    texts: Sequence[str],
    threshold: float = 0.5,
) -> List[str]:
    """
    Predice la lingua dei testi.
    Se la probabilità massima è inferiore a threshold, restituisce 'unknown'.
    """
    probas: NDArray[np.float_] = model.predict_proba(texts)
    max_probs: NDArray[np.float_] = probas.max(axis=1)
    preds: NDArray[np.int_] = model.predict(texts)
    pred_labels: NDArray[np.str_] = le.inverse_transform(preds)

    return [
        label if prob >= threshold else "unknown"
        for label, prob in zip(pred_labels.tolist(), max_probs.tolist())
    ]
