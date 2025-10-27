from __future__ import annotations
from typing import Iterable, Protocol, Sequence
import re
import string
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder


WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def clean_texts(texts: Iterable[str]) -> list[str]:
    """Normalizza testi rimuovendo punteggiatura ed eccessi di spazi.

    Parameters
    ----------
    texts : Iterable[str]
        Testi grezzi da processare.

    Returns
    -------
    list[str]
        Testi ripuliti in minuscolo e con la punteggiatura rimossa.
    """
    return [
        WHITESPACE_PATTERN.sub(" ", t.lower().translate(PUNCTUATION_TABLE)).strip()
        for t in texts
    ]

class ProbabilisticTextClassifier(Protocol):
    """Protocol describing the scikit-learn interface we rely on."""

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float_]:
        ...

    def predict(self, texts: Sequence[str]) -> NDArray[np.int_]:
        ...
        

def predict_language_safe(
    model: ProbabilisticTextClassifier,
    le: LabelEncoder,
    texts: Sequence[str],
    threshold: float = 0.5,
) -> list[str]:
    """Genera predizioni sulla lingua di un testo gestendo i casi in cui la lingua non e' conosciuta.

    Parameters
    ----------
    model : ProbabilisticTextClassifier
        Classificatore con metodi `predict` e `predict_proba`.
    le : LabelEncoder
        Encoder per convertire caratteri in numeri.
    texts : Sequence[str]
        Testi su cui stimare la lingua.
    threshold : float, optional
        Probabilità minima accettata prima di etichettare come `unknown` un testo.

    Returns
    -------
    list[str]
        Predizioni di una lingua oppure `unknown` se sotto soglia di confidenza.
    """
    probas: NDArray[np.float_] = model.predict_proba(texts)
    max_probs: NDArray[np.float_] = probas.max(axis=1)
    preds: NDArray[np.int_] = model.predict(texts)
    pred_labels: NDArray[np.str_] = le.inverse_transform(preds)

    return [label if prob >= threshold else "unknown" for label, prob in zip(pred_labels, max_probs)]
