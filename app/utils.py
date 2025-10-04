import re
import string

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


def predict_language_safe(model, le, texts, threshold=0.5):
    """
    Predice la lingua dei testi.
    Se la probabilità massima è inferiore a threshold, restituisce 'unknown'.
    """
    probas = model.predict_proba(texts)
    max_probs = probas.max(axis=1)
    preds = model.predict(texts)
    pred_labels = le.inverse_transform(preds)

    return [
        label if prob >= threshold else "unknown"
        for label, prob in zip(pred_labels, max_probs)
    ]
