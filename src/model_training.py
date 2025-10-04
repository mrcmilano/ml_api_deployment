# ===============================
# Multi-class Language Classification
# ===============================
import pprint
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Caricamento dati
# -------------------------------
# Devo esistere le colonne 'text' e 'language'
# TODO: aggiungere sanity check sul dataset per verificare che le colonne esistano
df = pd.read_csv('../data/lang_detection.csv')
df.columns = [x.lower() for x in df.columns]

# -------------------------------
# Label Encoding della variabile target
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(df["language"])  # trasforma le lingue in numeri interi

# -------------------------------
# Funzione di pulizia testo (vectorized) + standardizzazione UTF-8
# -------------------------------
def clean_texts(texts):
    return (
        pd.Series(texts)
        .astype(str)
        .map(lambda x: x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )

# -------------------------------
# Pipeline compatta
# -------------------------------
pipeline = Pipeline([
    ("cleaner", FunctionTransformer(clean_texts)),
    ("vectorizer", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# -------------------------------
# Parametri da ottimizzare con GridSearchCV
# -------------------------------
param_grid = {
    "vectorizer__ngram_range": [(1,1), (1,2)],
    "vectorizer__max_df": [0.9, 1.0],
    "vectorizer__min_df": [1, 2],
    "clf__alpha": [0.1, 0.5, 1.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# -------------------------------
# Fit sul dataset
# -------------------------------
grid.fit(df["text"], y)
pprint("✅ Migliori parametri:", grid.best_params_)
pprint("✅ Miglior score (CV):", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(df["text"])

# -------------------------------
# Classification Report
# -------------------------------
pprint("\nClassification Report:")
pprint(classification_report(y, y_pred, target_names=le.classes_, digits=4))

# -------------------------------
# Confusion Matrix visuale
# -------------------------------
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Multiclass Language Classification")
plt.tight_layout()
plt.show()