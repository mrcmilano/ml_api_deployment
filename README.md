# 🌍 ML Model Deployment App

## 📝 Descrizione

Questa app espone una semplice API REST basata su **FastAPI** per il riconoscimento automatico della lingua di un testo. Nato come education project per l'implementazione di best practice in ambito deployment di modelli ML.
Il modello di classificazione è allenato su 4 lingue: **inglese, francese, italiano e spagnolo** utilizzando **Multinomial Naive Bayes** e pipeline di preprocessing con **TF-IDF**.

Il progetto include:
- Script per training del modello di riconoscimento della lingua (con GridSearchCV per ottimizzare MultinomialNB) con pipeline di scikit-learn.
- Semplice API REST per per esposizione endpoint di predizione
- Endpoint di health check dell'API e endpoint per ottenere la versione del modello
- Deployment con Docker

⚠️ **WARNING** : This is a WIP

---

## 🚀 Funzionalità

- Allenamento e salvataggio modello di classificazione
- Endpoint `/language_detection` per la predizione della lingua di uno o più testi  
- Endpoint `/status` per verificare lo stato del servizio  
- Endpoint `/model_version` per ottenere la versione del modello attualmente in uso  
- Deploy containerizzato con **Docker** per un avvio rapido e portabile  
- Logging centralizzato per monitorare richieste e performance  

---

## ⚙️ Installazione e avvio locale

#### 1️⃣ Clona il repository

```bash
git clone https://github.com/mrcmilano/ml_api_deployment.git
cd ml_api_deployment
```

#### 2️⃣ Crea ed attiva un ambiente virtuale

```bash
python -m venv venv
source venv/bin/activate  # su macOS/Linux
venv\Scripts\activate     # su Windows
```

#### 3️⃣ Installa le dipendenze

```bash
pip install -r requirements.txt
```

#### 4️⃣ Allena modello basato su MultinomialNB

```bash
python src/model_training.py
```

#### 5️⃣ Avvia API in locale

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🐳 Esecuzione con Docker

#### Build dell'immagine

```bash
docker build -t language-api .
```

#### Avvio del container

```bash
docker run -d -p 8000:8000 language-api
```

### 📦 Versionamento dei modelli

L'API carica gli artefatti dal percorso `models/text/language_classification/<MODEL_VERSION>`.  
Imposta `MODEL_VERSION` (o `MODEL_DIR`) come variabile d'ambiente per scegliere quale modello servire:

```bash
docker run -e MODEL_VERSION=v2 -p 8000:8000 language-api
```

Assicurati che la cartella corrispondente contenga `best_classifier.pkl` e `label_encoder.pkl`.

### 🔁 Development con Docker

Per iterare rapidamente senza rebuild continui, usa l'immagine di sviluppo e monta il codice locale:

```bash
docker build -f Dockerfile.dev -t language-api-dev .
docker run --rm -it \
  -v $(pwd)/app:/app/app \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  -p 8000:8000 \
  language-api-dev
```

`uvicorn --reload` rileva automaticamente le modifiche nella directory montata. Imposta `-e MODEL_VERSION=vX` se vuoi testare un modello diverso.

## 🧠 Training del modello

Usa `src/model_training.py` per addestrare un nuovo classificatore partendo dalla configurazione YAML:

```bash
python src/model_training.py --config configs/model_config.yaml
```

Il training produce `best_classifier.pkl` e `label_encoder.pkl` nella cartella `output.model_dir` definita nel file di config, oltre a salvare un file di risultati con timestamp. Aggiorna `MODEL_VERSION` (o `MODEL_DIR`) per puntare al modello appena creato.

## 📡 Endpoint disponibili

| Endpoint              | Metodo | Descrizione                         |
| --------------------- | ------ | ----------------------------------- |
| `/status`             | GET    | Verifica che il servizio sia attivo |
| `/language_detection` | POST   | Rileva la lingua di uno o più testi |
| `/model_version`      | GET    | Restituisce la versione del modello |

### Esempio richiesta API

#### POST /language_detection

##### Richiesta

```bash
curl -X POST "http://127.0.0.1:8000/language_detection" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Bonjour à tous!", "Ciao come va?", "Hello world!"]}'
```

## ✅ Test

Per eseguire unit tests:

```bash
pytest -v
```

---
☑️ **DISCLAIMER** : Vibe-coded with OpenAI ChatGPT - v5 and v4.1
