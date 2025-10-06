# 🌍 ML Model Deployment App

## 📝 Descrizione 

Questa app espone una semplice API REST basata su **FastAPI** per il riconoscimento automatico della lingua di un testo. Nato come toy project per l'implementazione di best practice in ambito deployment di modelli ML.
Il modello di classificazione è allenato su 4 lingue: **inglese, francese, italiano e spagnolo** utilizzando **Multinomial Naive Bayes** e pipeline di preprocessing con **TF-IDF**.

Il progetto include:
- Script per training pipeline del modello di riconoscimento della lingua (con GridSearchCV per ottimizzare MultinomialNB) con pipeline di scikit-learn. 
- Semplice API REST per per esposizione endpoint di predizione via FastAPI, pronta per essere deployata con Docker
- Endpoint di health check dell'API e per ottenere la versione del modello

⚠️ __WARNING__ : This is a WIP

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

```
git clone https://github.com/mrcmilano/ml_api_deployment.git
cd ml_api_deployment
```

#### 2️⃣ Crea ed attiva un ambiente virtuale

```
python -m venv venv
source venv/bin/activate  # su macOS/Linux
venv\Scripts\activate     # su Windows
```

#### 3️⃣ Installa le dipendenze

```
pip install -r requirements.txt
```

#### 4️⃣ Allena modello basato su MultinomialNB

```
python src/model_training.py
```

#### 5️⃣ Avvia API in locale

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🐳 Esecuzione con Docker

#### Build dell'immagine

```
docker build -t language-api .
```

#### Avvio del container

```
docker run -d -p 8000:8000 language-api
```

## 📡 Endpoint disponibili

| Endpoint              | Metodo | Descrizione                         |
| --------------------- | ------ | ----------------------------------- |
| `/status`             | GET    | Verifica che il servizio sia attivo |
| `/language_detection` | POST   | Rileva la lingua di uno o più testi |
| `/model_version`      | GET    | Restituisce la versione del modello |

### Esempio richiesta API 
#### POST /language_detection
##### Richiesta

```
curl -X POST "http://127.0.0.1:8000/language_detection" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Bonjour à tous!", "Ciao come va?", "Hello world!"]}'
```

## ✅ Test

Per eseguire unit tests:
```
pytest -v
```

---
☑️ __DISCLAIMER__ : Vibe-coded with ChatGPT - v5 and v4.1
