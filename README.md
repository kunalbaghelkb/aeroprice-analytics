# 🚀 Aeroprice Analytics: ML-Powered Vehicle Pricing Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)
![Redis](https://img.shields.io/badge/Upstash_Redis-Serverless-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)

**Aeroprice Analytics** is an enterprise-grade, serverless Machine Learning API designed to predict real-time used car prices. Built with a modern MLOps architecture, it leverages an optimized XGBoost algorithm, advanced K-Fold Target Encoding, and a Redis caching layer for ultra-low latency inference.

---

## 🏗️ System Architecture

1. **The ML Pipeline:** Memory-efficient data ingestion, robust preprocessing (handling outliers and missing values), and cross-validated hyperparameter tuning.
2. **The API Layer:** Headless serverless architecture using FastAPI, fully decoupled from the ML training logic.
3. **The Cache Layer:** Upstash Redis integration to serve redundant requests in milliseconds, significantly reducing ML inference compute costs.
4. **The MLOps Tracker:** Remote MLflow integration via DagsHub for seamless experiment tracking and model registry.

---

## 🛠️ Tech Stack

* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
* **Backend Framework:** FastAPI, Uvicorn, Pydantic (Strict Schema Validation)
* **Caching & DB:** Redis (Upstash Serverless)
* **MLOps & Tracking:** MLflow, DagsHub, Joblib

---

## 📂 Datasets & Resources Used

To run this project locally, you will need to download the following datasets and place them in the `data/raw` folder as per the structure mentioned below.

### Insurance Fraud Detection Dataset (CSV)
This dataset is used to train the model for predicting vehicle price.
- **Source:** Kaggle (Craigslist Cars and Trucks Data)
- **Download Link:** [Click Here to Download CSV](https://www.kaggle.com/datasets/prena0808/craigslist-cars-and-trucks-data)
- **Placement:** Extract the file `vehicles.csv` inside `data/raw/`.

---

## 📂 Project Structure
The project follows a modular, production-ready structure:

    aeroprice_analytics/
    ├── api/
    │   ├── main.py                # FastAPI Application & Routes
    │   ├── schemas.py             # Pydantic Input/Output Validation
    │   └── redis_cache.py         # Upstash Redis Fallback Logic
    ├── src/
    │   ├── components/            # Isolated ML Lego Blocks
    │   │   ├── data_loader.py
    │   │   ├── preprocessor.py
    │   │   └── model_trainer.py
    │   ├── pipelines/             # Orchestrators
    │   │   ├── train_pipeline.py  # End-to-end ML Training Loop
    │   │   └── predict_pipeline.py# API Inference Handler
    │   ├── constants.py           # Global Configuration
    │   └── logger.py              # Custom Logging Setup
    ├── models/                    # Serialized Artifacts (.pkl)
    ├── notebooks/                 # EDA & Statistical Hypothesis Testing
    ├── .env                       # Environment Secrets (Git Ignored)
    └── requirements.txt

---

## ⚙️ Installation & Setup

1. Clone the Repository
    ```bash
    git clone https://github.com/your-username/aeroprice-analytics.git && cd aeroprice-analytics

2. Create Virtual Environment
    ```bash
    python3.11 -m venv .venv

    # Windows
    .venv\Scripts\activate

    # Mac/Linux
    source .venv/bin/activate

3. Install Dependencies
    ```bash
    pip install -r requirements.txt

4. Set Environment Variables
Create a .env file in the root directory and add your keys
    ```bash
    # DagsHub MLflow Credentials
    MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/aeroprice-analytics.mlflow
    MLFLOW_TRACKING_USERNAME=<your_username>
    MLFLOW_TRACKING_PASSWORD=<your_dagshub_token_or_password>

5. Generate Model
Execute this file to generate the model (Ensure that all required datasets and resources have been added as specified.)
    ```bash
    python -m src.pipelines.train_pipeline

6. Run the Application
    ```bash
    uvicorn api.main:app --reload

---

## ⚡ API Usage Example

**Endpoint:** `POST /predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2010,
    "odometer": 0,
    "manufacturer": "honda",
    "model": "civic",
    "condition": "unknown",
    "cylinders": "unknown",
    "fuel": "gas",
    "title_status": "clean",
    "transmission": "automatic",
    "drive": "fwd",
    "size": "compact",
    "type": "sedan",
    "paint_color": "unknown",
    "state": "ca",
    "region": "los angeles"
  }'
```

---

## Response (JSON)
    {
    "predicted_price": 11372.4,
    "currency": "USD",
    "source": "model"
    }

> **Note:** Subsequent identical requests will return `"source": "cache"` with near-zero latency.

---

## 👨‍💻 Author
**Kunal Baghel**

*Associate Data Scientist & AI/ML Engineer*

[LinkedIn](https://linkedin.com/in/kunalbaghelz) | [GitHub](http://github.com/kunalbaghelkb)