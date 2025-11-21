# Premier League Match Prediction (XGBoost Hybrid)

This project is a **Premier League match prediction system** built around a hybrid XGBoost model.  
It:

- Loads a hybrid ensemble model (75% sharp odds, 25% full-history)
- Generates predictions for upcoming PL fixtures
- Tracks live accuracy vs actual results
- Provides a backtest view to see performance before the model went live
- Displays everything in a Streamlit web application

## Project Structure

```
project/
├── app/
│   └── app.py
│
├── scripts/
│   ├── predict_hybrid.py
│   ├── backtest_hybrid.py
│   └── archive_live_predictions.py
│
├── data/
│   ├── live/
│   │   ├── pl_results_from_api.csv
│   │   ├── pl_fixtures_from_api.csv
│   │   ├── pl_predictions.csv
│   │   └── live_predictions_history.csv
│   │
│   └── processed/
│       ├── match_features_super.csv
│       ├── pl_fixtures_features.csv
│       └── backtest_hybrid.csv
│
├── models/
│   └── xgb_super_hybrid.pkl
│
├── scripts/pipeline.py
├── requirements.txt
└── .gitignore
```



## Installation

### 1. Clone the repo
```bash
git clone <your-repo-url>.git
cd <your-repo-name>
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```



## Running the Pipeline

Run all model steps (predict → archive → optional backtest):

```bash
python scripts/pipeline.py
```

This performs:

1. Generate predictions:
```bash
python scripts/predict_hybrid.py
```

2. Archive finished matches:
```bash
python scripts/archive_live_predictions.py
```

3. (Optional) Re-run backtest if enabled in pipeline.py
```bash
python scripts/backtest_hybrid.py
```



## Running the Streamlit App

```bash
streamlit run app/app.py
```

The app provides:
- **Backtest page** (GW < 12)
- **Live model page** (GW ≥ 12)
- Accuracy, predictions, confidence values, match tables




## Deployment (Streamlit Cloud)

1. Push this project to GitHub  
2. Go to https://streamlit.io/cloud  
3. Choose **New App**  
4. Set **Main file path** to:

```
app/app.py
```

5. Deploy — Streamlit installs from `requirements.txt` automatically.
