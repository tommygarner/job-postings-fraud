# Fake Job Posting Detection: Advanced ML with Interpretability

**Team**: [@tommygarner](https://github.com/tommygarner)

[@ethandavenport](https://github.com/ethandavenport)

[@nkfavoriti](https://github.com/nkfavoriti)

[@sebaspalacino](https://github.com/sebaspalacino)

[@slefebre10](https://github.com/slefebre10)

**Course**: Advanced Machine Learning (UT Austin)  
**Project Duration**: November 2025 - December 2025

## Project Overview

Streamlit app and ML pipeline to flag fraudulent job ads using Naive Bayes, LSTM, and MiniLM.

### Dataset

- **Source**: [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: 17,880 postings with 18 features
- **Fraud Rate**: 4.84% (severe class imbalance)
- **Target**: Binary classification (fraudulent: 0/1)

## Quick Start: Run the Streamlit App Locally

### 1. Clone the repository

```
git clone https://github.com/tommygarner/job-postings-fraud.git
cd job-postings-fraud
```

### 2. Create and activate a virtual environment (Python 3.10)

This project is tested with Python 3.10. Other versions (3.12–3.14) may have issues installing NumPy / TensorFlow.

On Windows (PowerShell):

```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:

```
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```
python -m pip install --upgrade pip
pip install setuptools wheel
pip install -r requirements_app.txt
```

This installs Streamlit, TensorFlow 2.12, scikit‑learn, transformers, and other packages needed by the app.

### 4. Run the Streamlit app

From the repo root:

```
python -m streamlit run app/app.py
```

Streamlit will print a local URL such as `http://localhost:8501`.  
Open that URL in your browser to use the Fake Job Posting Detection app.

## Methods (High Level)

- **Models**: Naive Bayes + TF‑IDF, LSTM, MiniLM transformer  
- **Ensemble**: NB 5%, LSTM 45%, MiniLM 50%; LOW / MEDIUM / HIGH risk bands  
- **Interpretability**: Naive Bayes word‑level contributions, LSTM and MiniLM explanations via LIME/SHAP, integrated gradients for MiniLM, model‑agreement narratives, and structured‑feature commentary with simple confidence plots.
- **Tooling**: Pandas, NumPy, scikit‑learn, TensorFlow, PyTorch + Transformers, Streamlit

---

## Repo Structure

| Folder | Purpose |
|--------|---------|
| `app/` | Streamlit app |
| `models/` | Trained model artifacts |
| `notebooks/` | EDA and training |
| `data/` | Raw / processed data (local only or .gitignored) |
| `results/` | Metrics and outputs |

---

## References

**Existing Kaggle Solutions**:
- https://github.com/38832/Fake-Job-Posting-Prediction
- https://github.com/Anshupriya2694/Fake-Job-Posting-Prediction

**Last Updated**: November 29, 2025  
**Status**: Active Development
