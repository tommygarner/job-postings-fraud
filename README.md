# Fake Job Posting Detection: Advanced ML with Interpretability

**Team**: [@tommygarner](https://github.com/tommygarner)
[@ethandavenport](https://github.com/ethandavenport)
[@nkfavoriti](https://github.com/nkfavoriti)
[@sebaspalacino](https://github.com/sebaspalacino)
[@slefebre10](https://github.com/slefebre10)

**Course**: Advanced Machine Learning (UT Austin)
**Status**: Completed December 2025

## Project Overview

Streamlit app and ML pipeline to flag fraudulent job ads using Naive Bayes, LSTM, and MiniLM.

### Dataset

- **Source**: [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: 17,880 postings with 18 features
- **Fraud Rate**: 4.84% (severe class imbalance)

## Methods

* **Models**: Naive Bayes + TF-IDF, LSTM (Keras), MiniLM transformer (HuggingFace)

* **Ensemble Strategy**: Weighted blend (NB 30%, LSTM 60%, MiniLM 10%) → LOW / MEDIUM / HIGH fraud-risk bands. Decision threshold: 40%

* **Interpretability**: Word-level TF-IDF contributions, LIME/SHAP text explanations, Integrated Gradients token attributions, model-agreement narratives, confidence plots

## Model Performance

| Metric     | Value  |
|------------|--------|
| Accuracy   | 97.52% |
| F1-Score   | 0.7140 |
| Precision  | 80.98% |
| Recall     | 63.85% |

Risk-based detection rates:
- **HIGH risk**: 59.2% of frauds caught, 0.3% false positive rate
- **MEDIUM risk**: 17.7% additional frauds flagged
- **LOW risk**: 95% of legitimate posts correctly classified

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/tommygarner/job-postings-fraud.git
cd job-postings-fraud
```

### 2. Create virtual environment (Python 3.10)

**Windows (PowerShell):**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install and run

```bash
pip install -r requirements_app.txt
python -m streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

## App Features

The Streamlit app has three main tabs:

- **Prediction**: Enter job posting details and get fraud risk assessment with confidence scores
- **Feature Visuals**: View model confidence plots and feature importance explanations
- **Common Signals**: See typical fraud indicators and patterns found in the training data

## Repo Structure

| Folder | Purpose |
| --- | --- |
| `app/` | Streamlit application |
| `models/` | Saved model artifacts (NB, LSTM, MiniLM) |
| `notebooks/` | EDA, preprocessing, training, interpretability |
| `results/` | LIME, SHAP, Integrated Gradients outputs |
| `requirements_app.txt` | Dependencies for the Streamlit app |

## References

- [Fake Job Posting Prediction (38832)](https://github.com/38832/Fake-Job-Posting-Prediction)
- [Fake Job Posting Prediction (Anshupriya2694)](https://github.com/Anshupriya2694/Fake-Job-Posting-Prediction)

---

**Last Updated**: February 2026
**Status**: Complete

*This repository was refined and cleaned up with Claude Code*
