# Fake Job Posting Detection: Advanced ML with Interpretability

**Team**: [@tommygarner](https://github.com/tommygarner)

[@ethandavenport](https://github.com/ethandavenport)

[@nkfavoriti](https://github.com/nkfavoriti)

[@sebaspalacino](https://github.com/sebaspalacino)

**Course**: Advanced Machine Learning (UT Austin)  
**Project Duration**: November 2025 - December 2025

## Project Overview

This project builds a machine learning system to detect fraudulent job postings using 
a Kaggle dataset of 17,880 job postings (4.84% fraudulent). Our novel approach combines 
**missingness-based feature engineering** with **interpretability analysis** to identify 
both amateur and sophisticated job posting scams.

### Key Contributions

1. **Novel Feature Engineering**: 16 engineered features from missing data patterns
2. **U-Shaped Fraud Pattern Discovery**: Identified dual fraud archetypes
   - Amateur scams (completeness score 0): 22.2% fraud rate
   - Sophisticated scams (completeness score 10): 9.3% fraud rate
3. **Interpretability-First Design**: Sentence-level fraud explanations using LIME/SHAP
4. **Statistically Validated**: Chi-square tests prove feature significance (p < 0.001)

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
..venv\Scripts\Activate.ps1
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

### 4. Ensure model files are present

The repository ships with pre‑trained models under `models/`:

- `naive_bayes_model.pkl`
- `vectorizer.pkl`
- `nb_pipeline.pkl`
- `tokenizer.pkl`
- `lstm_model.h5`
- `model_miniLM_final/` (MiniLM transformer + tokenizer)

As long as these files are present in `models/`, no retraining is required to run the app.

### 5. Run the Streamlit app

From the repo root:



```
python -m streamlit run app/app.py
```

Streamlit will print a local URL such as `http://localhost:8501`.  
Open that URL in your browser to use the Fake Job Posting Detection app.

## Project Phases

### Phase 1: EDA & Feature Engineering
- Analyzed missing value patterns (83.96% salary_range missing)
- Discovered U-shaped fraud-completeness relationship
- Created 16 novel features (10 binary indicators + completeness score + patterns)
- Statistical validation with chi-square tests

### Phase 2: Baseline Modeling
- Logistic Regression with polynomial features
- Random Forest and XGBoost models
- Class imbalance handling (focal loss, cost-weighting, SMOTE)
- Performance on AUROC, PR-AUC, F1-score

### Phase 3: Advanced Interpretability (Next)
- LIME/SHAP explanations for individual predictions
- Sentence-level fraud indicators
- Interactive visualization dashboard
- User feedback loop for model improvement

### Phase 4: Documentation & Blog (Final)
- Project blog post explaining findings
- Code walkthroughs and methodology
- Presentation slides for in-class demo
- Peer evaluation

## Repository Structure

| Folder | Purpose |
|--------|---------|
| `notebooks/` | Jupyter notebooks for analysis pipeline |
| `src/` | Reusable Python modules for modeling |
| `data/` | Raw and processed datasets |
| `models/` | Trained model artifacts |
| `visualizations/` | EDA and model performance plots |
| `results/` | Metrics, feature importance, predictions |
| `docs/` | Documentation and blog content |

## Methods & Technologies

- **Data Processing**: Pandas, NumPy
- **Modeling**: Scikit-learn, XGBoost, LightGBM
- **Imbalance Handling**: SMOTE, focal loss, class weighting
- **Interpretability**: LIME, SHAP
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Documentation**: Jupyter, Markdown

## Module 1: Interpretable Ensemble

### Features
- **Multi-level Interpretability**: Feature importance, sentence attribution, linguistic patterns, red flags
- **Three-tier Risk Classification**: HIGH/MEDIUM/LOW risk levels with actionable recommendations
- **Rule-Based Boosting**: 8 red flags detect fraud indicators (urgency, payment requests, missing credentials)
- **Batch Processing**: Optimized implementation 20-50x faster than single predictions

### Performance
- **Accuracy**: 97.52%
- **F1-Score**: 0.714
- **Precision**: 80.98%
- **Total Fraud Detection**: 76.9% (HIGH + MEDIUM)
- **False Positive Rate**: 0.3% (HIGH), 4.5% (MEDIUM)

See `notebooks/06_module1_interpretability.ipynb` for full implementation and examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

**Existing Kaggle Solutions**:
- https://github.com/38832/Fake-Job-Posting-Prediction
- https://github.com/Anshupriya2694/Fake-Job-Posting-Prediction

**Last Updated**: November 6, 2025  
**Status**: Active Development
