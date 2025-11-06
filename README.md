# Fake Job Posting Detection: Advanced ML with Interpretability

**Team**: [Add team members]  
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

- **Source**: [Kaggle - Real or Fake Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: 17,880 postings with 18 features
- **Fraud Rate**: 4.84% (severe class imbalance)
- **Target**: Binary classification (fraudulent: 0/1)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-job-posting-detection.git
cd fake-job-posting-detection
```

### 2. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Using kaggle CLI
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction
unzip real-or-fake-fake-jobposting-prediction.zip -d data/raw/
```

### 4. Run EDA
```bash
cd notebooks
jupyter notebook 01_eda_summary.ipynb
```

## Project Phases

### Phase 1: EDA & Feature Engineering ✓ (Complete)
- Analyzed missing value patterns (83.96% salary_range missing)
- Discovered U-shaped fraud-completeness relationship
- Created 16 novel features (10 binary indicators + completeness score + patterns)
- Statistical validation with chi-square tests

### Phase 2: Baseline Modeling (In Progress)
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

## Key Findings

### The Fraud Paradox
Both extremely low completeness (score 0, 22.2% fraud) and perfect completeness (score 10, 9.3% fraud) 
are fraud risks. Legitimate postings cluster around scores 6-7 (2.5% fraud rate).

**Implication**: Linear models fail; tree-based models recommended.

### Strongest Individual Predictors
- Missing company_profile: +15.83pp fraud rate increase
- no_company_info pattern: 24.45% fraud rate (5x baseline)
- completeness_score: U-shaped relationship (p < 0.001)

### Two Fraud Archetypes
1. **Amateur Scams**: Bare bones postings with minimal information
2. **Sophisticated Scams**: Detailed, "too good to be true" offerings

## Methods & Technologies

- **Data Processing**: Pandas, NumPy
- **Modeling**: Scikit-learn, XGBoost, LightGBM
- **Imbalance Handling**: SMOTE, focal loss, class weighting
- **Interpretability**: LIME, SHAP
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Documentation**: Jupyter, Markdown

## Performance Goals

| Metric | Target |
|--------|--------|
| AUROC | > 0.95 |
| PR-AUC | > 0.85 |
| Precision | > 0.90 |
| Recall | > 0.80 |
| F1-Score | > 0.85 |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Timeline

| Date | Milestone |
|------|-----------|
| Nov 6 | Repo setup, team organization |
| Nov 15 | EDA complete, features engineered |
| Nov 22 | Baseline models trained |
| Nov 29 | Advanced models & interpretability |
| Dec 6 | Final presentation ready |
| Dec 11 | Blog post & code submission |

## References

**Existing Kaggle Solutions**:
- https://github.com/38832/Fake-Job-Posting-Prediction
- https://github.com/Anshupriya2694/Fake-Job-Posting-Prediction

**Last Updated**: November 6, 2025  
**Status**: Active Development
