# Project Setup Guide for Team

## For Each Team Member

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-job-posting-detection.git
cd fake-job-posting-detection
```

### 2. Run Setup Script
```bash
bash setup.sh
source venv/bin/activate
```

### 3. Download Dataset
```bash
# Install Kaggle CLI if you don't have it
pip install kaggle

# Download dataset (need Kaggle account)
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction

# Extract to data/raw/
unzip real-or-fake-fake-jobposting-prediction.zip -d data/raw/
```

### 4. Verify Installation
```bash
python -c "import pandas, sklearn, xgboost; print('✅ All packages installed!')"
```

## Team Folder Assignments (Suggested)

- **Data Lead**: `data/`, `notebooks/01_eda_summary.ipynb`
- **ML Engineer**: `src/model_training.py`, `notebooks/03_04_models.ipynb`
- **Interpretability Lead**: `src/interpretability.py`, `notebooks/05_interpretability.ipynb`
- **Documentation Lead**: `docs/`, `README.md` updates

## File Structure Summary

```
fake-job-posting-detection/
├── data/
│   ├── raw/                    → Raw Kaggle dataset
│   ├── processed/              → Engineered features (CSV)
│   └── DATASET_INFO.md
├── notebooks/                  → Jupyter analysis
├── src/                        → Python modules
├── models/                     → Trained models
├── visualizations/             → Plots & charts
├── results/                    → Metrics, feature importance
├── docs/                       → Blog, documentation
├── README.md                   → Project overview
├── requirements.txt            → Package dependencies
├── .gitignore                  → What to ignore in git
├── CONTRIBUTING.md             → Contribution guidelines
└── project_outline.md          → Project plan
```

## Common Commands

```bash
# Update your local copy
git pull origin main

# Create new feature branch
git checkout -b feature/your-task

# View your changes
git status

# Stage and commit
git add .
git commit -m "Your message"

# Push to GitHub
git push origin feature/your-task

# Create Pull Request on GitHub web interface
```

## Weekly Sync Checklist

- [ ] What did each person complete?
- [ ] What blockers came up?
- [ ] What's the priority for next week?
- [ ] Any merge conflicts to discuss?

## Questions?

Create a GitHub Issue in the repository!
