# Project Outline

## Problem Statement
Fraudulent job postings mislead job seekers. We build an ML system to detect fake postings 
with interpretable explanations.

## Key Hypotheses

1. **H1**: Missing metadata fields are stronger fraud predictors than text features
   - Status: VALIDATED (chi-square p < 0.001)

2. **H2**: Tree-based models outperform linear models due to U-shaped relationship
   - Status: TESTING (Phase 2)

3. **H3**: Completeness score + pattern features outperform baseline by >5%
   - Status: PENDING (Phase 2)

## Deliverables

### Technical
- [ ] Feature-engineered dataset
- [ ] Trained models (XGBoost, Random Forest, ensemble)
- [ ] Interpretability pipeline (LIME/SHAP)
- [ ] Performance report
- [ ] Clean GitHub repository

### Documentation
- [ ] Blog post
- [ ] Presentation slides (15-20 min)
- [ ] Code walkthroughs
- [ ] Model card

## Timeline

| Week | Focus | Tasks |
|------|-------|-------|
| 1 (Nov 6-10) | Setup | Repo, roles, approach |
| 2 (Nov 11-17) | EDA | Features, validation |
| 3 (Nov 18-24) | Baseline | Logistic Reg, RF, XGBoost |
| 4 (Nov 25-Dec 1) | Advanced | Focal loss, ensemble |
| 5 (Dec 2-8) | Interpretability | LIME/SHAP |
| 6 (Dec 9-11) | Finalization | Blog, presentation |

## Success Criteria

- AUROC > 0.95, PR-AUC > 0.85
- 16 engineered features not in baseline repos
- Users understand why posting is flagged
- Clear, reproducible code
- All members contributed meaningfully

## Team Roles (Suggested)

- **Data Lead**: EDA, feature engineering
- **ML Engineer**: Model training, hyperparameter tuning
- **Interpretability Lead**: LIME/SHAP, visualization
- **Documentation Lead**: Blog, presentation
