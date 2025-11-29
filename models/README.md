## Trained Models

This directory contains trained models and preprocessing artifacts for fake job posting detection.

## Files

| File                    | Description                                | Size (approx.) | Format     |
|-------------------------|--------------------------------------------|----------------|------------|
| `naive_bayes_model.pkl` | Naive Bayes on TF-IDF + numeric features   | ~5–10 MB       | Pickle     |
| `lstm_model.h5`         | LSTM for sequential fraud patterns         | ~50–100 MB     | Keras HDF5 |
| `vectorizer.pkl`        | TF-IDF vectorizer (~5,000 features)        | ~20–30 MB      | Pickle     |
| `tokenizer.pkl`         | Keras tokenizer (vocab_size ~10,000)       | ~2–5 MB        | Pickle     |

## Model Performance

### Final Ensemble (NB + LSTM + Rules)

| Metric          | Value   | Description                         |
|-----------------|---------|-------------------------------------|
| Accuracy        | 97.52%  | Overall correct predictions         |
| F1-Score        | 0.7140  | Balance of precision and recall     |
| Precision       | 80.98%  | Correct when flagged as fraud       |
| Recall          | 63.85%  | Frauds directly caught              |
| Total Detection | 76.9%   | Frauds in HIGH or MEDIUM risk       |

## Risk-Based Performance

| Risk Level | Fraud Detection (n=260) | False Positive Rate (n=5,104) |
|-----------|--------------------------|-------------------------------|
| HIGH   | 59.2% (154/260)          | 0.3% (17/5,104)               |
| MEDIUM | 17.7% (46/260)           | 4.5% (229/5,104)              |
| LOW    | 23.1% (60/260)           | 95.2% (4,858/5,104)           |

## Individual Models

| Model        | Mean Score (Fraud) | Mean Score (Real) | Strength                     |
|--------------|---------------------|--------------------|------------------------------|
| Naive Bayes  | 31.76%              | 1.49%              | Fast, interpretable TF-IDF   |
| LSTM         | 60.43%              | 2.82%              | Strong on sequence patterns  |

## Architectures

### Naive Bayes

- Input: TF-IDF (~5,000 features) + numeric signals  
- Role: 25% of ensemble weight  

### LSTM

- Input: Tokenized, padded job descriptions via `tokenizer.pkl`  
- Role: 75% of ensemble weight  

### Ensemble + Rules

- Blend: `0.25 * NB_score + 0.75 * LSTM_score`  
- Risk: Rules map scores to HIGH / MEDIUM / LOW  

## Strengths

- Extremely low false positive rate in the HIGH-risk tier, indicating strong reliability when the model identifies a posting as high-risk.
- High precision, meaning that when the model flags a job as fraudulent, it is usually correct.
- The risk-tier system provides operational value by enabling differentiated levels of scrutiny.
- The ensemble combines complementary model types, capturing both TF-IDF features and sequential text patterns.
- High overall accuracy despite dataset imbalance.

## Weaknesses

- Recall is moderate, meaning a significant portion of fraudulent postings are not detected.
- The Medium-risk tier has a higher false positive rate, potentially increasing manual review workload.
- A notable proportion of fraudulent postings fall into the Low-risk tier, limiting full automation.
- The F1-score indicates a stronger focus on precision than comprehensive fraud detection.
- The ensemble relies heavily on the LSTM’s performance, limiting diversity in model behavior.
