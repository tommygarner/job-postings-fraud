## Trained Models

This directory contains trained models and preprocessing artifacts for fake job posting detection [attached_file:1].

## Files

| File                  | Description                                  | Size (approx.) | Format  |
|-----------------------|----------------------------------------------|----------------|---------|
| `naive_bayes_model.pkl` | Naive Bayes on TF-IDF + numeric features [attached_file:1]. | ~5–10 MB       | Pickle |
| `lstm_model.h5`       | LSTM for sequential fraud patterns [attached_file:1].        | ~50–100 MB     | Keras HDF5 |
| `vectorizer.pkl`      | TF-IDF vectorizer (~5,000 features) [attached_file:1].       | ~20–30 MB      | Pickle |
| `tokenizer.pkl`       | Keras tokenizer (vocab_size ~10,000) [attached_file:1].      | ~2–5 MB        | Pickle |

## Model Performance

### Final Ensemble (NB + LSTM + Rules)

| Metric          | Value   | Description                                  |
|-----------------|---------|----------------------------------------------|
| Accuracy        | 97.52%  | Overall correct predictions [attached_file:1]. |
| F1-Score        | 0.7140  | Harmonic mean of precision and recall [attached_file:1]. |
| Precision       | 80.98%  | Correct when flagged as fraud ~81% of time [attached_file:1]. |
| Recall          | 63.85%  | Directly catches ~64% of frauds [attached_file:1]. |
| Total Detection | 76.9%   | Frauds in HIGH or MEDIUM risk [attached_file:1]. |

## Risk-Based Performance

| Risk Level      | Fraud Detection (260 fraud) | False Positive Rate (5,104 real) |
|-----------------|-----------------------------|-----------------------------------|
| 🔴 **HIGH**     | 59.2% (154/260)             | 0.3% (17/5,104) [attached_file:1]. |
| 🟡 **MEDIUM**   | 17.7% (46/260)              | 4.5% (229/5,104) [attached_file:1]. |
| 🟢 **LOW**      | 23.1% (60/260)              | 95.2% (4,858/5,104) [attached_file:1]. |

## Individual Models

| Model        | Mean Score (Fraud) | Mean Score (Real) | Strength                               |
|-------------|--------------------|-------------------|----------------------------------------|
| **Naive Bayes** | 31.76%          | 1.49%             | Fast, interpretable TF-IDF features [attached_file:1]. |
| **LSTM**        | 60.43%          | 2.82%             | Strong on text sequence patterns [attached_file:1]. |

## Architectures

### Naive Bayes

- Input: TF-IDF (≈5,000 features) + selected numeric signals [attached_file:1].  
- Role: 25% of ensemble weight; interpretable baseline [attached_file:1].  

### LSTM

- Input: Tokenized, padded job descriptions via `tokenizer.pkl` [attached_file:1].  
- Role: 75% of ensemble weight; captures scam phrasing [attached_file:1].  

### Ensemble + Rules

- Blend: `0.25 * NB_score + 0.75 * LSTM_score` [attached_file:1].  
- Risk: Post-processing rules → HIGH / MEDIUM / LOW buckets [attached_file:1].  
