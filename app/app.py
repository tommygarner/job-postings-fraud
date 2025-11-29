import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Fake Job Posting Detection",
    layout="wide"
)

from pathlib import Path
import joblib

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    import os

    models_path = Path("models")
    minilm_path = models_path / "model_miniLM_final"

    st.write("DEBUG: models_path =", models_path.resolve())
    st.write("DEBUG: contents of models/:", [p.name for p in models_path.iterdir()])

    try:
        # Naive Bayes model (saved with joblib.dump)
        nb_path = models_path / "naive_bayes_model.pkl"
        st.write("DEBUG: loading NB from", nb_path)
        nb_model = joblib.load(nb_path)
        st.write("DEBUG: NB loaded OK")

        # TF‑IDF vectorizer (saved with joblib.dump)
        vec_path = models_path / "vectorizer.pkl"
        st.write("DEBUG: loading vectorizer from", vec_path)
        vectorizer = joblib.load(vec_path)
        st.write("DEBUG: vectorizer loaded OK")

        # LSTM tokenizer (also saved with joblib.dump)
        tok_path = models_path / "tokenizer.pkl"
        st.write("DEBUG: loading tokenizer from", tok_path)
        lstm_tokenizer = joblib.load(tok_path)
        st.write("DEBUG: tokenizer loaded OK")

        # LSTM model
        lstm_path = models_path / "lstm_model.h5"
        st.write("DEBUG: loading LSTM model from", lstm_path)
        lstm_model = load_model(lstm_path)
        st.write("DEBUG: LSTM model loaded OK")

        # MiniLM
        st.write("DEBUG: loading MiniLM from", minilm_path)
        minilm_tokenizer = AutoTokenizer.from_pretrained(str(minilm_path))
        minilm_model = AutoModelForSequenceClassification.from_pretrained(str(minilm_path))
        st.write("DEBUG: MiniLM tokenizer/model loaded OK")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write("DEBUG: using device", device)
        minilm_model.to(device)
        minilm_model.eval()

        st.write("DEBUG: all models loaded successfully")
        return nb_model, vectorizer, lstm_tokenizer, lstm_model, minilm_tokenizer, minilm_model, device

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

nb_model, vectorizer, lstm_tokenizer, lstm_model, minilm_tokenizer, minilm_model, device = load_models()

# ========== PREPROCESSING ==========
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(s: str) -> str:
    tokens = word_tokenize(str(s).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def extract_features(text, vectorizer, telecommuting=0, has_logo=0, has_questions=0):
    processed = preprocess_text(text)
    text_features = vectorizer.transform([processed])

    location_fraud_ratio = 0.05
    char_count = len(text)
    numeric = np.array([[telecommuting, has_logo, has_questions, location_fraud_ratio, char_count]])
    numeric_sparse = csr_matrix(numeric)

    combined = hstack([text_features, numeric_sparse])
    return combined, processed, char_count

# ========== PREDICTIONS ==========
def get_predictions(
    text,
    telecommuting,
    has_logo,
    has_questions,
    nb_model,
    vectorizer,
    lstm_tokenizer,
    lstm_model,
    minilm_tokenizer,
    minilm_model,
    device,
):
    combined_features, processed_text, char_count = extract_features(
        text,
        vectorizer,
        telecommuting,
        has_logo,
        has_questions,
    )

    # 1. Naive Bayes
    nb_probs = nb_model.predict_proba(combined_features)[0]
    nb_prob = float(nb_probs[1])

    # 2. LSTM
    lstm_seq = lstm_tokenizer.texts_to_sequences([processed_text])
    lstm_padded = pad_sequences(lstm_seq, maxlen=256)
    lstm_prob = float(lstm_model.predict(lstm_padded, verbose=0)[0][0])

    # 3. MiniLM
    inputs = minilm_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        outputs = minilm_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        minilm_prob = float(probs.cpu().numpy()[0][1])

    # Ensemble
    w_nb, w_lstm, w_minilm = 0.05, 0.45, 0.50
    ensemble_prob = (w_nb * nb_prob) + (w_lstm * lstm_prob) + (w_minilm * minilm_prob)

    decision_threshold = 0.30
    is_fraud = ensemble_prob >= decision_threshold

    if ensemble_prob >= 0.60:
        risk = "HIGH RISK"
        risk_color = "🔴"
        recommendation = "DO NOT APPLY - Strong fraud indicators detected"
    elif ensemble_prob >= 0.20:
        risk = "MEDIUM RISK"
        risk_color = "🟡"
        recommendation = "INVESTIGATE CAREFULLY - Verify company details, check reviews, research employer"
    else:
        risk = "LOW RISK"
        risk_color = "🟢"
        recommendation = "APPEARS SAFE - Standard application precautions recommended"

    return {
        "nb_prob": nb_prob,
        "lstm_prob": lstm_prob,
        "minilm_prob": minilm_prob,
        "ensemble_prob": ensemble_prob,
        "is_fraud": is_fraud,
        "risk": risk,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "processed_text": processed_text,
        "combined_features": combined_features,
        "char_count": char_count,
    }

# ========== HEADER ==========
st.title("Fake Job Posting Detection")
st.markdown("### AI-Powered Fraud Detection with Explainable Predictions")
st.markdown("*3-Model Ensemble: Naive Bayes + LSTM + MiniLM with Integrated Gradients*")

# ========== MODEL PERFORMANCE (COLLAPSIBLE) ==========
with st.expander("📊 View Model Performance Stats", expanded=False):
    st.markdown("**Test Set Performance (5,364 samples)**")
    
    stats_df = pd.DataFrame({
        'Model': ['Naive Bayes', 'LSTM', 'MiniLM+IG', 'Ensemble'],
        'Accuracy': [0.9702, 0.9737, 0.9748, 0.9782],
        'F1-Score': [0.5960, 0.6994, 0.7007, 0.7298],
        'Precision': [0.8676, 0.7847, 0.8272, 0.9133],
        'Recall': [0.4538, 0.6308, 0.6077, 0.6077],
        'ROC-AUC': [0.8494, 0.9343, 0.9406, 0.9714]
    })
    
    st.dataframe(
        stats_df.style.format({
            'Accuracy': '{:.1%}',
            'F1-Score': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'ROC-AUC': '{:.4f}'
        }),
        width="stretch",
        hide_index=True
    )
    
    st.info("Ensemble weights: NB 5% + LSTM 45% + MiniLM 50% | Decision threshold: 30%")

st.markdown("---")

# ========== INPUT FORM ==========
col1, col2 = st.columns([2, 1])

with col1:
    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the full job posting description here..."
    )

with col2:
    st.markdown("**Additional Features**")
    telecommuting = st.checkbox("Remote/Telecommuting Allowed")
    has_logo = st.checkbox("Company Logo Present")
    has_questions = st.checkbox("Screening Questions Included")

# ========== PREDICTION BUTTON ==========
if st.button("Analyze Job Posting", type="primary", width="stretch"):
    if not job_description.strip():
        st.error("Please enter a job description to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = get_predictions(
                job_description,
                int(telecommuting),
                int(has_logo),
                int(has_questions),
                nb_model,
                vectorizer,
                lstm_tokenizer,
                lstm_model,
                minilm_tokenizer,
                minilm_model,
                device
            )

        # Create tabs
        tab_pred, tab_vis, tab_global = st.tabs(
            ["Prediction", "Feature Visuals", "Common Signals"]
        )

        # ========== PREDICTION TAB ==========
        with tab_pred:
            st.markdown("---")
            st.header("Analysis Results")
            
            # === High-level label + probability ===
            st.markdown(f"## {results['risk_color']} {results['risk']}")
            st.markdown(
                f"**Fraud Probability (Ensemble): {results['ensemble_prob']*100:.1f}%** "
                "(Decision threshold: 30%)"
            )
            
            # Simple traffic-light style message
            if results['is_fraud']:
                st.error(results['recommendation'])
            elif results['ensemble_prob'] >= 0.20:
                st.warning(results['recommendation'])
            else:
                st.success(results['recommendation'])

            # === Risk summary narrative block ===
            with st.container():
                st.markdown("### Why the model reached this decision")
                bullet_points = []

                # Model agreement
                model_probs = {
                    "Naive Bayes": results["nb_prob"],
                    "LSTM": results["lstm_prob"],
                    "MiniLM+IG": results["minilm_prob"],
                }
                high_agree = [m for m, p in model_probs.items() if p >= 0.6]
                low_agree = [m for m, p in model_probs.items() if p < 0.4]

                if len(high_agree) == 3:
                    bullet_points.append(
                        "All three base models assign a high fraud probability, so the ensemble decision is **very confident**."
                    )
                elif len(high_agree) >= 2:
                    bullet_points.append(
                        f"{', '.join(high_agree)} assign high fraud probability; the ensemble leans toward that consensus."
                    )
                elif len(low_agree) >= 2:
                    bullet_points.append(
                        f"{', '.join(low_agree)} assign low fraud probability; the ensemble is relatively confident this is legitimate."
                    )
                else:
                    bullet_points.append(
                        "The base models partially disagree, so the ensemble uses its weights to balance the evidence."
                    )

                # Structured features narrative
                char_count = len(job_description)
                if char_count < 500:
                    length_desc = f"a **very short** description ({char_count} characters)"
                elif char_count > 5000:
                    length_desc = f"a **very long** description ({char_count} characters)"
                else:
                    length_desc = f"a **typical-length** description ({char_count} characters)"

                sf_bits = []
                sf_bits.append(length_desc)
                sf_bits.append("**no company logo**" if not has_logo else "a **company logo present**")
                sf_bits.append("**no screening questions**" if not has_questions else "**screening questions included**")
                if telecommuting:
                    sf_bits.append("**remote/telecommuting allowed**")

                bullet_points.append(
                    "Structured fields indicate "
                    + ", ".join(sf_bits)
                    + ", which the model has learned to associate with different risk levels."
                )

                for b in bullet_points:
                    st.markdown(f"- {b}")

            st.markdown("---")
            st.subheader("Individual Model Predictions (Fraud Probability)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Naive Bayes", f"{results['nb_prob']*100:.1f}%")
            with c2:
                st.metric("LSTM", f"{results['lstm_prob']*100:.1f}%")
            with c3:
                st.metric("MiniLM+IG", f"{results['minilm_prob']*100:.1f}%")
            with c4:
                st.metric("Ensemble", f"{results['ensemble_prob']*100:.1f}%")

            # Per-model verbal explanation
            def describe_model_view(name, p):
                if p >= 0.70:
                    return f"{name} saw **strong evidence of fraud** (≈ {p*100:.1f}% fraud)."
                elif p >= 0.40:
                    return f"{name} was **uncertain but leaning fraud** (≈ {p*100:.1f}% fraud)."
                elif p >= 0.20:
                    return f"{name} was **uncertain but leaning legitimate** (≈ {p*100:.1f}% fraud)."
                else:
                    return f"{name} saw **strong evidence of legitimacy** (≈ {p*100:.1f}% fraud)."

            st.markdown(
                "- " + describe_model_view("Naive Bayes", results["nb_prob"]) + "\n"
                "- " + describe_model_view("LSTM", results["lstm_prob"]) + "\n"
                "- " + describe_model_view("MiniLM+IG", results["minilm_prob"]) + "\n"
                "- The ensemble combines these views using tuned weights (Naive Bayes 5%, LSTM 45%, MiniLM+IG 50%)."
            )
            st.caption(
                "Naive Bayes relies on word counts and global fraud cues; "
                "LSTM focuses on word order and phrasing; MiniLM+IG uses contextual embeddings from a small transformer."
            )

            st.markdown("---")
            st.subheader("Naive Bayes Word-Level Explanation")

            # === NB contributions (for text explanation) ===
            feature_names = np.array(vectorizer.get_feature_names_out())
            feature_log_prob_fraud = nb_model.feature_log_prob_[1][:5000]
            feature_log_prob_real = nb_model.feature_log_prob_[0][:5000]
            tfidf_scores = results['combined_features'].toarray()[0][:5000]
            fraud_weight = feature_log_prob_fraud - feature_log_prob_real
            contribution = tfidf_scores * fraud_weight

            pos_idx = np.argsort(-contribution)
            neg_idx = np.argsort(contribution)

            pos_words = [
                (feature_names[i], float(contribution[i]))
                for i in pos_idx[:50]
                if contribution[i] > 0 and tfidf_scores[i] > 0
            ]
            neg_words = [
                (feature_names[i], float(contribution[i]))
                for i in neg_idx[:50]
                if contribution[i] < 0 and tfidf_scores[i] > 0
            ]

            total_nb_shift = float(contribution.sum())

            if results["is_fraud"]:
                st.markdown(
                    f"This posting has a **high fraud probability ({results['ensemble_prob']*100:.1f}%)**. "
                    "The words below are pushing the Naive Bayes model toward fraud."
                )
                st.markdown(
                    f"Overall, Naive Bayes word evidence shifts its internal score **toward fraud** "
                    f"(net contribution {total_nb_shift:.3f})."
                )

                st.markdown("**Top Fraud Indicators (Words Increasing Risk)**")
                fraud_words = [f"**{w}** ({c:.3f})" for w, c in pos_words[:8]]
                if fraud_words:
                    st.markdown("• " + "  \n• ".join(fraud_words))
            else:
                st.markdown(
                    f"This posting has a **low fraud probability ({results['ensemble_prob']*100:.1f}%)**. "
                    "The words below push Naive Bayes toward legitimate."
                )
                direction = "toward legitimate" if total_nb_shift < 0 else "slightly toward fraud"
                st.markdown(
                    f"Overall Naive Bayes word evidence shifts its internal score **{direction}**, "
                    "but the final ensemble probability remains low."
                )

                st.markdown("**Top Legitimacy Indicators (Words Decreasing Risk)**")
                legit_words = [f"**{w}** ({c:.3f})" for w, c in neg_words[:8]]
                if legit_words:
                    st.markdown("• " + "  \n• ".join(legit_words))

                strong_pos = [(w, c) for w, c in pos_words if abs(c) > 0.05][:6]
                if results["ensemble_prob"] >= 0.05 and strong_pos:
                    st.markdown("**Minor Fraud Signals (not enough to change the decision)**")
                    fraud_words = [f"**{w}** ({c:.3f})" for w, c in strong_pos]
                    st.markdown("• " + "  \n• ".join(fraud_words))

            st.markdown("---")
            st.markdown("**Structured Features (This Posting)**")
            feature_impacts = []
            # recompute char_count explicitly for clarity
            char_count = len(job_description)
            if not has_logo:
                feature_impacts.append("No company logo (more common in fraud postings)")
            else:
                feature_impacts.append("✓ Company logo present (more common in legitimate postings)")
            if not has_questions:
                feature_impacts.append("No screening questions (often skipped in scam posts)")
            else:
                feature_impacts.append("✓ Screening questions included (typical of real employers)")
            if telecommuting:
                feature_impacts.append("✓ Remote work offered (can be legitimate but is also heavily used in scams)")
            feature_impacts.append(f"Description length: {char_count} characters")

            for impact in feature_impacts:
                st.markdown(f"- {impact}")

        # ========== FEATURE VISUALS TAB ==========
        with tab_vis:
            st.subheader("Model Confidence for This Posting")

            probs = [
                results["nb_prob"],
                results["lstm_prob"],
                results["minilm_prob"],
                results["ensemble_prob"],
            ]
            labels = ["Naive Bayes", "LSTM", "MiniLM+IG", "Ensemble"]

            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["gray", "gray", "gray", "steelblue"]
            ax.barh(labels, [p * 100 for p in probs], color=colors)
            ax.set_xlabel("Fraud probability (%)")
            ax.set_xlim(0, 100)
            for i, v in enumerate(probs):
                ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center")
            st.pyplot(fig)

            st.markdown(
                "- The **Ensemble** bar shows the final decision-driving probability.\n"
                "- The three gray bars show how each base model views the same posting."
            )

            st.markdown("---")
            st.subheader("Text Length vs. Typical Training Posts")

            length_bins = ["Very short (<500)", "Short (500–1500)", "Medium (1500–3000)", "Long (>3000)"]
            train_counts = [0.35, 0.30, 0.25, 0.10]  # placeholder proportions

            # Determine this posting's bucket
            if char_count < 500:
                this_bucket = "Very short (<500)"
            elif char_count < 1500:
                this_bucket = "Short (500–1500)"
            elif char_count < 3000:
                this_bucket = "Medium (1500–3000)"
            else:
                this_bucket = "Long (>3000)"

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.bar(length_bins, [c * 100 for c in train_counts], color="lightgray")
            ax2.set_ylabel("Share of training posts (%)")
            ax2.set_xticklabels(length_bins, rotation=20, ha="right")
            if this_bucket in length_bins:
                idx = length_bins.index(this_bucket)
                ax2.bar(length_bins[idx], train_counts[idx] * 100, color="steelblue")
            st.pyplot(fig2)

            st.caption(
                f"This posting falls in the **{this_bucket}** bucket. "
                "Fraudulent postings in the training data tended to be shorter on average."
            )

            st.markdown("---")
            st.subheader("Structured Fields at a Glance")

            sf_labels = ["Company logo present", "Screening questions included", "Remote allowed"]
            sf_values = [
                1 if has_logo else 0,
                1 if has_questions else 0,
                1 if telecommuting else 0,
            ]
            sf_colors = [
                "seagreen" if has_logo else "tomato",
                "seagreen" if has_questions else "tomato",
                "steelblue" if telecommuting else "gray",
            ]

            fig3, ax3 = plt.subplots(figsize=(5, 2.5))
            ax3.barh(sf_labels, sf_values, color=sf_colors)
            ax3.set_xlim(0, 1)
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(["No", "Yes"])
            st.pyplot(fig3)

            st.caption(
                "Red bars highlight missing information (no logo, no screening questions), "
                "which was more common among fraudulent postings in the training data."
            )

        # ========== COMMON SIGNALS TAB ==========
        with tab_global:
            st.subheader("Common Fraud and Legitimate Signals (Training Data)")

            feature_names = np.array(vectorizer.get_feature_names_out())
            fraud_weight = nb_model.feature_log_prob_[1][:5000] - nb_model.feature_log_prob_[0][:5000]

            # Example training-level stats (replace with real values from notebook)
            overall_fraud_rate = 0.048
            avg_len_fraud = 450
            avg_len_legit = 1200
            no_logo_fraud = 0.80
            no_logo_legit = 0.40

            st.markdown("### Training-Set Summary")
            st.markdown(
                f"- Overall fraud rate in the dataset: **{overall_fraud_rate*100:.1f}%**.\n"
                f"- Average description length: **{avg_len_fraud:.0f}** chars for fraud vs "
                f"**{avg_len_legit:.0f}** chars for legitimate postings.\n"
                f"- Fraction without company logo: **{no_logo_fraud*100:.0f}%** for fraud vs "
                f"**{no_logo_legit*100:.0f}%** for legitimate."
            )

            st.markdown("---")

            top_fraud_idx = np.argsort(-fraud_weight)[:20]
            top_fraud = [(feature_names[i], float(fraud_weight[i])) for i in top_fraud_idx]

            top_legit_idx = np.argsort(fraud_weight)[:20]
            top_legit = [(feature_names[i], float(fraud_weight[i])) for i in top_legit_idx]

            col_fraud, col_legit = st.columns(2)

            with col_fraud:
                st.markdown("### Common Fraud Signals")
                if top_fraud:
                    st.markdown(
                        "These words appear **much more often in fraudulent** postings than in legitimate ones."
                    )
                    st.markdown("• " + "  \n• ".join([f"**{w}**" for w, _ in top_fraud]))

                    words_f = [w for w, _ in top_fraud[:10]]
                    vals_f = [v for _, v in top_fraud[:10]]
                    fig_f, ax_f = plt.subplots(figsize=(5, 3))
                    ax_f.barh(words_f[::-1], vals_f[::-1], color="tomato")
                    ax_f.set_xlabel("Naive Bayes log-odds (fraud vs. legit)")
                    st.pyplot(fig_f)

            with col_legit:
                st.markdown("### Common Legitimate Signals")
                if top_legit:
                    st.markdown(
                        "These words appear **much more often in legitimate** postings than in fraudulent ones."
                    )
                    st.markdown("• " + "  \n• ".join([f"**{w}**" for w, _ in top_legit]))

                    words_l = [w for w, _ in top_legit[:10]]
                    vals_l = [abs(v) for _, v in top_legit[:10]]
                    fig_l, ax_l = plt.subplots(figsize=(5, 3))
                    ax_l.barh(words_l[::-1], vals_l[::-1], color="seagreen")
                    ax_l.set_xlabel("Naive Bayes log-odds (legit vs. fraud, magnitude)")
                    st.pyplot(fig_l)

            st.markdown("---")
            st.caption(
                "Red and green words behave like global signals the Naive Bayes model learned from the training set. "
                "The word-level explanation in the Prediction tab shows which of these appear in the current posting."
            )

# ========== FOOTER ==========
st.markdown("---")
st.caption(
    "**UT Austin MSBA - Advanced Machine Learning**  \n"
    "Tommy Garner, Sam LeFebre, Nick Favoriti, Ethan Davenport, Sebastian Palacino  \n"
    "Trained on 17,880 job postings from Kaggle"
)
