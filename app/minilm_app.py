import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import numpy as np
import joblib
import shap
import scipy.sparse as sp

st.write("✅ Script imports completed")

# ------------------------------
# 1. Load MiniLM model and tokenizer
# ------------------------------
@st.cache_resource
def load_minilm():
    try:
        st.write("🔄 Attempting to load MiniLM model from models/model_miniLM_final...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/model_miniLM_final"
        )
        st.write("✅ MiniLM model loaded")

        tokenizer = AutoTokenizer.from_pretrained(
            "models/model_miniLM_final"
        )
        st.write("✅ MiniLM tokenizer loaded")

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        st.write(f"✅ MiniLM model moved to device: {device}")
        return model, tokenizer, device
    except Exception as e:
        st.error("❌ Error loading MiniLM model:")
        st.exception(e)
        st.stop()

st.write("🔄 Calling load_minilm()...")
model, tokenizer, device = load_minilm()
st.write("✅ MiniLM ready")

# Integrated Gradients instance (fraud class = index 0)
ig = IntegratedGradients(
    lambda inputs_embeds, attention_mask: F.softmax(
        model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits,
        dim=1,
    )[:, 0]
)

# ------------------------------
# 1b. Load Naive Bayes + vectorizer + build SHAP
# ------------------------------
@st.cache_resource
def load_nb_with_shap():
    st.write("🔄 Loading Naive Bayes + vectorizer...")
    nb_model = joblib.load("models/naive_bayes_model.pkl")
    nb_vectorizer = joblib.load("models/vectorizer.pkl")
    st.write("✅ Naive Bayes + vectorizer loaded")

    # Use the model's feature dimension to build background
    expected = getattr(nb_model, "n_features_in_", None)
    if expected is None:
        expected = len(nb_vectorizer.vocabulary_)

    import scipy.sparse as sp
    st.write(f"🔄 Building NB SHAP background with {expected} features...")
    X_bg_sparse = sp.csr_matrix((50, expected))  # 50 rows, expected feature dim
    X_bg_dense = X_bg_sparse.toarray()

    def nb_predict_proba_pos(X_dense):
        X_csr = sp.csr_matrix(X_dense)
        exp = getattr(nb_model, "n_features_in_", X_csr.shape[1])
        if X_csr.shape[1] != exp:
            # pad or truncate to match expected
            if X_csr.shape[1] < exp:
                pad_width = exp - X_csr.shape[1]
                X_csr = sp.hstack([X_csr, sp.csr_matrix((X_csr.shape[0], pad_width))])
            else:
                X_csr = X_csr[:, :exp]
        return nb_model.predict_proba(X_csr)[:, 1]

    st.write("🔄 Building NB SHAP KernelExplainer (this may take a moment)...")
    nb_explainer = shap.KernelExplainer(nb_predict_proba_pos, X_bg_dense)
    st.write("✅ NB SHAP explainer ready")

    return nb_model, nb_vectorizer, nb_explainer

nb_model, nb_vectorizer, nb_explainer = load_nb_with_shap()

# ------------------------------
# 2. MiniLM helper: predict fraud prob
# ------------------------------
def predict_fraud_prob(text: str) -> float:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)
        fraud_prob = probs[0, 0].item()  # class 0 = fraud
    return fraud_prob

# ------------------------------
# 3. MiniLM helper: IG word attributions
# ------------------------------
def get_ig_attributions(text: str, n_steps: int = 50):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    ).to(device)

    word_embeddings = model.base_model.get_input_embeddings()
    inputs_embeds = word_embeddings(encoded["input_ids"]).detach().requires_grad_(True)
    baselines = torch.zeros_like(inputs_embeds).to(device)

    attributions, delta = ig.attribute(
        inputs=inputs_embeds,
        baselines=baselines,
        additional_forward_args=(encoded["attention_mask"],),
        return_convergence_delta=True,
        n_steps=n_steps,
    )

    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    denom = torch.norm(attributions_sum) + 1e-8
    attributions_norm = attributions_sum / denom

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attrs = attributions_norm.detach().cpu().numpy()

    token_attr = [
        (tok, float(attr))
        for tok, attr in zip(tokens, attrs)
        if tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    return token_attr, float(delta.item())

# ------------------------------
# 3b. Naive Bayes + SHAP helper
# ------------------------------
def predict_nb_with_shap(text: str, top_k: int = 15, nsamples: int = 200):
    X_sparse = nb_vectorizer.transform([text])
    expected = getattr(nb_model, "n_features_in_", X_sparse.shape[1])
    if X_sparse.shape[1] != expected:
        if X_sparse.shape[1] < expected:
            pad_width = expected - X_sparse.shape[1]
            X_sparse = sp.hstack([X_sparse, sp.csr_matrix((1, pad_width))])
        else:
            X_sparse = X_sparse[:, :expected]
    X_dense = X_sparse.toarray()

    # Assume class 1 = fraud
    probs = nb_model.predict_proba(X_sparse)[0]
    nb_fraud_prob = float(probs[1])

    # SHAP values (KernelExplainer returns list)
    vals = nb_explainer.shap_values(X_dense, nsamples=nsamples)
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.array(vals).reshape(-1)

    try:
        terms = nb_vectorizer.get_feature_names_out()
    except Exception:
        terms = np.array([f"f{i}" for i in range(X_dense.shape[1])])

    if len(terms) != expected:
        terms = terms[:expected]
        vals = vals[:expected]

    idx = np.argsort(-np.abs(vals))[:top_k]
    token_attr_sorted = [(terms[i], float(vals[i])) for i in idx]

    return nb_fraud_prob, token_attr_sorted

# ------------------------------
# 4. Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fake Job Posting Detector (MiniLM + NB)", layout="wide")

st.title("Fake Job Posting Detector – MiniLM + Naive Bayes")
st.markdown(
    "Paste a job posting below to see fraud probability and which words "
    "contribute most to the prediction. MiniLM uses Integrated Gradients; "
    "Naive Bayes uses SHAP."
)

job_text = st.text_area(
    "Job Posting Text",
    height=250,
    placeholder="Paste full job description here...",
)

analyze = st.button("Analyze Job Posting")

if analyze:
    if not job_text.strip():
        st.warning("Please enter a job posting.")
    else:
        try:
            with st.spinner("Analyzing with MiniLM + Naive Bayes..."):
                fraud_prob = predict_fraud_prob(job_text)
                token_attr_sorted, delta = get_ig_attributions(job_text)

                nb_fraud_prob = None
                nb_token_attr_sorted = []
                try:
                    nb_fraud_prob, nb_token_attr_sorted = predict_nb_with_shap(job_text, nsamples=50)
                except Exception as e:
                    st.warning("Naive Bayes + SHAP explanation failed on this platform; showing MiniLM only.")
                    st.text(str(e))

                if nb_fraud_prob is not None:
                    ensemble_prob = (fraud_prob + nb_fraud_prob) / 2.0
                else:
                    ensemble_prob = fraud_prob
        except Exception as e:
            st.exception(e)
        else:
            # -------- Prediction panel --------
            if ensemble_prob > 0.60:
                risk_label = "HIGH RISK"
                risk_color = "red"
                rec = "DO NOT APPLY – strong indicators of fraud."
            elif ensemble_prob > 0.30:
                risk_label = "MEDIUM RISK"
                risk_color = "orange"
                rec = "INVESTIGATE CAREFULLY – some suspicious patterns."
            else:
                risk_label = "LOW RISK"
                risk_color = "green"
                rec = "APPEARS SAFE – no major red flags detected."

            st.subheader("Prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MiniLM Fraud", f"{fraud_prob*100:.1f}%")
            with col2:
                st.metric("Naive Bayes Fraud", f"{nb_fraud_prob*100:.1f}%")
            with col3:
                st.metric("Ensemble Fraud", f"{ensemble_prob*100:.1f}%")

            st.markdown(
                f"<h3 style='color:{risk_color}; margin-top:0.25rem;'>Overall Risk: {risk_label}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Recommendation:** {rec}")

            # -------- Combined overlay (MiniLM IG + NB SHAP) --------
            st.markdown("---")
            st.subheader("Word Importance Overlay (MiniLM + Naive Bayes)")

            # Map NB SHAP scores by lowercased term
            nb_scores = {}
            for term, val in nb_token_attr_sorted:
                nb_scores[term.lower()] = val

            spans = []
            for tok, ig_attr in token_attr_sorted:
                clean_tok = tok.replace("##", "")
                if clean_tok.strip() == "":
                    continue

                key = clean_tok.lower()

                ig = ig_attr
                nb = nb_scores.get(key, 0.0)

                # simple combination: average
                combined = 0.5 * ig + 0.5 * nb

                # map magnitude to opacity
                mag = min(abs(combined) * 5.0, 1.0)  # scale and clamp [0,1]
                if combined > 0:
                    # red for fraud‑pushing tokens
                    alpha = 0.15 + 0.35 * mag
                    bg = f"rgba(255,0,0,{alpha})"
                elif combined < 0:
                    # green for legit‑pushing tokens
                    alpha = 0.15 + 0.35 * mag
                    bg = f"rgba(0,128,0,{alpha})"
                else:
                    bg = ""

                label = f"{combined:+.2f}"

                if bg:
                    spans.append(
                        f"<span style='background-color:{bg}; padding:2px 3px; "
                        f"border-radius:3px; margin:1px; display:inline-block;'>"
                        f"{clean_tok}<span style='font-size:0.7rem; opacity:0.7;'> "
                        f"{label}</span></span>"
                    )
                else:
                    spans.append(clean_tok)

            html_text = " ".join(spans)

            st.markdown(
                "<div style='line-height:1.6; font-size:0.95rem; padding:0.75rem; "
                "border:1px solid #ddd; border-radius:4px; background-color:#fafafa;'>"
                + html_text +
                "</div>",
                unsafe_allow_html=True,
            )
