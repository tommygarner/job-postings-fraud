import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import numpy as np

import os

st.set_page_config(page_title="Fake Job Posting Detector (MiniLM)", layout="wide")

st.write(f"🔍 Current working directory: {os.getcwd()}")
st.write(f"🔍 App file location: {__file__}")
st.write("✅ Script imports completed")

from pathlib import Path

# Get the repository root (parent of the app directory)
REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "model_miniLM_final"


# ------------------------------
# 1. Load MiniLM model and tokenizer
# ------------------------------
@st.cache_resource
def load_minilm():
    try:
        st.write(f"🔄 Attempting to load MiniLM model from {MODEL_PATH}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH)  # Convert Path to string
        )
        st.write("✅ MiniLM model loaded")

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH)
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
# 4. Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fake Job Posting Detector (MiniLM)", layout="wide")

st.title("Fake Job Posting Detector – MiniLM")
st.markdown(
    "Paste a job posting below to see fraud probability and which words "
    "contribute most to the prediction using Integrated Gradients."
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
            with st.spinner("Analyzing with MiniLM..."):
                fraud_prob = predict_fraud_prob(job_text)
                token_attr_sorted, delta = get_ig_attributions(job_text)
        except Exception as e:
            st.exception(e)
        else:
            # -------- Prediction panel --------
            if fraud_prob > 0.60:
                risk_label = "HIGH RISK"
                risk_color = "red"
                rec = "DO NOT APPLY – strong indicators of fraud."
            elif fraud_prob > 0.30:
                risk_label = "MEDIUM RISK"
                risk_color = "orange"
                rec = "INVESTIGATE CAREFULLY – some suspicious patterns."
            else:
                risk_label = "LOW RISK"
                risk_color = "green"
                rec = "APPEARS SAFE – no major red flags detected."

            st.subheader("Prediction")
            st.metric("MiniLM Fraud Probability", f"{fraud_prob*100:.1f}%")

            st.markdown(
                f"<h3 style='color:{risk_color}; margin-top:0.25rem;'>Overall Risk: {risk_label}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Recommendation:** {rec}")

            # -------- IG word overlay --------
            st.markdown("---")
            st.subheader("Word Importance (Integrated Gradients)")

            spans = []
            for tok, ig_attr in token_attr_sorted:
                clean_tok = tok.replace("##", "")
                if clean_tok.strip() == "":
                    continue

                # map magnitude to opacity
                mag = min(abs(ig_attr) * 5.0, 1.0)
                if ig_attr > 0:
                    alpha = 0.15 + 0.35 * mag
                    bg = f"rgba(255,0,0,{alpha})"
                elif ig_attr < 0:
                    alpha = 0.15 + 0.35 * mag
                    bg = f"rgba(0,128,0,{alpha})"
                else:
                    bg = ""

                label = f"{ig_attr:+.2f}"

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
