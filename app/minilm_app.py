import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import numpy as np

st.write("✅ Script imports completed")

# ------------------------------
# 1. Load model and tokenizer
# ------------------------------
@st.cache_resource
def load_minilm():
    try:
        st.write("🔄 Attempting to load MiniLM model from models/model_miniLM_final...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/model_miniLM_final"
        )
        st.write("✅ Model loaded")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "models/model_miniLM_final"
        )
        st.write("✅ Tokenizer loaded")
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        st.write(f"✅ Model moved to device: {device}")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Error loading model:")
        st.exception(e)
        st.stop()

st.write("🔄 Calling load_minilm()...")
model, tokenizer, device = load_minilm()
st.write("✅ Model, tokenizer, device ready")

# rest of your code unchanged...

# Integrated Gradients instance
ig = IntegratedGradients(
    lambda inputs_embeds, attention_mask: F.softmax(
        model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits,
        dim=1,
    )[:, 1]
)

# ------------------------------
# 2. Helper: predict fraud prob
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
        fraud_prob = probs[0, 1].item()
    return fraud_prob

# ------------------------------
# 3. Helper: IG word attributions
# ------------------------------
def get_ig_attributions(text: str, n_steps: int = 50):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    ).to(device)

    # Get embeddings using public API (safer than .embeddings.word_embeddings)
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
    attributions_norm = attributions_sum / torch.norm(attributions_sum)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    attrs = attributions_norm.detach().cpu().numpy()

    token_attr = [
        (tok, float(attr))
        for tok, attr in zip(tokens, attrs)
        if tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    token_attr_sorted = sorted(token_attr, key=lambda x: abs(x[1]), reverse=True)
    return token_attr_sorted, float(delta.item())

# ------------------------------
# 4. Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fake Job Posting Detector (MiniLM)", layout="wide")

st.title("Fake Job Posting Detector – MiniLM + Integrated Gradients")
st.markdown(
    "Paste a job posting below to see fraud probability and which words "
    "contribute most to the prediction."
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
            # Risk level
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

            # Layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
            with col2:
                st.markdown(
                    f"<h3 style='color:{risk_color};'>Risk Level: {risk_label}</h3>",
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Recommendation:** {rec}")

            st.markdown("---")
            st.subheader("Word-level Importance (Integrated Gradients)")

            # Show top 15 fraud / legit indicators
            top_k = 15
            top_tokens = token_attr_sorted[:top_k]
            st.markdown("**Top tokens by absolute importance:**")
            for tok, attr in top_tokens:
                color = "red" if attr > 0 else "green"
                st.markdown(
                    f"- <span style='color:{color};'>{tok}</span> → {attr:+.4f}",
                    unsafe_allow_html=True,
                )

            st.caption(f"Convergence delta: {delta:.6f} (closer to 0 is better).")

            # Optional: simple inline highlighted text
            st.subheader("Highlighted Job Text (experimental)")

            highlighted = []
            for tok, attr in token_attr_sorted:
                color = ""
                if attr > 0.03:
                    color = "rgba(255,0,0,0.25)"  # red background
                elif attr < -0.03:
                    color = "rgba(0,128,0,0.25)"  # green background

                clean_tok = tok.replace("##", "")
                if color:
                    highlighted.append(
                        f"<span style='background-color:{color};'>{clean_tok}</span>"
                    )
                else:
                    highlighted.append(clean_tok)

            st.markdown(
                "<p style='line-height:1.6;'>"
                + " ".join(highlighted)
                + "</p>",
                unsafe_allow_html=True,
            )
