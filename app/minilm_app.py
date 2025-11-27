import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import numpy as np

import os

st.set_page_config(page_title="Fake Job Posting Detector (MiniLM)", layout="wide")

with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    ### Training Data
    - 17,880 job postings in total
    - Fraudulent: 866 (4.84%)
    - Legitimate: 17,014 (95.16%)

    ### Fraud Patterns (EDA)
    - Missing company profile → **17.7%** fraud rate
    - Missing employment type → **6.9%** fraud rate
    - Missing experience requirements → **6.2%** fraud rate
    - Very bare-bones postings → **22.2%** fraud rate
    - Extremely complete postings → **9.3%** fraud rate

    ### MiniLM Performance
    - Accuracy: **98.4%**
    - Precision (fraud): **99.8%**
    - Recall (fraud): **97.0%**
    - F1-score (fraud): **98.4%**

    ### Method
    - Fine-tuned MiniLM text classifier
    - Trained on the Employment Scam Aegean dataset
    - Integrated Gradients for word-level explanations
    """)


#st.write(f"🔍 Current working directory: {os.getcwd()}")
#st.write(f"🔍 App file location: {__file__}")
#st.write("✅ Script imports completed")

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
        #st.write(f"🔄 Attempting to load MiniLM model from {MODEL_PATH}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH)  # Convert Path to string
        )
        #st.write("✅ MiniLM model loaded")

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH)
        )

        #st.write("✅ MiniLM tokenizer loaded")

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #st.write(f"✅ MiniLM model moved to device: {device}")
        return model, tokenizer, device
    except Exception as e:
        st.error("❌ Error loading MiniLM model:")
        st.exception(e)
        st.stop()

#st.write("🔄 Calling load_minilm()...")
model, tokenizer, device = load_minilm()
#st.write("✅ MiniLM ready")

# Integrated Gradients instance (fraud class = index 1)
ig = IntegratedGradients(
    lambda inputs_embeds, attention_mask: F.softmax(
        model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits,
        dim=1,
    )[:, 1]
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
        fraud_prob = probs[0, 1].item()  # class 1 = fraud
    return fraud_prob

# ------------------------------
# 3. MiniLM helper: IG word attributions
# ------------------------------
def get_ig_attributions(text: str, n_steps: int = 5):
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

import re

def detect_fraud_patterns(text: str) -> dict:
    t = text.lower()
    patterns = {
        "Urgent / pressure language": bool(re.search(r"\b(urgent|immediate|asap|limited time|act now|don't miss)\b", t)),
        "Personal email domain": bool(re.search(r"@(gmail|yahoo|hotmail|outlook)\.com", t)),
        "Upfront payment / fee": bool(re.search(r"\b(fee|registration fee|processing fee|send.*money|wire transfer|western union|paypal|venmo)\b", t)),
        "Guaranteed high income": bool(re.search(r"\b(guaranteed|easy money|\$\s*\d{3,}\s*(per|/)\s*(week|day|hour))\b", t)),
        "Vague company info": not bool(re.search(r"\b(inc|llc|corp|ltd|corporation|company)\b", t)),
    }
    return patterns

def get_text_stats(text: str) -> dict:
    words = text.split()
    return {
        "word_count": len(words),
        "has_salary": bool(re.search(r"\$\s*\d+", text)),
        "has_contact_email": bool(re.search(r"@", text)),
        "has_requirements_keyword": "requirement" in text.lower() or "qualification" in text.lower(),
    }


# ------------------------------
# 4. Streamlit UI
# ------------------------------

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
                token_attr_sorted, delta = get_ig_attributions(job_text, n_steps=5)
                
                # Free GPU/CPU memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
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

            with st.expander("📊 How to interpret this score"):
                if fraud_prob < 0.30:
                    st.write(
                        "Scores below 30% usually correspond to well-structured postings "
                        "with realistic requirements, salary, and clear company information."
                    )
                elif fraud_prob < 0.60:
                    st.write(
                        "Scores between 30–60% often indicate some missing details or "
                        "mild red flags. Further manual verification is recommended."
                    )
                else:
                    st.write(
                        "Scores above 60% typically show multiple risk factors such as "
                        "missing company info, unusual contact methods, or unrealistic offers."
                    )

            # ---- Simple job characteristics ----
            stats = get_text_stats(job_text)
            st.markdown("---")
            st.subheader("Job Posting Characteristics")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Word count", stats["word_count"])
            c2.metric("Mentions salary", "Yes" if stats["has_salary"] else "No")
            c3.metric("Has email/contact", "Yes" if stats["has_contact_email"] else "No")
            c4.metric("Has requirements section", "Yes" if stats["has_requirements_keyword"] else "No")

            # ---- Fraud pattern analysis ----
            st.markdown("---")
            st.subheader("Fraud Pattern Analysis")

            patterns = detect_fraud_patterns(job_text)
            triggered = [name for name, present in patterns.items() if present]

            if triggered:
                st.warning(f"Detected {len(triggered)} potential fraud indicator(s):")
                for name in triggered:
                    st.write(f"- {name}")
            else:
                st.success("No obvious manual fraud patterns detected.")

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
                    bg = f"rgba(255,0,0,{alpha})"  #red for fraud
                elif ig_attr < 0:
                    alpha = 0.15 + 0.35 * mag
                    bg = f"rgba(0,128,0,{alpha})"  #green for legit
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

            # ---- Top terms summary ----
            st.markdown("---")
            st.subheader("Most Influential Terms")

            # Sort by attribution value
            sorted_attrs = sorted(token_attr_sorted, key=lambda x: x[1])
            top_fraud = [f"{tok} ({attr:+.2f})" for tok, attr in sorted_attrs[:5]]
            top_legit = [f"{tok} ({attr:+.2f})" for tok, attr in sorted_attrs[-5:][::-1]]

            col_fraud, col_legit = st.columns(2)
            with col_fraud:
                st.markdown("**🔴 Words pushing toward FRAUD**")
                for t in top_fraud:
                    st.write(f"- {t}")
            with col_legit:
                st.markdown("**🟢 Words pushing toward LEGIT**")
                for t in top_legit:
                    st.write(f"- {t}")

