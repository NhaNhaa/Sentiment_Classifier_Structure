import streamlit as st
import shap
import numpy as np
from transformers import pipeline

# ------------------------------------------------------------
# Load Twitter‑RoBERTa model
# ------------------------------------------------------------
@st.cache_resource
def load_transformer_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True
    )

# ------------------------------------------------------------
# SHAP explanation (positive class index 2)
# ------------------------------------------------------------
def shap_explain_transformer_html(model, text):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer([text])
        tokens = shap_values.data[0]
        pos_shap = shap_values.values[0, :, 2]
        
        max_abs = max(abs(pos_shap)) if len(pos_shap) > 0 else 1e-5
        if max_abs < 1e-9:
            max_abs = 1e-5
        
        html = "<div style='font-size:16px; line-height:1.6; max-width:600px; word-wrap:break-word'>"
        for token, val in zip(tokens, pos_shap):
            intensity_ratio = min(1.0, abs(val) / max_abs) if max_abs > 0 else 0
            intensity = int(50 + intensity_ratio * 200)
            if val > 0:
                color = f"rgb(255, {255-intensity}, {255-intensity})"
            else:
                color = f"rgb({255-intensity}, {255-intensity}, 255)"
            token_escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f"<span style='background-color:{color}; padding:2px 4px; margin:2px; display:inline-block; border-radius:3px' title='SHAP: {val:.4f}'>{token_escaped}</span> "
        html += "</div>"
        if max_abs < 0.01:
            html += f"<p><small>⚠️ SHAP values very small (max {max_abs:.4f}) – prediction driven by model bias.</small></p>"
        return html
    except Exception as e:
        st.error(f"SHAP failed: {str(e)}")
        return None

# ------------------------------------------------------------
# Streamlit UI – clean & user‑friendly
# ------------------------------------------------------------
st.set_page_config(page_title="Sentiment Classifier", page_icon="🎭", layout="wide")

# Custom CSS to remove progress bar
st.markdown("""
    <style>
        .stProgress > div > div > div > div { display: none; }
        [data-testid="stMetricValue"] { font-size: 2rem; }
        .word-explanation {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎭 Sentiment Classifier")
st.markdown("**Understand the emotion behind any text** – powered by Twitter‑RoBERTa and SHAP.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    show_shap = st.checkbox("🔍 Show SHAP word explanation", value=False, 
                            help="Highlights which words influenced the prediction (slower first run).")
    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown("**Twitter‑RoBERTa** – fine‑tuned on 124M tweets, excellent at slang, sarcasm, and subtle sentiment.")
    st.markdown("### 💡 How SHAP works")
    st.markdown("- **Red** background → pushes toward **POSITIVE**")
    st.markdown("- **Blue** background → pushes toward **NEGATIVE**")
    st.markdown("- **Hover** over any word to see its exact contribution value.")
    st.markdown("---")
    st.markdown("### 📌 Try these examples")
    st.markdown("- *I don't hate this movie, it's actually pretty good.*")
    st.markdown("- *Oh great, another boring sequel.*")
    st.markdown("- *It's okay, nothing special, but watchable.*")
    st.markdown("---")
    st.markdown("### 🙏 Acknowledgments")
    st.markdown("- **Model**: [Twitter‑RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) by Cardiff NLP (Hugging Face)")
    st.markdown("- **Explainability**: [SHAP](https://shap.readthedocs.io/) by Scott Lundberg & Su-In Lee")
    st.markdown("- **Framework**: [Hugging Face Transformers](https://huggingface.co/docs/transformers) & [Streamlit](https://streamlit.io/)")

# Main area – two columns for input and instructions
col_input, col_instructions = st.columns([3, 1])

with col_input:
    review_text = st.text_area(
        "✍️ Enter your review or text",
        value="",
        height=180,
        placeholder="e.g., This movie was absolutely fantastic! The acting was superb.",
        label_visibility="collapsed"
    )
    
    MAX_CHARS = 2000
    current_len = len(review_text)
    if current_len > MAX_CHARS:
        st.error(f"❌ Text too long: {current_len} characters (max {MAX_CHARS}). Please shorten your review.")
        button_disabled = True
    else:
        button_disabled = False
    
    analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True, disabled=button_disabled)
    
    if current_len > 0 and current_len <= MAX_CHARS:
        st.caption(f"📏 {current_len} characters (max {MAX_CHARS})")

with col_instructions:
    st.markdown("### ⚡ Quick tips")
    st.markdown("- Be natural – the model understands slang and emojis.")
    st.markdown(f"- Keep under **{MAX_CHARS} characters** (~300 words).")
    st.markdown("- **SHAP** explains the model’s reasoning word by word.")

# Load model (cached)
transformer_model = load_transformer_model()

if analyze_clicked and review_text.strip() and not button_disabled:
    with st.spinner("🧠 Analyzing sentiment..."):
        raw_result = transformer_model(review_text)
        scores = {item['label']: item['score'] for item in raw_result}
        
        pos_score = scores.get('positive', scores.get('LABEL_2', 0.0))
        neg_score = scores.get('negative', scores.get('LABEL_0', 0.0))
        
        if pos_score == 0.0 and neg_score == 0.0:
            best_label, best_score = max(scores.items(), key=lambda x: x[1])
            if best_label in ('positive', 'LABEL_2'):
                pos_score = best_score
            elif best_label in ('negative', 'LABEL_0'):
                neg_score = best_score
            else:
                pos_score = best_score * 0.6
                neg_score = best_score * 0.4
        
        sentiment = "POSITIVE" if pos_score > neg_score else "NEGATIVE"
        confidence = max(pos_score, neg_score)
        confidence = float(confidence)
        
        print(f"DEBUG: pos={pos_score:.4f}, neg={neg_score:.4f}, raw={raw_result}")

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if sentiment == "POSITIVE":
            st.markdown(f"## 😊 {sentiment}")
        else:
            st.markdown(f"## 😞 {sentiment}")
        st.markdown(f"### Confidence: **{confidence:.2%}**")
    
    if show_shap:
        with st.spinner("🔎 Computing word‑by‑word explanation (may take 30‑60s on first run)..."):
            html_shap = shap_explain_transformer_html(transformer_model, review_text)
            if html_shap:
                with res_col2:
                    st.markdown("### 🔎 Word importance")
                    st.markdown(html_shap, unsafe_allow_html=True)
                    st.caption("🔴 Red = pushes POSITIVE | 🔵 Blue = pushes NEGATIVE")
            else:
                with res_col2:
                    st.error("SHAP explanation failed. Try a shorter sentence or disable SHAP.")
    else:
        with res_col2:
            st.info("💡 Enable **Show SHAP word explanation** in the sidebar to see which words influenced the prediction.")

elif analyze_clicked and not review_text.strip():
    st.warning("Please enter some text to classify.")

st.markdown("---")
st.caption("Built with Twitter‑RoBERTa and SHAP | No hardcoded rules – the model explains itself.")