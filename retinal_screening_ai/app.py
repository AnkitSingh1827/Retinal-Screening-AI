"""
app.py
======
AI Retinal Screening — Streamlit Dashboard

Run with:
    streamlit run app.py

Tabs
────
  🏠 Home          → Project overview, how it works, disease categories
  🔬 Screening     → Upload image, run prediction
  📊 Result        → Prediction, confidence, probability chart
  🔍 AI Explanation → Grad-CAM heatmap and overlay
  ℹ️ Model Info    → Architecture, training metrics
  📖 About         → Problem, solution, limitations, future scope
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Retinal Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Global background ── */
    .main { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%); }

    /* ── Hide Streamlit default footer ── */
    footer { visibility: hidden; }

    /* ── Cards ── */
    .card {
        background: linear-gradient(135deg, #1e2a3a 0%, #162032 100%);
        border: 1px solid #2a3f55;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a2a4a 0%, #0f1f35 100%);
        border: 1px solid #2e4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* ── Prediction result card ── */
    .result-card {
        border-radius: 16px;
        padding: 28px;
        margin: 12px 0;
        text-align: center;
    }
    .result-normal   { background: linear-gradient(135deg, #1a4a2e, #0f2d1a); border: 1px solid #2e7d45; }
    .result-diabetes { background: linear-gradient(135deg, #4a1a1a, #2d0f0f); border: 1px solid #c0392b; }
    .result-glaucoma { background: linear-gradient(135deg, #1a2a4a, #0f1a2d); border: 1px solid #2980b9; }
    .result-amd      { background: linear-gradient(135deg, #3a2a0a, #2d1f05); border: 1px solid #f39c12; }

    /* ── Disclaimer box ── */
    .disclaimer {
        background: linear-gradient(135deg, #2d1a00, #1a1000);
        border: 1px solid #f39c12;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #f0c060;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* ── Section heading ── */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e0eaff;
        margin-bottom: 8px;
    }

    /* ── Disease badge ── */
    .badge {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px;
    }
    .badge-normal   { background: #1b5e20; color: #a5d6a7; }
    .badge-diabetes { background: #7f0000; color: #ef9a9a; }
    .badge-glaucoma { background: #0d47a1; color: #90caf9; }
    .badge-amd      { background: #e65100; color: #ffcc80; }

    /* ── Confidence meter ── */
    .confidence-bar {
        background: #1a2a4a;
        border-radius: 50px;
        height: 14px;
        margin: 10px 0;
        overflow: hidden;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 20px;
        color: #5a6a7a;
        font-size: 0.80rem;
        border-top: 1px solid #1e2d3e;
        margin-top: 40px;
    }

    /* ── Streamlit button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976D2, #1565C0);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.4);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1b2a;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #7a8fa0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1565C0 !important;
        color: white !important;
    }

    /* ── Info box ── */
    .info-box {
        background: #0d2137;
        border-left: 4px solid #1976D2;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        color: #b0c8e0;
        font-size: 0.90rem;
    }

    /* ── Metric number ── */
    .big-metric {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #7a8fa0;
        margin-top: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _safe_load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _format_pct(val):
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _confidence_color(conf: float) -> str:
    """Return a CSS color string based on confidence level."""
    if conf >= 0.80:
        return "#4CAF50"   # green
    elif conf >= 0.60:
        return "#FF9800"   # amber
    else:
        return "#F44336"   # red


def _disease_emoji(name: str) -> str:
    return {"Normal": "✅", "Diabetes": "🩸", "Glaucoma": "👁", "AMD": "🌅"}.get(name, "❓")


def _disease_css_class(name: str) -> str:
    return {
        "Normal":   "result-normal",
        "Diabetes": "result-diabetes",
        "Glaucoma": "result-glaucoma",
        "AMD":      "result-amd",
    }.get(name, "result-normal")


# ─────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the RetinalPredictor once and cache it."""
    try:
        from src.predict import RetinalPredictor
        predictor = RetinalPredictor()
        return predictor
    except Exception as exc:
        return None


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 20px 0 10px;'>
                <div style='font-size: 3rem;'>👁️</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #e0eaff; margin-top:6px;'>
                    AI Retinal Screening
                </div>
                <div style='font-size: 0.78rem; color: #5a8fa0; margin-top:2px;'>
                    B.Tech Final Project
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("#### 📊 Dataset")
        st.markdown(
            """
            <div class='info-box'>
                <b>ODIR-5K</b><br>
                Ocular Disease Intelligent Recognition<br>
                5,000 patients · Fundus images
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🤖 Model")
        st.markdown(
            """
            <div class='info-box'>
                <b>EfficientNetB0</b><br>
                Transfer learning · ImageNet weights<br>
                Input: 224×224×3
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🏷 Classes")
        badges_html = """
        <div>
            <span class='badge badge-normal'>✅ Normal</span>
            <span class='badge badge-diabetes'>🩸 Diabetes</span><br>
            <span class='badge badge-glaucoma'>👁 Glaucoma</span>
            <span class='badge badge-amd'>🌅 AMD</span>
        </div>
        """
        st.markdown(badges_html, unsafe_allow_html=True)

        st.divider()
        st.markdown(
            """
            <div class='disclaimer'>
                ⚠️ <b>Medical Disclaimer</b><br>
                This tool is an AI-assisted screening prototype
                for educational/research purposes only. It does
                <b>not</b> provide a medical diagnosis. Always
                consult a qualified ophthalmologist.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model status
        st.markdown("#### 🟢 Model Status")
        from config.config import BEST_MODEL_PATH
        if BEST_MODEL_PATH.exists():
            st.success("Model loaded ✓")
        else:
            st.error("Model not found")
            st.info("Run: `python -m src.train`")


# ─────────────────────────────────────────────────────────
# TAB 1 — HOME
# ─────────────────────────────────────────────────────────

def tab_home():
    # Hero section
    st.markdown(
        """
        <div style='text-align:center; padding: 40px 0 30px;'>
            <div style='font-size: 4rem; margin-bottom:10px;'>👁️</div>
            <h1 style='font-size:2.6rem; font-weight:800; color:#e0eaff;
                        background: linear-gradient(90deg, #64b5f6, #42a5f5, #1e88e5);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        margin-bottom:8px;'>
                AI Retinal Screening
            </h1>
            <p style='font-size:1.15rem; color:#7a9ab0; max-width:600px; margin:0 auto;'>
                AI-Assisted Retinal Image Screening System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Disclaimer (prominent)
    st.markdown(
        """
        <div class='disclaimer'>
            ⚠️ <b>Important Medical Disclaimer</b> — This application is an AI-assisted screening
            prototype and is <b>not</b> a substitute for examination or diagnosis by a qualified
            ophthalmologist. All results must be interpreted by a licensed medical professional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # How it works
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(
            """
            <div class='card'>
            <div class='section-title'>🔬 How It Works</div>
            <br>
            <ol style='color:#b0c8e0; line-height:2.0; padding-left:20px;'>
                <li>Upload a retinal fundus image (JPG/PNG)</li>
                <li>The system checks image quality</li>
                <li>EfficientNetB0 analyses the image</li>
                <li>A prediction with confidence is shown</li>
                <li>Grad-CAM highlights influential regions</li>
                <li>A screening recommendation is provided</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class='card'>
            <div class='section-title'>🏥 Disease Categories</div>
            <br>
            <table style='width:100%; color:#b0c8e0; border-collapse:collapse;'>
                <tr>
                    <td style='padding:8px 4px;'>✅ <b>Normal</b></td>
                    <td style='padding:8px 4px; color:#7a9ab0;'>No signs of ocular disease</td>
                </tr>
                <tr>
                    <td style='padding:8px 4px;'>🩸 <b>Diabetes</b></td>
                    <td style='padding:8px 4px; color:#7a9ab0;'>Diabetic retinopathy indicators</td>
                </tr>
                <tr>
                    <td style='padding:8px 4px;'>👁 <b>Glaucoma</b></td>
                    <td style='padding:8px 4px; color:#7a9ab0;'>Optic nerve / pressure signs</td>
                </tr>
                <tr>
                    <td style='padding:8px 4px;'>🌅 <b>AMD</b></td>
                    <td style='padding:8px 4px; color:#7a9ab0;'>Age-related macular degeneration</td>
                </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Problem statement
    st.markdown("---")
    st.markdown("### 🌍 Problem Being Solved")
    cols = st.columns(3, gap="medium")
    problems = [
        ("👨‍⚕️", "Specialist Shortage", "Ophthalmologists are scarce in rural and developing regions"),
        ("⏱️", "Delayed Diagnosis", "Manual screening is slow; diseases are often detected late"),
        ("🌐", "Accessibility", "High cost of eye exams limits access for lower-income populations"),
    ]
    for col, (icon, title, desc) in zip(cols, problems):
        col.markdown(
            f"""
            <div class='metric-card' style='height:160px;'>
                <div style='font-size:2.2rem;'>{icon}</div>
                <div style='font-weight:700; color:#e0eaff; margin:10px 0 6px;'>{title}</div>
                <div style='font-size:0.85rem; color:#7a9ab0; line-height:1.5;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Solution
    st.markdown("---")
    st.markdown("### 💡 Our Solution")
    st.markdown(
        """
        <div class='card'>
        <p style='color:#b0c8e0; font-size:1.05rem; line-height:1.8;'>
            This system uses <b>deep learning transfer learning</b> on fundus retinal photographs
            to provide a fast, automated first-level screening for four major ocular conditions.
            By deploying this as a web application, it can assist healthcare workers in remote settings
            to identify high-risk patients who need urgent specialist referral.
        </p>
        <ul style='color:#b0c8e0; line-height:2.0; margin-top:10px;'>
            <li>🚀 Fast first-level screening in seconds</li>
            <li>🔍 Grad-CAM explainability for clinical transparency</li>
            <li>📊 Probability-based risk stratification</li>
            <li>🏥 Designed as a <em>support tool</em>, not a replacement for doctors</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_footer()


# ─────────────────────────────────────────────────────────
# TAB 2 — SCREENING
# ─────────────────────────────────────────────────────────

def tab_screening():
    st.markdown("## 🔬 Retinal Image Screening")
    st.markdown(
        """
        <div class='info-box'>
            Upload a <b>retinal fundus photograph</b> (JPG/JPEG/PNG).
            The image should be a clear, colour fundus image taken with an ophthalmoscope or fundus camera.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📁 Upload Image")
        uploaded = st.file_uploader(
            label="Choose a retinal fundus image",
            type=["jpg", "jpeg", "png"],
            key="uploaded_file",
            help="Supported formats: JPG, JPEG, PNG",
        )

        if uploaded is not None:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
                st.session_state["uploaded_pil"] = pil_img
                st.session_state["result"]        = None   # clear previous result
                st.success(f"✅ Image loaded: {uploaded.name}  ({pil_img.size[0]}×{pil_img.size[1]} px)")
            except Exception as exc:
                st.error(f"❌ Could not load image: {exc}")
                st.session_state["uploaded_pil"] = None

        # Quality check preview
        if st.session_state.get("uploaded_pil"):
            from src.preprocessing import check_image_quality
            pil_img = st.session_state["uploaded_pil"]
            ok, msg = check_image_quality(pil_img)
            if ok:
                st.success(f"🟢 Quality check passed: {msg}")
            else:
                st.warning(f"🔴 Quality warning: {msg}")

    with col_preview:
        st.markdown("### 🖼 Image Preview")
        if st.session_state.get("uploaded_pil"):
            st.image(
                st.session_state["uploaded_pil"],
                caption="Uploaded retinal fundus image",
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div style='border: 2px dashed #2a3f55; border-radius:12px;
                            height:300px; display:flex; align-items:center;
                            justify-content:center; color:#3a5a6e; flex-direction:column;'>
                    <div style='font-size:3rem;'>👁️</div>
                    <div style='margin-top:10px;'>Image preview will appear here</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Analyze button
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        analyze_btn = st.button("🔍 Analyze Retina", key="analyze_btn", type="primary")

    if analyze_btn:
        if not st.session_state.get("uploaded_pil"):
            st.error("Please upload an image first.")
        else:
            predictor = load_predictor()
            if predictor is None:
                st.error("Failed to initialize predictor. Check console for errors.")
            elif not predictor.is_ready():
                st.error(
                    "⚠️ Model not found. Please run training first:\n"
                    "```\npython -m src.train\n```"
                )
            else:
                with st.spinner("🤖 Analysing retinal image … this may take a few seconds"):
                    result = predictor.predict(
                        st.session_state["uploaded_pil"], run_gradcam=True
                    )
                st.session_state["result"] = result

                if result.get("error"):
                    if not result.get("quality_ok"):
                        st.warning(f"⚠️ Image Quality Issue: {result['error']}")
                    else:
                        st.error(f"❌ Prediction Error: {result['error']}")
                else:
                    pred = result["prediction"]
                    conf = result["confidence"] * 100
                    st.success(f"✅ Analysis complete! Predicted: **{pred}** ({conf:.1f}% confidence)")
                    st.info("👈 See the **Result** and **AI Explanation** tabs for detailed output.")

    _render_footer()


# ─────────────────────────────────────────────────────────
# TAB 3 — RESULT
# ─────────────────────────────────────────────────────────

def tab_result():
    st.markdown("## 📊 Screening Result")

    result = st.session_state.get("result")
    if result is None:
        st.info("No result yet. Go to the **Screening** tab and analyse an image.")
        return

    if result.get("error") and not result.get("prediction"):
        st.error(f"Analysis failed: {result['error']}")
        return

    pred  = result["prediction"]
    conf  = result["confidence"]
    probs = result["probabilities"]
    low_c = result["low_confidence"]

    # ── Prediction card ───────────────────────────────────
    css_class  = _disease_css_class(pred)
    emoji      = _disease_emoji(pred)
    conf_color = _confidence_color(conf)

    st.markdown(
        f"""
        <div class='result-card {css_class}'>
            <div style='font-size:3.5rem; margin-bottom:10px;'>{emoji}</div>
            <div style='font-size:1.2rem; color:#a0b8c8; margin-bottom:4px;'>Predicted Condition</div>
            <div style='font-size:2.4rem; font-weight:800; color:#e0eaff; margin-bottom:12px;'>
                Possible {pred}
            </div>
            <div style='font-size:1rem; color:#a0b8c8; margin-bottom:8px;'>Model Confidence</div>
            <div style='font-size:3rem; font-weight:800; color:{conf_color};'>
                {conf*100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Low confidence warning
    if low_c:
        st.warning(
            "⚠️ **Low-confidence prediction.** The model is uncertain about this image. "
            "Please obtain professional ophthalmic evaluation."
        )

    # ── Screening recommendation ──────────────────────────
    st.markdown("---")
    recommendations = {
        "Normal": (
            "🟢", "Screening Recommendation",
            "No immediate ocular pathology detected. "
            "Routine annual eye examination is still recommended."
        ),
        "Diabetes": (
            "🔴", "Screening Recommendation",
            "Possible diabetic retinopathy indicators detected. "
            "Further examination by a qualified ophthalmologist is strongly recommended. "
            "Early intervention can prevent vision loss."
        ),
        "Glaucoma": (
            "🔴", "Screening Recommendation",
            "Possible glaucoma-related features detected. "
            "Immediate consultation with an ophthalmologist is recommended for intraocular pressure measurement and full evaluation."
        ),
        "AMD": (
            "🟡", "Screening Recommendation",
            "Possible age-related macular degeneration features detected. "
            "Specialist assessment with OCT imaging is recommended."
        ),
    }

    icon, title, rec = recommendations.get(pred, ("ℹ️", "Recommendation", "Consult a specialist."))
    st.markdown(
        f"""
        <div class='card'>
            <div style='font-size:1.05rem; font-weight:600; color:#b0c8e0; margin-bottom:8px;'>
                {icon} {title}
            </div>
            <div style='color:#d0e0f0; line-height:1.7; font-size:1.0rem;'>
                {rec}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probability chart ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Class Probability Distribution")

    try:
        import plotly.graph_objects as go

        class_names = list(probs.keys())
        prob_vals   = [probs[c] * 100 for c in class_names]

        colors = {
            "Normal":   "#4CAF50",
            "Diabetes": "#F44336",
            "Glaucoma": "#2196F3",
            "AMD":      "#FF9800",
        }
        bar_colors = [colors.get(c, "#78909C") for c in class_names]

        fig = go.Figure(
            go.Bar(
                x=class_names,
                y=prob_vals,
                marker_color=bar_colors,
                marker_line_color="rgba(255,255,255,0.2)",
                marker_line_width=1,
                text=[f"{v:.1f}%" for v in prob_vals],
                textposition="outside",
                textfont_size=13,
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,25,40,0.7)",
            font=dict(color="#b0c8e0", family="Inter"),
            title=dict(text="Model Probability per Class", font_size=15, x=0.5),
            yaxis=dict(
                title="Probability (%)", range=[0, 110],
                gridcolor="rgba(42,63,85,0.5)", gridwidth=1,
            ),
            xaxis=dict(title="Condition"),
            margin=dict(t=60, b=40, l=40, r=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback if plotly not installed
        for name, prob in probs.items():
            st.metric(label=name, value=f"{prob*100:.1f}%")

    # ── Medical disclaimer ────────────────────────────────
    st.markdown(
        """
        <div class='disclaimer'>
            ⚠️ <b>Important</b> — The prediction above is generated by an AI model trained on
            the ODIR-5K research dataset. It is a <b>screening indicator only</b> and must
            <b>not</b> be treated as a clinical diagnosis. All findings must be confirmed by
            a qualified ophthalmologist before any medical decision is made.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_footer()


# ─────────────────────────────────────────────────────────
# TAB 4 — AI EXPLANATION (GRAD-CAM)
# ─────────────────────────────────────────────────────────

def tab_explanation():
    st.markdown("## 🔍 AI Explanation (Grad-CAM)")
    st.markdown(
        """
        <div class='info-box'>
            <b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> highlights the regions
            of the retinal image that most influenced the model's prediction.
            Warmer colours (red/yellow) indicate higher influence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.get("result")
    if result is None:
        st.info("No result yet. Go to the **Screening** tab and analyse an image.")
        return

    gradcam = result.get("gradcam")
    pred    = result.get("prediction", "Unknown")

    if gradcam is None or not gradcam.get("success"):
        msg = gradcam.get("message", "Grad-CAM was not generated.") if gradcam else "No Grad-CAM data."
        st.warning(f"⚠️ Grad-CAM unavailable: {msg}")
        return

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("##### 🖼 Original Image")
        original = gradcam.get("original")
        if original:
            st.image(original, caption="Preprocessed (224×224)", use_container_width=True)

    with col2:
        st.markdown("##### 🌡 Grad-CAM Heatmap")
        heatmap = gradcam.get("heatmap")
        if heatmap is not None:
            st.image(heatmap, caption="Activation heatmap", use_container_width=True)

    with col3:
        st.markdown("##### 🔀 Overlay")
        overlay = gradcam.get("overlay")
        if overlay:
            st.image(overlay, caption=f"Heatmap overlaid — predicted: {pred}", use_container_width=True)

    # Explanation text
    st.markdown("---")
    st.markdown(
        f"""
        <div class='card'>
            <div style='font-size:1.05rem; font-weight:600; color:#64b5f6; margin-bottom:10px;'>
                🧠 What This Shows
            </div>
            <p style='color:#b0c8e0; line-height:1.8;'>
                The heatmap overlay indicates that the model focused on the highlighted
                (warmer-coloured) regions when predicting <b>{pred}</b>.
                In retinal imaging, clinically significant structures such as the optic disc,
                macula, and vasculature often correspond to disease-relevant regions.
            </p>
            <div style='margin-top:14px; padding:12px; background:#1a2a1a; border-radius:8px;
                        border-left:4px solid #f39c12; color:#f0c060;'>
                ⚠️ <b>Disclaimer:</b> {gradcam['message']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Colour legend
    st.markdown("##### 🎨 Colour Legend")
    st.markdown(
        """
        <div style='display:flex; gap:20px; margin-top:6px;'>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:20px; height:20px; background:#FF0000; border-radius:4px;'></div>
                <span style='color:#b0c8e0; font-size:0.85rem;'>Highest influence</span>
            </div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:20px; height:20px; background:#FFFF00; border-radius:4px;'></div>
                <span style='color:#b0c8e0; font-size:0.85rem;'>High influence</span>
            </div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:20px; height:20px; background:#00FF00; border-radius:4px;'></div>
                <span style='color:#b0c8e0; font-size:0.85rem;'>Moderate influence</span>
            </div>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:20px; height:20px; background:#0000FF; border-radius:4px;'></div>
                <span style='color:#b0c8e0; font-size:0.85rem;'>Low influence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_footer()


# ─────────────────────────────────────────────────────────
# TAB 5 — MODEL INFORMATION
# ─────────────────────────────────────────────────────────

def tab_model_info():
    st.markdown("## ℹ️ Model Information")

    from config.config import (
        BEST_MODEL_PATH,
        TRAINING_META_PATH,
        EVAL_METRICS_PATH,
        CLASS_NAMES_PATH,
        TARGET_CLASSES,
        BATCH_SIZE,
        EPOCHS_HEAD,
        EPOCHS_FINETUNE,
        INPUT_SHAPE,
        DROPOUT_RATE,
        DENSE_UNITS,
        BASE_MODEL_NAME,
    )

    # ── Architecture ──────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🏗 Architecture")
        st.markdown(
            f"""
            <div class='card'>
            <table style='width:100%; color:#b0c8e0; border-collapse:collapse;'>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Base Model</b></td>
                    <td style='padding:8px;'>{BASE_MODEL_NAME}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Pretrained Weights</b></td>
                    <td style='padding:8px;'>ImageNet</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Input Shape</b></td>
                    <td style='padding:8px;'>{INPUT_SHAPE}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Pooling</b></td>
                    <td style='padding:8px;'>GlobalAveragePooling2D</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Dense Units</b></td>
                    <td style='padding:8px;'>{DENSE_UNITS}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Dropout Rate</b></td>
                    <td style='padding:8px;'>{DROPOUT_RATE}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Output Classes</b></td>
                    <td style='padding:8px;'>{len(TARGET_CLASSES)}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Activation</b></td>
                    <td style='padding:8px;'>Softmax</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Optimizer</b></td>
                    <td style='padding:8px;'>Adam</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Loss Function</b></td>
                    <td style='padding:8px;'>Sparse Categorical Crossentropy</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 📋 Training Configuration")
        st.markdown(
            f"""
            <div class='card'>
            <table style='width:100%; color:#b0c8e0; border-collapse:collapse;'>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Dataset</b></td>
                    <td style='padding:8px;'>ODIR-5K (Hugging Face)</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Classes</b></td>
                    <td style='padding:8px;'>Normal, Diabetes, Glaucoma, AMD</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Batch Size</b></td>
                    <td style='padding:8px;'>{BATCH_SIZE}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Head Epochs</b></td>
                    <td style='padding:8px;'>{EPOCHS_HEAD}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Fine-tune Epochs</b></td>
                    <td style='padding:8px;'>{EPOCHS_FINETUNE}</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Split Ratio</b></td>
                    <td style='padding:8px;'>70% / 15% / 15%</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Splitting Strategy</b></td>
                    <td style='padding:8px;'>Patient-level (no leakage)</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Class Imbalance</b></td>
                    <td style='padding:8px;'>Class weights + Augmentation</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Augmentation</b></td>
                    <td style='padding:8px;'>Rotation, Flip, Zoom, Brightness</td></tr>
                <tr><td style='padding:8px; color:#64b5f6;'><b>Explainability</b></td>
                    <td style='padding:8px;'>Grad-CAM</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Training metadata ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Training Results")

    meta = _safe_load_json(TRAINING_META_PATH)
    if meta:
        cols = st.columns(4, gap="medium")
        vals = [
            ("🎯 Best Val Accuracy", f"{meta.get('best_val_accuracy', 0)*100:.1f}%"),
            ("📉 Best Val Loss",     f"{meta.get('best_val_loss', 0):.4f}"),
            ("🏋 Train Samples",    f"{meta.get('train_samples', '?'):,}"),
            ("⏱ Training Time",     f"{meta.get('training_duration_s', 0)/60:.0f} min"),
        ]
        for col, (label, value) in zip(cols, vals):
            col.markdown(
                f"""
                <div class='metric-card'>
                    <div class='big-metric' style='color:#64b5f6;'>{value}</div>
                    <div class='metric-label'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Training metadata not available. Run: `python -m src.train`")

    # ── Evaluation metrics ────────────────────────────────
    st.markdown("### 🏆 Evaluation Metrics (Test Set)")

    eval_data = _safe_load_json(EVAL_METRICS_PATH)
    if eval_data:
        cols = st.columns(4, gap="medium")
        metric_vals = [
            ("✅ Accuracy",          f"{eval_data.get('accuracy', 0)*100:.1f}%",   "#4CAF50"),
            ("📏 Macro F1",          f"{eval_data.get('macro_f1', 0)*100:.1f}%",   "#2196F3"),
            ("🔍 Macro Recall",      f"{eval_data.get('macro_recall', 0)*100:.1f}%", "#FF9800"),
            ("🔑 ROC-AUC",           f"{eval_data.get('roc_auc_macro') or '—'}",   "#9C27B0"),
        ]
        for col, (label, val, color) in zip(cols, metric_vals):
            col.markdown(
                f"""
                <div class='metric-card'>
                    <div class='big-metric' style='color:{color};'>{val}</div>
                    <div class='metric-label'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Per-class metrics table
        report = eval_data.get("per_class_report", {})
        if report:
            st.markdown("#### Per-Class Breakdown")
            rows = []
            for cls in TARGET_CLASSES:
                r = report.get(cls, {})
                rows.append({
                    "Class":     cls,
                    "Precision": f"{r.get('precision', 0)*100:.1f}%",
                    "Recall":    f"{r.get('recall', 0)*100:.1f}%",
                    "F1-Score":  f"{r.get('f1-score', 0)*100:.1f}%",
                    "Support":   int(r.get("support", 0)),
                })
            import pandas as pd
            df_metrics = pd.DataFrame(rows)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class='info-box'>
                ⚠️ <b>Medical Note</b>: For screening applications, <b>Recall (Sensitivity)</b>
                is the most important metric. High recall minimises missed disease cases,
                which is critical in healthcare contexts.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Evaluation metrics not available. Run: `python -m src.evaluate`")

    # Training plots
    st.markdown("---")
    st.markdown("### 📈 Training History Plots")
    from config.config import TRAINING_ACC_PATH, TRAINING_LOSS_PATH, CONFUSION_MATRIX_PATH
    plot_cols = st.columns(3, gap="medium")
    for col, (path, title) in zip(
        plot_cols,
        [
            (TRAINING_ACC_PATH,    "Training Accuracy"),
            (TRAINING_LOSS_PATH,   "Training Loss"),
            (CONFUSION_MATRIX_PATH,"Confusion Matrix"),
        ],
    ):
        if Path(path).exists():
            col.image(str(path), caption=title, use_container_width=True)
        else:
            col.info(f"{title} not available yet.")

    _render_footer()


# ─────────────────────────────────────────────────────────
# TAB 6 — ABOUT
# ─────────────────────────────────────────────────────────

def tab_about():
    st.markdown("## 📖 About This Project")

    st.markdown(
        """
        <div class='card'>
            <div class='section-title'>🎓 B.Tech Final Year Project</div>
            <br>
            <p style='color:#b0c8e0; line-height:1.8; font-size:1.02rem;'>
                This system was developed as a final-year B.Tech project with the goal of
                demonstrating how artificial intelligence can assist in ophthalmic screening.
                The project uses the <b>ODIR-5K dataset</b> (Ocular Disease Intelligent Recognition),
                transfer learning on <b>EfficientNetB0</b>, and <b>Grad-CAM explainability</b> to
                create an end-to-end retinal disease screening prototype.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### ✅ Key Features")
        features = [
            "Transfer learning with EfficientNetB0",
            "Four-class ophthalmic screening",
            "Patient-level data splitting (no leakage)",
            "Class imbalance handling",
            "Grad-CAM visual explainability",
            "Image quality assessment",
            "Confidence-based uncertainty warning",
            "Interactive Streamlit dashboard",
            "Medical disclaimer integration",
        ]
        for f in features:
            st.markdown(f"✅ {f}")

    with col2:
        st.markdown("### ⚠️ Known Limitations")
        limitations = [
            "Trained only on ODIR-5K (may not generalise to all fundus cameras)",
            "Four-class prototype — does not screen all eye diseases",
            "AI confidence ≠ diagnostic certainty",
            "Image quality heavily affects prediction reliability",
            "Not validated in a clinical trial",
            "Should not replace specialist examination",
        ]
        for lim in limitations:
            st.markdown(f"⚠️ {lim}")

    st.markdown("---")
    st.markdown("### 🚀 Future Scope")
    future = [
        ("🔬", "Multi-label classification",        "Screen for all 8 ODIR disease categories simultaneously"),
        ("📱", "Mobile deployment",                  "Deploy as an Android/iOS app for field use"),
        ("🌍", "Multi-language support",             "Support local languages in rural healthcare settings"),
        ("🧪", "Clinical validation",               "Partner with hospitals for prospective validation studies"),
        ("📊", "Longitudinal tracking",             "Track disease progression over time per patient"),
        ("🔗", "EHR integration",                  "Connect with electronic health record systems"),
    ]
    cols = st.columns(3, gap="medium")
    for i, (icon, title, desc) in enumerate(future):
        cols[i % 3].markdown(
            f"""
            <div class='metric-card' style='margin-bottom:16px; height:150px;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-weight:700; color:#e0eaff; margin:8px 0 4px;'>{title}</div>
                <div style='font-size:0.8rem; color:#7a9ab0;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div class='disclaimer'>
            ⚠️ <b>Full Medical Disclaimer</b> — This application is an AI-assisted screening prototype
            developed for educational and research purposes. It does not provide medical diagnoses,
            treatment recommendations, or prescriptions. Retinal images and any AI-generated predictions
            must be reviewed and interpreted by a qualified, licensed ophthalmologist before any clinical
            decision is made. The developers and institution assume no medical or legal liability for
            outcomes based on this tool's predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_footer()


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────

def _render_footer():
    st.markdown(
        """
        <div class='footer'>
            AI-assisted screening prototype &nbsp;|&nbsp;
            For educational/research use only &nbsp;|&nbsp;
            Final assessment must be performed by a qualified ophthalmologist
            <br>
            <span style='opacity:0.5; font-size:0.75rem;'>
                Built with EfficientNetB0 · ODIR-5K Dataset · Grad-CAM · Streamlit
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────

def main():
    # Initialise session state
    if "uploaded_pil" not in st.session_state:
        st.session_state["uploaded_pil"] = None
    if "result" not in st.session_state:
        st.session_state["result"] = None

    # Sidebar
    render_sidebar()

    # Tabs
    tabs = st.tabs([
        "🏠 Home",
        "🔬 Screening",
        "📊 Result",
        "🔍 AI Explanation",
        "ℹ️ Model Info",
        "📖 About",
    ])

    with tabs[0]: tab_home()
    with tabs[1]: tab_screening()
    with tabs[2]: tab_result()
    with tabs[3]: tab_explanation()
    with tabs[4]: tab_model_info()
    with tabs[5]: tab_about()


if __name__ == "__main__":
    main()
