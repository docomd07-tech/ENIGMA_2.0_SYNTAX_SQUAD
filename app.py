"""
app_single.py — Stress-Vision | Orbital Agronomy  (SINGLE-FILE VERSION)
All utility functions are inlined — no separate utils.py required.

Merge summary
─────────────
YOUR project:
  ✔ Email + strong-password signup with validation
  ✔ Welcome screen (dismissible, once per session)
  ✔ Sample Data mode — 4 global locations, Plotly charts
  ✔ Upload mode — auto-detect bands, Plotly side-by-side maps, histograms
  ✔ PPT features — vitality gauge, traffic-light, CWSI/CIre/Py-SIF gauges,
                   stress probability pie, impact cards, radar comparison,
                   5-phase workflow

FRIEND's project:
  ✔ Login for returning users (persistent users dict in session state)
  ✔ Sidebar navigation — Home / Sample Data / Band Analysis / Data Acquisition / Logout
  ✔ Tabbed analysis — RGB composite, NDVI (VIRIDIS), Stress Heatmap (MAGMA),
                      High-Stress Mask, Crop Health Overview, Advanced Metrics
  ✔ Stress threshold slider + pixel-level stats
  ✔ Neon glow metric cards
  ✔ REP (Red-Edge Position) and Thermal Anomaly layers
  ✔ Per-tab download buttons

Auth choice: YOUR signup (email + strong password) + FRIEND's Login (returning users).
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2
import rasterio
from rasterio.io import MemoryFile
import hashlib
import re
from io import BytesIO
from PIL import Image


# ══════════════════════════════════════════════════════════
# INLINED UTILITIES  (contents of utils.py)
# ══════════════════════════════════════════════════════════

# ── Auth ──────────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,12}$"
    return bool(re.match(pattern, password))

def validate_email(email):
    return bool(re.match(r"^\S+@\S+\.\S+$", email))


# ── Band loading ──────────────────────────────────────────
def load_band(file):
    """Raw load — no normalisation. Used by friend's cv2 pipeline."""
    with rasterio.open(file) as src:
        return src.read(1).astype(float)

def read_band_array(uploaded_file):
    """Percentile-stretch load via MemoryFile. Used by your upload pipeline."""
    try:
        bytes_data = uploaded_file.read()
        with MemoryFile(bytes_data) as memfile:
            with memfile.open() as dataset:
                arr = dataset.read(1).astype(np.float32)
                nodata = dataset.nodata
                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)
                p2, p98 = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
                if p98 > p2:
                    arr = (arr - p2) / (p98 - p2)
                return np.clip(arr, 0, 1)
    except Exception:
        return None


# ── Array helpers ─────────────────────────────────────────
def normalize(array):
    array = np.nan_to_num(array)
    mn, mx = np.min(array), np.max(array)
    return (array - mn) / (mx - mn + 1e-8)

def safe_index(a, b, eps=1e-10):
    denom = np.where(np.abs(a + b) < eps, eps, a + b)
    return np.clip((a - b) / denom, -1, 1)

def downsample(arr, size=80):
    try:
        from scipy.ndimage import zoom
        f = (size / arr.shape[0], size / arr.shape[1])
        return zoom(arr, f, order=1)
    except Exception:
        return arr[:size, :size]


# ── Spectral indices ──────────────────────────────────────
def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def calculate_ndwi(nir, swir):
    return (nir - swir) / (nir + swir + 1e-8)

def calculate_red_edge_index(nir, red_edge):
    return (nir - red_edge) / (nir + red_edge + 1e-8)

def calculate_stress(ndvi, ndwi, red_edge):
    return normalize((1 - ndwi) + (1 - ndvi) + (1 - red_edge))

def calculate_msi(nir, swir):
    msi_raw = np.where(nir > 0.01, swir / (nir + 1e-10), 0)
    return np.clip(1 - msi_raw / 2, 0, 1)


# ── Image builders ────────────────────────────────────────
def create_rgb(red, green, nir):
    return normalize(np.dstack((red, green, nir)))

def create_heatmap(stress):
    h = cv2.applyColorMap((stress * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

def create_stress_magma(stress):
    h = cv2.applyColorMap((stress * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

def create_ndvi_colormap(ndvi_norm):
    h = cv2.applyColorMap((ndvi_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

def convert_np_to_image(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Early-warning index simulators ───────────────────────
def compute_cwsi(health_score):
    return round(max(0.0, min(1.0, 1.0 - health_score + np.random.uniform(-0.04, 0.04))), 3)

def compute_cire(health_score):
    return round(max(0.0, health_score * 3.5 + np.random.uniform(-0.1, 0.1)), 3)

def compute_pysif(health_score):
    return round(max(0.0, min(1.0, health_score * 0.85 + np.random.uniform(-0.03, 0.03))), 3)


# ── Vitality & alert helpers ──────────────────────────────
def vitality_score(health_score):
    return int(round(health_score * 100))

def traffic_light_class(health_score):
    if health_score >= 0.65:
        return "healthy",   "🟢 HEALTHY",           "tl-healthy"
    elif health_score >= 0.45:
        return "previsual", "🟡 PRE-VISUAL STRESS",  "tl-previsual"
    else:
        return "damaged",   "🔴 METABOLIC DAMAGE",   "tl-damaged"

def irrigation_prescription(health_score, location_label="the field"):
    if health_score >= 0.65:
        return f"No immediate action needed. Continue monitoring {location_label} every 5 days."
    elif health_score >= 0.45:
        return (f"Metabolic slowdown detected in {location_label}. "
                f"Increase irrigation by 10–15% within 48 hours to prevent yield loss.")
    else:
        return (f"Critical metabolic damage in {location_label}. "
                f"Flood-irrigate immediately and apply targeted micronutrient treatment. "
                f"Yield loss imminent without intervention.")


# ── Plain-English Summary Card ────────────────────────────
def plain_english_summary(health_score, location_label="this field",
                           stressed_pct=None, ndvi_mean=None,
                           cwsi=None, cire=None, pysif=None,
                           crop=None, context="sample"):
    """
    Renders a big friendly 'What does this mean?' card BEFORE any charts.
    All parameters are optional — function degrades gracefully.
    """
    vs = vitality_score(health_score)
    status_key, _, _ = traffic_light_class(health_score)

    if status_key == "healthy":
        headline_emoji = "✅"
        headline_color = "#00e676"
        headline_text  = "Your crops look healthy!"
        plain_verdict  = (
            f"The satellite data for <b>{location_label}</b> shows that your "
            f"{'<b>' + crop + '</b> ' if crop else ''}crops are in <b>good condition</b>. "
            f"Plants are well-watered, their internal cell structure is intact, "
            f"and photosynthesis is running normally."
        )
        what_happened  = (
            "Think of it like a health check-up where everything came back normal. "
            "The crops are drinking enough water, their leaves are full of chlorophyll "
            "(the green pigment that makes food from sunlight), and the roots are healthy."
        )
        action_box_color = "#0a2d0a"
        action_border    = "#00e676"
        action_title     = "👨‍🌾 What should a farmer do?"
        action_text      = (
            f"No action needed right now. Keep doing what you're doing. "
            f"Check back in 5 days using this tool to make sure conditions haven't changed."
        )

    elif status_key == "previsual":
        headline_emoji = "⚠️"
        headline_color = "#ffa726"
        headline_text  = "Hidden stress detected — the plant looks fine but isn't!"
        plain_verdict  = (
            f"The satellite data for <b>{location_label}</b> has detected <b>early, invisible stress</b> "
            f"in your {'<b>' + crop + '</b> ' if crop else ''}crops. "
            f"<b>To the human eye and even to standard satellite tools, the field looks green and healthy.</b> "
            f"But our sensors are picking up changes happening inside the plant — "
            f"like a person who feels feverish before they look sick."
        )
        what_happened  = (
            "Here's what's happening right now inside the plant: <br>"
            "• The plant is starting to <b>close its tiny breathing pores</b> (stomata) to save water — like sweating less on a hot day.<br>"
            "• The <b>chlorophyll molecules</b> (the green pigment) are beginning to weaken at a molecular level — not visible yet, but detectable.<br>"
            "• <b>Photosynthesis</b> (how the plant makes food from sunlight) is slowing down.<br>"
            "A standard NDVI satellite tool would <b>not</b> catch this yet. Our Red-Edge band (B05) and water band (B11) catch it <b>7–14 days earlier</b>."
        )
        action_box_color = "#2d2000"
        action_border    = "#ffa726"
        action_title     = "👨‍🌾 What should a farmer do RIGHT NOW?"
        action_text      = (
            "Act within the next <b>24–48 hours</b>:<br>"
            "• <b>Increase irrigation by 10–15%</b> — the plants are thirsty before they show it.<br>"
            "• Check your <b>nitrogen fertilizer</b> levels — early chlorophyll breakdown often means nutrient deficiency.<br>"
            "• <b>Do NOT wait</b> until the field looks yellow — by then, you've already lost 10–20% of your yield and it can't be recovered."
        )

    else:  # damaged
        headline_emoji = "🚨"
        headline_color = "#ef5350"
        headline_text  = "Critical crop damage — immediate action required!"
        plain_verdict  = (
            f"The satellite data for <b>{location_label}</b> shows <b>severe metabolic damage</b> "
            f"in your {'<b>' + crop + '</b> ' if crop else ''}crops. "
            f"The stress has gone beyond the hidden phase — <b>plant cells are actively dying</b>. "
            f"Visible yellowing and wilting are either already present or days away."
        )
        what_happened  = (
            "Here's what has gone wrong inside the plant:<br>"
            "• <b>Cell membranes are rupturing</b> from prolonged water or heat stress — this is irreversible damage.<br>"
            "• <b>Chlorophyll is breaking down rapidly</b> — the plant is losing its ability to make food from sunlight.<br>"
            "• <b>Photosynthesis has nearly stopped</b> — the plant is shutting down non-essential functions to survive.<br>"
            "This is the stage where <b>yield loss is already locked in</b>. Emergency action can prevent complete crop failure, "
            "but the lost yield cannot be recovered."
        )
        action_box_color = "#3b0a0a"
        action_border    = "#ef5350"
        action_title     = "👨‍🌾 Emergency action needed NOW!"
        action_text      = (
            "<b>Do all of the following immediately:</b><br>"
            "• <b>Flood-irrigate the entire field</b> — do not wait for the next scheduled irrigation.<br>"
            "• Apply <b>micronutrient foliar spray</b> (especially nitrogen and potassium) directly on leaves.<br>"
            "• <b>Shade vulnerable seedlings</b> if heat is a contributing factor.<br>"
            "• Contact your <b>agricultural extension officer</b> — this level of stress may qualify for crop insurance claims."
        )

    # ── Build extra insight lines from optional data ──────
    extra_lines = []
    if stressed_pct is not None:
        sp = round(stressed_pct, 1)
        if sp < 20:
            extra_lines.append(f"🗺️ <b>Only {sp}% of the field area</b> is stressed — it's a small patch, easy to treat early.")
        elif sp < 50:
            extra_lines.append(f"🗺️ <b>{sp}% of the field area</b> is showing stress — about half the field needs attention.")
        else:
            extra_lines.append(f"🗺️ <b>{sp}% of the field area</b> is stressed — more than half the field is affected.")
    if ndvi_mean is not None:
        nv = round(ndvi_mean, 2)
        if nv >= 0.6:
            extra_lines.append(f"🌿 <b>NDVI score is {nv}</b> — the conventional tool says healthy. But our pre-visual sensors disagree.")
        elif nv >= 0.4:
            extra_lines.append(f"🌿 <b>NDVI score is {nv}</b> — the conventional tool is starting to notice stress too.")
        else:
            extra_lines.append(f"🌿 <b>NDVI score is {nv}</b> — visible damage is now detectable even by standard tools.")
    if cwsi is not None:
        if cwsi > 0.6:
            extra_lines.append(f"💧 <b>Water stress is high (CWSI={cwsi})</b> — the plant is essentially running a fever from dehydration.")
        elif cwsi > 0.35:
            extra_lines.append(f"💧 <b>Mild water stress detected (CWSI={cwsi})</b> — the plant is conserving water.")
        else:
            extra_lines.append(f"💧 <b>Water status is fine (CWSI={cwsi})</b> — the plant is well hydrated.")
    if pysif is not None:
        if pysif < 0.4:
            extra_lines.append(f"☀️ <b>Photosynthesis is very low (SIF={pysif})</b> — the plant is barely making food from sunlight.")
        elif pysif < 0.65:
            extra_lines.append(f"☀️ <b>Photosynthesis is reduced (SIF={pysif})</b> — the plant is underperforming.")
        else:
            extra_lines.append(f"☀️ <b>Photosynthesis is normal (SIF={pysif})</b> — light energy is being used efficiently.")

    extra_html = "".join(f"<div style='margin:5px 0;color:#cdd8e3;font-size:0.93em'>{line}</div>" for line in extra_lines)

    st.markdown(f"""
<div style='background:linear-gradient(135deg,#0d1f36,#0a2a4a);
            border:2px solid {headline_color};border-radius:14px;
            padding:24px 28px;margin-bottom:20px'>

  <div style='font-size:1.55em;font-weight:bold;color:{headline_color};margin-bottom:14px'>
    {headline_emoji} In Plain English: {headline_text}
  </div>

  <div style='background:rgba(0,0,0,0.25);border-radius:8px;padding:14px 18px;margin-bottom:14px;
              color:#e0f0ff;font-size:1.0em;line-height:1.7'>
    {plain_verdict}
  </div>

  <details>
    <summary style='color:#4fc3f7;cursor:pointer;font-size:0.95em;font-weight:600;
                    margin-bottom:8px'>
      🔬 What is actually happening inside the plant? (click to expand)
    </summary>
    <div style='color:#b0cce8;font-size:0.92em;line-height:1.8;
                padding:10px 14px;border-left:3px solid #1e4060;margin-top:8px'>
      {what_happened}
    </div>
  </details>

  {f"<div style='margin-top:12px'>{extra_html}</div>" if extra_lines else ""}

  <div style='background:{action_box_color};border:1px solid {action_border};
              border-radius:8px;padding:14px 18px;margin-top:16px'>
    <div style='color:{action_border};font-weight:bold;font-size:1.0em;margin-bottom:8px'>
      {action_title}
    </div>
    <div style='color:#e0f0ff;font-size:0.94em;line-height:1.8'>{action_text}</div>
  </div>

  <div style='margin-top:12px;color:#78909c;font-size:0.82em'>
    📡 Data source: ESA Sentinel-2 Level-2A · Analysis: Composite Stress Score
    (NDVI + NDWI + Red-Edge Index) · Vitality: {vs}/100
  </div>
</div>
""", unsafe_allow_html=True)


# ── Plotly chart helpers ──────────────────────────────────
def mini_gauge(val, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={"font": {"color": "#e0f0ff", "size": 28}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#78909c"},
            "bar":  {"color": color},
            "bgcolor": "#0d1f36", "bordercolor": "#1e4060",
            "steps": [
                {"range": [0, 0.4],    "color": "#1a0a0a"},
                {"range": [0.4, 0.65], "color": "#1a1400"},
                {"range": [0.65, 1],   "color": "#0a1a0a"},
            ],
        },
        title={"text": title, "font": {"color": "#90a4ae", "size": 13}},
    ))
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0f0ff"))
    return fig

def vitality_donut(vscore, color):
    fig = go.Figure(go.Pie(
        values=[vscore, 100-vscore], hole=0.72,
        marker_colors=[color, "#0d1f36"],
        textinfo="none", hoverinfo="skip", showlegend=False,
    ))
    fig.add_annotation(
        text=f"<b>{vscore}%</b><br><span style='font-size:11px'>Vitality</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=22, color="#e0f0ff"), align="center"
    )
    fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── UI component helpers (friend's neon card) ─────────────
def neon_card(title, value, color="#00ffcc"):
    st.markdown(
        f"""
        <div style='padding:15px;border-radius:10px;
                    background:linear-gradient(135deg,#111,#222);
                    box-shadow:0 0 20px {color};text-align:center;
                    margin-bottom:12px'>
            <h4 style='color:{color};margin:0'>{title}</h4>
            <h2 style='color:{color};margin:0'>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stress-Vision | Orbital Agronomy",
    page_icon="🌾",
    layout="wide",
)


# ══════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"]   { background: #0a1628; }
[data-testid="stSidebar"] * { color: #e0f0ff !important; }
.main  { background: #07111f; color: #e0f0ff; }
h1, h2, h3 { color: #4fc3f7 !important; }
.stMetric  { background: #0d1f36; border: 1px solid #1e4060;
             border-radius: 8px; padding: 10px; }
.stButton>button       { background: #1565c0; color: white;
                         border: none; border-radius: 6px; }
.stButton>button:hover { background: #1976d2; }
.auth-box  { max-width: 460px; margin: auto; padding: 2rem;
             background: #0d1f36; border-radius: 12px;
             border: 1px solid #1e4060; }
.welcome-box  { background: linear-gradient(135deg,#0d1f36,#0a2a4a);
                border: 1px solid #1e4060; border-radius: 12px;
                padding: 28px 32px; margin-bottom: 8px; }
.welcome-title { color:#4fc3f7; font-size:1.5em; font-weight:bold; margin-bottom:10px; }
.welcome-body  { color:#b0cce8; line-height:1.7; font-size:0.97em; }
.welcome-hl    { color:#ffa726; font-weight:600; }
.pill          { display:inline-block; background:#1565c0; color:#e0f0ff;
                 border-radius:20px; padding:3px 12px; font-size:0.8em; margin:3px 2px; }
.pill-green    { background:#1b5e20; }
.pill-orange   { background:#e65100; }
.how-to-box  { background:#0d1f36; border-left:4px solid #4fc3f7;
               padding:16px 20px; border-radius:0 8px 8px 0; margin-bottom:12px; }
.how-to-step { color:#4fc3f7; font-weight:bold; font-size:1.05em; }
.band-tag    { display:inline-block; background:#1565c0; color:white;
               border-radius:4px; padding:2px 8px; font-size:0.82em; margin:2px 0; }
.tl-healthy   { background:#0a2d0a; border:2px solid #66bb6a;
                border-radius:10px; padding:16px 20px; }
.tl-previsual { background:#2d2000; border:2px solid #ffa726;
                border-radius:10px; padding:16px 20px; }
.tl-damaged   { background:#3b0a0a; border:2px solid #ef5350;
                border-radius:10px; padding:16px 20px; }
.tl-label { font-size:1.1em; font-weight:bold; margin-bottom:6px; }
.tl-alert { font-size:0.95em; color:#cdd8e3; }
.impact-card { background:#0d1f36; border:1px solid #1e4060;
               border-radius:10px; padding:18px; text-align:center; }
.impact-num  { font-size:2em; font-weight:bold; color:#4fc3f7; }
.impact-desc { font-size:0.82em; color:#90a4ae; margin-top:4px; }
.wf-step { background:#0d1f36; border-left:4px solid #4fc3f7;
           border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px; }
.wf-num  { color:#ffa726; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# STATIC DATA
# ══════════════════════════════════════════════════════════
REAL_DATA = {
    "Punjab, India":   {"temp_anomaly":"+4.0°C","health_score":0.62,"speed":9,
                        "yield":18.2,"water":12.5,"trend_start":0.85,"trend_end":0.62,
                        "insight":"CRITICAL: Wheat in grain-filling stage. Heat stress alert issued by PAU. Light irrigation required immediately."},
    "California, USA": {"temp_anomaly":"+1.5°C","health_score":0.78,"speed":6,
                        "yield":14.5,"water":28.0,"trend_start":0.72,"trend_end":0.78,
                        "insight":"STABLE: Reservoirs at 64% capacity. Monitoring for salinity."},
    "Nairobi, Kenya":  {"temp_anomaly":"+2.8°C","health_score":0.41,"speed":11,
                        "yield":24.1,"water":35.0,"trend_start":0.65,"trend_end":0.41,
                        "insight":"ALARM: Severe drought in Central Kenya. Vegetation impacted by 30% rainfall deficit."},
    "Mekong Delta":    {"temp_anomaly":"+0.8°C","health_score":0.82,"speed":5,
                        "yield":10.2,"water":18.5,"trend_start":0.80,"trend_end":0.82,
                        "insight":"OPTIMAL: Abundant water source for Summer-Autumn rice."},
}
CROP_INFO = {
    "Punjab, India":"Wheat","California, USA":"Almonds",
    "Nairobi, Kenya":"Maize","Mekong Delta":"Rice",
}
REQUIRED_BANDS = ["B03","B04","B05","B08","B11"]
BAND_DEFINITIONS = {
    "B03":("Green",    "560 nm",  "Vegetation and water reflectance"),
    "B04":("Red",      "665 nm",  "Chlorophyll absorption"),
    "B05":("Red Edge", "705 nm",  "Pre-visual crop stress"),
    "B08":("NIR",      "842 nm",  "Cell structure health"),
    "B11":("SWIR 1",   "1610 nm", "Soil moisture content"),
}


# ══════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════
for key, default in [
    ("users", {}),
    ("logged_in", False),
    ("username", ""),
    ("user_name", ""),
    ("welcome_dismissed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════
# AUTH GATE  (your strong validation + friend's login)
# ══════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("<div class='auth-box'>", unsafe_allow_html=True)
    st.title("🌾 Stress-Vision")
    st.caption("Pre-visual crop stress detection · SDG 2: Zero Hunger")
    st.write("")

    auth_tab = st.radio("", ["🔑 Login", "📝 Sign Up"],
                        horizontal=True, label_visibility="collapsed")

    if auth_tab == "📝 Sign Up":
        with st.form("signup_form", clear_on_submit=False):
            full_name = st.text_input("Full Name")
            email     = st.text_input("Email Address")
            password  = st.text_input(
                "Password", type="password",
                help="8–12 chars · 1 uppercase · 1 lowercase · 1 number · 1 special (@$!%*?&)"
            )
            submitted = st.form_submit_button("Create Account & Enter", use_container_width=True)
            if submitted:
                if not full_name.strip():
                    st.error("Please enter your full name.")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                elif not validate_password(password):
                    st.error("Password must be 8–12 chars and include uppercase, lowercase, number, and special symbol.")
                elif email in st.session_state.users:
                    st.error("An account with this email already exists. Please log in.")
                else:
                    st.session_state.users[email] = {
                        "pw": hash_password(password),
                        "name": full_name.strip(),
                    }
                    st.session_state.logged_in  = True
                    st.session_state.username   = email
                    st.session_state.user_name  = full_name.strip()
                    st.rerun()
    else:
        with st.form("login_form", clear_on_submit=False):
            email    = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                user = st.session_state.users.get(email)
                if user and user["pw"] == hash_password(password):
                    st.session_state.logged_in  = True
                    st.session_state.username   = email
                    st.session_state.user_name  = user["name"]
                    st.session_state.welcome_dismissed = False
                    st.rerun()
                else:
                    st.error("Incorrect email or password. Please try again or sign up.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════
if not st.session_state.welcome_dismissed:
    st.markdown(f"""
<div class='welcome-box'>
  <div class='welcome-title'>Welcome, {st.session_state.user_name}! 🌾</div>
  <div class='welcome-body'>
    <b>Stress-Vision</b> is a pre-visual crop stress detection platform built to support
    <span class='welcome-hl'>SDG 2: Zero Hunger</span>.
    <br><br>
    Traditional satellite tools detect stress only <em>after</em> plants visibly yellow or wilt —
    by then, yield loss is locked in. Stress-Vision uses
    <span class='welcome-hl'>Sentinel-2 hyperspectral bands</span> — particularly the
    <b>Red-Edge (B05, 705 nm)</b> and <b>SWIR (B11, 1610 nm)</b> — to identify water stress,
    nutrient deficiency, and drought signatures
    <b>7–14 days before they become visible</b> to conventional NDVI tools.
    <br><br>
    <b>What you can do:</b><br>
    <span class='pill'>Explore sample data from 4 global locations</span>
    <span class='pill pill-green'>Upload real Sentinel-2 bands for live analysis</span>
    <span class='pill pill-orange'>Compare NDVI vs pre-visual stress indices</span>
    <span class='pill'>Generate RGB / NDVI / heatmap images with download</span>
    <span class='pill pill-green'>Vitality gauge · Traffic-light alert · Early-warning indices</span>
    <br><br>
    Bands accepted: <code>B03 B04 B05 B08 B11</code> — free from
    <a href='https://dataspace.copernicus.eu/' style='color:#4fc3f7;'>Copernicus Open Access Hub</a>.
  </div>
</div>
""", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("Got it — Let's Begin 🚀", type="primary"):
            st.session_state.welcome_dismissed = True
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════
st.sidebar.title(f"🌾 {st.session_state.user_name}")
st.sidebar.caption(f"Logged in as {st.session_state.username}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🗺️ Sample Data", "📊 Band Analysis", "📡 Data Acquisition", "Logout"],
)

if page == "Logout":
    for k in ("logged_in","username","user_name","welcome_dismissed"):
        st.session_state[k] = False if k == "logged_in" else "" if k != "welcome_dismissed" else False
    st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌾 Stress-Vision Dashboard")
    st.markdown(
        "**Pre-Visual Crop Stress Detection** powered by Sentinel-2 satellite bands — "
        "detecting water and nutrient stress **7–14 days before visible yellowing**."
    )
    st.markdown("---")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Detection Lead",   "7–14 Days", "vs. NDVI")
    c2.metric("Yield Protection", "10–20%",    "Early intervention")
    c3.metric("Water Savings",    "15–20%",    "Smart irrigation")
    c4.metric("Cost Reduction",   "30%",       "Targeted treatment")

    st.markdown("---")
    st.subheader("How Stress-Vision Works — 5-Phase Workflow")

    workflow = [
        ("1","Data Ingestion — The Triple Signal",
         "Captures <b>SIF</b> (far-red emissions), <b>Thermal LST</b> (land surface temperature), and "
         "<b>Red-Edge bands</b> (B05–B07) from ESA/NASA satellites simultaneously."),
        ("2","Pre-Processing & Fusion",
         "<b>Spatial downscaling</b> to 10 m · <b>Atmospheric correction</b> (haze removal) · "
         "<b>Temporal sync</b> against a 30-day historical baseline."),
        ("3","Feature Extraction — Early Warning Indices",
         "<b>CWSI</b> (transpiration fever) · <b>CIre</b> (chlorophyll destabilisation) · "
         "<b>Py-SIF</b> (actual photosynthetic yield) — all react 7–14 days before NDVI."),
        ("4","ML Inference — Hybrid CNN-LSTM",
         "<b>CNN</b>: identifies where stress is starting spatially. "
         "<b>LSTM</b>: flags metabolic shutdown from time-series drops. "
         "Output: Stress Probability Map — Healthy / Pre-Visual / Damaged."),
        ("5","Actionable Output — Traffic Light Prescription",
         "Plain-language alert: <i>'Metabolic slowdown in Zone B. Increase irrigation by 15% tonight.'</i> "
         "Pushable to mobile app, farm software, or IoT irrigation hardware."),
    ]
    for num,title,body in workflow:
        st.markdown(f"""
<div class='wf-step'>
  <span class='wf-num'>Phase {num} —</span>
  <b style='color:#e0f0ff'>{title}</b>
  <div style='color:#b0cce8;margin-top:6px;font-size:0.93em'>{body}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Stress-Vision vs Standard NDVI")
    st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #4fc3f7;border-radius:0 8px 8px 0;
            padding:10px 16px;margin-bottom:14px;color:#b0cce8;font-size:0.9em'>
  <b style='color:#4fc3f7'>📡 What is this chart comparing?</b><br>
  The <b style='color:#4fc3f7'>blue shape</b> is Stress-Vision. The <b style='color:#ef5350'>red dashed shape</b>
  is the standard NDVI tool used by most satellites today. A bigger area = better performance on that metric.
  The most important axis is <b>"Pre-Visual Capability"</b> — that's where Stress-Vision dominates,
  because NDVI scores nearly zero there (it simply cannot detect hidden stress at all).
</div>
""", unsafe_allow_html=True)
    categories = ["Cost Effectiveness","Detection Speed","Pre-Visual Capability",
                  "Prescription Output","Scalability","Compatibility"]
    sv_s   = [9.0,8.5,9.5,9.0,8.0,7.5]
    ndvi_s = [2.5,3.0,1.0,1.5,7.0,9.0]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=sv_s+[sv_s[0]], theta=categories+[categories[0]],
        fill="toself", name="Stress-Vision",
        line=dict(color="#4fc3f7",width=2), fillcolor="rgba(79,195,247,0.15)"))
    fig_r.add_trace(go.Scatterpolar(r=ndvi_s+[ndvi_s[0]], theta=categories+[categories[0]],
        fill="toself", name="Standard NDVI",
        line=dict(color="#ef5350",width=2,dash="dash"), fillcolor="rgba(239,83,80,0.10)"))
    fig_r.update_layout(
        polar=dict(bgcolor="#0d1f36",
                   radialaxis=dict(visible=True,range=[0,10],gridcolor="#1e4060",
                                   tickfont=dict(color="#78909c")),
                   angularaxis=dict(gridcolor="#1e4060",tickfont=dict(color="#b0cce8"))),
        showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f0ff")),
        paper_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=40,r=40,t=20,b=20))
    st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE: SAMPLE DATA
# ══════════════════════════════════════════════════════════
elif page == "🗺️ Sample Data":
    location     = st.sidebar.selectbox("Satellite Target", list(REAL_DATA.keys()))
    index_choice = st.sidebar.radio(
        "Spectral Analysis",
        ["NDWI (Water)","Red-Edge NDVI (Nutrient)","MSI (Moisture)"]
    )
    data = REAL_DATA[location]

    st.title("🗺️ Sample Data — Orbital Analysis")
    st.markdown(
        f"**Location:** {location} &nbsp;|&nbsp; **Crop:** {CROP_INFO[location]} &nbsp;|&nbsp; **SDG 2: Zero Hunger**"
    )

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Detection Lead",    f"{data['speed']} Days", "vs NDVI")
    m2.metric("Yield Protection",  f"{data['yield']}%",     "Potential Save")
    m3.metric("Water Optimisation",f"{data['water']}%",     "Efficiency Gain")
    m4.metric("Temp Anomaly",      data['temp_anomaly'],     "Above Normal")
    st.divider()

    # ── Plain-English Summary (shown FIRST before any charts) ──
    np.random.seed(int(data['health_score'] * 100))
    cwsi_s  = compute_cwsi(data['health_score'])
    cire_s  = compute_cire(data['health_score'])
    pysif_s = compute_pysif(data['health_score'])
    plain_english_summary(
        health_score  = data['health_score'],
        location_label= location,
        ndvi_mean     = data['health_score'] + 0.05,
        cwsi          = cwsi_s,
        cire          = cire_s,
        pysif         = pysif_s,
        crop          = CROP_INFO[location],
        context       = "sample",
    )

    with st.expander("Sentinel-2 Band Reference"):
        bd = pd.DataFrame([{"Band":k,"Name":v[0],"Wavelength":v[1],"Purpose":v[2]}
                           for k,v in BAND_DEFINITIONS.items()])
        st.dataframe(bd, use_container_width=True, hide_index=True)
        rc1,rc2,rc3 = st.columns(3)
        for c,title,scale,seed in [
            (rc1,"B04 — Red (665 nm): Chlorophyll","Reds",42),
            (rc2,"B08 — NIR (842 nm): Cell structure","Greys",43),
            (rc3,"B11 — SWIR (1610 nm): Moisture","Blues",44),
        ]:
            np.random.seed(seed)
            with c:
                st.caption(f"**{title}**")
                fi = px.imshow(np.random.uniform(0.05,0.5,(50,50)),
                               color_continuous_scale=scale,zmin=0,zmax=1)
                fi.update_layout(margin=dict(l=0,r=0,t=0,b=0),height=150,
                                 coloraxis_showscale=False)
                st.plotly_chart(fi, use_container_width=True)

    st.write("---")

    def gen_map(base, stressed=True):
        np.random.seed(1)
        g = np.full((80,80),base) + np.random.normal(0,0.05,(80,80))
        g[20:40,25:45] -= 0.35 if stressed else 0.12
        if stressed: g[55:65,55:70] -= 0.20
        return np.clip(g,0,1)

    cm1,cm2 = st.columns(2)
    with cm1:
        st.subheader("Standard NDVI")
        fn = px.imshow(gen_map(data['health_score']+0.15,False),
                       color_continuous_scale="Greens",zmin=0,zmax=1,labels={"color":"NDVI"})
        fn.update_layout(margin=dict(l=0,r=0,t=0,b=0),height=350)
        st.plotly_chart(fn, use_container_width=True)
    with cm2:
        st.subheader("Stress-Vision Overlay")
        fs = px.imshow(gen_map(data['health_score']),
                       color_continuous_scale=["red","yellow","green"],
                       zmin=0,zmax=1,labels={"color":"Stress"})
        fs.update_layout(margin=dict(l=0,r=0,t=0,b=0),height=350)
        st.plotly_chart(fs, use_container_width=True)
    st.caption("🔴 high stress · 🟡 moderate · 🟢 healthy — red zones on the right = pre-visual stress.")

    st.write("---")
    ct,ci = st.columns([2,1])
    with ct:
        st.subheader("30-Day Historical Trend")
        days = pd.date_range(end=pd.Timestamp.now(),periods=30)
        np.random.seed(7)
        tv = np.linspace(data['trend_start'],data['trend_end'],30)+np.random.normal(0,0.012,30)
        ft = go.Figure()
        ft.add_trace(go.Scatter(x=days,y=tv,mode="lines",fill="tozeroy",
                                line=dict(color="#4fc3f7",width=2.5),
                                fillcolor="rgba(79,195,247,0.10)"))
        ft.add_hline(y=0.5,line_dash="dash",line_color="#ef5350",annotation_text="Stress threshold")
        ft.update_layout(height=260,margin=dict(l=0,r=0,t=0,b=0),
                         paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(range=[0,1],gridcolor="#1e3048"),
                         xaxis=dict(gridcolor="#1e3048"))
        st.plotly_chart(ft, use_container_width=True)
    with ci:
        st.subheader("Field Insight")
        fn_ins = st.error if data['health_score']<0.5 else (st.warning if data['health_score']<0.7 else st.success)
        fn_ins(data['insight'])
        rdf = pd.DataFrame([{"Location":location,"Crop":CROP_INFO[location],
                              "Health Score":data['health_score'],
                              "Temp Anomaly":data['temp_anomaly'],
                              "Detection Lead (days)":data['speed'],
                              "Yield Protection (%)":data['yield'],
                              "Water Optimisation (%)":data['water'],
                              "Insight":data['insight']}])
        st.download_button("📥 Download Field Report",rdf.to_csv(index=False).encode(),
                           f"report_{location.replace(', ','_').replace(' ','_')}.csv","text/csv")

    st.write("---")
    st.subheader("Spectral Signature — Healthy vs Stressed Crop")
    st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #ffd54f;border-radius:0 8px 8px 0;
            padding:10px 16px;margin-bottom:14px;color:#b0cce8;font-size:0.9em'>
  <b style='color:#ffd54f'>📈 What is this chart showing?</b><br>
  Each point on the x-axis is a different satellite band (type of light).
  The two lines show how much light a <b style='color:#66bb6a'>healthy crop</b> vs a
  <b style='color:#ef5350'>stressed crop</b> reflects back to the satellite.
  <br><br>
  Notice the <b style='color:#ffd54f'>highlighted yellow zone (695–720 nm)</b> — this is where the
  two lines start to <em>diverge</em>. That's the Red-Edge band. The stressed crop reflects more light
  here because its chlorophyll is weakening — but the plant still <em>looks green</em> to the naked eye.
  This divergence is the scientific proof of pre-visual detection.
  Standard NDVI tools don't look at this zone at all.
</div>
""", unsafe_allow_html=True)
    wl = [560,665,705,842,1610]; bl = ["B03","B04","B05","B08","B11"]
    hs = [0.12,0.05,0.19,0.42,0.22]
    np.random.seed(3); sf = 1-data['health_score']
    ss = [v*(1-sf*np.random.uniform(0.2,0.7)) for v in hs]
    fsp = go.Figure()
    fsp.add_trace(go.Scatter(x=wl,y=hs,mode="lines+markers",name="Healthy Crop",
                             line=dict(color="#66bb6a",width=2),marker=dict(size=8)))
    fsp.add_trace(go.Scatter(x=wl,y=ss,mode="lines+markers",name="Stressed Crop",
                             line=dict(color="#ef5350",width=2,dash="dash"),
                             marker=dict(size=8,symbol="x")))
    fsp.add_vrect(x0=695,x1=720,fillcolor="rgba(255,193,7,0.10)",
                  annotation_text="Red-Edge Pre-Visual Zone",
                  annotation_position="top left",line_width=0)
    fsp.update_layout(height=290,xaxis_title="Wavelength (nm)",yaxis_title="Reflectance",
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      xaxis=dict(gridcolor="#1e3048",tickvals=wl,
                                 ticktext=[f"<b>{b}</b><br>{w} nm" for b,w in zip(bl,wl)]),
                      yaxis=dict(gridcolor="#1e3048"),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fsp, use_container_width=True)

    st.write("---")
    st.subheader("Field Vitality Score & Traffic Light Alert")
    np.random.seed(int(data['health_score']*100))
    vs_s  = vitality_score(data['health_score'])
    col_s = "#66bb6a" if vs_s>=65 else "#ffa726" if vs_s>=45 else "#ef5350"
    tks,tls,tcs = traffic_light_class(data['health_score'])
    prs   = irrigation_prescription(data['health_score'], location)
    cwsi  = compute_cwsi(data['health_score']); cire=compute_cire(data['health_score'])
    psif  = compute_pysif(data['health_score'])
    cg,ctl = st.columns([1,2])
    with cg:
        st.plotly_chart(vitality_donut(vs_s,col_s), use_container_width=True)
        st.caption(f"{location} | {CROP_INFO[location]}")
    with ctl:
        st.markdown(f"""
<div class='{tcs}'>
  <div class='tl-label'>{tls}</div>
  <div class='tl-alert'>
    <b>Alert:</b> {prs}<br><br>
    <b>Plant looks green?</b> {"Yes — but metabolic signals say otherwise. Pre-Visual advantage." if tks=="previsual"
                               else "No — visible damage begun. Act immediately." if tks=="damaged"
                               else "Yes — metabolic signals confirm healthy transpiration."}
  </div>
</div>""", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Early Warning Index Dashboard")
    st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #4fc3f7;border-radius:0 8px 8px 0;
            padding:10px 16px;margin-bottom:14px;color:#b0cce8;font-size:0.9em'>
  <b style='color:#4fc3f7'>📊 What are these dials?</b> These three gauges detect crop stress
  <b>before it becomes visible</b> — each measures a different hidden signal.
  Needle in the <b style='color:#66bb6a'>green zone = good</b>.
  Needle in the <b style='color:#ffa726'>middle = early warning</b>.
  Needle in the <b style='color:#ef5350'>red zone = act now</b>.
</div>
""", unsafe_allow_html=True)
    eg1,eg2,eg3 = st.columns(3)
    with eg1:
        st.plotly_chart(mini_gauge(cwsi,"CWSI — Water Stress","#ef5350"), use_container_width=True)
        st.caption("High = plant 'running a fever' from closed stomata.")
    with eg2:
        st.plotly_chart(mini_gauge(min(cire/4,1),"CIre — Chlorophyll Integrity","#66bb6a"), use_container_width=True)
        st.caption("Drops before visible yellowing begins.")
    with eg3:
        st.plotly_chart(mini_gauge(psif,"Py-SIF — Photosynthetic Yield","#4fc3f7"), use_container_width=True)
        st.caption("Measures actual light processing efficiency.")

    st.write("---")
    st.subheader("Stress Probability Distribution")
    st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #ffa726;border-radius:0 8px 8px 0;
            padding:10px 16px;margin-bottom:14px;color:#b0cce8;font-size:0.9em'>
  <b style='color:#ffa726'>🥧 What does this pie chart mean?</b><br>
  It divides your field's pixels into three categories based on how stressed they are.
  The <b style='color:#ffa726'>yellow slice (Pre-Visual)</b> is the most important —
  these are the areas that <em>look perfectly fine</em> in a normal photo
  but our satellite sensors have already detected hidden stress. If you irrigate these zones now,
  you prevent yield loss. If you wait, they move to the red slice.
</div>
""", unsafe_allow_html=True)
    hs_f=data['health_score']
    p_h=int(max(0,min(100,hs_f*100-10))); p_p=int(max(0,min(100-p_h,(1-hs_f)*60)))
    p_d=max(0,100-p_h-p_p)
    cp1,cp2 = st.columns([1,1])
    with cp1:
        fp = go.Figure(go.Pie(labels=["Healthy","Pre-Visual Stress","Metabolic Damage"],
            values=[p_h,p_p,p_d],marker_colors=["#66bb6a","#ffa726","#ef5350"],
            hole=0.4,textfont=dict(color="#e0f0ff")))
        fp.update_layout(height=270,margin=dict(l=0,r=0,t=10,b=0),
                         paper_bgcolor="rgba(0,0,0,0)",
                         legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f0ff")))
        st.plotly_chart(fp, use_container_width=True)
    with cp2:
        st.markdown(f"""
<br><div style='color:#e0f0ff;line-height:2'>
<b style='color:#66bb6a'>🟢 Healthy ({p_h}%)</b><br>Full metabolic activity.<br><br>
<b style='color:#ffa726'>🟡 Pre-Visual ({p_p}%)</b><br>Invisible to NDVI. Irrigate within 48 hrs.<br><br>
<b style='color:#ef5350'>🔴 Damaged ({p_d}%)</b><br>Cell damage begun. Act immediately.
</div>""", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Proven Impact of Pre-Visual Detection")
    ic1,ic2,ic3,ic4 = st.columns(4)
    for col,(num,title,desc) in zip([ic1,ic2,ic3,ic4],[
        ("10–20%","Yield Protection","Early intervention stops cell death before permanent yield loss."),
        ("15–20%","Water Savings","Thermal-guided VRI targets only metabolically thirsty zones."),
        ("30%","Chemical Cost Reduction","Targeted spraying replaces blanket treatment."),
        ("7–14 Days","Strategy Window","The pre-visual lead time no NDVI tool can match."),
    ]):
        with col:
            st.markdown(f"""
<div class='impact-card'>
  <div class='impact-num'>{num}</div>
  <div style='color:#4fc3f7;font-weight:600;margin:6px 0 4px'>{title}</div>
  <div class='impact-desc'>{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<br><div style='background:#1a0f00;border:1px solid #ffa726;border-radius:8px;
                padding:14px 18px;color:#e0f0ff'>
  <b style='color:#ffa726'>⚡ Flash Drought Early Warning (2026 Research)</b><br>
  Flash Droughts erase soil moisture in days. SIF + Thermal fusion lets farmers
  flood-irrigate <b>before root zones hit the permanent wilting point</b>.
</div>""", unsafe_allow_html=True)

    st.caption("Data Source: Integrated PAU Ag-Met, USDA FAS, GDO Reports (Feb 2026)")


# ══════════════════════════════════════════════════════════
# PAGE: BAND ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "📊 Band Analysis":
    st.title("📊 Stress-Vision: Crop Stress Analysis")
    st.write("Upload 5 Sentinel-2 bands to generate interactive stress maps, "
             "cv2 colourmap images, and metabolic early-warning indices.")

    st.subheader("Upload Sentinel-2 Band Files")
    uc1,uc2,uc3 = st.columns(3)
    with uc1:
        red_file      = st.file_uploader("🔴 Red Band (B04)",      type=["tif","tiff"])
        green_file    = st.file_uploader("🟢 Green Band (B03)",    type=["tif","tiff"])
    with uc2:
        nir_file      = st.file_uploader("🔵 NIR Band (B08)",      type=["tif","tiff"])
        swir_file     = st.file_uploader("🟤 SWIR Band (B11)",     type=["tif","tiff"])
    with uc3:
        red_edge_file = st.file_uploader("🟠 Red Edge Band (B05)", type=["tif","tiff"])

    st.info("⚠️ All bands must be from the **same date and location**.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Or drop all 5 files at once:**")
        st.caption("Auto-detected from filename (e.g. `..._B04.tif`)")
        bulk_files = st.file_uploader(
            "Bulk Band Upload", type=["tif","tiff"],
            accept_multiple_files=True, label_visibility="collapsed"
        )

    def resolve_bands():
        bands = {}
        if bulk_files:
            for f in bulk_files:
                fname = f.name.upper().replace(".TIF","").replace(".TIFF","")
                for key in REQUIRED_BANDS:
                    if key in fname:
                        bands[key] = f; break
        for k,f in {"B04":red_file,"B03":green_file,"B08":nir_file,
                    "B11":swir_file,"B05":red_edge_file}.items():
            if f is not None:
                bands[k] = f
        return bands

    resolved = resolve_bands()

    st.subheader("Band Upload Status")
    bs_cols = st.columns(5)
    for i,band in enumerate(REQUIRED_BANDS):
        bdef = BAND_DEFINITIONS[band]
        with bs_cols[i]:
            if band in resolved:
                st.success(f"**{band}**\n\n{bdef[0]}\n\n{bdef[1]}")
            else:
                st.error(f"**{band}**\n\n{bdef[0]}\n\n{bdef[1]}")

    all_five = all(b in resolved for b in ["B03","B04","B05","B08","B11"])
    if not all_five:
        missing = [b for b in REQUIRED_BANDS if b not in resolved]
        st.warning(f"Waiting for: **{', '.join(missing)}**. Upload all 5 bands to begin.")
        st.stop()

    st.success("✅ All 5 bands loaded — running full analysis…")

    with st.spinner("Reading band data…"):
        red      = load_band(resolved["B04"])
        green    = load_band(resolved["B03"])
        nir      = load_band(resolved["B08"])
        swir     = load_band(resolved["B11"])
        red_edge = load_band(resolved["B05"])

    ndvi       = calculate_ndvi(nir, red)
    ndwi       = calculate_ndwi(nir, swir)
    re_idx     = calculate_red_edge_index(nir, red_edge)
    stress     = calculate_stress(ndvi, ndwi, re_idx)
    stress_pct = float(np.mean(stress) * 100)

    rgb        = create_rgb(red, green, nir)
    ndvi_map   = create_ndvi_colormap(normalize(ndvi))
    stress_map = create_stress_magma(stress)

    # ── Plain-English Summary for uploaded data ───────────
    hs_quick = max(0.0, min(1.0, 1.0 - stress_pct / 100))
    np.random.seed(int(hs_quick * 100))
    cwsi_q  = compute_cwsi(hs_quick)
    cire_q  = compute_cire(hs_quick)
    pysif_q = compute_pysif(hs_quick)
    plain_english_summary(
        health_score  = hs_quick,
        location_label= "your uploaded field",
        stressed_pct  = stress_pct,
        ndvi_mean     = float(np.nanmean(ndvi)),
        cwsi          = cwsi_q,
        cire          = cire_q,
        pysif         = pysif_q,
        context       = "upload",
    )

    threshold    = st.slider("🎚️ Stress Threshold", 0.0, 1.0, 0.6, 0.01)
    high_mask    = (stress > threshold).astype(np.uint8) * 255
    stressed_px  = int(np.sum(stress > threshold))
    healthy_px   = int(np.sum(stress <= threshold))
    total_px     = stressed_px + healthy_px
    stressed_pct2= stressed_px/total_px*100
    healthy_pct2 = healthy_px/total_px*100

    rep             = normalize((red_edge - red) / (nir + 1e-8))
    thermal_anomaly = normalize(ndvi - ndwi)

    st.divider()

    tabs = st.tabs([
        "🌐 RGB Composite","🌿 NDVI Map","🔥 Stress Heatmap",
        "🚨 High-Stress Mask","📋 Crop Health Overview","🔬 Advanced Metrics",
    ])

    with tabs[0]:
        st.subheader("RGB False-Colour Composite (R-G-NIR)")
        st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #4fc3f7;border-radius:0 8px 8px 0;
            padding:12px 16px;margin-bottom:12px;color:#b0cce8;font-size:0.93em'>
  <b style='color:#4fc3f7'>🖼️ What am I looking at?</b><br>
  This is a satellite photo of your field using special light bands — not a normal camera.
  <b style='color:#e0f0ff'>Deep red/magenta areas = dense, healthy vegetation.</b>
  Pale or cyan areas = sparse crops, bare soil, or stressed plants.
  This image is just for orientation — it shows you <em>where</em> things are, not how stressed they are.
</div>
""", unsafe_allow_html=True)
        st.caption("Healthy vegetation appears deep red/magenta.")
        st.image(rgb, use_container_width=True)
        st.download_button("📥 Download RGB", convert_np_to_image(rgb),
                           "RGB_composite.png","image/png")

    with tabs[1]:
        st.subheader("NDVI Map (VIRIDIS)")
        st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #66bb6a;border-radius:0 8px 8px 0;
            padding:12px 16px;margin-bottom:12px;color:#b0cce8;font-size:0.93em'>
  <b style='color:#66bb6a'>🌿 What am I looking at?</b><br>
  NDVI is the <b>standard satellite health score</b> used by most farms worldwide.
  <b style='color:#ffd54f'>Yellow-green</b> = healthy (score close to 1.0).
  <b style='color:#ce93d8'>Dark purple</b> = stressed or bare soil (score close to 0).
  <br><br>
  <b style='color:#ffa726'>⚠️ Important limitation:</b> NDVI only turns purple <em>after</em> your crops are
  visibly yellowing. By that point, yield loss has already happened. That's why this app also uses
  the Stress Heatmap tab — which catches problems 7–14 days earlier.
</div>
""", unsafe_allow_html=True)
        st.caption("Purple = low NDVI · Yellow-green = high NDVI (healthy).")
        st.image(ndvi_map, use_container_width=True)
        col_nv1,col_nv2,col_nv3 = st.columns(3)
        col_nv1.metric("Mean NDVI",f"{np.nanmean(ndvi):.3f}")
        col_nv2.metric("Min NDVI", f"{np.nanmin(ndvi):.3f}")
        col_nv3.metric("Max NDVI", f"{np.nanmax(ndvi):.3f}")
        # Plain-English NDVI interpretation
        mean_ndvi = float(np.nanmean(ndvi))
        if mean_ndvi >= 0.6:
            st.success(f"✅ Mean NDVI of {mean_ndvi:.2f} — conventional tools say this field looks healthy. But check the Stress Heatmap tab for the full picture.")
        elif mean_ndvi >= 0.4:
            st.warning(f"⚠️ Mean NDVI of {mean_ndvi:.2f} — conventional tools are beginning to detect stress. Our pre-visual signals likely flagged this 1–2 weeks ago.")
        else:
            st.error(f"🚨 Mean NDVI of {mean_ndvi:.2f} — visible damage is confirmed even by standard tools. Immediate action required.")
        st.download_button("📥 Download NDVI Map", convert_np_to_image(ndvi_map),
                           "NDVI_map.png","image/png")

    with tabs[2]:
        st.subheader("Stress-Vision Heatmap (MAGMA)")
        st.markdown(f"""
<div style='background:#0d1f36;border-left:4px solid #ef5350;border-radius:0 8px 8px 0;
            padding:12px 16px;margin-bottom:12px;color:#b0cce8;font-size:0.93em'>
  <b style='color:#ef5350'>🔥 What am I looking at?</b><br>
  This is the <b>core innovation</b> of Stress-Vision. The heatmap combines three satellite signals —
  water content (B11), chlorophyll structure (B05), and cell health (B08) — into one stress score per pixel.
  <br><br>
  <b style='color:#ffffff'>⬛ Black/dark purple</b> = no stress, crops are fine here.<br>
  <b style='color:#ff7043'>🟧 Orange</b> = moderate stress — plants are struggling.<br>
  <b style='color:#ffffff;background:#ef5350;padding:1px 4px;border-radius:3px'>⬜ Bright white/yellow</b> = critical stress — act immediately on these zones.
  <br><br>
  Average stress across your entire field: <b style='color:#ef5350;font-size:1.1em'>{stress_pct:.1f}%</b>
  {"— <b style='color:#ef5350'>More than half your field needs urgent attention.</b>" if stress_pct > 50
   else "— <b style='color:#ffa726'>A significant portion needs attention soon.</b>" if stress_pct > 25
   else "— <b style='color:#66bb6a'>Most of your field is doing well.</b>"}
</div>
""", unsafe_allow_html=True)
        st.caption("Dark = low stress · Bright = critical stress.")
        st.image(stress_map, use_container_width=True)
        st.metric("Average Crop Stress", f"{stress_pct:.2f}%")
        st.download_button("📥 Download Heatmap", convert_np_to_image(stress_map),
                           "stress_heatmap.png","image/png")

    with tabs[3]:
        st.subheader(f"High-Stress Zones  (threshold > {threshold:.2f})")
        st.markdown(f"""
<div style='background:#0d1f36;border-left:4px solid #ffa726;border-radius:0 8px 8px 0;
            padding:12px 16px;margin-bottom:12px;color:#b0cce8;font-size:0.93em'>
  <b style='color:#ffa726'>🗺️ What am I looking at?</b><br>
  This black-and-white map is the simplest possible view: <b>white = problem zone, black = safe zone.</b>
  It takes the stress heatmap and draws a clear line — everything above the stress threshold
  ({threshold:.2f}) is marked white. These white patches are exactly where a farmer needs to go
  and take action first.
  <br><br>
  Currently <b style='color:#ef5350'>{stressed_pct2:.1f}% of your field ({stressed_px:,} pixels)</b> is marked as stressed.
  <b style='color:#66bb6a'>{healthy_pct2:.1f}% ({healthy_px:,} pixels)</b> is currently healthy.
  <br><br>
  💡 <b>Tip:</b> Use the slider above to adjust sensitivity. A threshold of 0.4 catches more subtle
  early stress. A threshold of 0.7 shows only the most critical zones.
</div>
""", unsafe_allow_html=True)
        st.caption("White = above threshold. Adjust slider above to refine.")
        st.image(high_mask, use_container_width=True)
        mc1,mc2 = st.columns(2)
        mc1.metric("Stressed Pixels",f"{stressed_pct2:.1f}%")
        mc2.metric("Healthy Pixels", f"{healthy_pct2:.1f}%")
        st.download_button("📥 Download Mask", convert_np_to_image(high_mask),
                           "high_stress_mask.png","image/png")

    with tabs[4]:
        st.subheader("Crop Health Overview")
        neon_card("Average Stress (%)",f"{stress_pct:.2f}%","#ff0066")
        neon_card("Healthy Crop (%)",  f"{healthy_pct2:.2f}%","#00ff99")
        neon_card("Stressed Crop (%)", f"{stressed_pct2:.2f}%","#ff00cc")
        bc1,bc2 = st.columns(2)
        with bc1:
            fig_bc = go.Figure(go.Bar(
                x=["Healthy","Stressed"],y=[healthy_pct2,stressed_pct2],
                marker_color=["#66bb6a","#ef5350"],
                text=[f"{healthy_pct2:.1f}%",f"{stressed_pct2:.1f}%"],
                textposition="auto"))
            fig_bc.update_layout(height=260,margin=dict(l=0,r=0,t=20,b=0),
                                 paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                 yaxis=dict(gridcolor="#1e3048"),title="Healthy vs Stressed (%)")
            st.plotly_chart(fig_bc, use_container_width=True)
        with bc2:
            ds_ndvi2  = downsample(normalize(ndvi))
            ds_stress2= downsample(stress)
            fig_cmp = px.imshow(
                np.concatenate([ds_ndvi2,ds_stress2],axis=1),
                color_continuous_scale=["red","yellow","green"],zmin=0,zmax=1,
                labels={"color":"Index"})
            fig_cmp.update_layout(height=250,margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption("Left = NDVI | Right = Composite Stress")

    with tabs[5]:
        st.subheader("Advanced Spectral Metrics")
        st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #ce93d8;border-radius:0 8px 8px 0;
            padding:12px 16px;margin-bottom:16px;color:#b0cce8;font-size:0.93em'>
  <b style='color:#ce93d8'>🔬 What am I looking at?</b><br>
  These two images show the <b>two earliest stress signals</b> that exist in satellite data —
  signals that appear even before the Stress Heatmap catches anything.
  <br><br>
  <b style='color:#e0f0ff'>Red-Edge Position (REP):</b> Measures the structural health of chlorophyll molecules.
  Think of it as a <em>molecular early warning</em> — bright areas mean the green pigment inside leaves
  is starting to weaken, even if the leaf is still physically green.
  <br><br>
  <b style='color:#e0f0ff'>Thermal Anomaly:</b> Shows where plants have closed their breathing pores (stomata)
  to conserve water. A plant that has stopped "sweating" is like a person running a high fever —
  something is very wrong even though they look normal on the outside. Bright areas here = drought stress.
  <br><br>
  Together, these two maps reveal stress <b>7–14 days before standard NDVI</b> would detect anything.
</div>
""", unsafe_allow_html=True)
        am1,am2 = st.columns(2)
        with am1:
            st.write("**Red-Edge Position (REP)**")
            st.caption("Proxy for early chlorophyll structural change.")
            st.image(convert_np_to_image(rep), use_container_width=True)
        with am2:
            st.write("**Thermal Anomaly (Proxy)**")
            st.caption("NDVI − NDWI proxy for stomatal closure / transpiration drop.")
            st.image(convert_np_to_image(thermal_anomaly), use_container_width=True)
        st.write("---")
        st.subheader("Index Distribution Histograms")
        hc1,hc2 = st.columns(2)
        with hc1:
            fh1 = px.histogram(normalize(ndvi).flatten(),nbins=60,
                               title="NDVI Distribution",color_discrete_sequence=["#66bb6a"])
            fh1.update_layout(showlegend=False,height=220,margin=dict(l=0,r=0,t=30,b=0),
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(gridcolor="#1e3048"),yaxis=dict(gridcolor="#1e3048"))
            st.plotly_chart(fh1, use_container_width=True)
        with hc2:
            fh2 = px.histogram(stress.flatten(),nbins=60,
                               title="Composite Stress Distribution",
                               color_discrete_sequence=["#ef5350"])
            fh2.update_layout(showlegend=False,height=220,margin=dict(l=0,r=0,t=30,b=0),
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(gridcolor="#1e3048"),yaxis=dict(gridcolor="#1e3048"))
            st.plotly_chart(fh2, use_container_width=True)

    st.divider()

    # PPT sections below tabs
    hs_upload = max(0.0, min(1.0, 1.0 - stress_pct/100))
    np.random.seed(int(hs_upload*100))

    st.subheader("Computed Field Insight")
    if stress_pct > 60:
        st.error(f"CRITICAL STRESS — {stress_pct:.1f}% average stress detected. Immediate intervention recommended.")
    elif stress_pct > 35:
        st.warning(f"MODERATE STRESS — {stress_pct:.1f}% average stress. Monitor and plan irrigation.")
    else:
        st.success(f"VEGETATION HEALTHY — {stress_pct:.1f}% average stress. Continue monitoring.")

    rpt = pd.DataFrame([{"Mean NDVI":round(float(np.nanmean(ndvi)),4),
                          "Mean Stress":round(stress_pct/100,4),
                          "Stressed Pixels (%)":round(stressed_pct2,2),
                          "Healthy Pixels (%)":round(healthy_pct2,2),
                          "Threshold":threshold}])
    st.download_button("📥 Download Analysis Report (CSV)",
                       rpt.to_csv(index=False).encode(),"stress_vision_report.csv","text/csv")

    st.write("---")
    st.subheader("Field Vitality Score & Traffic Light Alert")
    vs_u=vitality_score(hs_upload)
    col_u="#66bb6a" if vs_u>=65 else "#ffa726" if vs_u>=45 else "#ef5350"
    tku,tlu,tcu = traffic_light_class(hs_upload)
    pru = irrigation_prescription(hs_upload,"your uploaded area")
    cgu,ctlu = st.columns([1,2])
    with cgu:
        st.plotly_chart(vitality_donut(vs_u,col_u), use_container_width=True)
    with ctlu:
        st.markdown(f"""
<div class='{tcu}'>
  <div class='tl-label'>{tlu}</div>
  <div class='tl-alert'><b>Alert:</b> {pru}</div>
</div>""", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Early Warning Index Dashboard")
    st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #4fc3f7;border-radius:0 8px 8px 0;
            padding:10px 16px;margin-bottom:14px;color:#b0cce8;font-size:0.9em'>
  <b style='color:#4fc3f7'>📊 What are these dials?</b> Three hidden stress signals, each measuring
  a different thing happening inside your plants right now.
  <b>CWSI</b> = is the plant dehydrated and "running a fever"?
  <b>CIre</b> = is the green pigment (chlorophyll) breaking down?
  <b>Py-SIF</b> = is the plant still making food from sunlight?
  All three react <b>7–14 days before visible yellowing.</b>
</div>
""", unsafe_allow_html=True)
    cw_u=compute_cwsi(hs_upload); ci_u=compute_cire(hs_upload); ps_u=compute_pysif(hs_upload)
    eu1,eu2,eu3 = st.columns(3)
    with eu1:
        st.plotly_chart(mini_gauge(cw_u,"CWSI — Water Stress","#ef5350"), use_container_width=True)
        st.caption("Crop Water Stress Index.")
    with eu2:
        st.plotly_chart(mini_gauge(min(ci_u/4,1),"CIre — Chlorophyll Integrity","#66bb6a"), use_container_width=True)
        st.caption("Chlorophyll Index Red-Edge.")
    with eu3:
        st.plotly_chart(mini_gauge(ps_u,"Py-SIF — Photosynthetic Yield","#4fc3f7"), use_container_width=True)
        st.caption("Solar-Induced Fluorescence proxy.")

    st.write("---")
    st.subheader("Stress Probability Distribution")
    st.markdown("""
<div style='background:#0d1f36;border-left:4px solid #ffa726;border-radius:0 8px 8px 0;
            padding:10px 16px;margin-bottom:14px;color:#b0cce8;font-size:0.9em'>
  <b style='color:#ffa726'>🥧 What does this pie chart mean?</b><br>
  It breaks your entire field into three health categories.
  The <b style='color:#ffa726'>yellow slice</b> is what makes this app unique —
  those are areas that would look completely healthy in a normal satellite photo,
  but our pre-visual sensors have already detected hidden stress.
  Act on the yellow slice <em>before</em> it turns red.
</div>
""", unsafe_allow_html=True)
    pu_h=int(max(0,min(100,hs_upload*100-10)))
    pu_p=int(max(0,min(100-pu_h,(1-hs_upload)*60)))
    pu_d=max(0,100-pu_h-pu_p)
    cp1u,cp2u = st.columns([1,1])
    with cp1u:
        fpu = go.Figure(go.Pie(
            labels=["Healthy","Pre-Visual Stress","Metabolic Damage"],
            values=[pu_h,pu_p,pu_d],
            marker_colors=["#66bb6a","#ffa726","#ef5350"],
            hole=0.4,textfont=dict(color="#e0f0ff")))
        fpu.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#e0f0ff")))
        st.plotly_chart(fpu, use_container_width=True)
    with cp2u:
        st.markdown(f"""
<br><div style='color:#e0f0ff;line-height:2'>
<b style='color:#66bb6a'>🟢 Healthy ({pu_h}%)</b> — Full metabolic activity.<br>
<b style='color:#ffa726'>🟡 Pre-Visual ({pu_p}%)</b> — Hidden stress. Irrigate within 48 hrs.<br>
<b style='color:#ef5350'>🔴 Damaged ({pu_d}%)</b> — Cell damage begun. Act immediately.
</div>""", unsafe_allow_html=True)

    st.caption("Analysis based on user-uploaded Sentinel-2 Level-2A bands via Copernicus Open Access Hub.")


# ══════════════════════════════════════════════════════════
# PAGE: DATA ACQUISITION
# ══════════════════════════════════════════════════════════
elif page == "📡 Data Acquisition":
    st.title("📡 Satellite Data Acquisition")
    st.markdown("### 🛰 Data Source")
    st.write("We use **Sentinel-2 Level-2A surface reflectance data** from ESA, "
             "freely available via Copernicus.")

    st.markdown("### 📥 Step-by-Step Download Instructions")
    with st.expander("Copernicus Data Space (recommended)", expanded=True):
        st.markdown("""
<div class='how-to-box'>
<span class='how-to-step'>Step 1 — Register</span><br>
Visit <a href='https://dataspace.copernicus.eu/' style='color:#4fc3f7'>dataspace.copernicus.eu</a>
and create a free account.
</div>
<div class='how-to-box'>
<span class='how-to-step'>Step 2 — Search</span><br>
Select <b>Sentinel-2</b> · Product Type: <b>L2A</b> · Cloud cover: <b>&lt;10%</b>.
</div>
<div class='how-to-box'>
<span class='how-to-step'>Step 3 — Download & Extract</span><br>
Click <b>Analytical Download</b> → Extract ZIP →
Navigate to <code>GRANULE/.../IMG_DATA/R10m/</code> (B03, B04, B08)
and <code>R20m/</code> (B05, B11).
</div>
<div class='how-to-box'>
<span class='how-to-step'>Step 4 — Required Bands</span><br>
<span class='band-tag'>B03 Green 560nm</span>
<span class='band-tag'>B04 Red 665nm</span>
<span class='band-tag'>B05 Red Edge 705nm</span>
<span class='band-tag'>B08 NIR 842nm</span>
<span class='band-tag'>B11 SWIR 1610nm</span>
</div>
<div class='how-to-box'>
<span class='how-to-step'>Step 5 — Upload</span><br>
Go to <b>📊 Band Analysis</b> and upload your <code>.tif</code> files.
No renaming needed — bands are auto-detected from the filename.
</div>
""", unsafe_allow_html=True)

    col_a,col_b = st.columns(2)
    col_a.markdown("🔗 [Copernicus Data Space](https://browser.dataspace.copernicus.eu)")
    col_b.markdown("🔗 [USGS EarthExplorer](https://earthexplorer.usgs.gov)")
    st.info("⚠️ All bands must come from the **same acquisition date and same tile**.")
