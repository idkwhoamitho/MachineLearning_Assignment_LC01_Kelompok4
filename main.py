import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Diabetes Prediction Program", page_icon="🩺", layout="wide")

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    /* Variables for adaptive modes */
    :root {
        --bg-color: #f8fafc;
        --panel-bg: #ffffff;
        --text-color: #000000;
        --border-color: #0f766e;
        --primary: #10b981;
        --secondary: #34d399;
        --accent: #84cc16;
        --danger: #ef4444;
        --shadow-color: #cbd5e1;
        --hover-bg: #fef08a; /* Soft yellow hover */
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-color: #022c22;
            --panel-bg: #064e3b;
            --text-color: #f8fafc;
            --border-color: #34d399;
            --primary: #10b981;
            --secondary: #4ade80;
            --accent: #bbf7d0;
            --danger: #f87171;
            --shadow-color: #020617;
            --hover-bg: #86efac; /* Light green hover */
        }
    }

    /* Global Typography and Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: linear-gradient(rgba(34, 197, 94, 0.05) 50%, transparent 50%);
        background-size: 100% 4px; /* Scanline effect */
    }

    /* ========================
       FONT FIX (SAFE TARGETING)
       ======================== */
    /* ========================
       FONT FIX (CLEAN TARGETING)
       ======================== */
    /* Target common user text elements directly to prevent breaking internal Streamlit icons */
    h1, h2, h3, h4, h5, h6, p, label, li, input, textarea, th, td, 
    .stMarkdown, .stText, .stMetric, div[data-baseweb="select"],
    .hero-title, .hero-subtitle, .card-title, .metric-value, .metric-label, .result-title, .result-prob {
        font-family: 'VT323', monospace !important;
        letter-spacing: 1px !important;
    }
    
    div[data-testid="stMetricValue"], 
    div[data-testid="stMetricLabel"], 
    div.stDataFrame div, 
    div.stTable div,
    div.stButton > button {
        font-family: 'VT323', monospace !important;
    }

    /* Centered Container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Pixel Panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--panel-bg);
        border: 4px solid var(--border-color) !important;
        border-radius: 0px !important;
        box-shadow: 6px 6px 0px var(--shadow-color) !important;
        padding: 24px !important;
        margin-bottom: 24px;
        transition: transform 0.2s;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0px var(--shadow-color) !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: var(--panel-bg) !important;
        border: 4px solid var(--border-color) !important;
        border-radius: 0px !important;
        box-shadow: 4px 4px 0px var(--shadow-color) !important;
        color: var(--text-color) !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        text-transform: uppercase;
        font-size: 1.4rem !important;
        transition: all 0.1s !important;
    }
    div.stButton > button:hover {
        background-color: var(--hover-bg) !important;
        color: #000000 !important;
        border-color: var(--border-color) !important;
    }
    div.stButton > button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px var(--shadow-color) !important;
    }
    
    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: #000000 !important;
        border-color: var(--border-color) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--hover-bg) !important;
        color: #000000 !important;
    }

    /* Sidebar Menu */
    [data-testid="stSidebar"] {
    background-color: var(--panel-bg);
    border-right: 4px solid var(--border-color);
    height: 100vh !important;
    overflow: hidden !important;
    display: flex;
    flex-direction: column;
    scroll-bar: 
    }

    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        flex: 1 1 auto;
        padding-bottom: 10px;
    }

    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--text-color) !important;
        padding: 8px 12px !important;
        margin-bottom: 4px !important;
        font-size: 0.8rem !important;
        white-space: nowrap;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--hover-bg) !important;
        color: #000000 !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--primary) !important;
        color: #000000 !important;
        border: 4px solid var(--border-color) !important;
        box-shadow: 4px 4px 0px var(--shadow-color) !important;
    }

    /* Titles */
    .hero-title {
        font-size: 3.5rem;
        font-weight: bold;
        color: var(--text-color);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.6rem;
        color: var(--secondary);
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-size: 2rem;
        font-weight: bold;
        color: var(--text-color);
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-bottom: 4px dashed var(--border-color);
        padding-bottom: 0.5rem;
    }

    /* Cursor blink */
    .blink {
        animation: blink-animation 1s steps(2, start) infinite;
    }
    @keyframes blink-animation {
        to { visibility: hidden; }
    }

    /* Metrics */
    .metrics-wrapper {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    .metric-box {
        background: var(--panel-bg);
        border: 4px solid var(--border-color);
        box-shadow: 4px 4px 0px var(--shadow-color);
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 1.3rem;
        color: var(--text-color);
        text-transform: uppercase;
    }
    .color-acc { color: var(--secondary); }
    .color-prec { color: var(--primary); }
    .color-rec { color: var(--accent); }
    .color-f1 { color: var(--danger); }

    /* Prediction Cards */
    .result-card {
        padding: 30px;
        border: 4px solid var(--border-color);
        box-shadow: 8px 8px 0px var(--shadow-color);
        text-align: center;
        margin-top: 24px;
        animation: fadeIn 0.4s ease-out;
    }
    .result-high {
        background-color: var(--danger);
        color: #ffffff;
    }
    .result-low {
        background-color: var(--primary);
        color: #ffffff;
    }
    .result-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 12px;
        text-transform: uppercase;
        color: inherit;
    }
    .result-prob {
        font-size: 1.6rem;
        font-weight: bold;
        color: inherit;
    }

    /* Image wrapper */
    .pixel-img-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    .pixel-img {
        max-height: 220px !important;
        border: 4px solid var(--border-color);
        box-shadow: 6px 6px 0px var(--shadow-color);
        object-fit: contain;
    }

    /* ========================
       INPUT & SELECTBOX FIX
       ======================== */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: var(--panel-bg) !important;
        border: 3px solid var(--border-color) !important;
        border-radius: 0px !important;
        box-shadow: 3px 3px 0px var(--shadow-color) !important;
    }

    /* Ensure inputs, selectbox text, and labels have strong contrast */
    div[data-baseweb="input"] input, 
    div[data-baseweb="base-input"] input,
    div[data-baseweb="select"] * {
        color: var(--text-color) !important;
    }
    
    label, 
    div[role="radiogroup"] *, 
    div[data-baseweb="radio"] *, 
    div[data-baseweb="checkbox"] * {
        color: var(--text-color) !important;
    }

    /* ========================
       PLACEHOLDER FIX
       ======================== */
    ::placeholder {
        color: #1e293b !important; /* Dark slate for high contrast in light mode */
        opacity: 1 !important;
    }
    @media (prefers-color-scheme: dark) {
        ::placeholder {
            color: #94a3b8 !important;
        }
    }

    /* ========================
       DROPDOWN MENU FIX
       ======================== */
    ul[data-baseweb="menu"] {
        background-color: var(--panel-bg) !important;
        border: 4px solid var(--border-color) !important;
        border-radius: 0px !important;
    }
    ul[data-baseweb="menu"] li {
        color: var(--text-color) !important;
        font-family: 'VT323', monospace !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: var(--hover-bg) !important;
        color: #000000 !important;
    }

    /* Cleaned up complex icon overrides; VT323 is now strictly applied only to text components. */
    
    /* Heartbeat Grid */
    .heartbeat-box {
        background-color: var(--panel-bg);
        border: 2px solid var(--border-color);
        box-shadow: 4px 4px 0px var(--shadow-color);
        padding: 15px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        min-height: 100px;
        transition: transform 0.2s ease-in-out;
    }
    .heartbeat-box:hover {
        transform: translateY(-5px);
        box-shadow: 6px 6px 0px var(--shadow-color);
    }
    .heartbeat-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='100' viewBox='0 0 200 100'%3E%3Cpath d='M0 60 L30 60 L40 20 L55 90 L65 60 L200 60' fill='none' stroke='%2310b981' stroke-width='4' stroke-linecap='round' stroke-linejoin='round' opacity='0.15'/%3E%3C/svg%3E");
        background-repeat: repeat-x;
        background-position: center;
        background-size: 200px 100px;
        z-index: 0;
        pointer-events: none;
    }
    .heartbeat-content {
        position: relative;
        z-index: 1;
        font-size: 1.1rem;
        line-height: 1.4;
        color: var(--text-color);
        font-family: 'VT323', monospace !important;
        letter-spacing: 1px;
    }
    .heartbeat-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: var(--text-color);
        margin-bottom: 5px;
        border-bottom: 2px dashed var(--border-color);
        display: inline-block;
        padding-bottom: 2px;
        font-family: 'VT323', monospace !important;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def init_state():
    defaults = {
        'max_step': 0, 'current_step': 0, 'show_about': False, 'df': None, 'selected_features': [],
        'model': None, 'imputer': None, 'scaler': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<div style='padding: 5px 0 5px 0;'><h1 style='margin:0; color:var(--text-color); font-family: \"VT323\", monospace;'>SYSTEM MENU</h1></div>", unsafe_allow_html=True)

progress_percent = int((st.session_state.current_step / 6.0) * 100)
st.sidebar.progress(st.session_state.current_step / 6.0, text=f"WORKFLOW PROGRESS: {progress_percent}%")

step_names = [
    "🏠 HOME",
    "🔬 EDA",
    "🧪 FEATURES",
    "⚙️ PREPROCESS",
    "🤖 MODELLING",
    "📊 EVALUATION",
    "🧬 PREDICTION"
]

for i, name in enumerate(step_names):
    if i <= st.session_state.max_step:
        label = f"{name}"
    else:
        label = f"🔒 {name}"
        
    disabled = i > st.session_state.max_step
    btn_type = "primary" if (i == st.session_state.current_step and not st.session_state.show_about) else "secondary"
    
    if st.sidebar.button(label, disabled=disabled, use_container_width=True, key=f"nav_{i}", type=btn_type):
        st.session_state.show_about = False
        st.session_state.current_step = i
        st.rerun()

if st.sidebar.button("🔄️ REBOOT SYSTEM", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("<hr style='margin: 10px 0; border: 2px dashed var(--border-color);'>", unsafe_allow_html=True)
about_btn_type = "primary" if st.session_state.show_about else "secondary"
if st.sidebar.button("ℹ️ ABOUT", use_container_width=True, type=about_btn_type):
    st.session_state.show_about = True
    st.rerun()

# --- STEPS RENDERING ---

def render_step_0():
    st.markdown('<div class="hero-title">DIABETES PREDICTION PROGRAM<span class="blink">_</span></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">SYSTEM OVERVIEW</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="heartbeat-box">
                <div class="heartbeat-content">
                    <div class="heartbeat-title">[PROGRAM FUNCTION]</div>
                    <br>This terminal is a machine learning-powered application designed to assist healthcare professionals in predicting the onset of diabetes based on medical diagnostic measurements.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="heartbeat-box">
                <div class="heartbeat-content">
                    <div class="heartbeat-title">[EXPECTED OUTPUT]</div>
                    <br>The system will output a definitive diagnostic prediction (Positive or Negative for Diabetes) along with the statistical probability of the disease risk.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="heartbeat-box">
                <div class="heartbeat-content">
                    <div class="heartbeat-title">[OBJECTIVES]</div>
                    <br>The main goal is to provide a rapid, accurate, and early-warning screening tool that analyzes clinical data to identify patients at high risk for diabetes.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="heartbeat-box">
                <div class="heartbeat-content">
                    <div class="heartbeat-title">[DATASET DESCRIPTION]</div>
                    <br>The models are trained using the Pima Indians Diabetes Database. It contains essential metabolic predictor variables collected specifically to study diabetes patterns.
                    <br><br><b>[DATA SOURCE LINK]:</b> <a href="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database" target="_blank" style="color: inherit; text-decoration: underline; font-weight: bold;">Kaggle: Pima Indians Database</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 1.4rem; font-family: VT323, monospace; color: var(--text-color); text-align: center; margin-top: 15px; letter-spacing: 1px;'> > PRESS INITIALIZE TO BEGIN SYSTEM DIAGNOSTICS... </div><br>", unsafe_allow_html=True)
        
        st.write("")
        if st.button("INITIALIZE DIAGNOSTICS", type="primary", use_container_width=True):
            st.session_state.max_step = max(st.session_state.max_step, 1)
            st.session_state.current_step = 1
            st.rerun()

    if st.session_state.df is None:
        try:
            st.session_state.df = pd.read_csv("diabetes.csv")
        except FileNotFoundError:
            try:
                st.session_state.df = pd.read_csv("cleaned_data.csv")
            except Exception as e:
                st.error(f"Dataset not found: {e}")
                return

def render_step_1():
    st.markdown('<div class="hero-title">EXPLORATORY DATA ANALYSIS<span class="blink">_</span></div>', unsafe_allow_html=True)
    df = st.session_state.df
    
    with st.container(border=True):
        st.markdown('<div class="card-title">DATA TERMINAL</div>', unsafe_allow_html=True)
        st.markdown("<b style='font-size: 1.2rem; color: var(--text-color);'>> SHOWING TOP 5 RECORDS:</b>", unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)
        
        if st.button("VIEW FULL DATA"):
            st.dataframe(df, use_container_width=True)
            
    with st.container(border=True):
        st.markdown('<div class="card-title">DATASET OVERVIEW</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<b style='font-size: 1.2rem; color: var(--text-color);'>> DATA TYPES & MISSING VALUES:</b>", unsafe_allow_html=True)
            info_df = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)
            
        with col2:
            st.markdown("<b style='font-size: 1.2rem; color: var(--text-color);'>> DATASET SHAPE:</b>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 1.6rem; font-family: \"VT323\", monospace; color: var(--text-color); margin-bottom: 20px;'>{df.shape[0]} ROWS × {df.shape[1]} COLUMNS</div>", unsafe_allow_html=True)
            
        st.markdown("<b style='font-size: 1.2rem; color: var(--text-color);'>> STATISTICAL SUMMARY:</b>", unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
            
    with st.container(border=True):
        st.markdown('<div class="card-title">VISUAL DIAGNOSTICS</div>', unsafe_allow_html=True)
        plot_type = st.radio("SELECT PLOT:", ["Univariate", "Bivariate", "Multivariate"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if plot_type == "Univariate":
            col = st.selectbox("SELECT FEATURE:", df.columns)
            fig, ax = plt.subplots(figsize=(4.5, 2.5))
            sns.histplot(df[col], kde=True, ax=ax, color='#16a34a', edgecolor='gray', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=False)
            
        elif plot_type == "Bivariate":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-AXIS FEATURE:", df.columns)
            with col2:
                y_col = st.selectbox("Y-AXIS FEATURE:", [c for c in df.columns if c != x_col])
            fig, ax = plt.subplots(figsize=(4.5, 2.5))
            sns.scatterplot(data=df, x=x_col, y=y_col, hue="Outcome", palette=["#38bdf8", "#ef4444"], s=40, edgecolor='gray', linewidth=1, ax=ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=False)
            
        elif plot_type == "Multivariate":
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            sns.heatmap(df.corr(), annot=True, cmap="Greens", fmt=".2f", ax=ax, linewidths=1, linecolor='gray')
            st.pyplot(fig, use_container_width=False)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("PROCEED TO FEATURE SELECTION", type="primary", use_container_width=True):
        st.session_state.max_step = max(st.session_state.max_step, 2)
        st.session_state.current_step = 2
        st.rerun()

def render_step_2():
    st.markdown('<div class="hero-title">FEATURES SELECTION<span class="blink">_</span></div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">ISOLATE VARIABLES</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.4rem; color: var(--text-color);'><b>> SELECT FEATURES FOR DIAGNOSTIC MODEL:</b></p>", unsafe_allow_html=True)
        
        df = st.session_state.df
        features = [col for col in df.columns if col != 'Outcome']
        selected = st.multiselect("VARIABLES", features, default=features, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("CONFIRM VARIABLES", type="primary", use_container_width=True):
        if len(selected) == 0:
            st.error("ERROR: NO VARIABLES SELECTED.")
        else:
            st.session_state.selected_features = selected
            st.session_state.max_step = max(st.session_state.max_step, 3)
            st.session_state.current_step = 3
            st.rerun()

def render_step_3():
    st.markdown('<div class="hero-title">PREPROCESSING<span class="blink">_</span></div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    selected_features = st.session_state.selected_features
    
    with st.container(border=True):
        st.markdown('<div class="card-title">PIPELINE CONFIG</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            test_size = st.slider("TEST SET SIZE (%)", 10, 50, 20) / 100.0
        with col2:
            scaler_choice = st.selectbox("SCALING ALGORITHM", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
            
    X = df[selected_features].copy()
    y = df['Outcome'].copy()
    
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
        
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">PROCESSED DATA OVERVIEW</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(f"<div style='font-size: 1.6rem; font-family: \"VT323\", monospace; color: var(--text-color);'><b>> TRAINING SAMPLES:</b> {X_train_scaled.shape[0]}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='font-size: 1.6rem; font-family: \"VT323\", monospace; color: var(--text-color);'><b>> TESTING SAMPLES:</b> {X_test_scaled.shape[0]}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<b style='font-size: 1.2rem; color: var(--text-color);'>> PREPROCESSED SAMPLE (TOP 5):</b>", unsafe_allow_html=True)
        st.dataframe(X_train_scaled.head(5), use_container_width=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("INITIALIZE MODELLING", type="primary", use_container_width=True):
        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.imputer = imputer
        st.session_state.scaler = scaler
        st.session_state.max_step = max(st.session_state.max_step, 4)
        st.session_state.current_step = 4
        st.rerun()

def render_step_4():
    st.markdown('<div class="hero-title">MODEL SELECTION<span class="blink">_</span></div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">ALGORITHM PARAMETERS</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5], gap="large")
        with col1:
            algo = st.selectbox("SELECT ALGORITHM", ["Random Forest", "SVM", "Logistic Regression", "KNN"])
            
        with col2:
            if algo == "Random Forest":
                n_estimators = st.number_input("TREES (n_estimators)", 10, 500, 100)
                max_depth = st.number_input("MAX DEPTH", 1, 50, 10)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif algo == "SVM":
                C = st.number_input("REGULARIZATION (C)", 0.1, 100.0, 1.0)
                kernel = st.selectbox("KERNEL", ["linear", "poly", "rbf", "sigmoid"], index=2)
                model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
            elif algo == "Logistic Regression":
                C_lr = st.number_input("INVERSE REG. (C)", 0.01, 100.0, 1.0)
                model = LogisticRegression(C=C_lr, random_state=42)
            elif algo == "KNN":
                n_neighbors = st.number_input("NEIGHBORS", 1, 50, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("COMPILE & TRAIN MODEL", type="primary", use_container_width=True):
        with st.spinner("TRAINING IN PROGRESS..."):
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model = model
            st.session_state.max_step = max(st.session_state.max_step, 5)
            st.session_state.current_step = 5
        st.rerun()

def render_step_5():
    st.markdown('<div class="hero-title">EVALUATION MODEL<span class="blink">_</span></div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("SYSTEM ERROR: NO MODEL TRAINED.")
        return
        
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics_html = f"""
    <div class="metrics-wrapper">
        <div class="metric-box">
            <div class="metric-value color-acc">{acc*100:.1f}%</div>
            <div class="metric-label">ACCURACY</div>
        </div>
        <div class="metric-box">
            <div class="metric-value color-prec">{prec*100:.1f}%</div>
            <div class="metric-label">PRECISION</div>
        </div>
        <div class="metric-box">
            <div class="metric-value color-rec">{rec*100:.1f}%</div>
            <div class="metric-label">RECALL</div>
        </div>
        <div class="metric-box">
            <div class="metric-value color-f1">{f1*100:.1f}%</div>
            <div class="metric-label">F1-SCORE</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">CONFUSION MATRIX</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        annot_labels = np.array([
            [f"TRUE NEG (TN)\n{cm[0,0]}", f"FALSE POS (FP)\n{cm[0,1]}"],
            [f"FALSE NEG (FN)\n{cm[1,0]}", f"TRUE POS (TP)\n{cm[1,1]}"]
        ])
        
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Greens', ax=ax, cbar=False, 
                    annot_kws={"size": 10, "weight": "bold", "family": "monospace"}, linewidths=1, linecolor='gray')
        ax.set_xlabel('PREDICTED LABEL', fontweight='bold', labelpad=12)
        ax.set_ylabel('TRUE LABEL', fontweight='bold', labelpad=12)
        st.pyplot(fig, use_container_width=False)
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("LAUNCH DEPLOYMENT SYSTEM", type="primary", use_container_width=True):
        st.session_state.max_step = max(st.session_state.max_step, 6)
        st.session_state.current_step = 6
        st.rerun()

def render_step_6():
    st.markdown('<div class="hero-title">PREDICTION<span class="blink">_</span></div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">INPUT PATIENT DATA</div>', unsafe_allow_html=True)
        selected_features = st.session_state.selected_features
        user_input = {}
        
        cols = st.columns(3, gap="medium")
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                if feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age']:
                    user_input[feature] = st.number_input(f"{feature}", value=0, step=1)
                else:
                    user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)
                    
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("EXECUTE SCAN", type="primary", use_container_width=True)
        
    if submit_btn:
        input_df = pd.DataFrame([user_input])
        
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(0, np.nan)
                
        imputed_input = pd.DataFrame(st.session_state.imputer.transform(input_df), columns=input_df.columns)
        scaled_input = pd.DataFrame(st.session_state.scaler.transform(imputed_input), columns=input_df.columns)
        
        model = st.session_state.model
        prob = model.predict_proba(scaled_input)[0][1]
        pred = model.predict(scaled_input)[0]
        
        if pred == 1:
            res_html = f"""
            <div class="result-card result-high">
                <div class="result-title">⚠️ HIGH RISK DETECTED ⚠️</div>
                <div class="result-prob">DIAGNOSIS: DIABETIC</div>
                <div style="font-size:1.4rem; font-family: 'VT323', monospace; margin-top:10px; color:#0f172a;">PROBABILITY: {prob*100:.1f}%</div>
            </div>
            """
        else:
            res_html = f"""
            <div class="result-card result-low">
                <div class="result-title">✅ PATIENT STABLE ✅</div>
                <div class="result-prob">DIAGNOSIS: NON-DIABETIC</div>
                <div style="font-size:1.4rem; font-family: 'VT323', monospace; margin-top:10px; color:#0f172a;">PROBABILITY: {(1-prob)*100:.1f}%</div>
            </div>
            """
        st.markdown(res_html, unsafe_allow_html=True)

def render_about_page():
    st.markdown('<div class="hero-title">ABOUT PROJECT<span class="blink">_</span></div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">PROJECT DESCRIPTION</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="heartbeat-box">
            <div class="heartbeat-content">
                This application was developed as part of a Machine Learning course assignment at Bina Nusantara University. The project focuses on implementing a complete machine learning pipeline, starting from Exploratory Data Analysis (EDA) to model deployment. Through this application, we aim to demonstrate a practical use of predictive analytics in the healthcare domain, specifically for diabetes risk prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">MAIN OBJECTIVES</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="heartbeat-box">
            <div class="heartbeat-content">
                <ul>
                    <li>Understand and implement the end-to-end machine learning pipeline from Exploratory Data Analysis (EDA) to deployment.</li>
                    <li>Evaluate and compare multiple classification algorithms for medical diagnostic accuracy.</li>
                    <li>Develop an interactive dashboard to visualize data patterns and provide real-time risk assessments.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="card-title">DATASET INFO</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="heartbeat-box">
                <div class="heartbeat-content">
                    <b>Name:</b> Pima Indians Diabetes Dataset<br>
                    <b>Source:</b> UCI Machine Learning Repository<br>
                    <b>Scope:</b> Medical diagnostic measurements of Pima Indian heritage.<br>
                    <b>Link: <a href="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database" target="_blank" style="color: inherit; text-decoration: underline; font-weight: bold;">Kaggle: Pima Indians Database</a></b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="card-title">TEAM MEMBERS</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="heartbeat-box">
                <div class="heartbeat-content">
                    <b>Kelompok 4 - LC01</b><br>
                    • 2802397306 - Edwin Antonie<br>
                    • 2802529203 - Hasan<br>
                    • 2802401846 - Wesley Sumedha Deano<br>
                    • 2802391006 - Maximilianus Ronald
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">TECHNOLOGIES USED</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="heartbeat-box">
            <div class="heartbeat-content">
                <span style="color:var(--primary); font-weight:bold;">Streamlit</span> (User Interface), 
                <span style="color:var(--secondary); font-weight:bold;">Scikit-learn</span> (Machine Learning), 
                <span style="color:var(--accent); font-weight:bold;">Pandas</span> (Data Processing), 
                <span style="color:var(--primary); font-weight:bold;">NumPy</span> (Numerical Computation), 
                <span style="color:var(--secondary); font-weight:bold;">Matplotlib & Seaborn</span> (Data Visualization)
            </div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.show_about:
    render_about_page()
elif st.session_state.current_step == 0:
    render_step_0()
elif st.session_state.current_step == 1:
    render_step_1()
elif st.session_state.current_step == 2:
    render_step_2()
elif st.session_state.current_step == 3:
    render_step_3()
elif st.session_state.current_step == 4:
    render_step_4()
elif st.session_state.current_step == 5:
    render_step_5()
elif st.session_state.current_step == 6:
    render_step_6()
