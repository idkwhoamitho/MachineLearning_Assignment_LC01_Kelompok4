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

st.set_page_config(page_title="Diabetes Risk Prediction System", page_icon="🧬", layout="wide")

custom_css = """
<style>
    /* Global Background and Font */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Centered Container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Card styling leveraging Streamlit's border containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
        border: 1px solid #f1f5f9 !important;
        padding: 24px !important;
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    /* Images */
    img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
        border-radius: 12px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        padding: 10px 24px !important;
    }
    div.stButton > button:hover {
        transform: scale(1.03) !important;
    }
    
    /* Primary Button Override */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
        color: white !important;
    }

    /* Sidebar Stepper styles */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #64748b !important;
        box-shadow: none !important;
        padding: 14px 16px !important;
        margin-bottom: 6px !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f8fafc !important;
        color: #0f172a !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #eff6ff !important;
        color: #2563eb !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 4px 8px 8px 4px !important;
        font-weight: 700 !important;
    }

    /* Custom HTML Classes */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.8rem;
    }

    /* Metrics Grid */
    .metrics-wrapper {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    .metric-box {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 6px;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
    }
    .color-acc { color: #3b82f6; }
    .color-prec { color: #10b981; }
    .color-rec { color: #f59e0b; }
    .color-f1 { color: #8b5cf6; }

    /* Prediction Result Cards */
    .result-card {
        padding: 32px;
        border-radius: 16px;
        text-align: center;
        margin-top: 24px;
        animation: fadeIn 0.4s ease-out;
    }
    .result-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #fca5a5;
        color: #991b1b;
    }
    .result-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce3 100%);
        border: 2px solid #86efac;
        color: #166534;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .result-prob {
        font-size: 1.25rem;
        font-weight: 500;
        opacity: 0.95;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def init_state():
    defaults = {
        'max_step': 0, 'current_step': 0, 'df': None, 'selected_features': [],
        'model': None, 'imputer': None, 'scaler': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<div style='padding: 10px 0 20px 0;'><h2 style='margin:0; color:#0f172a;'>Progress Tracker</h2></div>", unsafe_allow_html=True)

step_names = [
    "Home",
    "Exploratory Data Analysis",
    "Feature Selection",
    "Preprocessing",
    "Modelling",
    "Evaluation",
    "Deployment"
]

for i, name in enumerate(step_names):
    if i < st.session_state.current_step:
        label = f"✓  {name}"
    elif i == st.session_state.current_step:
        label = f"●  {name}"
    elif i <= st.session_state.max_step:
        label = f"○  {name}"
    else:
        label = f"🔒  {name}"
        
    disabled = i > st.session_state.max_step
    btn_type = "primary" if i == st.session_state.current_step else "secondary"
    
    if st.sidebar.button(label, disabled=disabled, use_container_width=True, key=f"nav_{i}", type=btn_type):
        st.session_state.current_step = i
        st.rerun()

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
if st.sidebar.button("↻ Reset Progress", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- STEPS RENDERING ---

def render_step_0():
    col1, col2 = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with col1:
        st.markdown('<div class="hero-title">Diabetes Risk Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Advanced Health Analytics Platform</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #475569; font-size: 1.05rem; line-height: 1.7; margin-bottom: 24px;'>
        This modern, AI-powered application leverages advanced machine learning techniques to evaluate medical predictor variables and securely predict the likelihood of diabetes in patients.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("Begin Analysis", type="primary"):
            st.session_state.max_step = max(st.session_state.max_step, 1)
            st.session_state.current_step = 1
            st.rerun()
            
    with col2:
        st.markdown("""
        <div style='display: flex; justify-content: center; align-items: center; padding: 20px;'>
            <img src='https://drive.google.com/uc?export=view&id=1eYm4Aca0sOwEcC6NElXjxIXEP7DxQOOc' style='max-width: 100%; max-height: 300px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); object-fit: cover;' />
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">Platform Overview</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("""
            <h4 style='color: #1e293b; margin-bottom: 8px;'>Expected Output</h4>
            <p style='color: #64748b; line-height: 1.6;'>Users receive a precise probability percentage indicating the risk of diabetes, supported by robust evaluation metrics and strict data validation protocols.</p>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <h4 style='color: #1e293b; margin-bottom: 8px;'>Target Audience & Data</h4>
            <p style='color: #64748b; line-height: 1.6;'>Designed for Medical Professionals and Clinics. Built upon the renowned Pima Indians Diabetes Database containing comprehensive metabolic features.</p>
            """, unsafe_allow_html=True)
        
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
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 24px;'>Analyze the distributions and relationships of the predictor variables.</p>", unsafe_allow_html=True)
    df = st.session_state.df
    
    with st.container(border=True):
        st.markdown('<div class="card-title">Dataset Overview</div>', unsafe_allow_html=True)
        with st.expander("🔍 View Raw Dataset", expanded=False):
            st.dataframe(df, use_container_width=True)
        with st.expander("📊 View Statistical Summary", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)
            
    with st.container(border=True):
        st.markdown('<div class="card-title">Interactive Visualizations</div>', unsafe_allow_html=True)
        plot_type = st.radio("Select Plot Type", ["Univariate", "Bivariate", "Multivariate"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if plot_type == "Univariate":
            col = st.selectbox("Select Feature for Distribution", df.columns)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col], kde=True, ax=ax, color='#3b82f6', edgecolor='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        elif plot_type == "Bivariate":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis Feature", df.columns)
            with col2:
                y_col = st.selectbox("Y-axis Feature", [c for c in df.columns if c != x_col])
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df, x=x_col, y=y_col, hue="Outcome", palette=["#3b82f6", "#ef4444"], ax=ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        elif plot_type == "Multivariate":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax, linewidths=0.5)
            st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue to Feature Selection", type="primary"):
        st.session_state.max_step = max(st.session_state.max_step, 2)
        st.session_state.current_step = 2
        st.rerun()

def render_step_2():
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">Feature Selection</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 24px;'>Isolate the most crucial variables for model training.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">Configure Predictor Variables</div>', unsafe_allow_html=True)
        st.markdown("<p style='color: #475569; margin-bottom: 16px;'>Select the relevant features. The Target variable (Outcome) is automatically isolated and preserved.</p>", unsafe_allow_html=True)
        
        df = st.session_state.df
        features = [col for col in df.columns if col != 'Outcome']
        selected = st.multiselect("Predictor Variables", features, default=features, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue to Preprocessing", type="primary"):
        if len(selected) == 0:
            st.error("Please select at least one feature to proceed.")
        else:
            st.session_state.selected_features = selected
            st.session_state.max_step = max(st.session_state.max_step, 3)
            st.session_state.current_step = 3
            st.rerun()

def render_step_3():
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">Data Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 24px;'>Cleanse, split, and scale the data to prevent leakage and optimize training.</p>", unsafe_allow_html=True)
    
    df = st.session_state.df
    selected_features = st.session_state.selected_features
    
    with st.container(border=True):
        st.markdown('<div class="card-title">Pipeline Configuration</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100.0
        with col2:
            scaler_choice = st.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
            
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
        st.markdown('<div class="card-title">Processed Data Overview</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(f"<div style='font-size: 1.1rem;'><b>Training Samples:</b> <span style='color:#3b82f6;'>{X_train_scaled.shape[0]}</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='font-size: 1.1rem;'><b>Testing Samples:</b> <span style='color:#10b981;'>{X_test_scaled.shape[0]}</span></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("View Preprocessed Training Sample"):
            st.dataframe(X_train_scaled.head(), use_container_width=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue to Modelling", type="primary"):
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
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">Model Training</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 24px;'>Configure hyperparameters and fit the machine learning algorithm.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">Algorithm Configuration</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5], gap="large")
        with col1:
            algo = st.selectbox("Select Algorithm", ["Random Forest", "SVM", "Logistic Regression", "KNN"])
            
        with col2:
            if algo == "Random Forest":
                n_estimators = st.number_input("Trees (n_estimators)", 10, 500, 100)
                max_depth = st.number_input("Max Depth", 1, 50, 10)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif algo == "SVM":
                C = st.number_input("Regularization (C)", 0.1, 100.0, 1.0)
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
                model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
            elif algo == "Logistic Regression":
                C_lr = st.number_input("Inverse Reg. (C)", 0.01, 100.0, 1.0)
                model = LogisticRegression(C=C_lr, random_state=42)
            elif algo == "KNN":
                n_neighbors = st.number_input("Neighbors", 1, 50, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Train Model & Proceed", type="primary"):
        with st.spinner("Training model..."):
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model = model
            st.session_state.max_step = max(st.session_state.max_step, 5)
            st.session_state.current_step = 5
        st.rerun()

def render_step_5():
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 24px;'>Review the performance metrics of your trained model.</p>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model in the Modelling step first.")
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
            <div class="metric-value color-acc">{acc:.4f}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-value color-prec">{prec:.4f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-box">
            <div class="metric-value color-rec">{rec:.4f}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-box">
            <div class="metric-value color-f1">{f1:.4f}</div>
            <div class="metric-label">F1-Score</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        annot_labels = np.array([
            [f"True Negative (TN)\n{cm[0,0]}", f"False Positive (FP)\n{cm[0,1]}"],
            [f"False Negative (FN)\n{cm[1,0]}", f"True Positive (TP)\n{cm[1,1]}"]
        ])
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', ax=ax, cbar=False, 
                    annot_kws={"size": 11, "weight": "bold"}, linewidths=1)
        ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=12, color='#1e293b')
        ax.set_ylabel('True Label', fontweight='bold', labelpad=12, color='#1e293b')
        st.pyplot(fig)
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Unlock Deployment", type="primary"):
        st.session_state.max_step = max(st.session_state.max_step, 6)
        st.session_state.current_step = 6
        st.rerun()

def render_step_6():
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 24px;'>Enter real patient details below to generate a diagnosis probability.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="card-title">Patient Input Form</div>', unsafe_allow_html=True)
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
        submit_btn = st.button("Generate Prediction", type="primary", use_container_width=True)
        
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
                <div class="result-title">⚠️ High Risk</div>
                <div class="result-prob">The model predicts the patient is <b>Diabetic</b> with a {prob*100:.1f}% probability.</div>
            </div>
            """
        else:
            res_html = f"""
            <div class="result-card result-low">
                <div class="result-title">✅ Low Risk</div>
                <div class="result-prob">The model predicts the patient is <b>Non-Diabetic</b> with a {(1-prob)*100:.1f}% probability.</div>
            </div>
            """
        st.markdown(res_html, unsafe_allow_html=True)

if st.session_state.current_step == 0:
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
