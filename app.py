import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlycoAID – Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #f7f9fc; }

    .hero-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-header h1 { font-family: 'DM Serif Display', serif; font-size: 2.4rem; margin-bottom: 0.4rem; color: white; }
    .hero-header p  { font-size: 1rem; opacity: 0.8; margin: 0; color: #cde8f0; }

    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align: center;
        border-left: 4px solid #2c5364;
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; color: #2c5364; font-weight: 600; }
    .metric-card p  { font-size: 0.78rem; color: #888; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }

    .result-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white;
        padding: 1.8rem; border-radius: 14px; text-align: center;
        font-size: 1.5rem; font-weight: 600; margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255,65,108,0.35);
    }
    .result-low {
        background: linear-gradient(135deg, #11998e, #38ef7d); color: white;
        padding: 1.8rem; border-radius: 14px; text-align: center;
        font-size: 1.5rem; font-weight: 600; margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(17,153,142,0.35);
    }

    .xai-card {
        background: #fff; border-radius: 12px; padding: 1.2rem 1.5rem;
        margin: 0.5rem 0; border-left: 4px solid #203a43;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06); font-size: 0.93rem; color: #333;
    }
    .xai-card b { color: #2c5364; }

    .section-title {
        font-family: 'DM Serif Display', serif; font-size: 1.4rem;
        color: #1a1a2e; margin-bottom: 0.8rem;
        border-bottom: 2px solid #e0e7ef; padding-bottom: 0.4rem;
    }

    .stButton>button {
        background: linear-gradient(135deg, #203a43, #2c5364);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 500;
        width: 100%; transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2c5364, #38729b);
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(44,83,100,0.35);
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Wrangle function — exactly as in notebook ──────────────────────────────────
@st.cache_data
def wrangle(URL):
    df = pd.read_csv(URL)
    zeros_column = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zeros_column] = df[zeros_column].replace(0, np.nan)
    return df


# ── Train both models — mirrors notebook exactly ───────────────────────────────
@st.cache_resource
def train_models():
    # Load data
    data = wrangle("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

    # ── Decision Tree ──────────────────────────────────────────────────────────
    X = data.drop(columns="Outcome")
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42)
    )
    params = {
        "simpleimputer__strategy": ["mean", "median"],
        "decisiontreeclassifier__max_depth": range(1, 21, 2)
    }
    model = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1, verbose=0, scoring="recall")
    model.fit(X_train, y_train)

    # ── Logistic Regression ────────────────────────────────────────────────────
    data_log = data.copy()
    X_log = data_log.drop(columns="Outcome")
    y_log = data_log['Outcome']

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
    )

    clf_log = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
    )
    log_params = {
        "simpleimputer__strategy": ["mean", "median"],
        "logisticregression__C": [0.01, 0.1, 1, 10, 100]
    }
    model_log = GridSearchCV(clf_log, param_grid=log_params, cv=5, n_jobs=-1, verbose=0, scoring='recall')
    model_log.fit(X_train_log, y_train_log)

    return model, model_log, X_train, X_test, y_train, y_test, X.columns.tolist()




# ── Gauge chart ────────────────────────────────────────────────────────────────
def draw_gauge(probability):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1]*200, color='#e0e7ef', linewidth=18, solid_capstyle='round')
    fill_end = np.pi - (probability * np.pi)
    theta_fill = np.linspace(np.pi, fill_end, 200)
    color = '#ff416c' if probability > 0.5 else '#11998e'
    ax.plot(theta_fill, [1]*200, color=color, linewidth=18, solid_capstyle='round')
    needle_angle = np.pi - (probability * np.pi)
    ax.annotate('', xy=(needle_angle, 0.85), xytext=(needle_angle, 0),
                arrowprops=dict(arrowstyle='->', color='#1a1a2e', lw=2.5))
    ax.text(np.pi,    1.25, '0%',   ha='center', va='center', fontsize=9, color='#888')
    ax.text(np.pi/2,  1.25, '50%',  ha='center', va='center', fontsize=9, color='#888')
    ax.text(0,        1.25, '100%', ha='center', va='center', fontsize=9, color='#888')
    ax.text(0, 0, f'{probability*100:.1f}%', ha='center', va='center',
            fontsize=22, fontweight='bold', color=color)
    ax.set_ylim(0, 1.4)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ── XAI explanation ────────────────────────────────────────────────────────────
def explain_prediction(input_values, feature_names):
    thresholds = {
        'Glucose':                  (140, 'High glucose (>{:.0f} mg/dL) is the primary diabetes indicator.'),
        'BMI':                      (30,  'BMI above {:.0f} indicates obesity — a key diabetes risk factor.'),
        'Age':                      (45,  'Age above {:.0f} significantly increases diabetes risk.'),
        'Pregnancies':              (5,   'More than {:.0f} pregnancies increases gestational diabetes risk.'),
        'BloodPressure':            (80,  'Elevated blood pressure (>{:.0f} mmHg) is linked to diabetes.'),
        'DiabetesPedigreeFunction': (0.5, 'Pedigree score >{:.1f} suggests a strong family history of diabetes.'),
        'Insulin':                  (166, 'Insulin level >{:.0f} µU/mL may indicate insulin resistance.'),
        'SkinThickness':            (29,  'Skin thickness >{:.0f} mm correlates with higher body fat levels.'),
    }
    vals = dict(zip(feature_names, input_values[0]))
    explanations = []
    for feat, (thresh, msg) in thresholds.items():
        val = vals.get(feat, 0)
        if val > thresh:
            explanations.append(f"<b>{feat} = {val:.1f}</b> — ⚠️ {msg.format(thresh)}")
        else:
            explanations.append(f"<b>{feat} = {val:.1f}</b> — ✅ Within normal range.")
    return explanations


# ── Metrics helper ─────────────────────────────────────────────────────────────
def get_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    metrics = {
        'Accuracy':  round(accuracy_score(y_test, preds), 4),
        'Precision': round(report['1']['precision'], 4),
        'Recall':    round(report['1']['recall'], 4),
        'F1 Score':  round(report['1']['f1-score'], 4),
        'AUC-ROC':   round(roc_auc_score(y_test, proba), 4),
    }
    return metrics, preds, proba


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD EVERYTHING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <h1>🩺 GlycoAID – Diabetes Risk Prediction System</h1>
    <p>AI-powered early screening system · Logistic Regression & Decision Tree · Built for GlycoAID Digital Health</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models — please wait..."):
    model, model_log, X_train, X_test, y_train, y_test, feature_names = train_models()
    primary_dt  = model
    primary_log = model_log


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📊 Model Comparison", "📈 Visualisations"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Patient Information</p>', unsafe_allow_html=True)
    st.caption("Enter the patient's clinical measurements and click Predict.")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies",           min_value=0,   max_value=20,  value=1,   step=1)
        Glucose     = st.number_input("Glucose (mg/dL)",       min_value=0,   max_value=300, value=110, step=1)
        BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=72,  step=1)
        SkinThickness = st.number_input("Skin Thickness (mm)",  min_value=0,  max_value=100, value=20,  step=1)
    with col2:
        Insulin     = st.number_input("Insulin (µU/mL)",              min_value=0,   max_value=900,  value=80,   step=1)
        BMI         = st.number_input("BMI",                          min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01, format="%.2f")
        Age         = st.number_input("Age (years)",                  min_value=1,   max_value=120,  value=30,   step=1)

    st.markdown("---")
    model_choice = st.radio(
        "Select model for prediction:",
        ["🌳 Decision Tree (Best Model)", "📈 Logistic Regression"],
        horizontal=True
    )

    if st.button("🔍 Predict Diabetes Risk"):
        input_array = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                  Insulin, BMI, DiabetesPedigreeFunction, Age]])
        input_df = pd.DataFrame(input_array, columns=feature_names)

        active_model = primary_dt if "Decision Tree" in model_choice else primary_log

        prediction  = active_model.predict(input_df)[0]
        probability = active_model.predict_proba(input_df)[0][1]

        st.markdown("---")
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(
                    f'<div class="result-high">⚠️ High Risk of Diabetes<br>'
                    f'<span style="font-size:1rem;opacity:0.9">{probability*100:.1f}% probability</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-low">✅ Low Risk of Diabetes<br>'
                    f'<span style="font-size:1rem;opacity:0.9">{probability*100:.1f}% probability</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown('<p class="section-title" style="margin-top:1.5rem">🧠 Why this prediction?</p>', unsafe_allow_html=True)
            explanations = explain_prediction(input_df.values, feature_names)
            for exp in explanations:
                st.markdown(f'<div class="xai-card">{exp}</div>', unsafe_allow_html=True)

        with gauge_col:
            st.markdown('<p class="section-title">Risk Probability Gauge</p>', unsafe_allow_html=True)
            st.pyplot(draw_gauge(probability), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Model Comparison Table</p>', unsafe_allow_html=True)
    st.caption("Both models tuned with GridSearchCV · scoring=recall · cv=5 · Primary metric: Recall")

    dt_metrics,  dt_preds,  dt_proba  = get_metrics(model,     X_test, y_test)
    lr_metrics,  lr_preds,  lr_proba  = get_metrics(model_log, X_test, y_test)

    comparison_df = pd.DataFrame({
        'Metric':              list(dt_metrics.keys()),
        'Decision Tree':       list(dt_metrics.values()),
        'Logistic Regression': list(lr_metrics.values()),
    })

    def highlight_best(row):
        dt_val = row['Decision Tree']
        lr_val = row['Logistic Regression']
        colors = ['', '', '']
        if dt_val > lr_val:
            colors[1] = 'background-color:#d4f5e9; font-weight:600'
        elif lr_val > dt_val:
            colors[2] = 'background-color:#d4f5e9; font-weight:600'
        return colors

    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True
    )
    st.info("🟢 Green = better performing model for that metric. **Recall** is the primary metric — minimises missed diabetes diagnoses.")

    # Dataset stats
    st.markdown('<p class="section-title" style="margin-top:2rem">Dataset Overview</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card"><h3>768</h3><p>Total samples</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h3>8</h3><p>Input features</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h3>65%</h3><p>No diabetes</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><h3>35%</h3><p>Diabetes</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    # Feature importance — Decision Tree
    st.markdown('<p class="section-title">Feature Importance — Decision Tree</p>', unsafe_allow_html=True)
    dt_step     = model.best_estimator_.named_steps['decisiontreeclassifier']
    importances_dt = dt_step.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances_dt})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colors = ['#2c5364' if i == len(feat_imp_df)-1 else '#b0c4d8' for i in range(len(feat_imp_df))]
    ax1.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=colors)
    ax1.set_xlabel('Importance Score', fontsize=10)
    ax1.set_title('Feature Importance — Decision Tree', fontsize=12, fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    # Odds Ratios — Logistic Regression
    st.markdown('<p class="section-title" style="margin-top:2rem">Feature Importance — Logistic Regression (Odds Ratios)</p>', unsafe_allow_html=True)
    features    = X_train.columns.to_list()
    importances = model_log.best_estimator_.named_steps["logisticregression"].coef_[0]
    odds_ratios = pd.Series(importances, index=features).sort_values()

    fig_lr, axes = plt.subplots(1, 2, figsize=(12, 4))
    odds_ratios.tail().plot(kind="barh", ax=axes[0], color='#2c5364')
    axes[0].set_xlabel("Odds Ratio")
    axes[0].set_title("Top 5 Features", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    odds_ratios.head().plot(kind="barh", ax=axes[1], color='#b0c4d8')
    axes[1].set_xlabel("Odds Ratio")
    axes[1].set_title("Bottom 5 Features", fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_lr, use_container_width=True)

    # ROC Curve
    st.markdown('<p class="section-title" style="margin-top:2rem">ROC Curve Comparison</p>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for name, proba, color in [
        ('Decision Tree',       dt_proba, '#2c5364'),
        ('Logistic Regression', lr_proba, '#ff416c')
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax2.set_xlabel('False Positive Rate', fontsize=10)
    ax2.set_ylabel('True Positive Rate', fontsize=10)
    ax2.set_title('ROC Curve — Decision Tree vs Logistic Regression', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    # Confusion Matrices
    st.markdown('<p class="section-title" style="margin-top:2rem">Confusion Matrices</p>', unsafe_allow_html=True)
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, dt_preds),
            display_labels=['No Diabetes', 'Diabetes']
        ).plot(ax=ax3, colorbar=False, cmap='Blues')
        ax3.set_title('Decision Tree', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with cm_col2:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, lr_preds),
            display_labels=['No Diabetes', 'Diabetes']
        ).plot(ax=ax4, colorbar=False, cmap='Reds')
        ax4.set_title('Logistic Regression', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "GlycoAID Diabetes Risk Prediction System · Dataset: github.com/plotly/datasets · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)
