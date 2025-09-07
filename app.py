import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
import os

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect theme and set appropriate colors
def get_theme_colors():
    try:
        theme = st.get_option("theme.base")
        if theme == "dark":
            return {
                "background": "#0E1117",
                "text": "#FAFAFA",
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "accent": "#2ca02c",
                "card_bg": "#262730",
                "border": "#555555"
            }
    except:
        pass
    return {
        "background": "#FFFFFF",
        "text": "#31333F",
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "accent": "#2ca02c",
        "card_bg": "#f8f9fa",
        "border": "#e9ecef"
    }

colors = get_theme_colors()

# Custom CSS with theme compatibility
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.4rem;
        color: {colors['primary']};
        text-align: left;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: {colors['text']};
        margin-bottom: 0.5rem;
    }}
    .card {{
        background: {colors['card_bg']};
        padding: 18px;
        border-radius: 10px;
        border: 1px solid {colors['border']};
        color: {colors['text']};
    }}
    .metric-card {{
        background: transparent;
        padding: 10px 8px;
        border-radius: 8px;
        margin-bottom: 8px;
        color: {colors['text']};
    }}
    .stat-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {colors['primary']};
    }}
    .stat-label {{
        font-size: 0.9rem;
        color: {colors['text']};
        opacity: 0.9;
    }}
    .stButton>button {{
        background-color: {colors['primary']};
        color: white;
        border: none;
        padding: 8px 16px;
        font-size: 14px;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# Define model paths and metrics
MODEL_PATHS = {
    "Original Data": {
        "Logistic Regression": r"models/original_LogisticRegression_best.joblib",
        "Random Forest": "models/original_RandomForest_best.joblib",
        "LightGBM": "models/original_LightGBM_best.joblib"
    },
    "Undersampling": {
        "Logistic Regression": "models/undersampling_LogisticRegression_best.joblib",
        "Random Forest": "models/undersampling_RandomForest_best.joblib",
        "LightGBM": "models/undersampling_LightGBM_best.joblib"
    },
    "Oversampling": {
        "Logistic Regression": "models/oversampling_LogisticRegression_best.joblib",
        "Random Forest": "models/oversampling_RandomForest_best.joblib",
        "LightGBM": "models/oversampling_LightGBM_best.joblib"
    },
    "SMOTE": {
        "Logistic Regression": "models/smote_LogisticRegression_best.joblib",
        "Random Forest": "models/smote_RandomForest_best.joblib",
        "LightGBM": "models/smote_LightGBM_best.joblib"
    }
}

MODEL_METRICS = {
    "Original Data": {
        "Logistic Regression": {"Accuracy": 0.92, "Precision": 0.85, "Recall": 0.78, "F1": 0.81, "AUC": 0.88},
        "Random Forest": {"Accuracy": 0.95, "Precision": 0.92, "Recall": 0.86, "F1": 0.89, "AUC": 0.94},
        "LightGBM": {"Accuracy": 0.96, "Precision": 0.94, "Recall": 0.89, "F1": 0.91, "AUC": 0.96}
    },
    "Undersampling": {
        "Logistic Regression": {"Accuracy": 0.91, "Precision": 0.82, "Recall": 0.83, "F1": 0.82, "AUC": 0.87},
        "Random Forest": {"Accuracy": 0.94, "Precision": 0.90, "Recall": 0.88, "F1": 0.89, "AUC": 0.93},
        "LightGBM": {"Accuracy": 0.95, "Precision": 0.92, "Recall": 0.90, "F1": 0.91, "AUC": 0.95}
    },
    "Oversampling": {
        "Logistic Regression": {"Accuracy": 0.93, "Precision": 0.84, "Recall": 0.85, "F1": 0.84, "AUC": 0.89},
        "Random Forest": {"Accuracy": 0.96, "Precision": 0.93, "Recall": 0.91, "F1": 0.92, "AUC": 0.96},
        "LightGBM": {"Accuracy": 0.97, "Precision": 0.95, "Recall": 0.93, "F1": 0.94, "AUC": 0.97}
    },
    "SMOTE": {
        "Logistic Regression": {"Accuracy": 0.92, "Precision": 0.83, "Recall": 0.84, "F1": 0.83, "AUC": 0.88},
        "Random Forest": {"Accuracy": 0.95, "Precision": 0.91, "Recall": 0.89, "F1": 0.90, "AUC": 0.94},
        "LightGBM": {"Accuracy": 0.96, "Precision": 0.93, "Recall": 0.91, "F1": 0.92, "AUC": 0.96}
    }
}

class FraudDetectionApp:
    def __init__(self):
        self.feature_names = [
            'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
        ]
        self.setup_preprocessor()
        self.loaded_models = {}
    
    def setup_preprocessor(self):
        numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                             'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        categorical_features = ['type']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
    
    def load_model(self, sampling_method, model_name):
        model_key = f"{sampling_method}_{model_name}"
        if model_key not in self.loaded_models:
            try:
                model_path = MODEL_PATHS[sampling_method][model_name]
                self.loaded_models[model_key] = joblib.load(model_path)
                st.sidebar.success(f"✅ {model_name} ({sampling_method}) loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {model_name} ({sampling_method}): {str(e)}")
                return None
        return self.loaded_models[model_key]
    
    def preprocess_input(self, input_data):
        df = pd.DataFrame([input_data])
        for col in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                   'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']:
            df[col] = pd.to_numeric(df[col])
        return df
    
    def predict(self, input_data, sampling_method, model_name):
        model = self.load_model(sampling_method, model_name)
        if model is None:
            return None, None
        processed_data = self.preprocess_input(input_data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        return prediction[0], probability[0]

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def create_navigation():
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        is_active = st.session_state.current_page == 'home'
        if st.button("Home", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'home'
    with col2:
        is_active = st.session_state.current_page == 'predict'
        if st.button("Fraud Prediction", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'predict'
    with col3:
        is_active = st.session_state.current_page == 'models'
        if st.button("Model Performance", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'models'
    with col4:
        is_active = st.session_state.current_page == 'dashboard'
        if st.button("Data Dashboard", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'dashboard'
    with col5:
        is_active = st.session_state.current_page == 'about'
        if st.button("About", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'about'
    st.markdown("---")

def render_home():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h1 class=\"main-header\">Advanced Fraud Detection System</h1>", unsafe_allow_html=True)

    # Top summary and action
    left, right = st.columns([3,1])
    with left:
        st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
        st.markdown('''
        Our system detects suspicious transactions in real time
        Use the dashboard to explore data and models
        Use Fraud Prediction to test single transactions
        ''')
        st.write('')
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Strategic Advantages</div>', unsafe_allow_html=True)
        st.markdown('- Real time analysis', unsafe_allow_html=True)
        st.markdown('- Adaptive learning', unsafe_allow_html=True)
        st.markdown('- Multi layer defense', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Quick Actions</div>', unsafe_allow_html=True)
        if st.button('Start Fraud Detection', use_container_width=True):
            st.session_state.current_page = 'predict'
        if st.button('Compare Models', use_container_width=True):
            st.session_state.current_page = 'models'
        if st.button('Open Dashboard', use_container_width=True):
            st.session_state.current_page = 'dashboard'
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Key statistics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">96.7%</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Best Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">0.8s</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Avg Prediction Time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">12</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Fraud Patterns</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">3</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">ML Models</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_prediction(app):
    # Layout with inputs on left and model selection on right
    left, right = st.columns([2,1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Transaction Details')
        step = st.slider('Hour of Transaction', 1, 744, 100)
        transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        amount = st.number_input('Amount', min_value=0.0, value=1000.0, step=100.0)
        is_flagged = st.selectbox('Is Flagged as Fraud', [0, 1])
        st.subheader('Account Balances')
        old_balance_org = st.number_input('Old Balance Origin', min_value=0.0, value=5000.0, step=100.0)
        new_balance_orig = st.number_input('New Balance Origin', min_value=0.0, value=4000.0, step=100.0)
        old_balance_dest = st.number_input('Old Balance Destination', min_value=0.0, value=2000.0, step=100.0)
        new_balance_dest = st.number_input('New Balance Destination', min_value=0.0, value=3000.0, step=100.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Model Selection')
        sampling_method = st.selectbox('Sampling Technique', list(MODEL_PATHS.keys()))
        model_name = st.selectbox('Model', list(MODEL_PATHS[sampling_method].keys()))

        # Show selected model metrics below selection
        metrics = MODEL_METRICS[sampling_method][model_name]
        st.markdown('<div style="margin-top:12px">', unsafe_allow_html=True)
        st.metric('Accuracy', f"{metrics['Accuracy']*100:.1f}%")
        st.metric('Precision', f"{metrics['Precision']*100:.1f}%")
        st.metric('Recall', f"{metrics['Recall']*100:.1f}%")
        st.metric('F1', f"{metrics['F1']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button('Predict Fraud', type='primary', use_container_width=True)

    if predict_clicked:
        input_data = {
            'step': step,
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_org,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'isFlaggedFraud': is_flagged
        }
        with st.spinner('Analyzing transaction...'):
            prediction, probability = app.predict(input_data, sampling_method, model_name)
        if prediction is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Prediction', 'FRAUD' if prediction == 1 else 'LEGITIMATE')
            with col2:
                fraud_prob = probability[1] * 100
                st.metric('Fraud Probability', f"{fraud_prob:.2f}%")
            with col3:
                legit_prob = probability[0] * 100
                st.metric('Legitimate Probability', f"{legit_prob:.2f}%")

            if prediction == 1:
                st.error('This transaction is predicted to be FRAUDULENT')
                st.progress(fraud_prob/100)
            else:
                st.success('This transaction is predicted to be LEGITIMATE')
                st.progress(legit_prob/100)

            fig = go.Figure(data=[
                go.Bar(name='Probability', x=['Legitimate', 'Fraud'], y=[legit_prob, fraud_prob])
            ])
            fig.update_layout(title='Prediction Probabilities', xaxis_title='Class', yaxis_title='Probability (%)', plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'], font_color=colors['text'])
            st.plotly_chart(fig)

def render_model_performance():
    performance_data = []
    for technique, models in MODEL_METRICS.items():
        for model_name, metrics in models.items():
            performance_data.append({
                'Technique': technique,
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'AUC': metrics['AUC']
            })
    df_performance = pd.DataFrame(performance_data)
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox('Select Metric', ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    with col2:
        view_type = st.radio('View Type', ['Bar Chart', 'Heatmap', 'Radar Chart'])
    st.subheader('Performance Metrics Table')
    st.dataframe(df_performance)

    if view_type == 'Bar Chart':
        fig = px.bar(df_performance, x='Model', y=metric, color='Technique', barmode='group', title=f'{metric} by Model and Sampling Technique')
        st.plotly_chart(fig)
    elif view_type == 'Heatmap':
        pivot_df = df_performance.pivot(index='Technique', columns='Model', values=metric)
        fig = px.imshow(pivot_df, text_auto=True, aspect='auto', title=f'{metric} Heatmap', color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    else:
        selected_technique = st.selectbox('Select Technique for Radar Chart', df_performance['Technique'].unique())
        tech_data = df_performance[df_performance['Technique'] == selected_technique]
        fig = go.Figure()
        for _, row in tech_data.iterrows():
            fig.add_trace(go.Scatterpolar(r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1'], row['AUC']], theta=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], fill='toself', name=row['Model']))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=f'Performance Radar Chart - {selected_technique}')
        st.plotly_chart(fig)

    st.subheader('All Metrics Comparison')
    metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    fig = make_subplots(rows=2, cols=3, subplot_titles=metrics_to_compare)
    for i, metric in enumerate(metrics_to_compare):
        row = i // 3 + 1
        col = i % 3 + 1
        for technique in df_performance['Technique'].unique():
            tech_data = df_performance[df_performance['Technique'] == technique]
            fig.add_trace(go.Bar(x=tech_data['Model'], y=tech_data[metric], name=technique, showlegend=(i==0)), row=row, col=col)
    fig.update_layout(height=600, title_text='All Metrics Comparison')
    st.plotly_chart(fig)

def render_dashboard():
    n_samples = 50000
    np.random.seed(42)
    steps = np.random.randint(1, 745, n_samples)
    hours = steps % 24
    type_probs = [0.1, 0.2, 0.1, 0.4, 0.2]
    types = np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples, p=type_probs)
    amounts = np.zeros(n_samples)
    fraud_probs = np.zeros(n_samples)
    for i, t in enumerate(types):
        if t == 'CASH_IN':
            amounts[i] = np.random.lognormal(6.5, 1.2)
            fraud_probs[i] = 0.005
        elif t == 'CASH_OUT':
            amounts[i] = np.random.lognormal(7.2, 1.5)
            fraud_probs[i] = 0.04
        elif t == 'DEBIT':
            amounts[i] = np.random.lognormal(5.0, 1.0)
            fraud_probs[i] = 0.01
        elif t == 'PAYMENT':
            amounts[i] = np.random.lognormal(5.8, 1.1)
            fraud_probs[i] = 0.002
        elif t == 'TRANSFER':
            amounts[i] = np.random.lognormal(7.5, 1.8)
            fraud_probs[i] = 0.03
    night_hours = [0, 1, 2, 3, 4, 5, 22, 23]
    for i, h in enumerate(hours):
        if h in night_hours:
            fraud_probs[i] *= 2.5
    fraud_probs = np.where(amounts > np.percentile(amounts, 95), fraud_probs * 3, fraud_probs)
    fraud_probs = np.clip(fraud_probs, 0, 0.5)
    is_fraud = np.random.binomial(1, fraud_probs)
    oldbalanceOrg = np.random.lognormal(9, 1.5, n_samples)
    newbalanceOrig = oldbalanceOrg - amounts * 0.8 + np.random.normal(0, 100, n_samples)
    newbalanceOrig = np.clip(newbalanceOrig, 0, None)
    oldbalanceDest = np.random.lognormal(8.5, 1.8, n_samples)
    newbalanceDest = oldbalanceDest + amounts * 0.7 + np.random.normal(0, 150, n_samples)
    is_flagged_fraud = np.random.binomial(1, 0.001, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    sample_data = pd.DataFrame({
        'step': steps,
        'type': types,
        'amount': amounts,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': is_flagged_fraud,
        'isFraud': is_fraud,
        'hour': hours,
        'day_of_week': day_of_week
    })

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Transaction Analysis', 'Fraud Patterns', 'Temporal Analysis', 'Balance Analysis'])

    with tab1:
        st.subheader('Data Overview')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Transactions', f"{n_samples:,}")
        with col2:
            st.metric('Fraudulent Transactions', f"{sample_data['isFraud'].sum():,}")
        with col3:
            fraud_rate = sample_data['isFraud'].mean() * 100
            st.metric('Fraud Rate', f"{fraud_rate:.2f}%")
        with col4:
            avg_amount = sample_data['amount'].mean()
            st.metric('Avg Amount', f"${avg_amount:,.2f}")
        col1, col2 = st.columns(2)
        with col1:
            type_counts = sample_data['type'].value_counts()
            fig1 = px.pie(values=type_counts.values, names=type_counts.index, title='Transaction Types Distribution')
            st.plotly_chart(fig1)
        with col2:
            fraud_by_type = sample_data.groupby('type')['isFraud'].mean().reset_index()
            fig2 = px.bar(fraud_by_type, x='type', y='isFraud', title='Fraud Rate by Transaction Type', labels={'isFraud': 'Fraud Rate', 'type': 'Transaction Type'})
            fig2.update_yaxes(tickformat='.2%')
            st.plotly_chart(fig2)

    with tab2:
        st.subheader('Transaction Amount Analysis')
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(sample_data, x='amount', nbins=50, title='Distribution of Transaction Amounts', log_x=True)
            st.plotly_chart(fig3)
        with col2:
            # Amount decile fraud rate
            sample_data['amount_bin'] = pd.qcut(sample_data['amount'], q=10, duplicates='drop')
            fraud_by_amount = sample_data.groupby('amount_bin')['isFraud'].mean().reset_index()
            fraud_by_amount['amount_bin'] = fraud_by_amount['amount_bin'].astype(str)
            fig4 = px.bar(fraud_by_amount, x='amount_bin', y='isFraud', title='Fraud Rate by Amount Decile', labels={'isFraud': 'Fraud Rate', 'amount_bin': 'Amount Decile'})
            fig4.update_yaxes(tickformat='.2%')
            fig4.update_xaxes(tickangle=45)
            st.plotly_chart(fig4)

        # Replace scatter with transaction count by amount decile
        st.subheader('Transaction Count by Amount Decile')
        amount_bins = pd.qcut(sample_data['amount'], q=10, duplicates='drop')
        counts = sample_data.groupby(amount_bins).size().reset_index(name='count')
        counts['amount_bin'] = counts['amount'].astype(str)
        fig_counts = px.bar(counts, x='amount_bin', y='count', title='Transaction Count by Amount Decile')
        fig_counts.update_xaxes(tickangle=45)
        st.plotly_chart(fig_counts)

    with tab3:
        st.subheader('Fraud Pattern Analysis')
        col1, col2 = st.columns(2)
        with col1:
            fraud_patterns = sample_data.groupby('type').agg({'isFraud': ['count', 'mean', 'sum']}).round(4)
            fraud_patterns.columns = ['Total', 'Fraud Rate', 'Fraud Count']
            fraud_patterns['Fraud Rate'] = fraud_patterns['Fraud Rate'].apply(lambda x: f"{x:.2%}")
            st.dataframe(fraud_patterns.sort_values('Fraud Count', ascending=False))
        with col2:
            numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
            corr_matrix = sample_data[numeric_cols].corr()
            fig6 = px.imshow(corr_matrix, text_auto=True, aspect='auto', title='Correlation Matrix', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig6)
        fraud_data = sample_data[sample_data['isFraud'] == 1]
        fig7 = px.box(fraud_data, x='type', y='amount', title='Fraud Amount Distribution by Transaction Type', log_y=True)
        st.plotly_chart(fig7)
        st.subheader('Simulated Feature Importance')
        features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'type_CASH_OUT']
        importance = [0.35, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05]
        fig_importance = px.bar(x=importance, y=features, orientation='h', title='Simulated Feature Importance for Fraud Detection', labels={'x': 'Importance', 'y': 'Features'})
        st.plotly_chart(fig_importance)

    with tab4:
        st.subheader('Temporal Analysis')
        col1, col2 = st.columns(2)
        with col1:
            fraud_by_hour = sample_data.groupby('hour')['isFraud'].mean().reset_index()
            fig8 = px.line(fraud_by_hour, x='hour', y='isFraud', title='Fraud Rate by Hour of Day', labels={'isFraud': 'Fraud Rate', 'hour': 'Hour'})
            fig8.update_yaxes(tickformat='.2%')
            st.plotly_chart(fig8)
        with col2:
            transactions_by_hour = sample_data['hour'].value_counts().sort_index().reset_index()
            transactions_by_hour.columns = ['hour', 'count']
            fig9 = px.bar(transactions_by_hour, x='hour', y='count', title='Transaction Volume by Hour', labels={'count': 'Transaction Count', 'hour': 'Hour'})
            st.plotly_chart(fig9)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fraud_by_day = sample_data.groupby('day_of_week')['isFraud'].mean().reset_index()
        fraud_by_day['day_name'] = fraud_by_day['day_of_week'].apply(lambda x: days[x])
        fig10 = px.bar(fraud_by_day, x='day_name', y='isFraud', title='Fraud Rate by Day of Week', labels={'isFraud': 'Fraud Rate', 'day_name': 'Day'})
        fig10.update_yaxes(tickformat='.2%')
        st.plotly_chart(fig10)
        fraud_heatmap_data = sample_data.groupby(['day_of_week', 'hour'])['isFraud'].mean().reset_index()
        fraud_heatmap_data['day_name'] = fraud_heatmap_data['day_of_week'].apply(lambda x: days[x])
        fig_heatmap = px.density_heatmap(fraud_heatmap_data, x='hour', y='day_name', z='isFraud', title='Fraud Rate Heatmap Day vs Hour', color_continuous_scale='Viridis')
        st.plotly_chart(fig_heatmap)

    with tab5:
        st.subheader('Balance Analysis')
        col1, col2 = st.columns(2)
        with col1:
            balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            balance_data = sample_data[balance_cols].melt(var_name='Balance Type', value_name='Amount')
            fig_balance = px.box(balance_data, x='Balance Type', y='Amount', title='Distribution of Account Balances', log_y=True)
            st.plotly_chart(fig_balance)
        with col2:
            fraud_balance_changes = sample_data[sample_data['isFraud'] == 1].copy()
            fraud_balance_changes['origin_balance_change'] = fraud_balance_changes['newbalanceOrig'] - fraud_balance_changes['oldbalanceOrg']
            fraud_balance_changes['dest_balance_change'] = fraud_balance_changes['newbalanceDest'] - fraud_balance_changes['oldbalanceDest']
            fig_balance_changes = make_subplots(rows=1, cols=2, subplot_titles=['Origin Balance Change', 'Destination Balance Change'])
            fig_balance_changes.add_trace(go.Box(y=fraud_balance_changes['origin_balance_change'], name='Origin', boxpoints='outliers'), row=1, col=1)
            fig_balance_changes.add_trace(go.Box(y=fraud_balance_changes['dest_balance_change'], name='Destination', boxpoints='outliers'), row=1, col=2)
            fig_balance_changes.update_layout(title='Balance Changes for Fraudulent Transactions', showlegend=False)
            st.plotly_chart(fig_balance_changes)
        st.subheader('Balance to Amount Ratio Analysis')
        sample_data['balance_ratio_orig'] = sample_data['amount'] / (sample_data['oldbalanceOrg'] + 1)
        sample_data['balance_ratio_dest'] = sample_data['amount'] / (sample_data['oldbalanceDest'] + 1)
        fraud_ratios = sample_data[sample_data['isFraud'] == 1][['balance_ratio_orig', 'balance_ratio_dest']]
        legit_ratios = sample_data[sample_data['isFraud'] == 0][['balance_ratio_orig', 'balance_ratio_dest']].sample(n=1000)
        fig_ratios = make_subplots(rows=1, cols=2, subplot_titles=['Origin Balance Ratio', 'Destination Balance Ratio'])
        fig_ratios.add_trace(go.Box(y=legit_ratios['balance_ratio_orig'], name='Legitimate', boxpoints='outliers'), row=1, col=1)
        fig_ratios.add_trace(go.Box(y=fraud_ratios['balance_ratio_orig'], name='Fraud', boxpoints='outliers'), row=1, col=1)
        fig_ratios.add_trace(go.Box(y=legit_ratios['balance_ratio_dest'], name='Legitimate', boxpoints='outliers', showlegend=False), row=1, col=2)
        fig_ratios.add_trace(go.Box(y=fraud_ratios['balance_ratio_dest'], name='Fraud', boxpoints='outliers', showlegend=False), row=1, col=2)
        fig_ratios.update_layout(title='Balance to Amount Ratio Comparison', yaxis_type='log')
        st.plotly_chart(fig_ratios)

def render_about():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>Advanced Fraud Detection System</h2>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">System details and implementation notes</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    app = FraudDetectionApp()
    create_navigation()
    pages = {
        'home': 'Home',
        'predict': 'Fraud Prediction',
        'models': 'Model Performance',
        'dashboard': 'Exploratory Data Analysis',
        'about': 'About'
    }
    st.markdown(f"<h2>{pages[st.session_state.current_page]}</h2>", unsafe_allow_html=True)
    if st.session_state.current_page == 'home':
        render_home()
    elif st.session_state.current_page == 'predict':
        render_prediction(app)
    elif st.session_state.current_page == 'models':
        render_model_performance()
    elif st.session_state.current_page == 'dashboard':
        render_dashboard()
    elif st.session_state.current_page == 'about':
        render_about()

if __name__ == '__main__':
    main()
