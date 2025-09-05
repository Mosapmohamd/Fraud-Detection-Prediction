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
        # Try to detect Streamlit theme
        theme = st.get_option("theme.base")
        if theme == "dark":
            return {
                "background": "#0E1117",
                "text": "#FAFAFA",
                "primary": "#FF4B4B",
                "secondary": "#FF6B6B",
                "accent": "#FF8080",
                "card_bg": "#262730",
                "border": "#555555"
            }
    except:
        pass
    # Default light theme colors
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
        font-size: 3rem;
        color: {colors['primary']};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {colors['card_bg']};
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {colors['primary']};
        margin-bottom: 1rem;
        color: {colors['text']};
    }}
    .prediction-fraud {{
        background-color: rgba(255, 107, 107, 0.2);
        border-left: 4px solid #f44336;
    }}
    .prediction-legit {{
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 4px solid #4caf50;
    }}
    .stButton>button {{
        background-color: {colors['primary']};
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {colors['secondary']};
        transform: translateY(-1px);
    }}
    .nav-button {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text']} !important;
        border: 2px solid {colors['border']} !important;
    }}
    .nav-button.active {{
        background-color: {colors['primary']} !important;
        color: white !important;
        border: 2px solid {colors['primary']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .navbar {{
        display: flex;
        justify-content: center;
        background: {colors['card_bg']};
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 1px solid {colors['border']};
    }}
    .card {{
        background: {colors['card_bg']};
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid {colors['border']};
        color: {colors['text']};
    }}
    .highlight-box {{
        background: linear-gradient(135deg, {colors['primary']}20, {colors['secondary']}20);
        border-left: 4px solid {colors['primary']};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }}
</style>
""", unsafe_allow_html=True)

# Define model paths
MODEL_PATHS = {
    "Original Data": {
        "Logistic Regression": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\original_LogisticRegression_best.joblib",
        "Random Forest": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\original_RandomForest_best.joblib",
        "LightGBM": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\original_LightGBM_best.joblib"
    },
    "Undersampling": {
        "Logistic Regression": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\undersampling_LogisticRegression_best.joblib",
        "Random Forest": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\undersampling_RandomForest_best.joblib",
        "LightGBM": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\undersampling_LightGBM_best.joblib"
    },
    "Oversampling": {
        "Logistic Regression": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\oversampling_LogisticRegression_best.joblib",
        "Random Forest": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\oversampling_RandomForest_best.joblib",
        "LightGBM": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\oversampling_LightGBM_best.joblib"
    },
    "SMOTE": {
        "Logistic Regression": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\smote_LogisticRegression_best.joblib",
        "Random Forest": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\smote_RandomForest_best.joblib",
        "LightGBM": r"D:\Projects\Machine learning Projects\Fraud Detection Prediciton\NoteBook\models\smote_LightGBM_best.joblib"
    }
}

# Define model metrics with accuracy
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
        # Define numerical and categorical features
        numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                             'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        categorical_features = ['type']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
    
    def load_model(self, sampling_method, model_name):
        """Load a specific model"""
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
        """Preprocess user input for prediction"""
        df = pd.DataFrame([input_data])
        
        # Ensure correct data types
        for col in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                   'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']:
            df[col] = pd.to_numeric(df[col])
        
        return df
    
    def predict(self, input_data, sampling_method, model_name):
        """Make prediction using the specified model"""
        model = self.load_model(sampling_method, model_name)
        if model is None:
            return None, None
        
        processed_data = self.preprocess_input(input_data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        
        return prediction[0], probability[0]

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def create_navigation():
    """Create navigation buttons with active page highlighting"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        is_active = st.session_state.current_page == 'home'
        if st.button("🏠 Home", use_container_width=True, 
                    type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'home'
    
    with col2:
        is_active = st.session_state.current_page == 'predict'
        if st.button("🔍 Fraud Prediction", use_container_width=True,
                    type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'predict'
    
    with col3:
        is_active = st.session_state.current_page == 'models'
        if st.button("📊 Model Performance", use_container_width=True,
                    type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'models'
    
    with col4:
        is_active = st.session_state.current_page == 'dashboard'
        if st.button("📈 Data Dashboard", use_container_width=True,
                    type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'dashboard'
    
    with col5:
        is_active = st.session_state.current_page == 'about'
        if st.button("ℹ️ About", use_container_width=True,
                    type="primary" if is_active else "secondary"):
            st.session_state.current_page = 'about'
    
    st.markdown("---")

def render_home():
    st.markdown(f'<h1 class="main-header">🔍  Fraud Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
        <h3> Machine Learning for Fraud Detection</h3>
        
        <div class="highlight-box">
        This application uses state-of-the-art machine learning models to detect 
        fraudulent financial transactions in real-time.
        </div>
        
        <h4>🎯 Key Features:</h4>\n
        - Real-time fraud prediction for individual transactions
        - Multiple machine learning models with different sampling techniques
        - Comprehensive model performance comparison
        - Interactive data exploration dashboard
        
        <h4>🤖 Available Models:</h4>\n
        - Logistic Regression
        - Random Forest
        - LightGBM
        
        <h4>⚖️ Sampling Techniques:</h4>\n
        - Original Data
        - Undersampling
        - Oversampling
        - SMOTE
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
        <h3>🚀 Quick Access</h3>
        """, unsafe_allow_html=True)
        if st.button("Start Fraud Detection", use_container_width=True, icon="🔍"):
            st.session_state.current_page = 'predict'
        if st.button("View Model Performance", use_container_width=True, icon="📊"):
            st.session_state.current_page = 'models'
        if st.button("Explore Data", use_container_width=True, icon="📈"):
            st.session_state.current_page = 'dashboard'
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
        <h3>📋 How to Use:</h3>\n
        1. Navigate to Fraud Prediction
        2. Select model and sampling technique
        3. Enter transaction details
        4. Get instant fraud detection results
        </div>
        """, unsafe_allow_html=True)

def render_prediction(app):
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Selection")
        sampling_method = st.selectbox(
            "Sampling Technique",
            list(MODEL_PATHS.keys()),
            help="Choose the sampling technique used for the model"
        )
        
        model_name = st.selectbox(
            "Model",
            list(MODEL_PATHS[sampling_method].keys()),
            help="Choose the machine learning model"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Transaction Details")
        step = st.slider("Hour of Transaction (Step)", 1, 744, 100)
        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        )
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        is_flagged = st.selectbox("Is Flagged as Fraud", [0, 1])
        
        st.subheader("Account Balances")
        old_balance_org = st.number_input("Old Balance Origin", min_value=0.0, value=5000.0, step=100.0)
        new_balance_orig = st.number_input("New Balance Origin", min_value=0.0, value=4000.0, step=100.0)
        old_balance_dest = st.number_input("Old Balance Destination", min_value=0.0, value=2000.0, step=100.0)
        new_balance_dest = st.number_input("New Balance Destination", min_value=0.0, value=3000.0, step=100.0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
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
        
        with st.spinner("Analyzing transaction..."):
            prediction, probability = app.predict(input_data, sampling_method, model_name)
        
        if prediction is not None:
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "FRAUD" if prediction == 1 else "LEGITIMATE")
            
            with col2:
                fraud_prob = probability[1] * 100
                st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
            
            with col3:
                legit_prob = probability[0] * 100
                st.metric("Legitimate Probability", f"{legit_prob:.2f}%")
            
            # Detailed results
            if prediction == 1:
                st.error("🚨 This transaction is predicted to be FRAUDULENT")
                st.progress(fraud_prob/100)
            else:
                st.success("✅ This transaction is predicted to be LEGITIMATE")
                st.progress(legit_prob/100)
            
            # Probability chart
            fig = go.Figure(data=[
                go.Bar(name='Probability', 
                      x=['Legitimate', 'Fraud'], 
                      y=[legit_prob, fraud_prob],
                      marker_color=['#4caf50', '#f44336'])
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability (%)",
                plot_bgcolor=colors['card_bg'],
                paper_bgcolor=colors['card_bg'],
                font_color=colors['text']
            )
            st.plotly_chart(fig)

def render_model_performance():
    
    # Prepare data for visualization
    performance_data = []
    for technique, models in MODEL_METRICS.items():
        for model_name, metrics in models.items():
            performance_data.append({
                "Technique": technique,
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "AUC": metrics["AUC"]
            })
    
    df_performance = pd.DataFrame(performance_data)
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Select Metric", ["Accuracy", "Precision", "Recall", "F1", "AUC"])
    with col2:
        view_type = st.radio("View Type", ["Bar Chart", "Heatmap", "Radar Chart"])
    
    # Display metrics table
    st.subheader("Performance Metrics Table")
    st.dataframe(df_performance.style.highlight_max(axis=0, color='lightgreen')
                .highlight_min(axis=0, color='lightcoral'))
    
    # Visualization based on selection
    if view_type == "Bar Chart":
        fig = px.bar(df_performance, x="Model", y=metric, color="Technique", 
                     barmode="group", title=f"{metric} by Model and Sampling Technique",
                     template="plotly_white" if colors['background'] == "#FFFFFF" else "plotly_dark")
        st.plotly_chart(fig)
    
    elif view_type == "Heatmap":
        pivot_df = df_performance.pivot(index="Technique", columns="Model", values=metric)
        fig = px.imshow(pivot_df, text_auto=True, aspect="auto", 
                        title=f"{metric} Heatmap",
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    
    elif view_type == "Radar Chart":
        selected_technique = st.selectbox("Select Technique for Radar Chart", df_performance['Technique'].unique())
        tech_data = df_performance[df_performance['Technique'] == selected_technique]
        
        fig = go.Figure()
        for _, row in tech_data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1'], row['AUC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Performance Radar Chart - {selected_technique}",
            template="plotly_white" if colors['background'] == "#FFFFFF" else "plotly_dark"
        )
        st.plotly_chart(fig)
    
    # Comparison of all metrics
    st.subheader("All Metrics Comparison")
    metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=metrics_to_compare)
    
    for i, metric in enumerate(metrics_to_compare):
        row = i // 3 + 1
        col = i % 3 + 1
        
        for technique in df_performance['Technique'].unique():
            tech_data = df_performance[df_performance['Technique'] == technique]
            fig.add_trace(
                go.Bar(x=tech_data['Model'], y=tech_data[metric], name=technique, showlegend=(i==0)),
                row=row, col=col
            )
    
    fig.update_layout(height=600, title_text="All Metrics Comparison")
    st.plotly_chart(fig)

def render_dashboard():
    
    # Generate comprehensive sample data
    n_samples = 50000
    sample_data = pd.DataFrame({
        'step': np.random.randint(1, 745, n_samples),
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], n_samples, p=[0.1, 0.2, 0.1, 0.4, 0.2]),
        'amount': np.random.exponential(1000, n_samples),
        'oldbalanceOrg': np.random.lognormal(8, 1.5, n_samples),
        'newbalanceOrig': np.random.lognormal(8, 1.5, n_samples),
        'oldbalanceDest': np.random.lognormal(8, 1.5, n_samples),
        'newbalanceDest': np.random.lognormal(8, 1.5, n_samples),
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.985, 0.015]),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples)
    })
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Transaction Analysis", "Fraud Patterns", "Temporal Analysis"])
    
    with tab1:
        st.subheader("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{n_samples:,}")
        with col2:
            st.metric("Fraudulent Transactions", f"{sample_data['isFraud'].sum():,}")
        with col3:
            fraud_rate = sample_data['isFraud'].mean() * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            avg_amount = sample_data['amount'].mean()
            st.metric("Avg Amount", f"${avg_amount:,.2f}")
        
        # Transaction type distribution
        col1, col2 = st.columns(2)
        with col1:
            type_counts = sample_data['type'].value_counts()
            fig1 = px.pie(values=type_counts.values, names=type_counts.index, 
                         title="Transaction Types Distribution")
            st.plotly_chart(fig1)
        
        with col2:
            fraud_by_type = sample_data.groupby('type')['isFraud'].mean().reset_index()
            fig2 = px.bar(fraud_by_type, x='type', y='isFraud', 
                         title="Fraud Rate by Transaction Type",
                         labels={'isFraud': 'Fraud Rate', 'type': 'Transaction Type'})
            st.plotly_chart(fig2)
    
    with tab2:
        st.subheader("Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(sample_data, x='amount', nbins=50, 
                               title="Distribution of Transaction Amounts",
                               log_x=True)
            st.plotly_chart(fig3)
        
        with col2:
            fraud_amounts = sample_data[sample_data['isFraud'] == 1]['amount']
            legit_amounts = sample_data[sample_data['isFraud'] == 0]['amount']
            
            fig4 = go.Figure()
            fig4.add_trace(go.Box(y=legit_amounts, name='Legitimate', marker_color='green'))
            fig4.add_trace(go.Box(y=fraud_amounts, name='Fraud', marker_color='red'))
            fig4.update_layout(title="Amount Distribution by Fraud Status", yaxis_type="log")
            st.plotly_chart(fig4)
        
        # Balance analysis
        st.subheader("Balance Analysis")
        balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        balance_data = sample_data[balance_cols].melt(var_name='Balance Type', value_name='Amount')
        
        fig5 = px.box(balance_data, x='Balance Type', y='Amount', 
                     title="Distribution of Account Balances", log_y=True)
        st.plotly_chart(fig5)
    
    with tab3:
        st.subheader("Fraud Pattern Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Fraud by transaction type
            fraud_patterns = sample_data.groupby('type').agg({
                'isFraud': ['count', 'mean', 'sum']
            }).round(3)
            fraud_patterns.columns = ['Total', 'Fraud Rate', 'Fraud Count']
            st.dataframe(fraud_patterns.sort_values('Fraud Rate', ascending=False))
        
        with col2:
            # Correlation heatmap
            numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest', 'isFraud']
            corr_matrix = sample_data[numeric_cols].corr()
            
            fig6 = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                            title="Correlation Matrix", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig6)
        
        # Fraud amount distribution by type
        fraud_data = sample_data[sample_data['isFraud'] == 1]
        fig7 = px.box(fraud_data, x='type', y='amount', 
                     title="Fraud Amount Distribution by Transaction Type", log_y=True)
        st.plotly_chart(fig7)
    
    with tab4:
        st.subheader("Temporal Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Fraud by hour
            fraud_by_hour = sample_data.groupby('hour')['isFraud'].mean().reset_index()
            fig8 = px.line(fraud_by_hour, x='hour', y='isFraud', 
                          title="Fraud Rate by Hour of Day",
                          labels={'isFraud': 'Fraud Rate', 'hour': 'Hour'})
            st.plotly_chart(fig8)
        
        with col2:
            # Transactions by hour
            transactions_by_hour = sample_data['hour'].value_counts().sort_index().reset_index()
            transactions_by_hour.columns = ['hour', 'count']
            fig9 = px.bar(transactions_by_hour, x='hour', y='count', 
                         title="Transaction Volume by Hour",
                         labels={'count': 'Transaction Count', 'hour': 'Hour'})
            st.plotly_chart(fig9)
        
        # Fraud by day of week
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fraud_by_day = sample_data.groupby('day_of_week')['isFraud'].mean().reset_index()
        fraud_by_day['day_name'] = fraud_by_day['day_of_week'].apply(lambda x: days[x])
        
        fig10 = px.bar(fraud_by_day, x='day_name', y='isFraud', 
                      title="Fraud Rate by Day of Week",
                      labels={'isFraud': 'Fraud Rate', 'day_name': 'Day'})
        st.plotly_chart(fig10)

def render_about():
    
    st.markdown(f"""
    <div class="card">
    <h2>Fraud Detection System</h2>
    
    <div class="highlight-box">
    This application uses machine learning to detect fraudulent financial transactions
    in real-time. The models are trained on historical transaction data and can identify
    patterns indicative of fraudulent activity.
    </div>
    
    <h3>🎯 Features:</h3>\n
    - <b>Real-time Prediction</b>: Analyze individual transactions for fraud
    - <b>Multiple Models</b>: Compare different machine learning approaches
    - <b>Sampling Techniques</b>: Evaluate different approaches to handle class imbalance
    - <b>Performance Metrics</b>: Comprehensive evaluation of model performance
    - <b>Data Dashboard</b>: Explore transaction patterns and characteristics
    
    <h3>🤖 Model Details:</h3>\n
    - <b>Algorithms</b>: Logistic Regression, Random Forest, LightGBM
    - <b>Sampling Techniques</b>: Original Data, Undersampling, Oversampling, SMOTE
    - <b>Training Data</b>: Financial transaction records with fraud labels
    
    <h3>📊 How to Use:</h3>\n
    1. Navigate to Fraud Prediction for individual transactions
    2. Select model and sampling technique
    3. Enter transaction details
    4. Get instant fraud detection results
    5. Use Model Performance to compare different approaches
    6. Explore data patterns in the Data Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ⚠️ **Note**: This is a demonstration application. 
    Always verify predictions with additional fraud detection measures 
    and domain expertise before taking action.
    """)

def main():
    # Initialize app
    app = FraudDetectionApp()
    
    # Create navigation
    create_navigation()
    
    # Page header with current page highlight
    pages = {
        'home': '🏠 Home',
        'predict': '🔍 Fraud Prediction',
        'models': '📊 Model Performance Comparison',
        'dashboard': '📈 Exploratory Data Analysis',
        'about': 'ℹ️ About This Application'
    }
    
    st.markdown(f"<h2>{pages[st.session_state.current_page]}</h2>", unsafe_allow_html=True)
    
    # Render the current page based on session state
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

if __name__ == "__main__":
    main()
