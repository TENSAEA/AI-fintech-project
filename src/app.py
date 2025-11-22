import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import graph_utils

# Page Config
st.set_page_config(page_title="Fraud Detection & AML System", layout="wide", page_icon="🛡️")

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.95);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stMetric label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    h1, h2, h3 {
        color: #667eea;
    }
    .info-box {
        background: #f0f4ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not os.path.exists('data/transactions.csv'):
        return pd.DataFrame()
    df = pd.read_csv('data/transactions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('models/model_xgb.pkl', 'rb') as f:
            models['xgb'] = pickle.load(f)
        with open('models/model_iso.pkl', 'rb') as f:
            models['iso'] = pickle.load(f)
    except FileNotFoundError:
        return None
    return models

def show_about():
    st.title("📚 About This Project")
    
    st.markdown("""
    ## 🎯 Project Overview
    This is a comprehensive **Fraud Detection & Anti-Money Laundering (AML)** system designed for financial services. 
    It combines multiple machine learning approaches to detect fraudulent transactions and identify suspicious networks.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ❓ Why This Project?
        
        Financial fraud is a **$5+ trillion** global problem affecting:
        - Banks and financial institutions
        - E-commerce platforms
        - Payment processors
        - Individual consumers
        
        **This project addresses:**
        - ⚖️ Highly imbalanced data (fraud is rare but costly)
        - 🔄 Evolving adversaries (fraudsters adapt)
        - 🕸️ Complex networks (fraud rings & mule accounts)
        - ⚡ Real-time detection needs
        """)
        
    with col2:
        st.markdown("""
        ### 🔬 Why Synthetic Data?
        
        We use **synthetic data** for several important reasons:
        
        1. **Privacy & Compliance** 🔒
           - Real financial data contains sensitive PII
           - GDPR, PCI-DSS, and banking regulations restrict sharing
        
        2. **Accessibility** 🌍
           - Real fraud datasets are rarely public
           - Enables learning and experimentation
        
        3. **Controlled Testing** 🧪
           - Known fraud patterns for validation
           - Test edge cases and rare scenarios
        
        4. **Reproducibility** 🔁
           - Consistent results across experiments
           - Academic and research purposes
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🚀 How This Helps You
    
    ### For Financial Institutions & Businesses:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Upload Your Data**
        - Replace synthetic data with real transaction logs
        - Same features: amount, timestamp, sender, receiver
        - Automatic fraud pattern detection
        """)
        
    with col2:
        st.markdown("""
        **🤖 Dual Detection**
        - **Supervised (XGBoost)**: Known fraud patterns
        - **Unsupervised (Isolation Forest)**: Novel attacks
        - Graph analytics for fraud rings
        """)
        
    with col3:
        st.markdown("""
        **⚡ Real-Time Scoring**
        - Instant risk assessment
        - API-ready predictions
        - Scalable architecture
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🛠️ Technical Approach
    
    ### Three-Layer Defense:
    """)
    
    tab1, tab2, tab3 = st.tabs(["1️⃣ Supervised Learning", "2️⃣ Unsupervised Detection", "3️⃣ Graph Analytics"])
    
    with tab1:
        st.markdown("""
        **XGBoost Classifier**
        - Learns from labeled fraud examples
        - High precision on known patterns
        - Handles imbalanced data with class weights
        - Features: amount, time patterns, velocity
        
        **Best for:** Detecting fraud types you've seen before
        """)
        
    with tab2:
        st.markdown("""
        **Isolation Forest**
        - Detects anomalies without labels
        - Identifies unusual transaction patterns
        - Catches new fraud methods (zero-day attacks)
        - Adapts to evolving threats
        
        **Best for:** Unknown fraud schemes and emerging threats
        """)
        
    with tab3:
        st.markdown("""
        **Network Analysis (NetworkX)**
        - Maps money flow between accounts
        - Detects circular transactions (fraud rings)
        - Identifies mule account networks
        - Community detection algorithms
        
        **Best for:** Organized crime and money laundering
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📁 Using Your Own Data
    
    **To adapt this system for your transactions:**
    
    1. **Format Requirements** (CSV with these columns):
       ```
       transaction_id, sender_id, receiver_id, amount, timestamp, is_fraud (optional)
       ```
    
    2. **Replace the data file:**
       ```bash
       cp your_transactions.csv data/transactions.csv
       ```
    
    3. **Retrain models:**
       ```bash
       python src/train_models.py
       ```
    
    4. **Launch dashboard:**
       ```bash
       streamlit run src/app.py
       ```
    
    **Note:** For production use, add more features (device fingerprint, geolocation, transaction history, etc.)
    """)
    
    st.info("💡 **Pro Tip:** This system can be extended with deep learning (autoencoders, LSTMs) for sequence-based fraud detection!")

def show_data_explorer(df):
    st.title("📊 Data Explorer")
    
    st.markdown("""
    Explore the transaction dataset in detail. This page helps you understand the data structure, 
    distributions, and identify patterns.
    """)
    
    # Summary Statistics
    st.header("📈 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Unique Accounts", f"{df['sender_id'].nunique() + df['receiver_id'].nunique():,}")
    with col3:
        fraud_count = df[df['is_fraud'] == 1].shape[0]
        st.metric("Fraud Cases", f"{fraud_count:,}")
    with col4:
        fraud_rate = (fraud_count / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    # Data Sample
    st.header("🔍 Sample Transactions")
    st.markdown("**View a sample of the dataset:**")
    
    col1, col2 = st.columns(2)
    with col1:
        show_fraud = st.selectbox("Filter by:", ["All Transactions", "Fraud Only", "Normal Only"])
    with col2:
        n_rows = st.slider("Number of rows to display:", 5, 100, 20)
    
    if show_fraud == "Fraud Only":
        display_df = df[df['is_fraud'] == 1].head(n_rows)
    elif show_fraud == "Normal Only":
        display_df = df[df['is_fraud'] == 0].head(n_rows)
    else:
        display_df = df.head(n_rows)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Statistical Analysis
    st.header("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Amount Distribution")
        fig = px.histogram(df, x="amount", color="is_fraud", 
                          nbins=50, 
                          labels={"is_fraud": "Fraud"},
                          color_discrete_map={0: "#3b82f6", 1: "#ef4444"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** Fraudulent transactions often have different amount patterns")
        
    with col2:
        st.subheader("Fraud vs Normal Statistics")
        stats_df = df.groupby('is_fraud')['amount'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        stats_df.index = ['Normal', 'Fraud']
        st.dataframe(stats_df, use_container_width=True)
        
        avg_fraud = df[df['is_fraud'] == 1]['amount'].mean()
        avg_normal = df[df['is_fraud'] == 0]['amount'].mean()
        st.metric("Average Fraud Amount", f"${avg_fraud:,.2f}", 
                 delta=f"{((avg_fraud - avg_normal) / avg_normal * 100):.1f}% vs Normal")
    
    # Time Analysis
    st.header("⏰ Temporal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df['hour'] = df['timestamp'].dt.hour
        hourly = df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly[0], name='Normal', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly[1], name='Fraud', fill='tozeroy'))
        fig.update_layout(title="Transactions by Hour of Day", xaxis_title="Hour", yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Pattern:** Fraudsters often operate during specific hours")
        
    with col2:
        df['day_name'] = df['timestamp'].dt.day_name()
        daily = df.groupby(['day_name', 'is_fraud']).size().unstack(fill_value=0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = daily.reindex([d for d in day_order if d in daily.index])
        
        fig = px.bar(daily, barmode='group', 
                    labels={'value': 'Count', 'day_name': 'Day'},
                    title="Transactions by Day of Week")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** Fraud patterns may vary by day of week")
    
    # Top Accounts
    st.header("🏆 Top Active Accounts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Active Senders")
        top_senders = df['sender_id'].value_counts().head(10)
        st.bar_chart(top_senders)
        
    with col2:
        st.subheader("Most Active Receivers")
        top_receivers = df['receiver_id'].value_counts().head(10)
        st.bar_chart(top_receivers)

def show_dashboard(df, models):
    st.title("📊 Dashboard")
    
    st.markdown("""
    **Real-time overview of transaction data and fraud metrics.**  
    Monitor key performance indicators and identify trends.
    """)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_txns = len(df)
    fraud_txns = df[df['is_fraud'] == 1].shape[0]
    fraud_rate = (fraud_txns / total_txns) * 100
    total_volume = df['amount'].sum()
    
    with col1:
        st.metric("Total Transactions", f"{total_txns:,}")
    with col2:
        st.metric("Fraud Cases", f"{fraud_txns:,}", delta=f"{fraud_rate:.2f}%", delta_color="inverse")
    with col3:
        st.metric("Total Volume", f"${total_volume:,.0f}")
    with col4:
        fraud_volume = df[df['is_fraud'] == 1]['amount'].sum()
        st.metric("Fraud Volume", f"${fraud_volume:,.0f}")
    
    st.markdown("---")
    
    # Time Series
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Transaction Timeline")
        df_daily = df.set_index('timestamp').resample('D').agg({'transaction_id': 'count', 'is_fraud': 'sum'})
        df_daily.columns = ['Total', 'Fraud']
        st.line_chart(df_daily)
        st.caption("Daily transaction volume with fraud overlay")
        
    with col2:
        st.subheader("💰 Amount Distribution by Type")
        fig = px.box(df, x="is_fraud", y="amount", 
                    labels={"is_fraud": "Transaction Type", "amount": "Amount ($)"},
                    color="is_fraud",
                    color_discrete_map={0: "#3b82f6", 1: "#ef4444"})
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Normal', 'Fraud'])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Box plot showing amount ranges")
    
    # Model Performance (if available)
    if models:
        st.header("🤖 Model Performance")
        st.markdown("**Current models are trained and ready for predictions.**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ XGBoost Model: Active")
            st.info("Supervised learning for known fraud patterns")
        with col2:
            st.success("✅ Isolation Forest Model: Active")
            st.info("Unsupervised anomaly detection")

def show_inference(models, df):
    st.title("⚡ Real-time Fraud Detection")
    
    st.markdown("""
    **Test the fraud detection models with custom transaction parameters.**  
    Enter transaction details below to get instant risk assessment.
    """)
    
    if models is None:
        st.error("❌ Models not found! Please train models first by running `python src/train_models.py`")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0, step=100.0)
        hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
        day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
        
        st.markdown("---")
        predict_btn = st.button("🔍 Analyze Transaction", use_container_width=True, type="primary")
        
    with col2:
        st.subheader("ℹ️ Understanding the Results")
        st.markdown("""
        **Fraud Probability (XGBoost)**  
        - 0-30%: Low Risk ✅
        - 30-70%: Medium Risk ⚠️
        - 70-100%: High Risk 🚨
        
        **Anomaly Detection (Isolation Forest)**  
        - Normal: Typical transaction pattern
        - Anomaly: Unusual behavior detected
        
        ---
        
        **Why use both models?**  
        - XGBoost catches **known** fraud types
        - Isolation Forest catches **new** patterns
        - Together they provide comprehensive coverage
        """)
    
    if predict_btn:
        # Prepare input
        input_data = pd.DataFrame({
            'amount': [amount],
            'hour': [hour],
            'day_of_week': [day_map[day]]
        })
        
        # Predictions
        prob = models['xgb'].predict_proba(input_data)[0][1]
        is_outlier = models['iso'].predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("🎯 Analysis Results")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.metric("Fraud Probability", f"{prob:.1%}")
            
            if prob > 0.7:
                st.error("🚨 **HIGH RISK** - Strong fraud indicators detected!")
                st.markdown("**Recommended Action:** Block transaction and flag for review")
            elif prob > 0.3:
                st.warning("⚠️ **MEDIUM RISK** - Some suspicious patterns")
                st.markdown("**Recommended Action:** Request additional verification")
            else:
                st.success("✅ **LOW RISK** - Transaction appears normal")
                st.markdown("**Recommended Action:** Approve transaction")
                
        with col_r2:
            anomaly_status = "🔴 Anomaly Detected" if is_outlier == -1 else "🟢 Normal Pattern"
            st.metric("Anomaly Status", anomaly_status)
            
            if is_outlier == -1:
                st.warning("⚠️ This transaction is unusual compared to typical patterns")
                st.markdown("**Note:** May indicate new fraud method or rare legitimate transaction")
            else:
                st.success("✅ Transaction follows expected patterns")
        
        # Additional Context
        st.markdown("---")
        st.subheader("📊 Context from Historical Data")
        
        # Compare with similar transactions
        similar = df[(df['amount'] > amount * 0.8) & (df['amount'] < amount * 1.2)]
        if len(similar) > 0:
            fraud_rate_similar = (similar['is_fraud'].sum() / len(similar)) * 100
            st.info(f"ℹ️ Among {len(similar)} similar transactions (±20% amount), **{fraud_rate_similar:.1f}%** were fraudulent")

def show_graph_analysis(df):
    st.title("🕸️ Network & Graph Analysis")
    
    st.markdown("""
    **Fraud Ring Detection using Graph Analytics**  
    Visualize transaction networks to identify suspicious patterns like circular transactions and mule account networks.
    """)
    
    st.markdown("""
    ### 🔍 What are Fraud Rings?
    
    Fraud rings are organized networks of accounts that transfer money in circular or layered patterns to:
    - Launder money
    - Obscure the source of funds
    - Cash out stolen funds through "mule" accounts
    
    **Graph analysis helps detect:**
    - Circular transaction patterns (A → B → C → A)
    - Hub accounts with many connections
    - Isolated clusters of suspicious activity
    """)
    
    # Sample selection
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider("Sample Size (for performance)", 50, 200, 100)
    with col2:
        show_only_fraud = st.checkbox("Show only fraud transactions", value=True)
    
    # Filter data
    if show_only_fraud:
        graph_df = df[df['is_fraud'] == 1].head(sample_size)
    else:
        graph_df = df.head(sample_size)
    
    if graph_df.empty:
        st.warning("No transactions to visualize. Try adjusting filters.")
        return
    
    # Build graph
    G = graph_utils.build_transaction_graph(graph_df)
    rings = graph_utils.detect_rings(G)
    
    st.markdown("---")
    
    # Graph Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes (Accounts)", G.number_of_nodes())
    with col2:
        st.metric("Edges (Transactions)", G.number_of_edges())
    with col3:
        st.metric("Detected Rings", len(rings))
    with col4:
        density = nx.density(G)
        st.metric("Network Density", f"{density:.3f}")
    
    if rings:
        st.success(f"✅ Found {len(rings)} potential fraud ring(s)")
        
        with st.expander("🔍 View Detected Rings"):
            for i, ring in enumerate(rings[:5]):  # Show first 5
                st.write(f"**Ring {i+1}:** {' → '.join(ring)} → {ring[0]}")
    else:
        st.info("ℹ️ No circular patterns detected in this sample")
    
    # Visualize
    st.subheader("🗺️ Network Visualization")
    highlight_nodes = list(set([n for r in rings for n in r])) if rings else []
    fig = graph_utils.visualize_graph_plotly(G, highlight_nodes=highlight_nodes)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("🔴 Red nodes = Part of detected fraud rings | 🔵 Blue nodes = Other accounts")

def main():
    # Sidebar Navigation
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["📚 About", "📊 Data Explorer", "📈 Dashboard", "⚡ Real-time Detection", "🕸️ Graph Analysis"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Quick Links:**
    - [Documentation](#)
    - [GitHub Repo](#)
    - [Report Issue](#)
    """)
    
    # Load data
    df = load_data()
    models = load_models()
    
    if df.empty and page not in ["📚 About"]:
        st.error("⚠️ Data not found! Please run `python src/data_generator.py` first.")
        st.stop()
    
    # Route to pages
    if page == "📚 About":
        show_about()
    elif page == "📊 Data Explorer":
        show_data_explorer(df)
    elif page == "📈 Dashboard":
        show_dashboard(df, models)
    elif page == "⚡ Real-time Detection":
        show_inference(models, df)
    elif page == "🕸️ Graph Analysis":
        show_graph_analysis(df)

if __name__ == "__main__":
    main()
