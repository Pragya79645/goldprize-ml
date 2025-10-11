import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="💰",
    layout="centered"
)

# Custom CSS for better styling and alignment
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding: 1rem 2rem;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    /* Header Styling */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .main-title {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF8C00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: #999;
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .subtitle {
            color: #aaa;
        }
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* Input Section */
    .input-section {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #f0f0f0;
    }
    
    .section-title {
        color: #333;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* Input Fields Alignment */
    .stNumberInput > div > div > input {
        text-align: left;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        border: 2px solid #e8e8e8;
        transition: all 0.3s ease;
        padding: 0.75rem 1rem;
        background: #fafafa;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #FFD700;
        box-shadow: 0 0 0 4px rgba(255, 215, 0, 0.15);
        background: white;
    }
    
    .stNumberInput > label {
        font-weight: 600;
        color: #333;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-bottom: 0.5rem;
    }
    
    /* Input control buttons */
    .stNumberInput button {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stNumberInput button:hover {
        background: #FFD700;
        border-color: #FFD700;
        color: #000;
    }
    
    /* Button Styling */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 2.5rem 0 !important;
    }
    
    .stButton > div {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    .stButton>button {
        width: 100% !important;
        max-width: 450px !important;
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 1.1rem 3rem !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 8px 25px rgba(255, 165, 0, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        display: block !important;
        margin: 0 auto !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(255, 165, 0, 0.5) !important;
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Prediction Box */
    .prediction-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        width: 100%;
        max-width: 600px;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-label {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #FFD700;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin: 1rem 0;
        line-height: 1;
    }
    
    .prediction-subtext {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        margin-top: 1rem;
        font-style: italic;
    }
    
    /* Metrics Section */
    .metrics-container {
        margin: 2rem 0;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 15px;
    }
    
    .metrics-title {
        text-align: center;
        color: #333;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFD700;
        text-align: center;
    }
    
    div[data-testid="stMetricLabel"] {
        text-align: center;
        font-weight: 600;
        color: #555;
        font-size: 0.9rem;
    }
    
    /* Info Box */
    .stExpander {
        background: white;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #FFD700;
    }
    
    .insight-title {
        font-weight: 700;
        color: #333;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-content {
        color: #555;
        line-height: 1.6;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .stExpander {
            background: #1e1e1e;
            border-color: #3a3a3a;
        }
        
        .input-section {
            background: #1e1e1e;
            border-color: #3a3a3a;
        }
        
        .stNumberInput > div > div > input {
            background: #2a2a2a;
            border-color: #3a3a3a;
            color: #e0e0e0;
        }
        
        .stNumberInput > div > div > input:focus {
            background: #1e1e1e;
        }
        
        .stNumberInput > label {
            color: #e0e0e0;
        }
        
        .section-title {
            color: #e0e0e0;
        }
        
        .metrics-container {
            background: #1e1e1e;
            border: 1px solid #3a3a3a;
        }
        
        .metrics-title {
            color: #e0e0e0;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #aaa;
        }
        
        .insight-card {
            background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
            border-left-color: #FFD700;
        }
        
        .insight-title {
            color: #e0e0e0;
        }
        
        .insight-content {
            color: #aaa;
        }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
    
    .footer em {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        text-align: center;
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
try:
    model = pickle.load(open('regressor_model.pkl', 'rb'))
    model_loaded = True
except:
    model_loaded = False

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">💰 Gold Price Predictor</h1>
        <p class="subtitle">Powered by Advanced Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model file not found. Please ensure 'regressor_model.pkl' is in the same directory.")
    st.stop()

# Information section
with st.expander("ℹ️ About This Application & How It Works"):
    st.markdown("""
    ### 🎯 Purpose
    This application uses a **Random Forest Regression** machine learning model to predict gold prices based on key market indicators. 
    It demonstrates how interconnected global markets influence precious metal valuations.
    
    ### 🧠 The Model
    The predictor uses **supervised learning** trained on historical market data. The algorithm learns complex, non-linear relationships 
    between input features and gold prices that aren't immediately obvious to human observers.
    
    ### 📊 Input Features Explained
    
    **1. SPX (S&P 500 Index)** 📈
    - Represents the overall health of the US stock market
    - **Inverse relationship**: When stocks fall, investors often move to gold as a "safe haven"
    - Gold typically rises during market uncertainty and economic downturns
    
    **2. USO (Oil Price)** 🛢️
    - Crude oil is denominated in US dollars
    - **Positive correlation**: Rising oil prices often indicate inflation, which increases gold's appeal
    - Both commodities are influenced by geopolitical events
    
    **3. SLV (Silver Price)** 🥈
    - Silver and gold are both precious metals with similar market drivers
    - **Strong positive correlation**: They tend to move together due to shared factors
    - Silver is more industrial, gold is more of a store of value
    
    **4. EUR/USD Exchange Rate** 💱
    - Gold is priced in US dollars globally
    - **Inverse relationship**: A weaker dollar makes gold cheaper for foreign buyers, increasing demand
    - Currency fluctuations directly impact gold's relative value
    
    ### 🔬 How the Prediction Works
    
    1. **Input Processing**: Your entered values are normalized to match the training data scale
    2. **Feature Weighting**: The model assigns importance to each indicator based on learned patterns
    3. **Ensemble Prediction**: Multiple decision trees vote on the predicted price
    4. **Output**: The model returns a price prediction in USD, which is converted to INR
    
    ### 📈 Why These Variables Matter
    
    Gold prices don't exist in isolation. They're influenced by:
    - **Economic uncertainty** → Investors seek gold as a hedge
    - **Inflation expectations** → Gold maintains purchasing power
    - **Currency movements** → Dollar strength affects gold denominated prices
    - **Commodity markets** → Interconnected global trading relationships
    
    ### 🎓 Learning Insights
    
    This project demonstrates:
    - Real-world application of regression algorithms
    - Feature engineering for financial prediction
    - Integration of multiple market indicators
    - Building user-friendly ML interfaces with Streamlit
    """)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Input section
st.markdown('<p class="section-title">📊 Market Indicators</p>', unsafe_allow_html=True)

# Create perfectly aligned input fields
col1, col2 = st.columns(2, gap="large")

with col1:
    spx = st.number_input("📈 S&P 500 Index (SPX)", min_value=0.0, max_value=10000.0, value=4500.0, step=10.0, help="Current S&P 500 index value", key="spx")
    slv = st.number_input("🥈 Silver Price (SLV)", min_value=0.0, max_value=100.0, value=24.0, step=0.5, help="Current silver price in USD", key="slv")

with col2:
    uso = st.number_input("🛢️ Oil Price (USO)", min_value=0.0, max_value=200.0, value=75.0, step=1.0, help="Current crude oil price", key="uso")
    eur_usd = st.number_input("💱 EUR/USD Rate", min_value=0.0, max_value=2.0, value=1.08, step=0.01, help="Current Euro to USD exchange rate", key="eur")

# USD to INR input
usd_inr = st.number_input("💵 USD to INR Rate", min_value=0.0, max_value=500.0, value=83.0, step=0.1, help="Current USD to INR exchange rate", key="usd_inr")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Centered predict button
predict_button = st.button("🔮 Predict Gold Price", use_container_width=True)

# Prediction results
if predict_button:
    with st.spinner("🔍 Analyzing market data..."):
        import time
        time.sleep(0.5)  # Brief pause for effect
        
        input_data = np.array([[spx, uso, slv, eur_usd]])
        prediction_usd = model.predict(input_data)[0]
        prediction_inr = prediction_usd * usd_inr
        
        st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-box">
                <div class="prediction-label">Predicted Gold Price</div>
                <div class="prediction-value">${prediction_usd:,.2f} / ₹{prediction_inr:,.2f}</div>
                <div class="prediction-subtext">Based on current market indicators</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics summary
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.markdown('<p class="metrics-title">📌 Input Summary</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("SPX", f"{spx:,.0f}")
        with col2:
            st.metric("USO", f"${uso:.2f}")
        with col3:
            st.metric("SLV", f"${slv:.2f}")
        with col4:
            st.metric("EUR/USD", f"{eur_usd:.4f}")
        with col5:
            st.metric("USD/INR", f"₹{usd_inr:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Understanding Your Prediction")
        
        # Calculate relative positions
        spx_high = spx > 4500
        uso_high = uso > 75
        slv_high = slv > 24
        eur_strong = eur_usd > 1.08
        
        # Market sentiment analysis
        st.markdown("#### 🔍 Market Sentiment Analysis")
        
        if spx_high:
            st.markdown("• **Strong stock market** may reduce gold's safe-haven appeal")
        else:
            st.markdown("• **Weak stock market** may increase demand for gold as a hedge")
            
        if uso_high:
            st.markdown("• **High oil prices** suggest inflationary pressure, supporting gold prices")
        else:
            st.markdown("• **Lower oil prices** indicate reduced inflation concerns")
            
        if slv_high:
            st.markdown("• **Silver strength** indicates bullish precious metals sentiment")
        else:
            st.markdown("• **Silver weakness** may signal bearish metals market")
            
        if eur_strong:
            st.markdown("• **Strong Euro** (weak dollar) typically supports higher gold prices")
        else:
            st.markdown("• **Weak Euro** (strong dollar) may pressure gold prices")
        
        # Calculation breakdown
        st.markdown("#### 🧮 How the Calculation Works")
        
        st.markdown(f"""
        **Step 1: Feature Input**  
        The model receives 4 normalized features: [SPX: {spx}, USO: {uso}, SLV: {slv}, EUR/USD: {eur_usd}]
        
        **Step 2: Random Forest Processing**  
        • Multiple decision trees (typically 100+) analyze the inputs  
        • Each tree predicts based on patterns learned from historical data  
        • Trees consider feature interactions (e.g., high oil + weak dollar = higher gold)
        
        **Step 3: Ensemble Averaging**  
        • All tree predictions are averaged to reduce overfitting  
        • This provides a robust, stable prediction: **${prediction_usd:,.2f}**
        
        **Step 4: Currency Conversion**  
        • USD prediction × INR exchange rate ({usd_inr}) = **₹{prediction_inr:,.2f}**
        """)
        
        # Why this matters
        st.markdown("#### 🎯 Practical Applications")
        
        st.markdown("""
        **For Investors:** Understand how macroeconomic factors influence gold prices  
        **For Traders:** Identify potential entry/exit points based on market conditions  
        **For Students:** Learn how ML models process multiple variables for predictions  
        **For Researchers:** Study correlation patterns between different asset classes
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>⚠️ <em>Disclaimer: This prediction is for educational and informational purposes only.<br>
    Always consult professional financial advisors before making investment decisions.</em></p>
    <p style="margin-top: 1rem; color: #999; font-size: 0.85rem;">
    Built with Streamlit • ML Model: Random Forest Regression • Data-driven insights for learning
    </p>
</div>
""", unsafe_allow_html=True)