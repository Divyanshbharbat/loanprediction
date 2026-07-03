import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

import db_helper
import ai_helper

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="FinQuest AI | Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background with a high-end deep-indigo/dark-space radial gradient */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #090d16 60%, #020617 100%) !important;
        color: #f8fafc !important;
    }
    
    /* Glassmorphism Containers & Cards */
    div.stCard, div[data-testid="stVerticalBlock"] > div:has(div.card-content) {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        padding: 28px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Styled Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
    }
    
    /* Elegant vibrant gradients for main headers */
    .gradient-text {
        background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 50%, #f472b6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 1.2rem !important;
        text-shadow: 0 2px 20px rgba(167, 139, 250, 0.1) !important;
    }
    
    .section-title {
        color: #c084fc !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(167, 139, 250, 0.3) !important;
        padding-bottom: 10px !important;
        margin-bottom: 20px !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Custom Assessment Results Glow Cards */
    .approval-card-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.08) 100%) !important;
        border: 2px solid rgba(16, 185, 129, 0.5) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        text-align: center !important;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.25) !important;
    }
    .approval-card-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.08) 100%) !important;
        border: 2px solid rgba(239, 68, 68, 0.5) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        text-align: center !important;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.25) !important;
    }
    
    /* Streamlit Form Styling */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Standard Buttons styling */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(139, 92, 246, 0.6) !important;
        color: #ffffff !important;
    }
    .stButton>button:active {
        transform: translateY(0px) !important;
    }
    
    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #060911 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Label Clarity & Visibility */
    .stWidgetFormLabel label, label[data-testid="stWidgetLabel"] {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* Input Fields Styling */
    .stTextInput input, .stNumberInput input, .stSelectbox [role="combobox"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput input:hover, .stNumberInput input:hover, .stSelectbox [role="combobox"]:hover {
        border-color: rgba(139, 92, 246, 0.5) !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox [role="combobox"]:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
    }
    
    /* Metric Cards readability */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        padding: 18px 24px !important;
        text-align: center !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }
    
    /* Tabs custom styling */
    button[data-baseweb="tab"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        padding: 12px 20px !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #c084fc !important;
        border-color: #8b5cf6 !important;
        background-color: rgba(139, 92, 246, 0.05) !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the ML Pipeline
@st.cache_resource
def load_ml_pipeline():
    try:
        with open("loan_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("ML model pipeline file 'loan_pipeline.pkl' not found. Please run the training script first.")
        return None

pipeline = load_ml_pipeline()

# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
    <h2 style="margin: 0; color: #a78bfa; font-weight: 800;">🏦 FinQuest AI</h2>
    <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #9ca3af;">Smart Loan Approval Prediction Platform</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "NAVIGATE SYSTEM",
    ["🎯 Loan Predictor", "📊 Analytics & History", "📚 About System"],
    index=0
)

# Loan Predictor Hub
if menu == "🎯 Loan Predictor":
    st.markdown('<h1 class="gradient-text">🎯 Loan Predictor Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9ca3af; margin-top:-15px; margin-bottom: 25px;">Enter applicant credentials to calculate loan eligibility and generate AI financial advice.</p>', unsafe_allow_html=True)
    
    if pipeline is None:
        st.warning("ML Pipeline is offline. Please make sure loan_pipeline.pkl is trained and exists.")
    else:
        # Form Container
        st.markdown('<div class="card-content">', unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 class="section-title">👤 Personal Profile</h3>', unsafe_allow_html=True)
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Marital Status", ["Married (Yes)", "Single (No)"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Employment Type", ["Salaried (No)", "Self-Employed (Yes)"])
                age = st.slider("Age of Applicant", min_value=18, max_value=80, value=35)
                
            with col2:
                st.markdown('<h3 class="section-title">💰 Financial Profile & Loan</h3>', unsafe_allow_html=True)
                # Map slider inputs for income
                app_income = st.number_input("Applicant Monthly Salary ($)", min_value=0, max_value=100000, value=5000, step=100)
                coapp_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, max_value=50000, value=0, step=100)
                credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700, help="Scores >= 650 generally represent prime history.")
                existing_debt = st.number_input("Existing Monthly Debt Payments ($)", min_value=0, max_value=20000, value=500, step=50)
                
                loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=1000000, value=150000, step=5000)
                loan_term = st.selectbox("Loan Term (Months)", [360, 240, 180, 120, 84, 60, 36, 12])
                property_area = st.selectbox("Property Location Area", ["Urban", "Semiurban", "Rural"])
                
            # Submit Button
            submitted = st.form_submit_button("⚡ Predict Loan Approval")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            # Process Form Values
            # Map forms to database/ML format
            db_married = "Yes" if "Married" in married else "No"
            db_self_employed = "Yes" if "Self-Employed" in self_employed else "No"
            db_dependents = "3" if dependents == "3+" else dependents
            
            # Map credit score to Credit_History string "1" or "0"
            db_credit_history = "1" if credit_score >= 650 else "0"
            
            # Convert loan amount to thousands for the ML model
            ml_loan_amount = loan_amount / 1000.0
            
            # Prepare dataframe for pipeline
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Married": db_married,
                "Dependents": db_dependents,
                "Education": education,
                "Self_Employed": db_self_employed,
                "ApplicantIncome": app_income,
                "CoapplicantIncome": coapp_income,
                "LoanAmount": ml_loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": db_credit_history,
                "Property_Area": property_area
            }])
            
            # Predict
            with st.spinner("Processing models and querying Gemini AI..."):
                try:
                    # Get prediction and probabilities
                    pred_class = pipeline.predict(input_df)[0]
                    pred_probs = pipeline.predict_proba(input_df)[0]
                    
                    # Convert to visual format
                    status = "Approved" if pred_class == 1 else "Rejected"
                    
                    # Compute confidence score and risk score
                    # For Approved, confidence is approval probability. For Rejected, it is rejection probability.
                    confidence = pred_probs[1] * 100 if pred_class == 1 else pred_probs[0] * 100
                    approval_prob = pred_probs[1]
                    
                    # Risk score is (1 - approval probability) * 100
                    risk_score = (1 - approval_prob) * 100
                    
                    # Gather applicant dictionary for AI helpers
                    applicant_data = {
                        "applicant_income": app_income,
                        "coapplicant_income": coapp_income,
                        "credit_score": credit_score,
                        "dependents": dependents,
                        "education": education,
                        "self_employed": db_self_employed,
                        "married": db_married,
                        "loan_amount": ml_loan_amount,
                        "loan_term": loan_term,
                        "property_area": property_area,
                        "existing_debt": existing_debt,
                        "age": age
                    }
                    
                    # Call LLM helpers
                    llm_explanation = ai_helper.generate_explanation(applicant_data, status, confidence)
                    ai_advice = ai_helper.generate_financial_advice(applicant_data, status)
                    
                    # Save to history database
                    db_helper.save_prediction(
                        username='admin',
                        gender=gender,
                        married=db_married,
                        dependents=dependents,
                        education=education,
                        self_employed=db_self_employed,
                        applicant_income=app_income,
                        coapplicant_income=coapp_income,
                        loan_amount=loan_amount,
                        loan_term=loan_term,
                        credit_score=credit_score,
                        property_area=property_area,
                        prediction_result=status,
                        confidence_score=confidence,
                        llm_explanation=llm_explanation,
                        ai_advice=ai_advice
                    )
                    
                    # Display Predictions Results Screen
                    st.markdown('<br><h2 class="section-title">📊 Assessment Results</h2>', unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns([3, 2])
                    
                    with res_col1:
                        # Success/Failure Card
                        if status == "Approved":
                            st.markdown(f"""
                            <div class="approval-card-approved">
                                <h1 style="margin:0; font-size:4rem;">✅ APPROVED</h1>
                                <p style="font-size:1.2rem; color: #a7f3d0; margin-top:10px;">
                                    Applicant meets core solvency parameters. Approval confidence is <b>{confidence:.1f}%</b>.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="approval-card-rejected">
                                <h1 style="margin:0; font-size:4rem;">❌ REJECTED</h1>
                                <p style="font-size:1.2rem; color: #fca5a5; margin-top:10px;">
                                    Application fails standard lending checks. Rejection confidence is <b>{confidence:.1f}%</b>.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # AI Explanation Card
                        st.markdown(f"""
                        <div class="card-content" style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius:12px; padding:20px;">
                            <h3 style="color:#60a5fa; margin-top:0; display:flex; align-items:center;">🤖 AI Decision Explanation</h3>
                            <p style="font-size:1.1rem; line-height:1.6; color:#e5e7eb; font-style:italic;">
                                "{llm_explanation}"
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI Advice Card
                        advice_html = ""
                        for bullet in ai_advice.split("\n"):
                            if bullet.strip():
                                # remove markdown bullet asterisk or dash if present
                                clean_bullet = bullet.strip().lstrip("*").lstrip("-").strip()
                                advice_html += f"<li style='margin-bottom:10px; color:#e5e7eb;'>{clean_bullet}</li>"
                                
                        st.markdown(f"""
                        <div class="card-content" style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius:12px; padding:20px; margin-top:20px;">
                            <h3 style="color:#a78bfa; margin-top:0;">💡 Personalized AI Recommendations</h3>
                            <ul style="padding-left:20px; font-size:1.05rem; line-height:1.5;">
                                {advice_html if advice_html else "<li>Maintain good credit practices.</li>"}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        # Risk Meter Chart
                        st.markdown('<div class="card-content" style="text-align:center;">', unsafe_allow_html=True)
                        st.markdown('<h3 style="margin-top:0;">🛡️ Portfolio Risk Meter</h3>', unsafe_allow_html=True)
                        
                        risk_level = "LOW RISK"
                        risk_color = "#10b981"
                        if risk_score > 70:
                            risk_level = "HIGH RISK"
                            risk_color = "#ef4444"
                        elif risk_score > 30:
                            risk_level = "MEDIUM RISK"
                            risk_color = "#f59e0b"
                            
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = risk_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Risk Index: {risk_level}", 'font': {'size': 20, 'color': risk_color}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9ca3af"},
                                'bar': {'color': risk_color},
                                'bgcolor': "rgba(255,255,255,0.05)",
                                'borderwidth': 2,
                                'bordercolor': "rgba(255,255,255,0.1)",
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                                ],
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#f3f4f6", 'family': "Outfit"},
                            height=280,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Debt-to-income stats
                        total_income = app_income + coapp_income
                        dti = (existing_debt / total_income * 100) if total_income > 0 else 0
                        loan_to_income = (loan_amount / (total_income * 12) * 100) if total_income > 0 else 0
                        
                        st.markdown(f"""
                        <div style="text-align: left; margin-top:20px; font-size:0.95rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top:15px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#9ca3af;">Debt-to-Income (DTI) Ratio:</span>
                                <span style="font-weight:600; color:{'#10b981' if dti < 36 else '#ef4444' if dti > 45 else '#f59e0b'};">{dti:.1f}%</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#9ca3af;">Loan-to-Annual-Income Ratio:</span>
                                <span style="font-weight:600; color:{'#10b981' if loan_to_income < 300 else '#ef4444'};">{loan_to_income:.1f}%</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <span style="color:#9ca3af;">Qualifying Credit Tier:</span>
                                <span style="font-weight:600; color:#60a5fa;">{ 'Super Prime (720+)' if credit_score >= 720 else 'Prime (650-719)' if credit_score >= 650 else 'Subprime (<650)' }</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Inference pipeline failed: {str(e)}")

# Analytics Dashboard & History
elif menu == "📊 Analytics & History":
    st.markdown('<h1 class="gradient-text">📊 Analytics & History</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9ca3af; margin-top:-15px; margin-bottom: 25px;">Track your evaluation database, review charts, and browse past predictions.</p>', unsafe_allow_html=True)
    
    # Load History Data
    df_history = db_helper.get_user_history('admin')
    
    if df_history.empty:
        st.info("You don't have any prediction history yet. Run a prediction in the Loan Predictor tab to see history analytics!")
    else:
        # Overview Cards
        total_runs = len(df_history)
        approvals = df_history[df_history['prediction_result'] == "Approved"]
        total_approvals = len(approvals)
        approval_rate = (total_approvals / total_runs) * 100 if total_runs > 0 else 0
        avg_score = df_history['credit_score'].mean()
        avg_loan = df_history['loan_amount'].mean()
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Queries", f"{total_runs}")
        with m_col2:
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        with m_col3:
            st.metric("Avg. Credit Score", f"{avg_score:.0f}")
        with m_col4:
            st.metric("Avg. Loan Request", f"${avg_loan:,.0f}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for Visualizations vs Data Table
        dash_tab1, dash_tab2 = st.tabs(["📉 Visual Insights", "📜 History Registry"])
        
        with dash_tab1:
            st.markdown('<div class="card-content">', unsafe_allow_html=True)
            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                # Result Distribution Pie
                fig_pie = px.pie(
                    df_history, 
                    names='prediction_result', 
                    title='Prediction Outcomes Distribution',
                    hole=0.4,
                    color='prediction_result',
                    color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'}
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#f3f4f6", 'family': "Outfit"},
                    legend=dict(orientation="h", y=-0.1)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Credit Score Histogram
                fig_hist = px.histogram(
                    df_history, 
                    x='credit_score', 
                    title='Distribution of Applicant Credit Scores',
                    color='prediction_result',
                    color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'},
                    nbins=15
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#f3f4f6", 'family': "Outfit"},
                    xaxis_title="Credit Score",
                    yaxis_title="Count",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with v_col2:
                # Scatter Plot: Income vs Loan Amount
                df_history['combined_income'] = df_history['applicant_income'] + df_history['coapplicant_income']
                fig_scatter = px.scatter(
                    df_history,
                    x='combined_income',
                    y='loan_amount',
                    color='prediction_result',
                    size='confidence_score',
                    hover_data=['credit_score', 'education'],
                    title='Combined Income vs. Requested Loan Amount',
                    labels={'combined_income': 'Total Income ($/mo)', 'loan_amount': 'Loan Amount ($)'},
                    color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'}
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#f3f4f6", 'family': "Outfit"},
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Bar Chart: Loan term distribution
                fig_term = px.bar(
                    df_history.groupby(['loan_term', 'prediction_result']).size().reset_index(name='count'),
                    x='loan_term',
                    y='count',
                    color='prediction_result',
                    title='Loan Requests by Repayment Term (Months)',
                    barmode='group',
                    color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'}
                )
                fig_term.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#f3f4f6", 'family': "Outfit"},
                    xaxis_title="Term in Months",
                    yaxis_title="Count",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_term, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with dash_tab2:
            st.markdown('<div class="card-content">', unsafe_allow_html=True)
            st.markdown('<h3 style="margin-top:0;">📜 Your Prediction Logs</h3>', unsafe_allow_html=True)
            st.write("Below is a log of all predictions run under your account. Select a row to inspect full AI reports.")
            
            # Clean up history view
            display_cols = ['id', 'created_at', 'loan_amount', 'loan_term', 'credit_score', 'prediction_result', 'confidence_score']
            df_display = df_history[display_cols].copy()
            df_display.columns = ['ID', 'Date/Time', 'Loan Requested ($)', 'Term (mo)', 'Credit Score', 'Result', 'Confidence (%)']
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Record Selection inspect panel
            record_id = st.selectbox("Select Prediction ID to Inspect Detail AI Reports", df_history['id'].tolist())
            if record_id:
                selected_rec = df_history[df_history['id'] == record_id].iloc[0]
                
                st.markdown("---")
                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.markdown(f"#### 👤 Profile Details (Record #{record_id})")
                    st.write(f"- **Salary:** ${selected_rec['applicant_income']:,}/mo")
                    st.write(f"- **Co-applicant Salary:** ${selected_rec['coapplicant_income']:,}/mo")
                    st.write(f"- **Credit Score:** {selected_rec['credit_score']}")
                    st.write(f"- **Education:** {selected_rec['education']}")
                    st.write(f"- **Self-Employed:** {selected_rec['self_employed']}")
                    st.write(f"- **Loan request:** ${selected_rec['loan_amount']:,} over {selected_rec['loan_term']:.0f} months")
                    st.write(f"- **Property Area:** {selected_rec['property_area']}")
                    
                    status_badge = "🟢 APPROVED" if selected_rec['prediction_result'] == "Approved" else "🔴 REJECTED"
                    st.markdown(f"- **System Result:** **{status_badge}** ({selected_rec['confidence_score']:.1f}% Confidence)")
                    
                with det_col2:
                    st.markdown("#### 🤖 Saved AI Report")
                    st.markdown(f"**Explanation:**")
                    st.info(selected_rec['llm_explanation'])
                    st.markdown(f"**Financial Recommendations:**")
                    
                    saved_advice_html = ""
                    for bullet in selected_rec['ai_advice'].split("\n"):
                        if bullet.strip():
                            clean_bullet = bullet.strip().lstrip("*").lstrip("-").strip()
                            saved_advice_html += f"<li style='margin-bottom:6px; font-size:0.95rem;'>{clean_bullet}</li>"
                    st.markdown(f"<ul>{saved_advice_html}</ul>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# About System Page
elif menu == "📚 About System":
    st.markdown('<h1 class="gradient-text">📚 About Loan Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<div class="card-content">', unsafe_allow_html=True)
    
    st.markdown("""
    ### ⚙️ System Architecture
    This application is an end-to-end Machine Learning (ML) and Large Language Model (LLM) hybrid deployment designed to automate credit analysis.
    
    ```
    ┌──────────────────────┐
    │  Streamlit Frontend  │
    └──────────┬───────────┘
               │ (Applicant parameters)
               ▼
    ┌──────────────────────┐
    │  Random Forest ML    ├────────► [Predicts Approval Status & Confidence]
    └──────────┬───────────┘
               │ (Solvency prediction + Applicant stats)
               ▼
    ┌──────────────────────┐
    │  Google Gemini LLM   ├────────► [Generates Custom Decision Explanations & Advice]
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   SQLite Database    ├────────► [Saves Query History and User Authentication]
    └──────────────────────┘
    ```
    
    ### 🛡️ Technologies Used
    - **UI Framework:** Streamlit (Python Web App Interface)
    - **Database Manager:** SQLite3 (Local file storage for user management & audit logging)
    - **ML Framework:** Scikit-learn (Data cleaning pipeline, imputer, categorical one-hot-encoders, Random Forest Classifier)
    - **AI Engine:** Google Gemini API (`gemini-2.5-flash` model via the new `google-genai` SDK)
    - **Visualizations:** Plotly Express & Plotly Graph Objects (Interactive charts and risk gauge)
    
    ### 📊 Random Forest Classifier Statistics
    The model was trained on the Kaggle Loan Approval dataset:
    - **Test Classification Accuracy:** `85.37%`
    - **Precision on Rejected Applications:** `92%`
    - **Recall on Approved Applications:** `98%`
    - **Preprocessing:** Categorical column mode imputers + standard scalers + label transformers are bundled in a serialized pipeline `loan_pipeline.pkl`.
    
    ### 👥 Developers Note
    Developed as a beginner-to-intermediate showcase project showing how business logic models (like Scikit-learn classifiers) can be seamlessly paired with generative AI assistants (like Google Gemini) to build cohesive, client-facing financial software.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
