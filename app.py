# app.py - DataSync Analytics Predictive Maintenance Dashboard (FAST DEPLOY)
# Copy-paste ready. No external deps beyond basics. Deploys <2 mins on Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="DataSync Analytics")

# McKinsey Dark Theme
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; color: white; border-radius: 10px;}
    .stMetric > div > div > div {color: white;}
    </style>
""", unsafe_allow_html=True)

# Predictive Engine (No external ML libs needed)
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2025-03-01', periods=n, freq='H')
    df = pd.DataFrame({
        'Date': dates,
        'Error_Rate': np.clip(np.random.exponential(0.1, n), 0.01, 0.4),
        'Sync_Latency': np.random.exponential(3, n),
        'Insight_Delay': np.random.exponential(1, n),
        'Maintenance_Cost': np.random.lognormal(10, 0.3, n),
        'Scalability_Score': np.random.uniform(2, 9, n),
        'Pred_Savings': np.random.lognormal(9, 0.5, n),
        'Tech_Comfort': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.4, 0.3])
    })
    # Realistic correlations
    df['Error_Rate'] = np.clip(df['Sync_Latency'] * 0.05 + np.random.normal(0, 0.03, n), 0.01, 0.4)
    df['Pred_Savings'] = df['Maintenance_Cost'] * df['Error_Rate'] * 4
    return df

# Fast Forecasting (Simple trend + noise)
def forecast_maintenance(df, periods=168):  # 7 days hourly
    last_week = df.tail(168).copy()
    trend = last_week['Maintenance_Cost'].mean() * np.linspace(1, 0.92, periods)
    forecast = pd.DataFrame({
        'Date': pd.date_range(df['Date'].max() + timedelta(hours=1), periods=periods, freq='H'),
        'Forecast_Cost': trend + np.random.normal(0, trend*0.1, periods),
        'Lower': trend * 0.9,
        'Upper': trend * 1.1
    })
    return forecast

# Anomaly Detection (Simple statistical)
def detect_anomalies(df):
    threshold = df['Error_Rate'].quantile(0.95)
    return df[df['Error_Rate'] > threshold][['Date', 'Error_Rate', 'Sync_Latency']]

# Load Data
df = generate_data()
forecast = forecast_maintenance(df)
anomalies = detect_anomalies(df)

st.title("🛠️ DataSync Analytics")
st.markdown("**Predictive Maintenance for E-commerce System Silos**")

# KPIs Row 1
col1, col2, col3, col4 = st.columns(4)
col1.metric("Error Rate", f"{df['Error_Rate'].mean():.1%}", "-1.2%")
col2.metric("Sync Latency", f"{df['Sync_Latency'].mean():.1f}min", "-0.3min")
col3.metric("Pred. Savings", f"${df['Pred_Savings'].sum():,.0f}", "+15%")
col4.metric("Health Score", f"{df['Scalability_Score'].mean():.1f}/10", "+0.4")

# Main Dashboard
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("📈 System Health Overview")
    
    # Maintenance Forecast
    fig1 = px.line(forecast, x='Date', y=['Forecast_Cost', 'Lower', 'Upper'], 
                   title="Next 7 Days: Maintenance Cost Prediction",
                   color_discrete_sequence=['#e91e63', '#9c27b0', '#9c27b0'])
    st.plotly_chart(fig1, use_container_width=True)
    
    # Error vs Savings
    fig2 = px.scatter(df.tail(500), x='Error_Rate', y='Pred_Savings', 
                      color='Tech_Comfort', size='Scalability_Score',
                      title="Error Rate vs Savings Potential",
                      hover_data=['Sync_Latency'])
    st.plotly_chart(fig2, use_container_width=True)

with colB:
    st.subheader("🚨 Live Alerts")
    
    # Health Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=df['Scalability_Score'].mean(),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Health"},
        delta={'reference': 6.0},
        gauge={'axis': {'range': [None, 10]},
               'bar': {'color': "#4caf50"},
               'steps': [{'range': [0, 4], 'color': "#ff5252"},
                        {'range': [4, 7], 'color': "#ff9800"},
                        {'range': [7, 10], 'color': "#4caf50"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8}}))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Top Anomalies
    st.dataframe(anomalies.head(10), use_container_width=True)
    
    if len(anomalies) > 5:
        st.error(f"🚨 {len(anomalies)} anomalies detected!")

# Correlation Matrix
st.subheader("🔗 Metric Relationships")
corr = df[['Error_Rate', 'Sync_Latency', 'Insight_Delay', 'Maintenance_Cost']].corr()
fig3 = px.imshow(corr, title="Correlation Heatmap", color_continuous_scale='RdBu_r', aspect="auto")
st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*DataSync Analytics | Dubai E-commerce Focus | McKinsey Data Analytics*")
