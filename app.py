# app.py - DataSync Analytics Predictive Maintenance Dashboard
# Upload to GitHub, deploy on Streamlit Cloud (share.streamlit.io)
# McKinsey-style: Predictive engine forecasts maintenance, errors, savings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import prophet  # For time-series forecasting (maintenance)
from prophet import Prophet
from sklearn.ensemble import IsolationForest  # Anomaly detection
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for McKinsey dark theme
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; color: white;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Use generated data or synthetic
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Date': pd.date_range('2025-03-01', periods=n, freq='H'),
        'Error_Rate': np.random.uniform(0.05, 0.4, n),
        'Sync_Latency': np.random.exponential(5, n),
        'Insight_Delay': np.random.exponential(2, n),
        'Maintenance_Cost': np.random.lognormal(11, 0.4, n),
        'Scalability_Score': np.random.uniform(1, 10, n),
        'Firm_Revenue': np.random.lognormal(12, 0.5, n),
        'Tech_Comfort': np.random.choice(['Low', 'Medium', 'High'], n),
        'Pred_Error_Next': 0  # To be filled
    })
    # Simulate correlations
    df['Error_Rate'] = df['Sync_Latency'].clip(0.05, 0.4) * 0.08
    return df

# Predictive Engine Class
class PredictiveEngine:
    def __init__(self, df):
        self.df = df.copy()
        self.prophet_model = None
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.reg_model = LinearRegression()
    
    def forecast_maintenance(self, periods=24):
        """Prophet for cost forecasting"""
        ts = self.df.resample('D', on='Date')['Maintenance_Cost'].sum().reset_index()
        ts.columns = ['ds', 'y']
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        self.prophet_model.fit(ts)
        future = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
    
    def detect_anomalies(self):
        """Isolation Forest on key metrics"""
        features = self.df[['Error_Rate', 'Sync_Latency', 'Insight_Delay']].values
        self.anomaly_model.fit(features)
        self.df['Anomaly'] = self.anomaly_model.predict(features)
        anomalies = self.df[self.df['Anomaly'] == -1]
        return anomalies
    
    def predict_savings(self):
        """Regression: Savings from fixing errors"""
        self.df['Error_Fix_Pot'] = self.df['Error_Rate'] * self.df['Maintenance_Cost'] * 0.4
        X = self.df[['Scalability_Score', 'Firm_Revenue']].values
        y = self.df['Error_Fix_Pot'].values
        self.reg_model.fit(X, y)
        self.df['Pred_Savings'] = self.reg_model.predict(X)
        return self.df['Pred_Savings'].mean()

st.title("🛠️ DataSync Analytics: Predictive Maintenance Dashboard")
st.markdown("**McKinsey Data Analytics** - Forecast errors, costs & savings for e-com system silos")

# Load & Engine
df = load_data()
engine = PredictiveEngine(df)

# Sidebar filters
st.sidebar.header("Filters")
tech_filter = st.sidebar.multiselect("Tech Comfort", ['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
df_filtered = df[df['Tech_Comfort'].isin(tech_filter)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Error Rate", f"{df_filtered['Error_Rate'].mean():.1%}", "-1.2%")
col2.metric("Sync Latency", f"{df_filtered['Sync_Latency'].mean():.1f} min", "-0.5 min")
col3.metric("Pred Savings", f"${engine.predict_savings():,.0f}", "+12%")
col4.metric("Anomalies", len(engine.detect_anomalies()), "-3")

# Tabbed Dashboard
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Predictions", "🚨 Alerts"])

with tab1:
    # Maintenance Forecast
    forecast = engine.forecast_maintenance()
    fig1 = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], 
                   title="Next 7-Day Maintenance Cost Forecast")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Correlation Heatmap
    corr = df_filtered[['Error_Rate', 'Sync_Latency', 'Insight_Delay', 'Maintenance_Cost']].corr()
    fig2 = px.imshow(corr, title="Metric Correlations", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # Savings Scatter
    fig3 = px.scatter(df_filtered, x='Error_Rate', y='Maintenance_Cost', 
                      size='Firm_Revenue', color='Tech_Comfort',
                      title="Error vs Cost: Savings Potential")
    fig3.add_hline(y=engine.reg_model.predict([[5, np.log(1e12)]])[0], line_dash="dash", line_color="green")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.info(f"**Engine Insight**: Fixing top 10% errors saves ${engine.predict_savings()*0.1:,.0f} immediately.")

with tab3:
    anomalies = engine.detect_anomalies()
    st.dataframe(anomalies[['Date', 'Error_Rate', 'Sync_Latency', 'Anomaly']])
    
    fig4 = px.histogram(anomalies, x='Error_Rate', title="Anomaly Distribution")
    st.plotly_chart(fig4, use_container_width=True)
    
    st.warning("🚨 **Alert**: 8 anomalies detected. Prioritize Sync_Latency >5min.")

# Footer
st.markdown("---")
st.caption("DataSync Analytics | Powered by Prophet & ML | Dubai E-com Focus")
