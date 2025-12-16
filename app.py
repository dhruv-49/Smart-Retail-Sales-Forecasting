import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.api import Holt, ExponentialSmoothing, SARIMAX
from sklearn.metrics import mean_squared_error
import math
import os

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide")
st.title("🛒 Walmart Sales Forecasting and Inventory Optimization")
st.markdown("""
This app lets you explore Walmart sales data (state + store), 
do EDA, and forecast future sales using different time-series models.
""")

st.sidebar.header("⚙️ Controls")

@st.cache_data
def load_data():
    # Try common sales filenames
    possible_files = ["sales.csv", "train.csv", "sales_train_validation.csv"]
    main_file = None
    for f in possible_files:
        if os.path.exists(f):
            main_file = f
            break
    if not main_file:
        st.error("No main sales file found. Put 'sales.csv' or 'sales_train_validation.csv' in this folder.")
        st.stop()

    df = pd.read_csv(main_file)
    df.columns = [c.lower() for c in df.columns]

    # If M5 format (d_1, d_2, ...), melt to long format
    if 'd_1' in df.columns:
        df = df.melt(
            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            var_name='d', value_name='sales'
        )

    # Add real dates using calendar.csv
    if os.path.exists("calendar.csv"):
        cal = pd.read_csv("calendar.csv")
        df = df.merge(cal[['d', 'date']], on='d', how='left')
    else:
        st.warning("calendar.csv not found. Using dummy dates.")
        df['date'] = pd.date_range(start='2011-01-29', periods=len(df), freq='D')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

data = load_data()

# Sidebar filters
states = sorted(data['state_id'].unique())
state = st.sidebar.selectbox("Select State", states)

stores = sorted(data[data['state_id'] == state]['store_id'].unique())
store = st.sidebar.selectbox("Select Store", stores)

forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 180, 60)
window = st.sidebar.slider("Moving Average Window", 3, 30, 7)

# Filter selected state + store and aggregate daily
subset = data[(data['state_id'] == state) & (data['store_id'] == store)]
subset = subset.groupby('date')['sales'].sum().reset_index()

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Forecast (Single Model)", "📈 Model Comparison"])

# ---------- EDA ----------
with tab1:
    st.subheader(f"EDA — {state} / {store}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Summary Statistics")
        st.write(subset.describe())

    with col2:
        st.write("Missing Values")
        st.write(subset.isnull().sum())

    fig = px.line(subset, x="date", y="sales",
                  title=f"{store} - Daily Sales", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    rolling_window = st.slider("Rolling Mean Window (days)", 3, 60, 14, key="rolling")
    subset["Rolling_Avg"] = subset["sales"].rolling(rolling_window).mean()
    fig2 = px.line(subset, x="date", y="Rolling_Avg",
                   title=f"{store} - {rolling_window}-Day Rolling Average")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Models ----------
def naive_forecast(df, periods):
    last_value = df["sales"].iloc[-1]
    future_dates = pd.date_range(df["date"].iloc[-1], periods=periods + 1)[1:]
    return pd.DataFrame({"date": future_dates, "forecast": [last_value] * periods})

def moving_average_forecast(df, window, periods):
    avg = df["sales"].tail(window).mean()
    future_dates = pd.date_range(df["date"].iloc[-1], periods=periods + 1)[1:]
    return pd.DataFrame({"date": future_dates, "forecast": [avg] * periods})

def holt_forecast(df, periods):
    model = Holt(df["sales"]).fit()
    forecast = model.forecast(periods)
    future_dates = pd.date_range(df["date"].iloc[-1], periods=periods + 1)[1:]
    return pd.DataFrame({"date": future_dates, "forecast": forecast})

def exp_smoothing_forecast(df, periods):
    model = ExponentialSmoothing(df["sales"], trend="add", seasonal=None).fit()
    forecast = model.forecast(periods)
    future_dates = pd.date_range(df["date"].iloc[-1], periods=periods + 1)[1:]
    return pd.DataFrame({"date": future_dates, "forecast": forecast})

def arima_forecast(df, periods):
    model = SARIMAX(df["sales"], order=(1, 1, 1)).fit(disp=False)
    forecast = model.forecast(steps=periods)
    future_dates = pd.date_range(df["date"].iloc[-1], periods=periods + 1)[1:]
    return pd.DataFrame({"date": future_dates, "forecast": forecast})

def prophet_forecast(df, periods):
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast"})
    return forecast.tail(periods)

# ---------- Single Model Forecast ----------
with tab2:
    model_choice = st.selectbox(
        "Select Forecasting Model",
        ["Naive", "Moving Average", "Holt Linear", "Exponential Smoothing", "ARIMA", "Prophet"]
    )

    if st.button("Run Forecast"):
        if model_choice == "Naive":
            forecast = naive_forecast(subset, forecast_days)
        elif model_choice == "Moving Average":
            forecast = moving_average_forecast(subset, window, forecast_days)
        elif model_choice == "Holt Linear":
            forecast = holt_forecast(subset, forecast_days)
        elif model_choice == "Exponential Smoothing":
            forecast = exp_smoothing_forecast(subset, forecast_days)
        elif model_choice == "ARIMA":
            forecast = arima_forecast(subset, forecast_days)
        elif model_choice == "Prophet":
            forecast = prophet_forecast(subset, forecast_days)

        fig3 = px.line()
        fig3.add_scatter(x=subset["date"], y=subset["sales"], name="Actual Sales")
        fig3.add_scatter(x=forecast["date"], y=forecast["forecast"],
                         name="Forecast", line=dict(dash="dot"))
        fig3.update_layout(
            title=f"{model_choice} Forecast for {store}",
            xaxis_title="Date", yaxis_title="Sales"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.write("Forecasted values")
        st.dataframe(forecast.tail())

# ---------- Model Comparison ----------
with tab3:
    st.subheader(f"Model Comparison — {state} / {store}")

    models = {
        "Naive": naive_forecast,
        "Moving Average": lambda df, p: moving_average_forecast(df, window, p),
        "Holt Linear": holt_forecast,
        "Exponential Smoothing": exp_smoothing_forecast,
        "ARIMA": arima_forecast,
        "Prophet": prophet_forecast,
    }

    comparison_df = pd.DataFrame()
    fig4 = px.line()
    fig4.add_scatter(x=subset["date"], y=subset["sales"],
                     name="Actual Sales", line=dict(width=2))

    for name, func in models.items():
        try:
            f = func(subset, forecast_days)
            fig4.add_scatter(x=f["date"], y=f["forecast"], name=name)
            if len(subset) > forecast_days:
                actual = subset["sales"].tail(forecast_days).values
                pred = f["forecast"].head(len(actual)).values
                rmse = math.sqrt(mean_squared_error(actual, pred))
            else:
                rmse = None
            comparison_df = pd.concat(
                [comparison_df, pd.DataFrame({"Model": [name], "RMSE": [rmse]})]
            )
        except Exception as e:
            st.warning(f"Model {name} failed: {e}")

    fig4.update_layout(
        title=f"Forecast Comparison ({store})",
        xaxis_title="Date", yaxis_title="Sales"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.write("Model performance (lower RMSE is better):")
    st.dataframe(comparison_df)

st.markdown("---")
st.write("Made by **Dhruv Khatri** — Smart Retail Sales & Inventory Optimization")
