import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# Load Saved Models
# ----------------------------

@st.cache_resource
def load_arima():
    return joblib.load("models/arima_model_AAPL.joblib")

@st.cache_resource
def load_lstm():
    return load_model("models/lstm_model_AAPL.h5")

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("models/scaler_AAPL.joblib")
    except:
        return None  # if not available


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="AAPL Stock Forecast", layout="wide")
st.title("📈 AAPL Stock Forecast Web App")
st.write("Forecast future stock prices using ARIMA and LSTM models.")

# Sidebar
forecast_days = st.sidebar.number_input("Forecast Days", min_value=5, max_value=120, value=30)
model_choice = st.sidebar.selectbox("Choose Model", ["ARIMA", "LSTM"])

# Download AAPL historical data
st.subheader("📊 AAPL Stock Price History")
# Download AAPL data
df = yf.download("AAPL", start="2015-01-01")

# FIX: Flatten MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df = df[["Close"]]  # use Close price
df.dropna(inplace=True)

st.line_chart(df)

series = df["Close"]

# ----------------------------
# Produce Forecast
# ----------------------------

if st.sidebar.button("Generate Forecast"):
    st.subheader(f"🔮 Forecast for next {forecast_days} days using {model_choice}")

    if model_choice == "ARIMA":
        model = load_arima()
        forecast = model.predict(n_periods=forecast_days)
        forecast = np.array(forecast).reshape(-1)

    elif model_choice == "LSTM":
        model = load_lstm()
        scaler = load_scaler()

        # Scale data if scaler exists
        if scaler:
            scaled = scaler.transform(series.values.reshape(-1,1))
        else:
            scaled = (series.values - series.values.min()) / (series.values.max() - series.values.min())

        window = 20
        last_window = scaled[-window:]

        preds = []
        for _ in range(forecast_days):
            x = last_window.reshape(1, window, 1)
            pred = model.predict(x, verbose=0)[0][0]
            preds.append(pred)
            last_window = np.append(last_window[1:], pred)

        # Inverse scale if scaler exists
        if scaler:
            forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        else:
            forecast = preds * (series.max() - series.min()) + series.min()

    # Create forecast DataFrame
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                 periods=forecast_days, freq="B")
    
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast}).set_index("Date")

    # Plot the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name="Forecast"))
    fig.update_layout(title=f"AAPL Stock Forecast using {model_choice}",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Show table
    st.write("### 📄 Forecast Data")
    st.dataframe(forecast_df)

    # Download button
    csv = forecast_df.to_csv().encode()
    st.download_button("Download Forecast CSV", csv, "AAPL_forecast.csv", "text/csv")

    st.success("Forecast generated successfully!")
