import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import get_price_data
from utils.indicators import add_indicators, compute_rsi
from utils.modeling import prepare_ml_data, train_regression_model

st.set_page_config(page_title="MarketLens", layout="wide")

st.title("MarketLens")
st.subheader("Stock & Crypto Analytics Dashboard")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Enter ticker", value="AAPL")
    period = st.selectbox("Select time range", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("Select interval", ["1d", "1wk", "1mo"], index=0)

try:
    df = get_price_data(ticker, period=period, interval=interval)
    df = add_indicators(df)
    df["RSI14"] = compute_rsi(df["Close"])

    latest_close = df["Close"].iloc[-1]
    first_close = df["Close"].iloc[0]
    total_return = ((latest_close / first_close) - 1) * 100

    last_daily_return = df["Daily Return"].iloc[-1] * 100 if pd.notna(df["Daily Return"].iloc[-1]) else 0
    last_volatility = df["Volatility20"].iloc[-1] * 100 if pd.notna(df["Volatility20"].iloc[-1]) else 0
    period_high = df["Close"].max()
    period_low = df["Close"].min()

    tab1, tab2 = st.tabs(["Market Dashboard", "ML Prediction"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"${latest_close:,.2f}")
        col2.metric("Total Return", f"{total_return:.2f}%")
        col3.metric("Last Daily Return", f"{last_daily_return:.2f}%")
        col4.metric("Rolling Volatility", f"{last_volatility:.2f}%")

        st.markdown("---")

        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close"
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA20"],
                mode="lines",
                name="MA20"
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA50"],
                mode="lines",
                name="MA50"
            )
        )
        price_fig.update_layout(
            title=f"{ticker.upper()} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        st.plotly_chart(price_fig, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            returns_fig = go.Figure()
            returns_fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["Daily Return"],
                    mode="lines",
                    name="Daily Return"
                )
            )
            returns_fig.update_layout(
                title="Daily Returns",
                xaxis_title="Date",
                yaxis_title="Return",
                height=400
            )
            st.plotly_chart(returns_fig, use_container_width=True)

        with col_right:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["RSI14"],
                    mode="lines",
                    name="RSI(14)"
                )
            )
            rsi_fig.update_layout(
                title="RSI (14)",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=400
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Summary Stats")

        stats_col1, stats_col2 = st.columns(2)
        stats_col1.write(f"**Period High:** ${period_high:,.2f}")
        stats_col1.write(f"**Period Low:** ${period_low:,.2f}")
        stats_col2.write(f"**Rows Loaded:** {len(df)}")
        stats_col2.write(f"**Ticker:** {ticker.upper()}")

        st.markdown("---")
        st.subheader("Raw Data")
        st.dataframe(df.tail(30), use_container_width=True)

    # Second tab on the page 
    with tab2:
        st.subheader("Next-Day Return Prediction")
        st.caption("This is a baseline machine learning model for educational analysis, not true financial advice.")

        ml_df = prepare_ml_data(df)
        model, results_df, metrics, next_day_prediction = train_regression_model(ml_df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{metrics['rmse']:.5f}")
        m2.metric("Direction Accuracy", f"{metrics['direction_accuracy'] * 100:.2f}%")
        m3.metric("Train Rows", metrics["train_size"])
        m4.metric("Test Rows", metrics["test_size"])

        pred_percent = next_day_prediction * 100
        pred_label = "Up" if next_day_prediction > 0 else "Down"

        st.markdown("---")
        st.subheader("Tomorrow Forecast")
        st.metric("Predicted Next-Day Return", f"{pred_percent:.2f}%")
        st.write(f"**Predicted Direction:** {pred_label}")

        st.markdown("---")
        st.subheader("Actual vs Predicted Returns")

        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Scatter(
                x=results_df["Date"],
                y=results_df["Actual Return"],
                mode="lines",
                name="Actual Return"
            )
        )
        pred_fig.add_trace(
            go.Scatter(
                x=results_df["Date"],
                y=results_df["Predicted Return"],
                mode="lines",
                name="Predicted Return"
            )
        )
        pred_fig.update_layout(
            title="Model Performance on Test Set",
            xaxis_title="Date",
            yaxis_title="Next-Day Return",
            height=500
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Prediction Samples")
        display_df = results_df.copy()
        display_df["Actual Return"] = display_df["Actual Return"].map(lambda x: f"{x * 100:.2f}%")
        display_df["Predicted Return"] = display_df["Predicted Return"].map(lambda x: f"{x * 100:.2f}%")
        st.dataframe(display_df.tail(20), use_container_width=True)

        with st.expander("Model Features Used"):
            st.write(metrics["feature_cols"])
#Can't leave space empty 
except Exception as e:
    st.error(str(e))
    st.info("Try tickers like AAPL, MSFT, NVDA, BTC-USD, or ETH-USD. For ML, use 1y or 2y for better results.")
