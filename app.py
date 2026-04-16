import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import get_price_data
from utils.indicators import add_indicators, compute_rsi


st.set_page_config(page_title="MarketLens", layout="wide")

# Used streamlit for the UI 
st.title("MarketLens")
st.subheader("Stock & Crypto Analytics Dashboard")


with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Enter ticker", value="AAPL") # placeholder ticker for apple 
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
    high_52 = df["Close"].max()
    low_52 = df["Close"].min()

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
    stats_col1.write(f"**Period High:** ${high_52:,.2f}")
    stats_col1.write(f"**Period Low:** ${low_52:,.2f}")
    stats_col2.write(f"**Rows Loaded:** {len(df)}")
    stats_col2.write(f"**Ticker:** {ticker.upper()}")

    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(df.tail(30), use_container_width=True)

except Exception as e:
    st.error(str(e))
    st.info("Try tickers like AAPL, MSFT, NVDA, BTC-USD, or ETH-USD.")
