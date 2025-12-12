# renderers.py

import streamlit as st
import pandas as pd
from datetime import timedelta


from stocks import (
    stock_snapshot,
    portfolio_summary,
    fetch_history,
    linear_prediction
)

def render_community_portfolios(portfolios, go_to):
    from stocks import fetch_long_history, linear_prediction

    if not portfolios:
        st.info("No community portfolios available.")
        return

    for p in portfolios[:5]:
        owner = p.get("user_id", "Unknown")
        stocks = p.get("stocks", [])

        invested = sum(
            s.get("purchase_price", s.get("price", 0)) * s.get("shares", 1)
            for s in stocks
        )

        predicted = 0
        for s in stocks:
            hist = fetch_long_history(s["symbol"], years=1)
            if hist.empty or len(hist) < 30:
                predicted += s.get("price", 0) * s.get("shares", 1)
                continue

            pred = linear_prediction(hist["Close"])
            if pred:
                predicted += pred["predicted_price"] * s.get("shares", 1)
            else:
                predicted += s.get("price", 0) * s.get("shares", 1)

        pct = ((predicted - invested) / invested * 100) if invested else 0

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
            c1.write(f"**{owner}**")
            c2.metric("Purchase Value", f"${invested:,.2f}")
            c3.metric("Predicted (1Y)", f"${predicted:,.2f}", f"{pct:+.2f}%")
            c4.metric("Stocks", len(stocks))

            if c5.button("View", key=f"view_{p['_id']}"):
                st.session_state.media_portfolio_id = str(p["_id"])
                st.session_state.media_portfolio_owner = owner
                go_to("media_portfolio_view")

        st.markdown("---")

def render_global_market_dashboard(symbols_by_country, days=365):
    from stocks import fetch_history

    all_data = {}

    for country, symbols in symbols_by_country.items():
        data = {}
        for sym in symbols[:6]:
            df = fetch_history(sym, period="1y")
            if isinstance(df, pd.DataFrame) and not df.empty:
                data[sym] = df
        if data:
            all_data[country] = data

    if not all_data:
        st.warning("No global stock data available.")
        return

    tabs = st.tabs(list(all_data.keys()))

    for tab, (country, data) in zip(tabs, all_data.items()):
        with tab:
            st.write(f"### {country} Stock Market")
            cols = st.columns(3)

            for idx, (symbol, df) in enumerate(data.items()):
                with cols[idx % 3]:
                    try:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else latest
                        change = latest["Close"] - prev["Close"]
                        pct = (change / prev["Close"]) * 100 if prev["Close"] else 0

                        st.markdown(f"**{symbol}**")
                        st.metric("Price", f"${latest['Close']:.2f}", f"{pct:+.2f}%")

                        st.line_chart(df["Close"].tail(365), height=140)
                        st.markdown("---")

                    except Exception as e:
                        st.error(f"{symbol} failed: {e}")


def render_portfolio_overview_table(user_portfolios):
    import pandas as pd
    from stocks import fetch_long_history, linear_prediction

    rows = []

    for p in user_portfolios:
        stocks = p.get("stocks", [])
        invested = sum(
            s.get("purchase_price", s.get("price", 0)) * s.get("shares", 1)
            for s in stocks
        )

        predicted_total = 0
        for s in stocks:
            hist = fetch_long_history(s["symbol"], years=2)
            if hist.empty or len(hist) < 30:
                predicted_total += s.get("price", 0) * s.get("shares", 1)
                continue

            pred = linear_prediction(hist["Close"])
            if pred:
                predicted_total += pred["predicted_price"] * s.get("shares", 1)
            else:
                predicted_total += s.get("price", 0) * s.get("shares", 1)

        pct = ((predicted_total - invested) / invested)*100 if invested else 0

        top = ", ".join([f"{s['symbol']} ({s.get('shares',1)} sh)" for s in stocks[:3]])
        if len(stocks) > 3:
            top += f" +{len(stocks)-3} more"

        rows.append({
            "Portfolio Name": p.get("portfolio_name", "Unnamed Portfolio"),
            "Total Value": f"${invested:,.2f}",
            "Predicted (1Y)": f"${predicted_total:,.2f}",
            "Expected Change": f"{pct:+.2f}%",
            "Stocks": len(stocks),
            "Markets": ", ".join(p.get("countries", [])),
            "Top Holdings": top or "None",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_prediction_summary(symbol, price_series):
    """Display prediction summary metrics for a single stock."""
    if price_series is None or len(price_series) < 2:
        st.warning("Not enough data for prediction summary.")
        return
    
    latest = price_series.iloc[-1]
    previous = price_series.iloc[-2]
    change = latest - previous
    pct = (change / previous * 100) if previous != 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", f"${latest:.2f}", f"{change:+.2f}")
    with col2:
        st.metric("Daily Change (%)", f"{pct:+.2f}%")

# ==== STOCK TABLE RENDERER =====

def render_stock_table(stocks: list[dict], *, show_change=True):
    """
    Render a full HTML stock table from a stock list.
    Uses the stock_snapshot() function for calculations.
    """

    rows = []
    for s in stocks:
        snap = stock_snapshot(s)

        price_display = f"${snap['current_price']:.2f}"
        pct_display = (
            f"<span style='color:green'>{snap['gain_pct']:+.2f}%</span>"
            if snap["gain_pct"] > 0
            else f"<span style='color:red'>{snap['gain_pct']:+.2f}%</span>"
        )

        rows.append({
            "Symbol": s.get("symbol", ""),
            "Name": s.get("name", s.get("symbol", "")),
            "Shares": s.get("shares", 1),
            "Purchase Price": f"${snap['purchase_price']:.2f}",
            "Purchase Value": f"${snap['purchase_value']:.2f}",
            "Current Price": price_display,
            "Current Value": f"${snap['current_value']:.2f}",
            "Gain/Loss": f"${snap['gain']:+.2f}",
            "Change %": pct_display if show_change else "",
        })

    df = pd.DataFrame(rows)
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ==== PORTFOLIO SUMMARY METRICS =====

def render_portfolio_summary(stocks: list[dict]):
    """
    Render purchase value, current value, total gain/loss.
    """
    summary = portfolio_summary(stocks)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Purchase Value", f"${summary['total_purchase']:.2f}")

    with col2:
        st.metric("Current Value", f"${summary['total_current']:.2f}")

    with col3:
        st.metric(
            "Gain/Loss",
            f"${summary['gain']:+.2f}",
            f"{summary['gain_pct']:+.2f}%"
        )

    st.divider()


# ==== INDIVIDUAL STOCK PERFORMANCE GRID =====

def render_stock_performance_grid(stocks: list[dict]):
    """
    Render a grid of cards: each card shows a stock's metrics + mini chart.
    """

    cols = st.columns(2)  # two per row

    for idx, s in enumerate(stocks):
        with cols[idx % 2]:

            snap = stock_snapshot(s)
            symbol = s["symbol"]

            st.markdown(f"### {symbol}")

            colA, colB = st.columns(2)
            with colA:
                st.metric(
                    "Value",
                    f"${snap['current_value']:.2f}",
                    f"{snap['gain']:+.2f}"
                )
            with colB:
                st.markdown("**Change %**")
                color = "green" if snap["gain_pct"] > 0 else "red"
                st.markdown(
                    f"<span style='color:{color}'>{snap['gain_pct']:+.2f}%</span>",
                    unsafe_allow_html=True
                )

            st.write(f"**Shares:** {s.get('shares', 1)}")
            st.write(f"**Purchase:** ${snap['purchase_price']:.2f}")
            st.write(f"**Current:** ${snap['current_price']:.2f}")

            # Mini history chart
            hist = fetch_history(symbol)
            if not hist.empty:
                st.line_chart(hist["Close"].tail(30), height=150)

            st.markdown("---")


# ==== LINEAR PREDICTION SUMMARY BLOCK =====

def render_prediction_summary(stock_name: str, price_series: pd.Series):
    """
    Show prediction metrics (current price, predicted price, Δ, R²).
    """
    pred = linear_prediction(price_series)
    if not pred:
        st.warning(f"Not enough data to predict {stock_name}.")
        return None

    predicted = pred["predicted_price"]
    current = pred["current_price"]
    change = predicted - current
    pct = (change / current * 100) if current > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current:.2f}")
    with col2:
        st.metric(
            "Predicted (1 Year)",
            f"${predicted:.2f}",
            f"{change:+.2f} ({pct:+.2f}%)"
        )
    with col3:
        st.metric("Model R²", f"{pred['r_squared']:.4f}")

    return pred


# ==== PREDICTION CHART RENDERER =====

def render_prediction_chart(price_series: pd.Series, pred: dict, *, lookback_days=365):
    """
    Draw a combined historical + future prediction chart.
    """

    if pred is None:
        return

    # Historical data
    recent = price_series.tail(lookback_days)
    recent_dates = recent.index

    # Future dates
    last_date = recent_dates[-1]
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=len(pred["future_predictions"]),
        freq="D"
    )

    combined_dates = list(recent_dates) + list(future_dates)
    combined_actual = list(recent.values) + [None] * len(pred["future_predictions"])
    combined_pred = list(pred["reg_line"][-lookback_days:]) + list(pred["future_predictions"])

    df = pd.DataFrame({
        "Historical": combined_actual,
        "Predicted": combined_pred
    }, index=combined_dates)

    st.line_chart(df, height=350)
