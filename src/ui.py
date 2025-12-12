import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from urllib.parse import quote
from login import (
    get_user_portfolios, get_all_portfolios, get_portfolio_by_id,
    create_portfolio, update_portfolio, delete_portfolio,
    add_stock_to_portfolio, remove_stock_from_portfolio
)
from constants import STOCK_SYMBOLS_BY_COUNTRY, AVAILABLE_COUNTRIES
from stocks import (
    fetch_long_history,
    linear_prediction,
)

from renderers import (
    render_portfolio_summary,
    render_portfolio_overview_table,
    render_community_portfolios,
    render_global_market_dashboard,
    render_prediction_summary,
    render_prediction_chart,
    render_stock_table,
    render_stock_performance_grid,
)
from algorithms import insertion_sort_portfolios
from stocks import StockPredictor

def handle_logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "login"
    st.rerun()

def format_percentage_with_color(percentage):
    if percentage > 0:
        return f'<span style="color: green;">{percentage:+.2f}%</span>'
    elif percentage < 0:
        return f'<span style="color: red;">{percentage:+.2f}%</span>'
    else:
        return f'<span style="color: gray;">{percentage:.2f}%</span>'


# ==================== UI COMPONENTS AND SIDEBAR RENDERING ====================

def render_sidebar(page_title, actions=None, back_button=None):
    """Render a dynamic sidebar with page title, action buttons, navigation, and logout."""
    with st.sidebar:
        st.header(page_title)

        # Render action buttons if provided
        if actions:
            st.subheader("Actions")
            for action in actions:
                if st.button(action['label'], width="stretch", type=action.get('type', 'secondary'), key=action.get('key')):
                    action['callback']()
        st.divider()

        # Optional back button
        if back_button:
            if st.button(f"← {back_button['label']}", width="stretch"):
                back_button['callback']()

        # Navigation buttons
        if st.button("Dashboard", width="stretch"):
            st.session_state.page = "dashboard"
            st.rerun()
        if st.button("Logout", width="stretch"):
            handle_logout()

@st.dialog("Delete Portfolio")
def show_delete_confirmation_popup():
    portfolio_id = st.session_state.get('confirm_delete_portfolio')
    portfolio_name = st.session_state.get('confirm_delete_name', 'Unknown Portfolio')
    
    st.warning(f"**Confirm Deletion**")
    st.write(f"Are you sure you want to delete the portfolio **{portfolio_name}**?")
    st.write("**This action cannot be undone.**")
    st.write("")
    
    col_confirm, col_cancel = st.columns(2)
    
    with col_confirm:
        if st.button("Yes, Delete", type="primary", width="stretch"):
            try:
                success, message = delete_portfolio(portfolio_id, st.session_state.username)
                
                if success:
                    st.success(f"Portfolio '{portfolio_name}' deleted successfully!")
                    del st.session_state.confirm_delete_portfolio
                    del st.session_state.confirm_delete_name
                    st.rerun()
                else:
                    st.error(f"Failed to delete portfolio: {message}")
                    
            except Exception as e:
                st.error(f"Error deleting portfolio: {str(e)}")
    
    with col_cancel:
        if st.button("Cancel", width="stretch"):
            del st.session_state.confirm_delete_portfolio
            del st.session_state.confirm_delete_name
            st.rerun()

def get_company_news_link(symbol):
    try:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', symbol)
        except Exception:
            company_name = symbol
        
        search_query = f"{company_name} {symbol} stock"
        encoded_query = quote(search_query)
        google_news_url = f"https://news.google.com/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        return {
            'company_name': company_name,
            'symbol': symbol,
            'search_query': search_query,
            'news_url': google_news_url
        }
        
    except Exception as e:
        st.error(f"Error generating news link: {str(e)}")
        return None

def get_stock_info_with_history(symbol):
    try:
        ticker = yf.Ticker(symbol)
        
        info = ticker.info
        
        historical_data = fetch_long_history(symbol, years=25)
        
        recent_data = ticker.history(period="2d")
        
        stock_info = {
            "symbol": symbol,
            "name": info.get('longName', info.get('shortName', symbol)),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "historical_data": historical_data,
            "info": info
        }
        
        if not recent_data.empty and len(recent_data) >= 2:
            current_price = float(recent_data['Close'].iloc[-1])
            previous_price = float(recent_data['Close'].iloc[-2])
            change = current_price - previous_price
            
            stock_info.update({
                "price": current_price,
                "change": change,
                "previous_price": previous_price
            })
        
        return stock_info
        
    except Exception as e:
        st.error(f"Failed to fetch complete info for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_stocks_for_search(country):
    if country not in STOCK_SYMBOLS_BY_COUNTRY:
        return []
    
    symbols = STOCK_SYMBOLS_BY_COUNTRY[country]
    stock_data = []
    
    try:
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_string = " ".join(batch)
            
            try:
                tickers = yf.Tickers(batch_string)
                
                for symbol in batch:
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info
                        hist = ticker.history(period="2d")
                        
                        if not hist.empty and len(hist) >= 2:
                            current_price = float(hist['Close'].iloc[-1])
                            previous_price = float(hist['Close'].iloc[-2])
                            change = current_price - previous_price
                            
                            stock_data.append({
                                "symbol": symbol,
                                "name": info.get('longName', info.get('shortName', symbol)),
                                "price": current_price,
                                "change": change,
                                "country": country
                            })
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
                
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
    
    return stock_data

def login_page(go_to, verify_user, update_last_login):
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", type="primary", width="stretch"):
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    update_last_login(username)
                    st.success("Login successful!")
                    go_to("dashboard")
                else:
                    st.error("Invalid credentials")

    st.divider()

    st.write("Don't have an account?")
    if st.button("Register here", width="stretch"):
        go_to("register")

def register_page(go_to, register_user):
    st.title("Register")

    username = st.text_input("Choose a username")
    email = st.text_input("Email")
    password = st.text_input(
        "Choose a password",
        type="password",
        help="Must be at least 8 characters with uppercase, lowercase, number and special character",
    )
    confirm_password = st.text_input("Confirm password", type="password")

    if password and confirm_password:
        if password == confirm_password:
            st.success("Passwords match")
        else:
            st.error("Passwords don't match")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Register", type="primary", width="stretch"):
            success, message = register_user(username, password, email)
            if success:
                st.success(message)
                st.balloons()  # Celebration animation
                go_to("login")
            else:
                st.error(message)

    with col2:
        if st.button("Back to login", width="stretch"):
            go_to("login")

def dashboard_page(go_to, get_user_info, change_password):
    render_sidebar(
        "Menu",
        actions=[
            {'label': 'Detailed Stock Analysis', 'callback': lambda: go_to("stock_analysis")},
            {'label': 'Portfolios', 'callback': lambda: go_to("portfolios")}
        ],
    )

    st.title("Dashboard")

    user_info = get_user_info(st.session_state.username)
    st.markdown(f"### Welcome back, **{st.session_state.username}**!")

    st.divider()

    st.subheader("Quick Actions")

    user_portfolios = get_user_portfolios(st.session_state.username)

    if user_portfolios:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Create New Portfolio", type="primary", width="stretch"):
                go_to("create_portfolio")
        with col2:
            if st.button("View All Portfolios", width="stretch"):
                go_to("portfolios")
        with col3:
            if st.button("Search Stocks", width="stretch"):
                go_to("stock_analysis")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Your First Portfolio", type="primary", width="stretch"):
                go_to("create_portfolio")
        with col2:
            if st.button("Detailed Stock Analysis", width="stretch"):
                go_to("stock_analysis")

    st.divider()

    st.subheader("Your Portfolios Summary")

    if user_portfolios:
        # Combine all stocks from all portfolios
        all_stocks = [s for p in user_portfolios for s in p.get('stocks', [])]
        if all_stocks:
            render_portfolio_summary(all_stocks)
        else:
            st.info("No stocks in your portfolios yet.")
        
        st.divider()

        st.subheader("Portfolio Overview")
        render_portfolio_overview_table(user_portfolios)

    st.divider()

    st.subheader("Community Portfolios")

    all_portfolios = get_all_portfolios()
    if all_portfolios:
        other_user_portfolios = [p for p in all_portfolios if p.get("user_id") != st.session_state.username]
        render_community_portfolios(other_user_portfolios, go_to)

    st.divider()

    st.subheader("Global Stock Market Dashboard")

    render_global_market_dashboard(STOCK_SYMBOLS_BY_COUNTRY)

def stock_analysis_page(go_to, get_user_info, change_password):
    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("Stock Analysis")
        st.subheader("Stock Search")

        # Build full stock list
        all_stock_symbols = sorted({
            symbol for symbols in STOCK_SYMBOLS_BY_COUNTRY.values() for symbol in symbols
        })

        search_query = st.text_input(
            "Search Stock Symbol",
            value="AAPL",
            placeholder="Type to search (e.g., AAPL, GOOGL, TSLA...)"
        )

        # --- Filter matching symbols ---
        if search_query:
            filtered = [s for s in all_stock_symbols if search_query.upper() in s.upper()]
            if filtered:
                shown = filtered[:10]
                cols = st.columns(2)
                for i, sym in enumerate(shown):
                    with cols[i % 2]:
                        if st.button(sym, key=f"sym_{sym}", width="stretch"):
                            st.session_state.selected_stock_symbol = sym
                            st.rerun()
                selected = st.session_state.get("selected_stock_symbol", filtered[0])
            else:
                st.warning("No matching stocks found.")
                selected = "AAPL"
        else:
            selected = "AAPL"

        selected = st.session_state.get("selected_stock_symbol", selected)

        # ---- Analysis Tools ----
        st.subheader("Analysis Tools")
        show_volume = st.checkbox("Show Volume", True)
        show_moving_avg = st.checkbox("Show Moving Average", False)

        st.divider()

        if st.button("← Back to Dashboard", width="stretch"):
            go_to("dashboard")
        if st.button("Portfolios", width="stretch"):
            go_to("portfolios")
        if st.button("Logout", width="stretch"):
            handle_logout()

    # ---- MAIN CONTENT ----
    st.title(f"{selected} - Detailed Analysis")

    # Load 10yr history
    df = fetch_long_history(selected, years=10)

    if df is None or df.empty:
        st.error(f"Unable to load data for {selected}")
        return

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change = float(latest["Close"] - prev["Close"])
    pct = (change / float(prev["Close"]) * 100) if prev["Close"] else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Current Price**")
        st.markdown(f"${latest['Close']:.2f}")
        st.markdown(format_percentage_with_color(pct), unsafe_allow_html=True)
    with col2:
        st.metric("52-Week High", f"${df['High'].tail(252).max():.2f}")
    with col3:
        st.metric("52-Week Low", f"${df['Low'].tail(252).min():.2f}")
    with col4:
        vol = latest.get("Volume")
        st.metric("Volume", f"{int(vol):,}" if vol else "N/A")

    st.divider()

    # ---- PRICE CHART ----
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader(f"{selected} Price Chart")
        chart_df = pd.DataFrame({"Close": df["Close"]})
        if show_moving_avg and len(df) >= 20:
            chart_df["20-Day MA"] = df["Close"].rolling(20).mean()
        st.line_chart(chart_df, height=400)

    with col_b:
        if show_volume and "Volume" in df.columns:
            st.subheader("Volume")
            st.bar_chart(df["Volume"], height=400)
        else:
            st.subheader("High-Low Range")
            st.line_chart(df[["High", "Low"]], height=400)

    st.divider()

    # ---- RECENT PERFORMANCE ----
    col_l, col_r = st.columns([2, 3])

    with col_l:
        st.subheader("Recent Performance")
        periods = [1, 7, 30]
        perf = []
        for p in periods:
            if len(df) > p:
                old = df["Close"].iloc[-(p+1)]
                now = df["Close"].iloc[-1]
                pct = ((now - old) / old * 100)
                perf.append({"Period": f"{p} Days", "Change (%)": f"{pct:+.2f}%"})
        st.table(pd.DataFrame(perf))

    with col_r:
        st.subheader("Price Statistics")
        stats = pd.DataFrame([
            ["Average", f"${df['Close'].mean():.2f}"],
            ["Median", f"${df['Close'].median():.2f}"],
            ["Std Deviation", f"${df['Close'].std():.2f}"],
            ["Range", f"${(df['Close'].max() - df['Close'].min()):.2f}"],
        ], columns=["Metric", "Value"])
        st.table(stats)

    st.divider()

    # ---- PREDICTION ----
    st.subheader("Price Prediction (Linear Regression)")

    predictor = StockPredictor(selected)

    # Load historical data and generate prediction
    if predictor.load_history(years=10):
        pred = predictor.predict(future_days=365)
        
        if pred:
            # Display prediction summary using class method
            summary = predictor.get_prediction_summary()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${summary['current_price']:.2f}")
            with col2:
                st.metric(
                    "Predicted (1 Year)", 
                    f"${summary['predicted_price']:.2f}",
                    f"{summary['change']:+.2f} ({summary['change_percent']:+.2f}%)"
                )
            with col3:
                st.metric("Trend", summary['trend'])
            
            # Also show R² score
            st.write(f"**Model R² Score:** {summary['r_squared']:.4f}")
            
            # Use existing chart renderer
            price_series = df["Close"].dropna()
            render_prediction_chart(price_series, pred)
        else:
            st.warning("Not enough historical data to generate prediction.")
    else:
        st.warning(f"Could not load historical data for {selected}")

    st.divider()

    # ---- COMPANY NEWS ----
    st.subheader("Company News")
    info = get_company_news_link(selected)

    if info:
        st.write(f"**Company:** {info['company_name']}")
        st.link_button(
            f"View {info['company_name']} News on Google",
            info['news_url'],
            use_container_width=True
        )

def portfolios_page(go_to, get_user_info, change_password):
    """Refactored portfolio manager using renderers + unified prediction engine."""

    # ---- SIDEBAR ----
    render_sidebar(
        "Portfolio Manager",
        actions=[
            {'label': 'Create New Portfolio', 'callback': lambda: go_to("create_portfolio")},
            {'label': 'Portfolio Analytics', 'callback': lambda: go_to("portfolio_analytics")}
        ],
        back_button={'label': 'Back to Dashboard', 'callback': lambda: go_to("dashboard")}
    )

    st.title("Portfolio Management")

    # ---- Fetch user portfolios ----
    user_portfolios = get_user_portfolios(st.session_state.username)

    if not user_portfolios:
        st.info("You haven't created any portfolios yet.")
        if st.button("Create Your First Portfolio", type="primary"):
            go_to("create_portfolio")
        return

    # ---- 1. PORTFOLIO SUMMARY ----
    st.subheader("Your Portfolios Summary")
    
    # Combine all stocks from all portfolios
    all_stocks = [s for p in user_portfolios for s in p.get('stocks', [])]
    if all_stocks:
        render_portfolio_summary(all_stocks)
    else:
        st.info("No stocks in your portfolios yet.")

    st.divider()

    # ---- 2. PORTFOLIO OVERVIEW TABLE ----
    st.subheader("Portfolio Overview")
    render_portfolio_overview_table(user_portfolios)

    st.divider()

    # ---- 3. LIST PORTFOLIOS AS CARDS WITH ACTION BUTTONS ----
    st.subheader("My Portfolios")

    # Sort portfolios by calculated value (descending)
    sorted_portfolios = insertion_sort_portfolios(user_portfolios, key="value")

    for p in sorted_portfolios:
        _render_single_portfolio_card(p, go_to)

    # Popup for delete confirmation
    if st.session_state.get('confirm_delete_portfolio'):
        show_delete_confirmation_popup()

def _render_single_portfolio_card(portfolio, go_to):
    """Helper: render one portfolio with actions"""
    name = portfolio.get("portfolio_name", "Unnamed")
    p_id = str(portfolio["_id"])
    stocks = portfolio.get("stocks", [])
    created = portfolio.get("created_at")
    created_str = created.strftime("%Y-%m-%d") if created else "Unknown"

    total_value = sum(
        s.get("purchase_price", s.get("price", 0)) * s.get("shares", 1)
        for s in stocks
    )

    with st.container():
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown(f"### {name}")
            st.write(f"**Created:** {created_str}")
            st.write(f"**Holdings:** {', '.join([s['symbol'] for s in stocks]) or 'No Stocks'}")

        with col2:
            st.metric("Current Value", f"${total_value:,.2f}")

        # --- Action buttons ---
        col_view, col_edit, col_share, col_delete = st.columns(4)

        with col_view:
            if st.button("View", key=f"view_{p_id}"):
                st.session_state.view_portfolio_id = p_id
                st.session_state.view_portfolio_name = name
                go_to("portfolio_details")

        with col_edit:
            if st.button("Edit", key=f"edit_{p_id}"):
                st.session_state.edit_portfolio_id = p_id
                st.session_state.edit_portfolio_name = name
                go_to("edit_portfolio")

        with col_share:
            if st.button("Share", key=f"share_{p_id}"):
                st.session_state.share_portfolio = {
                    "_id": p_id,
                    "name": name,
                    "stocks": [s["symbol"] for s in stocks],
                    "value": total_value,
                }

        with col_delete:
            if st.button("Delete", key=f"delete_{p_id}", type="secondary"):
                st.session_state.confirm_delete_portfolio = p_id
                st.session_state.confirm_delete_name = name
                st.rerun()

        # --- Optional share panel ---
        if st.session_state.get("share_portfolio", {}).get("_id") == p_id:
            _render_share_panel(st.session_state.share_portfolio)

        st.markdown("---")


def _render_share_panel(portfolio_data):
    """Helper: sharing UI block"""
    name = portfolio_data["name"]
    p_id = portfolio_data["_id"]
    stocks = portfolio_data["stocks"]

    with st.expander(f"Share Portfolio: {name}", expanded=True):
        st.write("**Share your portfolio with others:**")

        url = f"https://yourapp.com/shared-portfolio/{p_id}"

        template = f"""
Investment Portfolio Template: "{name}"

Holdings ({len(stocks)} stocks):
{", ".join(stocks)}

Create your own version of this portfolio:
{url}
""".strip()

        left, right = st.columns(2)

        with left:
            st.text_area("Template", value=template, height=150, key=f"share_template_{p_id}")
            if st.button("Copy Text", key=f"copy_{p_id}", type="primary"):
                st.success("Copied!")
            if st.button("Close", key=f"close_share_{p_id}"):
                del st.session_state.share_portfolio
                st.rerun()

        with right:
            st.write("**Share URL:**")
            st.code(url)


def create_portfolio_page(go_to, get_user_info, change_password):
    """Page where users create a new portfolio with selected countries."""

    render_sidebar("Create New Portfolio", back_button={'label': 'Back to Portfolios', 'callback': lambda: go_to("portfolios")})
    
    st.title("Create New Portfolio")
    st.markdown("### Let's build your investment portfolio step by step")
    
    st.divider()
    
    with st.form("create_portfolio_form", clear_on_submit=False):
        
        st.subheader("Which countries would you like to invest in?")
        
        selected_countries = st.multiselect(
            "Select countries/regions for investment",
            options=AVAILABLE_COUNTRIES,
            default=[AVAILABLE_COUNTRIES[0]] if AVAILABLE_COUNTRIES else [],
            help="Choose the countries where you'd like to invest. This will help us recommend appropriate stocks and ETFs."
        )
        
        st.divider()
        
        st.subheader("Portfolio Details")
        portfolio_name = st.text_input(
            "Portfolio Name", 
            placeholder="e.g., My Global Growth Portfolio",
            help="Give your portfolio a memorable name"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("Create Portfolio", type="primary", use_container_width=True)
        
        with col2:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)
    
    if submitted:
        if portfolio_name and selected_countries:

            portfolio_data = {
                'name': portfolio_name,
                'countries': selected_countries,
                'stocks': []
            }
            
            success, message = create_portfolio(st.session_state.username, portfolio_data)
            
            if success:
                st.success("Portfolio created successfully!")
                st.balloons()
                
                st.subheader("Portfolio Summary")
                st.write(f"**Name:** {portfolio_name}")
                st.write(f"**Countries:** {', '.join(selected_countries)}")
                

                user_portfolios = get_user_portfolios(st.session_state.username)

                if user_portfolios:
                    latest_portfolio = user_portfolios[0]  # Sorted by created_at desc
                    st.session_state.current_portfolio = portfolio_data
                    st.session_state.current_portfolio['_id'] = str(latest_portfolio['_id'])
                else:
                    st.session_state.current_portfolio = portfolio_data
                    st.session_state.current_portfolio['_id'] = 'temp_id'

                go_to("my_stocks")
            else:
                st.error(f"{message}")
            
        else:
            st.error("Please fill in all required fields (Portfolio name and at least one country)")
    
    elif cancelled:
        go_to("portfolios")

def my_stocks_page(go_to, get_user_info, change_password):
    if "current_portfolio" not in st.session_state:
        st.error("No active portfolio.")
        go_to("portfolios")
        return

    p_id = st.session_state.current_portfolio.get("_id")
    portfolio = (
        get_portfolio_by_id(p_id)
        if p_id and p_id != "temp_id"
        else st.session_state.current_portfolio
    )

    render_sidebar(
        "My Stocks",
        actions=[
            {"label": "Add Stock", "callback": lambda: go_to("stock_search"), "type": "primary"},
            {"label": "Portfolio Analytics", "callback": lambda: go_to("portfolio_analytics")},
        ],
        back_button={"label": "Back to Portfolios", "callback": lambda: go_to("portfolios")},
    )

    st.title("My Portfolio")

    if not portfolio:
        st.error("Portfolio not found.")
        return

    stocks = portfolio.get("stocks", [])
    name = portfolio.get("portfolio_name", portfolio.get("name", "Unnamed Portfolio"))

    st.markdown(f"### Portfolio: **{name}**")
    render_portfolio_summary(stocks)

    st.divider()
    st.subheader("Stock Holdings")

    if stocks:
        render_stock_table(stocks)
    
        for s in stocks:
            if st.button("Remove", key=f"remove_{s['symbol']}"):
                _remove_stock(s["symbol"], portfolio)
    else:
        col = st.columns([1, 2, 1])[1]
        with col:
            if st.button("Add Your First Stock", type="primary"):
                go_to("stock_search")

    if st.session_state.get("show_my_stocks_analytics", False):
        st.divider()
        st.subheader("Portfolio Analytics & Predictions")

        total_current = 0
        total_predicted = 0
        predictions = []

        for s in stocks:
            hist = fetch_long_history(s["symbol"], years=2)
            if hist.empty or len(hist["Close"]) < 30:
                continue

            price_series = hist["Close"].dropna()
            pred = linear_prediction(price_series)
            if not pred:
                continue

            shares = s.get("shares", 1)
            c_val = pred["current_price"] * shares
            p_val = pred["predicted_price"] * shares

            predictions.append({
                "symbol": s["symbol"],
                "name": s.get("name", s["symbol"]),
                "shares": shares,
                "pred": pred,
                "price_series": price_series
            })

            total_current += c_val
            total_predicted += p_val

        if predictions:
            diff = total_predicted - total_current
            pct = (diff / total_current * 100) if total_current else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Portfolio Value", f"${total_current:,.2f}")
            with col2:
                st.metric("Predicted Value (1 Year)", f"${total_predicted:,.2f}", f"{diff:+.2f} ({pct:+.2f}%)")
            with col3:
                st.metric("Trend", "Upward" if diff > 0 else "Downward")

            st.divider()
            st.subheader("Individual Stock Predictions")

            for item in predictions:
                symbol = item["symbol"]
                name = item["name"]
                pred = item["pred"]
                price_series = item["price_series"]

                with st.expander(f"{symbol} — {name}"):
                    render_prediction_summary(symbol, price_series)
                    render_prediction_chart(price_series, pred)
        else:
            st.warning("No predictions available.")

        if st.button("Hide Analytics"):
            st.session_state.show_my_stocks_analytics = False
            st.rerun()


def _remove_stock(symbol, portfolio):
    p_id = portfolio.get("_id")
    if p_id and p_id != "temp_id":
        success, msg = remove_stock_from_portfolio(p_id, symbol)
        if success:
            st.success(f"Removed {symbol}")
            st.rerun()
        else:
            st.error(msg)
    else:
        portfolio["stocks"] = [s for s in portfolio["stocks"] if s["symbol"] != symbol]
        st.session_state.current_portfolio = portfolio
        st.rerun()

def show_stock_historical_data(symbol, name):
    st.subheader(f"Historical Analysis: {symbol}")
    
    stock_info = get_stock_info_with_history(symbol)

    if stock_info and not stock_info['historical_data'].empty:
        historical_data = stock_info['historical_data']

        if 'price' in stock_info:
            st.metric("Current Price", f"${stock_info['price']:.2f}")
        if 'change' in stock_info:
            change_pct = (stock_info['change'] / stock_info['previous_price'] * 100) if stock_info.get('previous_price', 0) > 0 else 0
            st.metric("Daily Change", f"${stock_info['change']:+.2f}", f"{change_pct:+.2f}%")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")

        st.divider()

        if len(historical_data) > 0:
            first_price = historical_data['Close'].iloc[0]
            last_price = historical_data['Close'].iloc[-1]
            total_return = ((last_price - first_price) / first_price) * 100
            years = len(historical_data) / 252  # Approximate trading days per year
            annualized_return = ((last_price / first_price) ** (1/years) - 1) * 100 if years > 0 else 0

            st.metric("Total Return", f"{total_return:+.2f}%")
            st.metric("Annualized Return", f"{annualized_return:+.2f}%")
            st.metric("All-Time High", f"${historical_data['High'].max():.2f}")
            st.metric("All-Time Low", f"${historical_data['Low'].min():.2f}")

            st.divider()

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price History", "Volume", "Returns", "Statistics", "Prediction"])

            with tab1:
                st.subheader("Stock Price Over Time")
                timeframe = st.selectbox(
                    "Select Timeframe",
                    ["All Time", "Last 10 Years", "Last 5 Years", "Last 2 Years"],
                    key=f"timeframe_{symbol}"
                )

                if timeframe == "All Time":
                    chart_data = historical_data
                elif timeframe == "Last 10 Years":
                    chart_data = historical_data.tail(10 * 252)
                elif timeframe == "Last 5 Years":
                    chart_data = historical_data.tail(5 * 252)
                else:  # Last 2 Years
                    chart_data = historical_data.tail(2 * 252)

                st.line_chart(chart_data['Close'], height=400)

                st.subheader("OHLC Data")
                st.write("**Open Prices**")
                st.line_chart(chart_data['Open'], height=150)
                st.write("**High Prices**")
                st.line_chart(chart_data['High'], height=150)
                st.write("**Low Prices**")
                st.line_chart(chart_data['Low'], height=150)
                st.write("**Close Prices**")
                st.line_chart(chart_data['Close'], height=150)

            with tab2:
                st.subheader("Trading Volume Over Time")
                st.bar_chart(chart_data['Volume'], height=400)

                avg_volume = chart_data['Volume'].mean()
                max_volume = chart_data['Volume'].max()
                st.metric("Average Volume", f"{avg_volume:,.0f}")
                st.metric("Maximum Volume", f"{max_volume:,.0f}")

            with tab3:
                st.subheader("Daily Returns Analysis")
                returns = chart_data['Close'].pct_change().dropna()

                st.line_chart(returns * 100, height=300)

                st.metric("Avg Daily Return", f"{returns.mean() * 100:.2f}%")
                st.metric("Volatility", f"{returns.std() * 100:.2f}%")
                st.metric("Best Day", f"{returns.max() * 100:.2f}%")
                st.metric("Worst Day", f"{returns.min() * 100:.2f}%")

            with tab4:
                st.subheader("Detailed Statistics")

                stats_data = {
                    "Metric": ["Current Price", "52-Week High", "52-Week Low", "All-Time High", "All-Time Low",
                             "Total Return", "Annualized Return", "Volatility", "Average Volume", "Market Cap"],
                    "Value": []
                }

                recent_year = historical_data.tail(252) if len(historical_data) > 252 else historical_data
                fifty_two_week_high = recent_year['High'].max()
                fifty_two_week_low = recent_year['Low'].min()

                stats_data["Value"] = [
                    f"${stock_info.get('price', last_price):.2f}",
                    f"${fifty_two_week_high:.2f}",
                    f"${fifty_two_week_low:.2f}",
                    f"${historical_data['High'].max():.2f}",
                    f"${historical_data['Low'].min():.2f}",
                    f"{total_return:+.2f}%",
                    f"{annualized_return:+.2f}%",
                    f"{returns.std() * 100:.2f}%",
                    f"{historical_data['Volume'].mean():,.0f}",
                    f"${stock_info.get('info', {}).get('marketCap', 'N/A')}"
                ]

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                if stock_info.get('info'):
                    st.subheader("Company Information")
                    info = stock_info['info']

                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
                    st.write(f"**Beta:** {info.get('beta', 'N/A')}")

                    if info.get('longBusinessSummary'):
                        st.subheader("Business Summary")
                        st.write(info['longBusinessSummary'])
            with tab5:
                st.subheader("Price Prediction Using Linear Regression")

                data = historical_data['Close'].dropna()

                if len(data) >= 30:
                    prediction = linear_prediction(data, future_days=365)

                    if prediction:
                        current_price = prediction['current_price']
                        predicted_1year = prediction['predicted_price']
                        predicted_change = predicted_1year - current_price
                        predicted_change_pct = (predicted_change / current_price) * 100

                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Predicted Price (1 Year)", f"${predicted_1year:.2f}", f"{predicted_change:+.2f} ({predicted_change_pct:+.2f}%)")
                        st.metric("Model R² Score", f"{prediction['r_squared']:.4f}")

                        st.divider()

                        historical_dates = data.index
                        last_date = historical_dates[-1]
                        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                                    periods=365, freq='D')

                        st.subheader("Historical Data with Regression Line")

                        historical_chart_data = pd.DataFrame({
                            'Actual Price': data.values,
                            'Regression Line': prediction['reg_line']
                        }, index=historical_dates)

                        st.line_chart(historical_chart_data, height=400)

                        st.subheader("Future Price Prediction (Next Year)")

                        lookback_days = min(504, len(data))
                        recent_data = data.tail(lookback_days)
                        recent_dates = recent_data.index
                        recent_X = np.arange(len(data) - lookback_days, len(data))
                        recent_pred = prediction['slope'] * recent_X + prediction['intercept']

                        combined_dates = list(recent_dates) + list(future_dates)
                        combined_actual = list(recent_data.values) + [None] * 365
                        combined_predicted = list(recent_pred) + list(prediction['future_predictions'])

                        combined_df = pd.DataFrame({
                            'Historical Price': combined_actual,
                            'Predicted Price': [None] * lookback_days + list(prediction['future_predictions'])
                        }, index=combined_dates)

                        st.line_chart(combined_df, height=400)

                        st.divider()
                        st.subheader("Model Details")
                        st.write(f"**Regression Equation:** Price = {prediction['slope']:.4f} × Days + {prediction['intercept']:.2f}")
                        st.write(f"**Daily Trend:** {'Upward' if prediction['slope'] > 0 else 'Downward'} (${prediction['slope']:.4f} per day)")
                        st.write(f"**Training Data Points:** {len(data)} days")
                        st.write(f"**Prediction Period:** 365 days (1 year)")
                else:
                    st.warning("Not enough historical data for regression analysis. Need at least 30 days.")

    else:
        st.error(f"No historical data available for {symbol}")

def stock_search_page(go_to, get_user_info, change_password):
    """Search available stocks from all markets and add them to a portfolio."""

    # ---- Ensure portfolio exists ----
    if 'current_portfolio' not in st.session_state:
        st.warning("No portfolio selected.")
        if st.button("Go to Portfolios"):
            go_to("portfolios")
        st.stop()

    portfolio = st.session_state.current_portfolio
    portfolio_id = portfolio.get("_id")

    # ---- Sidebar ----
    render_sidebar(
        "Stock Search",
        back_button={'label': 'Back to My Stocks', 'callback': lambda: go_to("my_stocks")}
    )

    st.title("Stock Search")
    st.markdown("### Find and add stocks to your portfolio")

    # ---- Country dropdown ----
    available_countries = ["All"] + list(STOCK_SYMBOLS_BY_COUNTRY.keys())
    selected_country = st.selectbox("Country", available_countries)

    # ---- Search box ----
    search_query = st.text_input(
        "Search for stocks (symbol or company name)",
        placeholder="e.g., AAPL, Apple, Tesla"
    )

    st.divider()

    # ---- Load stock list ----
    if selected_country != "All":
        all_stocks = get_stocks_for_search(selected_country)
    else:
        all_stocks = []
        for country in STOCK_SYMBOLS_BY_COUNTRY.keys():
            all_stocks.extend(get_stocks_for_search(country))

    # ---- Filter results ----
    if search_query:
        filtered = [
            s for s in all_stocks
            if search_query.upper() in s["symbol"] or search_query.lower() in s["name"].lower()
        ]
    else:
        filtered = all_stocks

    # ---- Display results ----
    for stock in filtered:
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 1.5, 1.5, 2, 1])

            with col1:
                st.write(f"**{stock['symbol']}**")
            with col2:
                st.write(f"{stock['country']}")
            with col3:
                st.write(f"${stock['price']:.2f}")
            with col4:
                st.write(f"{stock['change']:+.2f}")

            # ---- ADD STOCK FORM ----
            with col5:
                with st.form(key=f"add_{stock['symbol']}"):
                    shares = st.number_input("Shares", min_value=1, value=1)
                    purchase_price = st.number_input(
                        "Purchase Price ($)",
                        min_value=0.01,
                        value=float(stock['price']),
                        step=0.01
                    )

                    submitted = st.form_submit_button("Add", use_container_width=True)

                    if submitted:
                        new_stock = {
                            'symbol': stock['symbol'],
                            'name': stock['name'],
                            'purchase_price': purchase_price,
                            'current_price': stock['price'],
                            'price': purchase_price,
                            'shares': shares
                        }

                        if portfolio_id and portfolio_id != "temp_id":
                            # Saved portfolio — update in database
                            success, msg = add_stock_to_portfolio(portfolio_id, new_stock)
                            if success:
                                st.success(f"Added {shares} shares of {stock['symbol']}!")
                                st.rerun()
                            else:
                                st.error(f"Failed: {msg}")
                        else:
                            # Temp portfolio — update session state
                            if 'stocks' not in st.session_state.current_portfolio:
                                st.session_state.current_portfolio['stocks'] = []
                            st.session_state.current_portfolio['stocks'].append(new_stock)
                            st.success(f"Added {shares} shares of {stock['symbol']}!")
                            st.rerun()
            with col6:
                if st.button("History", key=f"history_{stock['symbol']}"):
                    show_stock_historical_data(stock['symbol'], stock['name'])

            st.markdown("---")

        
def edit_portfolio_page(go_to, get_user_info, change_password):
    """Page allowing user to rename portfolio and modify stock quantities."""

    # Must have portfolio loaded
    if 'edit_portfolio_id' not in st.session_state:
        st.error("No portfolio selected")
        go_to("portfolios")
        return
    
    portfolio = get_portfolio_by_id(st.session_state.edit_portfolio_id)
    if not portfolio:
        st.error("Portfolio not found")
        go_to("portfolios")
        return
    
    portfolio_id = st.session_state.edit_portfolio_id

    render_sidebar("Edit Portfolio", back_button={'label': 'Back to Portfolios', 'callback': lambda: go_to("portfolios")})
    
    st.title("Edit Portfolio")
    
    st.subheader("Portfolio Information")
    
    col_name, col_name_btn = st.columns([3, 1])
    with col_name:
        if st.session_state.get('editing_portfolio_name'):
            new_portfolio_name = st.text_input(
                "Portfolio Name",
                value=portfolio['portfolio_name'],
                key="portfolio_name_input",
                help="Enter the new name for your portfolio"
            )
        else:
            st.markdown(f"**Current Name:** {portfolio['portfolio_name']}")
    
    with col_name_btn:
        st.write("") # Spacing
        if st.session_state.get('editing_portfolio_name'):
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key="save_name"):
                    if 'portfolio_name_input' in st.session_state and st.session_state.portfolio_name_input.strip():
                        success, message = update_portfolio(portfolio_id, {'portfolio_name': st.session_state.portfolio_name_input.strip()})
                        
                        if success:
                            st.success(f"Portfolio name updated to '{st.session_state.portfolio_name_input}'!")
                            st.session_state.editing_portfolio_name = False
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"Failed to update name: {message}")
                    else:
                        st.error("Please enter a valid portfolio name")
            
            with col_cancel:
                if st.button("Cancel", key="cancel_name"):
                    st.session_state.editing_portfolio_name = False
                    st.rerun()
        else:
            if st.button("Edit Name", key="edit_name"):
                st.session_state.editing_portfolio_name = True
                st.rerun()
    
    st.divider()
    
    st.subheader("Portfolio Summary")
    total_value = sum(stock.get('price', 0) * stock.get('shares', 1) for stock in portfolio.get('stocks', []))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Value", f"${total_value:.2f}")
    with col2:
        st.metric("Number of Stocks", len(portfolio.get('stocks', [])))
    
    st.divider()
    
    st.subheader("Manage Stocks")
    
    if portfolio.get('stocks'):
        stocks = portfolio['stocks']
        
        if 'stock_changes' not in st.session_state:
            st.session_state.stock_changes = {}
        
        updated_stocks = []
        stocks_to_remove = []
        
        for idx, stock in enumerate(stocks):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1, 1.5, 1])
                
                with col1:
                    st.write(f"**{stock['symbol']}**")
                
                with col2:
                    st.write(f"${stock.get('price', 0):.2f}")
                
                with col3:
                    current_shares = stock.get('shares', 1)
                    new_shares = st.number_input(
                        "Shares",
                        min_value=0,
                        max_value=10000,
                        value=current_shares,
                        key=f"shares_{stock['symbol']}_{idx}",
                        help="Set to 0 to remove stock"
                    )
                    
                    if new_shares != current_shares:
                        st.session_state.stock_changes[stock['symbol']] = new_shares
                
                with col4:
                    shares_to_use = st.session_state.stock_changes.get(stock['symbol'], current_shares)
                    total_stock_value = stock.get('price', 0) * shares_to_use
                    st.write(f"${total_stock_value:.2f}")
                
                with col5:
                    if st.button("Remove", key=f"remove_{stock['symbol']}_{idx}"):
                        st.session_state.stock_changes[stock['symbol']] = 0
                        stocks_to_remove.append(stock['symbol'])
                
                final_shares = st.session_state.stock_changes.get(stock['symbol'], current_shares)
                if final_shares > 0:
                    updated_stock = stock.copy()
                    updated_stock['shares'] = final_shares
                    updated_stock['value'] = stock.get('price', 0) * final_shares
                    updated_stocks.append(updated_stock)
                
                st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", type="primary", width="stretch"):
                final_stocks = [s for s in updated_stocks if s['shares'] > 0]
                
                for symbol in stocks_to_remove:
                    success, message = remove_stock_from_portfolio(portfolio_id, symbol)
                    if success:
                        st.success(f"Removed {symbol}")
                    else:
                        st.error(f"Failed to remove {symbol}: {message}")
                
                update_data = {'stocks': final_stocks}
                success, message = update_portfolio(portfolio_id, update_data)
                
                if success:
                    st.success("Portfolio updated successfully!")
                    st.session_state.stock_changes = {}  # Clear changes
                    st.balloons()
                else:
                    st.error(f"Failed to update portfolio: {message}")
        
        with col2:
            if st.button("Cancel Changes", width="stretch"):
                st.session_state.stock_changes = {}  # Clear changes

        st.divider()
        
        # ---- Add More Stocks Button ----
        if st.button("Add More Stocks", width="stretch"):
            st.session_state.current_portfolio = {
                '_id': portfolio_id,
                'name': portfolio['portfolio_name'],
                'countries': portfolio.get('countries', []),
                'stocks': stocks
            }
            go_to("stock_search")
    
    else:
        
        if st.button("Add Your First Stock", type="primary", width="stretch"):
            st.session_state.current_portfolio = {
                '_id': portfolio_id,
                'name': portfolio['portfolio_name'],
                'countries': portfolio['countries'],
                'stocks': []
            }
            go_to("stock_search")

def portfolio_details_page(go_to, get_user_info, change_password):
    """Render detailed portfolio view with stock performance analysis (refactored)."""

    # -------------------------------------------------------------------
    # Validate portfolio selection
    # -------------------------------------------------------------------
    if "view_portfolio_id" not in st.session_state:
        st.error("No portfolio selected for viewing.")
        go_to("portfolios")
        return

    portfolio_id = st.session_state.view_portfolio_id
    portfolio = get_portfolio_by_id(portfolio_id)

    if not portfolio:
        st.error("Portfolio not found.")
        go_to("portfolios")
        return

    stocks = portfolio.get("stocks", [])

    # -------------------------------------------------------------------
    # Sidebar (Info + reusable sidebar renderer)
    # -------------------------------------------------------------------
    st.sidebar.markdown(f"**Portfolio:** {portfolio['portfolio_name']}")
    st.sidebar.markdown(
        f"**Created:** {portfolio['created_at'].strftime('%Y-%m-%d') if portfolio.get('created_at') else 'Unknown'}"
    )
    st.sidebar.divider()

    render_sidebar(
        page_title="Portfolio Details",
        actions=[
            {
                "label": "Edit Portfolio",
                "callback": lambda: (
                    setattr(st.session_state, "edit_portfolio_id", portfolio_id),
                    setattr(st.session_state, "edit_portfolio_name", portfolio["portfolio_name"]),
                    go_to("edit_portfolio")
                ),
                "type": "primary",
                "key": "edit_portfolio_btn",
            },
            {
                "label": "Portfolio Analytics",
                "callback": lambda: (
                    setattr(st.session_state, "analytics_portfolio_id", portfolio_id),
                    go_to("portfolio_analytics")
                ),
                "key": "portfolio_analytics_btn",
            },
        ],
        back_button={
            "label": "Back to Portfolios",
            "callback": lambda: go_to("portfolios"),
        }
    )

    # ---- Page Heading ----
    st.title("Portfolio Details")
    st.markdown(f"### {portfolio['portfolio_name']}")

    # ---- No stocks yet ----
    if not stocks:
        if st.button("Add Stocks", type="primary"):
            st.session_state.current_portfolio = {
                "_id": portfolio_id,
                "name": portfolio["portfolio_name"],
                "countries": portfolio["countries"],
                "stocks": [],
            }
            go_to("stock_search")
        return

    # ---- Summary Metrics (Purchase Value, Current Value, Gain/Loss) ----
    render_portfolio_summary(stocks)

    # ---- Table of Holdings ----
    st.subheader("Stock Holdings Detail")
    render_stock_table(stocks)

    st.divider()

    # ---- Individual Stock Performance Cards ----
    st.subheader("Individual Stock Performance")
    render_stock_performance_grid(stocks)

    # ---- Future expansion: analytics section toggle ----
    if st.session_state.get("show_portfolio_details_analytics", False):
        st.divider()
        st.subheader("Portfolio Analytics & Predictions")

        if st.button("Hide Analytics"):
            st.session_state.show_portfolio_details_analytics = False
            st.rerun()

def portfolio_analytics_page(go_to, get_user_info, change_password):
    render_sidebar(
        page_title="Portfolio Analytics",
        back_button={'label': "Back to Portfolios", 'callback': lambda: go_to("portfolios")}
    )

    st.title("Portfolio Analytics & Predictions")

    # Determine scope (single portfolio or all)
    if "analytics_portfolio_id" in st.session_state:
        portfolio = get_portfolio_by_id(st.session_state.analytics_portfolio_id)
        if not portfolio:
            st.error("Portfolio not found.")
            return
        st.markdown(f"### Analyzing: {portfolio['portfolio_name']}")
        stocks = portfolio.get('stocks', [])
    else:
        st.markdown("### Analyzing All Portfolios")
        portfolios = get_user_portfolios(st.session_state.username)
        if not portfolios:
            st.warning("No portfolios available.")
            return
        stocks = [s for p in portfolios for s in p.get("stocks", [])]

    if not stocks:
        st.info("No stocks available to analyze.")
        return

    # ---- Overall Prediction ----
    st.subheader("Overall Portfolio Value Prediction")

    total_current = 0
    total_predicted = 0
    predictions = []

    for s in stocks:
        hist = fetch_long_history(s["symbol"], years=2)

        if hist.empty or len(hist["Close"]) < 30:
            continue

        price_series = hist["Close"].dropna()
        pred = linear_prediction(price_series)

        if not pred:
            continue

        shares = s.get("shares", 1)

        c_val = pred["current_price"] * shares
        p_val = pred["predicted_price"] * shares

        total_current += c_val
        total_predicted += p_val

        predictions.append({
            "symbol": s["symbol"],
            "name": s.get("name", s["symbol"]),
            "shares": shares,
            "pred": pred,
            "price_series": price_series
        })

    if not predictions:
        st.warning("Not enough historical data for predictions.")
        return

    diff = total_predicted - total_current
    pct = (diff / total_current * 100) if total_current else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Portfolio Value", f"${total_current:,.2f}")
    with col2:
        st.metric("Predicted Value (1 Year)", f"${total_predicted:,.2f}",
                  f"{diff:+.2f} ({pct:+.2f}%)")
    with col3:
        st.metric("Trend", "Upward" if diff > 0 else "Downward")

    st.divider()

    # ---- Individual Prediction Panels ----
    st.subheader("Individual Stock Predictions")

    for item in predictions:
        symbol = item["symbol"]
        name = item["name"]
        price_series = item["price_series"]
        pred = item["pred"]

        with st.expander(f"{symbol} — {name}"):
            render_prediction_summary(symbol, price_series)
            render_prediction_chart(price_series, pred)

def media_portfolio_view_page(go_to, get_user_info, change_password):
    """Read-only community portfolio view — fully refactored."""

    # ---- Validate portfolio selection ----
    if "media_portfolio_id" not in st.session_state:
        st.error("No portfolio selected for viewing.")
        go_to("dashboard")
        return

    portfolio_id = st.session_state.media_portfolio_id
    portfolio = get_portfolio_by_id(portfolio_id)

    if not portfolio:
        st.error("Portfolio not found.")
        go_to("dashboard")
        return

    owner_username = st.session_state.get(
        "media_portfolio_owner",
        portfolio.get("user_id", "Unknown User")
    )

    stocks = portfolio.get("stocks", [])

    # ---- Sidebar (Info + reusable sidebar renderer) ----
    st.sidebar.markdown(f"**Owner:** {owner_username}")
    st.sidebar.markdown(
        f"**Created:** {portfolio['created_at'].strftime('%Y-%m-%d')}"
        if portfolio.get("created_at") else "**Created:** Unknown"
    )
    st.sidebar.divider()

    render_sidebar(
        page_title="Community Portfolio",
        actions=None,
        back_button={
            "label": "Back to Dashboard",
            "callback": lambda: go_to("dashboard")
        }
    )

    # ---- Page Header ----
    st.title(f"{owner_username}'s Portfolio")

    if not stocks:
        st.info("This portfolio contains no stocks.")
        return

    # Summary + Table + Individual stock cards
    render_portfolio_summary(stocks)

    st.subheader("Stock Holdings Detail")
    render_stock_table(stocks)

    st.divider()

    st.subheader("Individual Stock Performance")
    render_stock_performance_grid(stocks)

    st.divider()

    # ---- Prediction Analytics ----
    st.subheader("Portfolio Prediction Analytics")

    total_current = 0
    total_predicted = 0
    predictions = []

    for s in stocks:
        hist = fetch_long_history(s["symbol"], years=2)

        if hist.empty or len(hist["Close"]) < 30:
            continue

        price_series = hist["Close"].dropna()
        pred = linear_prediction(price_series)

        if not pred:
            continue

        shares = s.get("shares", 1)

        current_val = pred["current_price"] * shares
        predicted_val = pred["predicted_price"] * shares

        total_current += current_val
        total_predicted += predicted_val

        predictions.append({
            "symbol": s["symbol"],
            "name": s.get("name", s["symbol"]),
            "shares": shares,
            "pred": pred,
            "price_series": price_series
        })

    if not predictions:
        st.warning("No predictions available — insufficient historical data.")
        return

    # ---- Portfolio-level prediction summary ----
    diff = total_predicted - total_current
    pct = (diff / total_current * 100) if total_current else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Portfolio Value", f"${total_current:,.2f}")
    with col2:
        st.metric(
            "Predicted Value (1 Year)",
            f"${total_predicted:,.2f}",
            f"{diff:+.2f} ({pct:+.2f}%)"
        )
    with col3:
        st.metric("Trend", "Upward" if diff > 0 else "Downward")

    st.divider()

    # ---- Individual prediction panels ----
    st.subheader("Individual Stock Predictions")

    for item in predictions:
        symbol = item["symbol"]
        name = item["name"]
        series = item["price_series"]
        pred = item["pred"]

        with st.expander(f"{symbol} — {name}"):

            # Unified prediction metrics
            render_prediction_summary(symbol, series)

            # Unified prediction chart
            render_prediction_chart(series, pred)