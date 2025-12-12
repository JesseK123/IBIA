import streamlit as st

from login import (
    verify_user,
    register_user,
    get_user_info,
    change_password,
    update_last_login,
    create_portfolio,
    get_user_portfolios,
    get_all_portfolios,
    get_portfolio_by_id,
    update_portfolio,
    delete_portfolio,
    add_stock_to_portfolio,
    remove_stock_from_portfolio
)

from ui import (
    login_page,
    register_page,
    dashboard_page,
    stock_analysis_page,
    portfolios_page,
    create_portfolio_page,
    my_stocks_page,
    stock_search_page,
    edit_portfolio_page,
    portfolio_details_page,
    portfolio_analytics_page,
    media_portfolio_view_page,
)

from database import initialize_database


# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Investment Portfolio App",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="J",
)

# ---- INITIALIZE DATABASE ----
if "db_initialized" not in st.session_state:
    ok = initialize_database()
    st.session_state.db_initialized = ok

    if not ok:
        st.error("Failed to initialise database. Check MongoDB connection settings.")
        st.stop()

# ---- SESSION STATE DEFAULTS ----
st.session_state.setdefault("page", "login")
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("username", "")


# ---- ROUTING HELPER FUNCTION ----
def go_to(page: str):
    """Safe navigation (removes accidental trailing slashes)."""
    clean_page = page.strip().rstrip("/")
    st.session_state.page = clean_page
    st.rerun()


# ---- PAGE ROUTER (clean, scalable) ----
ROUTES = {
    "dashboard": dashboard_page,
    "stock_analysis": stock_analysis_page,
    "portfolios": portfolios_page,
    "create_portfolio": create_portfolio_page,
    "my_stocks": my_stocks_page,
    "stock_search": stock_search_page,
    "edit_portfolio": edit_portfolio_page,
    "portfolio_details": portfolio_details_page,
    "portfolio_analytics": portfolio_analytics_page,
    "media_portfolio_view": media_portfolio_view_page,
}


# ---- MAIN APPLICATION ----
def main():

    # -------------------------
    # CUSTOM CSS
    # -------------------------
    st.markdown("""
    <style>
    .status-indicator {
        position: fixed;
        top: 12px;
        right: 12px;
        background: #28a745;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        z-index: 9999;
    }
    .positive-percentage { color: #28a745; font-weight: bold; }
    .negative-percentage { color: #dc3545; font-weight: bold; }
    .neutral-percentage  { color: #6c757d; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # ---- CONNECTION BADGE ----
    if st.session_state.db_initialized:
        st.markdown('<div class="status-indicator">Connected</div>', unsafe_allow_html=True)

    # ---- AUTH ROUTING  ----
    if st.session_state.logged_in:

        page = st.session_state.get("page", "dashboard")
        handler = ROUTES.get(page)

        if handler:
            handler(go_to, get_user_info, change_password)
        else:
            # Safe fallback
            st.warning(f"Unknown page '{page}'. Redirecting to dashboard.")
            go_to("dashboard")

    else:

        # PUBLIC PAGES
        if st.session_state.page == "register":
            register_page(go_to, register_user)
        else:
            login_page(go_to, verify_user, update_last_login)


# ---- RUN MAIN FUNCTION ----
if __name__ == "__main__":
    main()
