import streamlit as st
import pandas as pd
import numpy as np
import portfolio as pf
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        font-weight: 500;
        color: #6c757d;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="title">Portfolio Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Track your investment performance and analyze your trading patterns</p>', unsafe_allow_html=True)

# Create sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Current price input
    current_price = st.number_input(
        "Current Market Price ($)",
        min_value=0.01,
        step=0.01,
        value=150.0,  # Set a default value
        help="Enter the current market price to update metrics"
    )
    
    # Add a button to refresh the analysis
    refresh = st.button("Refresh Analysis", use_container_width=True)
    
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This dashboard visualizes your portfolio metrics
    and trading patterns from your transaction history.
    
    Enter the current market price to see updated metrics.
    """)

# Initialize portfolio analyzer - don't use caching for now
def load_analyzer(price):
    return pf.PortfolioAnalyzer('transactions.csv', price)

analyzer = load_analyzer(current_price)

# Calculate metrics
basic_metrics = analyzer.calculate_basic_metrics()
returns = analyzer.calculate_portfolio_returns()
patterns = analyzer.analyze_trading_patterns()
final_metrics = returns['final_metrics']

# Display main metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Total Investment",
        f"${basic_metrics['total_investment']:.2f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    realized_pl = basic_metrics['realized_pl']
    delta_color = "normal" if realized_pl < 0 else "normal"
    st.metric(
        "Realized P/L",
        f"${realized_pl:.2f}",
        delta=None,
        delta_color=delta_color
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    unrealized_pl = final_metrics['unrealized_pl']
    delta_color = "normal" if unrealized_pl < 0 else "normal"
    st.metric(
        "Unrealized P/L",
        f"${unrealized_pl:.2f}",
        delta=None,
        delta_color=delta_color
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Current Shares",
        f"{final_metrics['current_shares']:.0f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Return metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Average Cost Basis",
        f"${final_metrics['avg_cost_basis']:.2f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    unrealized_return = final_metrics['unrealized_return_pct']
    delta_color = "normal" if unrealized_return < 0 else "normal"
    st.metric(
        "Unrealized Return",
        f"{unrealized_return:.2f}%",
        delta=None,
        delta_color=delta_color
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    combined_return = final_metrics['combined_return_pct']
    delta_color = "normal" if combined_return < 0 else "normal"
    st.metric(
        "Combined Return",
        f"{combined_return:.2f}%",
        delta=None,
        delta_color=delta_color
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Generate and display chart - don't use caching
st.markdown("### Portfolio Analysis")

# Create the chart
analyzer.plot_portfolio_analysis()

# Display the chart
try:
    chart = Image.open('portfolio_analysis.png')
    st.image(chart, use_column_width=True)
except Exception as e:
    st.error(f"Could not load chart: {e}")

# Add tabs for additional data
tab1, tab2 = st.tabs(["Trading Patterns", "Transaction History"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Trades", patterns['total_trades'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Buy Trades", patterns['buy_trades'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sell Trades", patterns['sell_trades'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Buy Size", f"${patterns['avg_buy_size']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Sell Size", f"${patterns['avg_sell_size']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        most_active = str(patterns['most_active_month'])
        st.metric("Most Active Month", most_active)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Load transaction data
    def load_transactions():
        df = pd.read_csv('transactions.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['Price_Per_Share'] = abs(df['Amount'] / df['Shares'])
        return df
    
    transactions = load_transactions()
    
    # Display transactions table with styling
    st.dataframe(
        transactions,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
            "Transaction Type": st.column_config.TextColumn("Type"),
            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("Shares", format="%.2f"),
            "Fee": st.column_config.NumberColumn("Fee", format="$%.2f"),
            "Price_Per_Share": st.column_config.NumberColumn("Price/Share", format="$%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )

# Add returns data visualization
st.markdown("### Returns Timeline")

# Format returns data
returns_data = returns['returns_data']
for entry in returns_data:
    if isinstance(entry['date'], str):
        entry['date'] = pd.to_datetime(entry['date'])

df_returns = pd.DataFrame(returns_data)

# Display returns timeline
if not df_returns.empty:
    # Format dataframe for display
    display_returns = df_returns.copy()
    display_returns['date'] = display_returns['date'].dt.strftime('%Y-%m-%d')
    
    # Display styled dataframe
    st.dataframe(
        display_returns,
        column_config={
            "date": st.column_config.TextColumn("Date"),
            "action": st.column_config.TextColumn("Action"),
            "shares": st.column_config.NumberColumn("Position Size", format="%.2f"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "avg_cost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
            "unrealized_return_pct": st.column_config.NumberColumn("Unrealized Return %", format="%.2f%%"),
            "realized_return_pct": st.column_config.NumberColumn("Realized Return %", format="%.2f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No returns data available.") 