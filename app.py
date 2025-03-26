import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

def get_historical_price(ticker, date):
    """Get historical price for a ticker on a specific date"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=date, end=date + pd.DateOffset(days=1))
        if not hist.empty:
            return hist['Close'].iloc[0]
        hist = stock.history(period='max')
        if not hist.empty:
            hist = hist[hist.index <= date]
            return hist['Close'].iloc[-1] if not hist.empty else None
        return None
    except Exception as e:
        st.error(f"Error fetching price for {ticker}: {e}")
        return None

def get_live_market_data():
    """Fetch live market data for key indicators"""
    data = {}
    try:
        # Get INR exchange rate first
        inr = yf.Ticker("INR=X")
        inr_hist = inr.history(period='1d')
        inr_rate = inr_hist['Close'].iloc[-1] if not inr_hist.empty else None
        
        # Nifty 50
        nifty = yf.Ticker("^NSEI")
        nifty_hist = nifty.history(period='1d')
        data['nifty'] = nifty_hist['Close'].iloc[-1] if not nifty_hist.empty else None
        
        # Gold Price (USD to INR converted)
        if inr_rate:
            gold = yf.Ticker("GC=F")
            gold_hist = gold.history(period='1d')
            if not gold_hist.empty:
                data['gold'] = gold_hist['Close'].iloc[-1] * inr_rate
        
        # Bitcoin (USD to INR converted)
        if inr_rate:
            bitcoin = yf.Ticker("BTC-USD")
            btc_hist = bitcoin.history(period='1d')
            if not btc_hist.empty:
                data['bitcoin'] = btc_hist['Close'].iloc[-1] * inr_rate
            
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
    
    return data

def format_currency(value):
    """Format numeric values as currency strings"""
    return f"₹{value:,.2f}" if not pd.isna(value) and value is not None else "N/A"

def user_inputs():
    """User input section with live market data display"""
    st.sidebar.header("Investor Profile")
    risk_level = st.sidebar.slider("Risk Tolerance (1-10)", 1, 10, 5)
    st.session_state.risk_level = risk_level
    
    # Live market data display
    st.header("Live Market Data")
    market_data = get_live_market_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        nifty_val = market_data.get('nifty', None)
        st.metric("Nifty 50", f"{nifty_val:.2f}" if nifty_val else "N/A")
    with col2:
        gold_val = market_data.get('gold', None)
        st.metric("Gold (per oz)", format_currency(gold_val))
    with col3:
        btc_val = market_data.get('bitcoin', None)
        st.metric("Bitcoin", format_currency(btc_val))
    
    # Portfolio input section
    st.header("Portfolio Input")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker_input = st.text_input("Stock Tickers (comma separated)", "TCS.NS,INFY.NS")
    with col2:
        date = st.date_input("Purchase Date", datetime(2020, 1, 1))
    with col3:
        amount = st.number_input("Total Amount (₹)", 1000, 10000000, 100000)
    
    if st.button("Add Investment"):
        raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        num_tickers = len(raw_tickers)
        
        if num_tickers == 0:
            st.error("Please enter at least one ticker")
            return
            
        amount_per_ticker = amount / num_tickers
        
        for raw_ticker in raw_tickers:
            ticker = raw_ticker
            if '.' not in ticker:
                ticker += '.NS'
            
            purchase_price = get_historical_price(ticker, date)
            
            if purchase_price and not np.isnan(purchase_price):
                st.session_state.portfolio.append({
                    'ticker': ticker,
                    'date': date,
                    'amount': amount_per_ticker,
                    'purchase_price': purchase_price
                })
                st.success(f"Added {ticker} with ₹{amount_per_ticker:,.2f}")
            else:
                st.error(f"Could not retrieve valid price for {ticker}")

def main_dashboard():
    """Main dashboard with portfolio analysis and optimization"""
    st.header("Portfolio Analysis")
    
    if not st.session_state.portfolio:
        st.info("Add investments using the form above")
        return
    
    # Current holdings display
    st.subheader("Current Holdings")
    holdings_df = pd.DataFrame(st.session_state.portfolio)
    
    # Calculate current values
    current_prices = {}
    for ticker in holdings_df['ticker'].unique():
        try:
            current_prices[ticker] = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        except:
            current_prices[ticker] = np.nan
    
    holdings_df['current_price'] = holdings_df['ticker'].map(current_prices)
    holdings_df['current_value'] = holdings_df['amount'] / holdings_df['purchase_price'] * holdings_df['current_price']
    holdings_df['pct_change'] = (holdings_df['current_price'] / holdings_df['purchase_price'] - 1) * 100
    
    # Format dataframe without using .style.format()
    formatted_df = holdings_df.copy()
    formatted_df['amount'] = formatted_df['amount'].apply(format_currency)
    formatted_df['purchase_price'] = formatted_df['purchase_price'].apply(lambda x: f"₹{x:.2f}")
    formatted_df['current_price'] = formatted_df['current_price'].apply(lambda x: f"₹{x:.2f}" if not pd.isna(x) else "N/A")
    formatted_df['current_value'] = formatted_df['current_value'].apply(format_currency)
    formatted_df['pct_change'] = formatted_df['pct_change'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(formatted_df)
    
    # Portfolio optimization
    st.subheader("Portfolio Optimization")
    unique_tickers = list(set(holdings_df['ticker']))
    
    try:
        # Get historical data
        data = yf.download(unique_tickers, period='5y')['Adj Close']
        data = data.dropna(axis=1, how='all')
        
        if data.empty:
            st.error("Insufficient historical data for optimization")
            return
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Risk-adjusted optimization
        risk_level = st.session_state.get('risk_level', 5)
        
        # Calculate target return based on risk profile
        ef_minvol = EfficientFrontier(mu, S)
        ef_minvol.min_volatility()
        min_ret, min_vol, _ = ef_minvol.portfolio_performance()
        max_ret = mu.max()
        target_ret = min_ret + (max_ret - min_ret) * (risk_level - 1) / 9
        
        # Perform optimization
        ef = EfficientFrontier(mu, S)
        ef.efficient_return(target_ret)
        weights = ef.clean_weights()
        
        # Prepare comparison data
        optimized_weights = pd.DataFrame.from_dict(weights, orient='index', columns=['Optimized Weight'])
        
        # Calculate current weights
        current_total = holdings_df['current_value'].sum()
        current_weights = holdings_df.groupby('ticker')['current_value'].sum() / current_total
        current_weights_df = pd.DataFrame(current_weights, columns=['Current Weight'])
        
        # Create comparison dataframe
        comparison_df = pd.concat([current_weights_df, optimized_weights], axis=1)
        comparison_df['Difference'] = comparison_df['Optimized Weight'] - comparison_df['Current Weight']
        
        # Format percentages
        comparison_df['Current Weight'] = comparison_df['Current Weight'].apply(lambda x: f"{x:.2%}")
        comparison_df['Optimized Weight'] = comparison_df['Optimized Weight'].apply(lambda x: f"{x:.2%}")
        comparison_df['Difference'] = comparison_df['Difference'].apply(lambda x: f"{x:+.2%}")
        
        # Display results
        st.write("Current vs. Optimized Allocation")
        st.dataframe(comparison_df)
        
        # Display performance metrics
        st.subheader("Optimized Portfolio Performance")
        ret, vol, sharpe = ef.portfolio_performance()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Annual Return", f"{ret:.2%}")
        with col2:
            st.metric("Annual Volatility", f"{vol:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Portfolio Optimizer", 
        layout="wide",
        page_icon="📈"
    )
    st.title("AI-Powered Portfolio Optimizer")
    
    user_inputs()
    main_dashboard()

if _name_ == "_main_":
    main()