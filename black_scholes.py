import streamlit as st
import numpy as np
import scipy.stats as si
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta


def fetch_stock_data(ticker):
    """
    Fetch stock data from Yahoo Finance
    """
    try:
        stock = yf.Ticker(ticker)
        # Get current price and other information
        info = stock.info
        
        # Get historical data to calculate volatility
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        hist = stock.history(start=start_date, end=end_date)
        
        # Calculate annual volatility from daily returns
        if len(hist) > 0:
            daily_returns = hist['Close'].pct_change().dropna()
            annual_volatility = daily_returns.std() * np.sqrt(252)  # Annualize daily volatility
        else:
            annual_volatility = None
            
        return {
            'price': info.get('regularMarketPrice', info.get('currentPrice', None)),
            'name': info.get('shortName', ticker),
            'volatility': annual_volatility,
            'dividend_yield': info.get('dividendYield', 0),
            'success': True
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return {'success': False}


def black_scholes(S, K, T, r, sigma, option_type="call", q=0):
    """
    Calculate option price and Greeks using Black-Scholes model
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity in years
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    option_type: "call" or "put"
    q: Dividend yield (decimal)
    
    Returns: Dictionary with price and Greeks
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if option_type == "call":
        price = S * np.exp(-q * T) * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        prob_itm = si.norm.cdf(d2)
        delta = np.exp(-q * T) * si.norm.cdf(d1)
        theta = -((S * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))) * si.norm.pdf(d1) - r * K * np.exp(-r * T) * si.norm.cdf(d2) + q * S * np.exp(-q * T) * si.norm.cdf(d1)
    else:  # put option
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * np.exp(-q * T) * si.norm.cdf(-d1)
        prob_itm = si.norm.cdf(-d2)
        delta = np.exp(-q * T) * (si.norm.cdf(d1) - 1)
        theta = -((S * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))) * si.norm.pdf(d1) + r * K * np.exp(-r * T) * si.norm.cdf(-d2) - q * S * np.exp(-q * T) * si.norm.cdf(-d1)
    
    # Common Greeks for both call and put
    gamma = (np.exp(-q * T) * si.norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * si.norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 to represent change per 1% volatility
    rho = K * T * np.exp(-r * T) * (si.norm.cdf(d2) if option_type == "call" else -si.norm.cdf(-d2)) / 100  # Divided by 100 to represent change per 1% rate
    
    return {
        "price": price,
        "prob_itm": prob_itm,
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,  # Convert to daily theta
        "vega": vega,
        "rho": rho
    }


def main():
    st.set_page_config(layout="wide", page_title="Black-Scholes Probability Matrix")
    
    st.title("Black-Scholes Probability Matrix")
    st.write("This app calculates the probability of options finishing in-the-money using the Black-Scholes model.")
    
    # Add stock ticker input
    st.subheader("Stock Selection")
    col_ticker, col_fetch = st.columns([3, 1])
    
    with col_ticker:
        ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
    
    with col_fetch:
        fetch_button = st.button("Fetch Stock Data")
    
    # Initialize stock data
    if "stock_data" not in st.session_state:
        st.session_state.stock_data = None
    
    # Fetch stock data when button is clicked
    if fetch_button:
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data = fetch_stock_data(ticker)
            if stock_data['success']:
                st.session_state.stock_data = stock_data
                st.success(f"Successfully fetched data for {stock_data['name']}")
            else:
                st.error(f"Failed to fetch data for {ticker}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Option Parameters")
        
        # Stock price input - use fetched data if available
        if st.session_state.stock_data and st.session_state.stock_data['price']:
            S = st.number_input("Current Stock Price (S)", 
                              min_value=0.01, 
                              value=float(st.session_state.stock_data['price']),
                              step=0.01,
                              format="%.2f")
            
            if st.session_state.stock_data['name']:
                st.caption(f"Stock: {st.session_state.stock_data['name']}")
        else:
            S = st.number_input("Current Stock Price (S)", min_value=0.01, value=100.0, step=0.01, format="%.2f")
        
        K = st.number_input("Strike Price (K)", min_value=0.01, value=S, step=1.0, format="%.2f")
        T = st.number_input("Time to Expiry (T) in years", min_value=0.001, max_value=10.0, value=1.0, step=0.01)
        r = st.number_input("Risk-Free Rate (r) as a decimal", min_value=0.0, max_value=0.5, value=0.05, step=0.001, format="%.3f")
        
        # Volatility input - use fetched data if available
        if st.session_state.stock_data and st.session_state.stock_data.get('volatility'):
            sigma = st.number_input("Volatility (sigma) as a decimal", 
                                  min_value=0.01, 
                                  max_value=2.0, 
                                  value=float(st.session_state.stock_data['volatility']),
                                  step=0.01,
                                  format="%.2f")
            st.caption(f"Historical 1-year volatility: {st.session_state.stock_data['volatility']:.2f}")
        else:
            sigma = st.number_input("Volatility (sigma) as a decimal", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
        
        # Dividend yield input - use fetched data if available
        if st.session_state.stock_data and st.session_state.stock_data.get('dividend_yield') is not None:
            q = st.number_input("Dividend Yield (q) as a decimal", 
                              min_value=0.0, 
                              max_value=100, #OLD VAL WAS 0.5 
                              value=float(st.session_state.stock_data['dividend_yield']),
                              step=0.001,
                              format="%.3f")
            if st.session_state.stock_data['dividend_yield'] > 0:
                st.caption(f"Current dividend yield: {st.session_state.stock_data['dividend_yield']:.3f}")
        else:
            q = st.number_input("Dividend Yield (q) as a decimal", min_value=0.0, max_value=0.5, value=0.0, step=0.001, format="%.3f")
            
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        # Calculate option price and Greeks
        results = black_scholes(S, K, T, r, sigma, option_type, q)
        
        # Display results
        st.subheader("Option Pricing Results")
        
        # Create a better layout for displaying all Greeks
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Option Price", f"${results['price']:.2f}")
            st.metric("Probability ITM", f"{results['prob_itm']*100:.1f}%")
        with col1b:
            st.metric("Delta", f"{results['delta']:.4f}")
            st.metric("Gamma", f"{results['gamma']:.4f}")
        
        # Add the remaining Greeks in a new row
        col1c, col1d, col1e = st.columns(3)
        with col1c:
            st.metric("Theta", f"{results['theta']:.4f}")
            st.caption("Daily time decay")
        with col1d:
            st.metric("Vega", f"{results['vega']:.4f}")
            st.caption("Per 1% vol change")
        with col1e:
            st.metric("Rho", f"{results['rho']:.4f}")
            st.caption("Per 1% rate change")

    with col2:
        # Probability Matrix
        st.subheader("In-The-Money Probability Matrix")
        
        # Number of data points to use for the matrix
        num_points = st.slider("Matrix Resolution", min_value=5, max_value=30, value=15)
        
        # Create ranges for stock price and strike price
        stock_range = np.linspace(max(0.6 * S, S - 2 * S * sigma * np.sqrt(T)), 
                                 S + 2 * S * sigma * np.sqrt(T), num_points)
        strike_range = np.linspace(max(0.6 * K, K - 2 * S * sigma * np.sqrt(T)), 
                                  K + 2 * S * sigma * np.sqrt(T), num_points)
        
        # Create the matrix
        Z = np.zeros((len(stock_range), len(strike_range)))
        
        for i in range(len(stock_range)):
            for j in range(len(strike_range)):
                Z[i, j] = black_scholes(stock_range[i], strike_range[j], T, r, sigma, option_type, q)["prob_itm"]
        
        # Format the stock and strike ranges for display
        stock_labels = [f"${x:.2f}" for x in stock_range]
        strike_labels = [f"${x:.2f}" for x in strike_range]
        
        # Add annotations to the heatmap
        annotations = []
        for i in range(len(stock_range)):
            for j in range(len(strike_range)):
                if i % 2 == 0 and j % 2 == 0:  # Only show some annotations to avoid overcrowding
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{Z[i, j]*100:.1f}%",
                            showarrow=False,
                            font=dict(size=8, color="black" if Z[i, j] < 0.5 else "white")
                        )
                    )
        
        # Create the heatmap with green-red colorscale
        fig = go.Figure(data=go.Heatmap(
            z=Z * 100,  # Convert to percentage
            x=strike_labels,
            y=stock_labels,
            colorscale=[[0, 'rgb(255, 100, 100)'], [0.5, 'rgb(255, 255, 200)'], [1, 'rgb(100, 200, 100)']],  # Red to Green
            zmin=0,
            zmax=100,
            colorbar=dict(title="Probability ITM (%)")
        ))

        # Update layout without annotations
        fig.update_layout(
            xaxis_title="Strike Price",
            yaxis_title="Stock Price",
            title=f"Probability of {option_type.capitalize()} Option Finishing In-The-Money (%)",
            height=600
        )
        
        # Mark the current stock price and strike price
        fig.add_shape(
            type="line",
            x0=0,
            y0=stock_labels.index(f"${S:.2f}") if f"${S:.2f}" in stock_labels else stock_labels.index(min(stock_labels, key=lambda x: abs(float(x.replace('$', '')) - S))),
            x1=len(strike_labels) - 1,
            y1=stock_labels.index(f"${S:.2f}") if f"${S:.2f}" in stock_labels else stock_labels.index(min(stock_labels, key=lambda x: abs(float(x.replace('$', '')) - S))),
            line=dict(color="black", width=2, dash="dash"),
        )
        
        fig.add_shape(
            type="line",
            x0=strike_labels.index(f"${K:.2f}") if f"${K:.2f}" in strike_labels else strike_labels.index(min(strike_labels, key=lambda x: abs(float(x.replace('$', '')) - K))),
            y0=0,
            x1=strike_labels.index(f"${K:.2f}") if f"${K:.2f}" in strike_labels else strike_labels.index(min(strike_labels, key=lambda x: abs(float(x.replace('$', '')) - K))),
            y1=len(stock_labels) - 1,
            line=dict(color="black", width=2, dash="dash"),
        )
        
        # Add a marker for the current option parameters
        current_strike_idx = strike_labels.index(f"${K:.2f}") if f"${K:.2f}" in strike_labels else strike_labels.index(min(strike_labels, key=lambda x: abs(float(x.replace('$', '')) - K)))
        current_stock_idx = stock_labels.index(f"${S:.2f}") if f"${S:.2f}" in stock_labels else stock_labels.index(min(stock_labels, key=lambda x: abs(float(x.replace('$', '')) - S)))
        
        fig.add_annotation(
            x=current_strike_idx,
            y=current_stock_idx,
            text="Current",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            font=dict(size=12, color="black"),
            ax=20,
            ay=-30
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of the matrix
        st.write("""
        ### Understanding the Probability Matrix:
        - The matrix shows the probability of the option finishing In-The-Money at expiration.
        - The y-axis represents different stock prices, and the x-axis represents different strike prices.
        - Green areas indicate higher probability of finishing in-the-money (profitable).
        - Red areas indicate lower probability of finishing in-the-money (unprofitable).
        - The current stock price and strike price are marked with black dashed lines.
        - The probability changes as stock price and strike price vary.
        """)
    
    # Add educational information
    with st.expander("About the Black-Scholes Model"):
        st.write("""
        ## Black-Scholes Option Pricing Model
        
        The Black-Scholes model is a mathematical model used for pricing options contracts. It was developed by Fischer Black and Myron Scholes in 1973 and later expanded by Robert Merton.
        
        ### Key Assumptions:
        1. The stock follows a geometric Brownian motion with constant volatility
        2. No transaction costs or taxes
        3. Risk-free interest rate is constant
        4. No dividends during the option's life (though this app includes a dividend yield adjustment)
        5. European-style options (can only be exercised at expiration)
        
        ### Key Parameters:
        - **S**: Current stock price
        - **K**: Strike price
        - **T**: Time to maturity (in years)
        - **r**: Risk-free interest rate
        - **σ**: Volatility of the underlying asset
        - **q**: Dividend yield
        
        ### Greeks Explained:
        - **Delta**: Rate of change of option price with respect to the underlying asset's price
        - **Gamma**: Rate of change of delta with respect to the underlying asset's price
        - **Theta**: Rate of change of option price with respect to time (time decay)
        - **Vega**: Rate of change of option price with respect to volatility
        - **Rho**: Rate of change of option price with respect to the risk-free interest rate
        """)
    
    st.caption("Note: This app uses the yfinance library to fetch real-time stock data. Accuracy depends on the data available from Yahoo Finance.")


if __name__ == "__main__":
    main()
