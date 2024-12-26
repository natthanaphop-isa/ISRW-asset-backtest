import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import streamlit as st

# Function to plot the price chart
def plot_price_chart(ticker, stock_prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices.index, y=stock_prices, mode='lines', name='Price', line=dict(color='blue')))
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Function to calculate and display backtest results
def backtestStocks_plotly(stocks, start, end):
    try:
        stock_data = yf.download(stocks, start=start, end=end)['Close']
        returns = stock_data.pct_change()

        results = {}

        for ticker in stocks:
            if ticker not in stock_data.columns:
                st.error(f"No data available for {ticker}.")
                continue

            stock_prices = stock_data[ticker]
            stock_returns = returns[ticker]

            # Calculate years
            difference = end - start
            years = difference.total_seconds() / (365.25 * 24 * 3600)

            # CAGR
            start_price = stock_prices.iloc[0]
            end_price = stock_prices.iloc[-1]
            cagr = ((end_price / start_price) ** (1 / years)) - 1

            # Annualized Volatility
            annual_volatility = np.std(stock_returns) * np.sqrt(252)

            # Drawdown Calculation
            rolling_max = stock_prices.cummax()
            drawdown = (stock_prices / rolling_max) - 1
            max_drawdown = drawdown.min()

            # Recovery Period
            max_drawdown_start = drawdown.idxmin()
            recovery_date = None
            for date, price in stock_prices[max_drawdown_start:].items():
                if price >= rolling_max[max_drawdown_start]:
                    recovery_date = date
                    break
            recovery_period = None
            if recovery_date:
                recovery_period = (recovery_date - max_drawdown_start).days / 30.22

            # Calendar Year Returns
            monthly_returns = stock_returns.resample('ME').sum()
            annual_returns = stock_returns.resample('YE').sum()

            # Store results
            results[ticker] = {
                'Compound Annual Growth Rate (CAGR)': cagr,
                'Annualized Risk': annual_volatility,
                'Max Drawdown': max_drawdown,
                'Recovery Period (Months)': recovery_period
            }

            # Generate plots
            st.subheader(f"{ticker} Analysis")
            plot_price_chart(ticker, stock_prices)
            plot_annual_returns(ticker, annual_returns)
            plot_drawdown_and_underwater(ticker, drawdown, drawdown.index, [0]*len(drawdown))
            plot_seasonality_and_table(ticker, monthly_returns)

        return results

    except Exception as e:
        st.error(f"An error occurred during backtesting: {e}")
        return None

# Main Streamlit App
def main():
    st.title("Stock Analysis and Backtesting")

    # User inputs
    tickers = st.text_input("Enter ticker symbols (comma-separated):", "SPY")
    start_date = st.date_input("Start date:", dt.date.today() - dt.timedelta(days=365 * 10))
    end_date = st.date_input("End date:", dt.date.today())

    if st.button("Run Backtest"):
        stocks = [ticker.strip().upper() for ticker in tickers.split(',')]
        results = backtestStocks_plotly(stocks, start_date, end_date)

        if results:
            st.subheader("Backtest Results")
            for ticker, metrics in results.items():
                st.write(f"### {ticker}")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        if 'CAGR' in metric or 'Risk' in metric:
                            st.write(f"{metric}: {value * 100:.2f}%")
                        elif 'Months' in metric:
                            st.write(f"{metric}: {value:.2f} Months")
                        else:
                            st.write(f"{metric}: {value:.2f}")
                    else:
                        st.write(f"{metric}: {value}")

if __name__ == "__main__":
    main()
