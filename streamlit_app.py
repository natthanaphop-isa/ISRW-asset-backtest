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

# Function to plot drawdown and underwater periods
def plot_drawdown_and_underwater(ticker, drawdown, underwater_x, underwater_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        mode='lines',
        fill='tozeroy',
        fillcolor='pink',
        name='Drawdown',
        line=dict(color='red'),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%"
    ))
    fig.update_layout(
        title=f"{ticker} Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=underwater_x,
        y=underwater_y,
        mode='lines',
        name='Underwater Duration',
        line=dict(color='green'),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Duration: %{y:.1f} Months"))
    fig.update_layout(
        title=f"{ticker} Underwater Period Duration",
        xaxis_title="Date",
        yaxis_title="Duration (Months)",
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Function to plot annual returns chart
def plot_annual_returns(ticker, annual_return):
    annual_return_percentage = annual_return * 100
    years = annual_return.index.year
    values = annual_return_percentage.values

    full_years = np.arange(years.min(), years.max() + 1)
    return_dict = dict(zip(years, values))
    all_values = [return_dict.get(year, 0) for year in full_years]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=full_years,
        y=all_values,
        name=f'{ticker}',
        marker=dict(color=["red" if v <= 0 else "green" for v in all_values],
        )
    ))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(
        title=f"{ticker} Annual Returns by Calendar Year",
        xaxis_title="Year",
        yaxis_title="Annual Returns (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig)

    positive_returns = [v for v in values if v > 0]
    negative_returns = [v for v in values if v <= 0]

    fig = go.Figure()

    # Add positive returns histogram
    if positive_returns:
        fig.add_trace(go.Histogram(
            x=positive_returns,
            marker=dict(color="green"),
            xbins=dict(
                size=bin_size  # Set bin size
            ),
            name = f'{ticker}',
            hovertemplate="Count: {y}"
        ))

    # Add negative returns histogram
    if negative_returns:
        fig.add_trace(go.Histogram(
            x=negative_returns,
            marker=dict(color="red"),
            xbins=dict(
                size=bin_size  # Set bin size
            ),
            name = f'{ticker}',
            hovertemplate="Count: {y}"
        ))

    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        barmode='overlay',
        title=f"{ticker} Annual Returns Distribution",
        xaxis_title="Annual Return (%)",
        yaxis_title="Frequency",
        bargap=0.2,  # Add spacing between bars
        template="plotly_white",
        showlegend=False  # Remove legend
    )
    st.plotly_chart(fig)

# Function to plot seasonality histogram
def plot_seasonality_and_table(ticker, monthly_returns):
    monthly_avg = monthly_returns.groupby(monthly_returns.index.month).mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_avg_filled = monthly_avg.reindex(range(1, 13)).fillna(0)

    fig = px.bar(
        x=months,
        y=monthly_avg_filled.values * 100,
        labels={'x': 'Month', 'y': 'Monthly Average Return (%)'},
        title=f"{ticker} Seasonality Analysis",
        template="plotly_white"
    )
    fig.update_traces(marker=dict(color=["green" if val > 0 else "red" for val in monthly_avg_filled]),
                      hovertemplate="Month: %{x}<br>Monthly Return: %{y:.2f}%")
    st.plotly_chart(fig)

# Main Streamlit App
def main():
    st.title("Stock Analysis and Backtesting")

    # User inputs
    tickers = st.text_input("Enter ticker symbols (comma-separated):", "SPY")
    start_date = st.date_input("Start date:", dt.date.today() - dt.timedelta(days=365 * 10))
    end_date = st.date_input("End date:", dt.date.today())

    if st.button("Run Backtest"):
        try:
            stocks = [ticker.strip().upper() for ticker in tickers.split(',')]
            stock_data = yf.download(stocks, start=start_date, end=end_date)['Close']
            returns = stock_data.pct_change()

            for ticker in stocks:
                if ticker not in stock_data.columns:
                    st.error(f"No data available for {ticker}.")
                    continue

                stock_prices = stock_data[ticker]
                stock_returns = returns[ticker]

                # Drawdown Calculation
                rolling_max = stock_prices.cummax()
                drawdown = (stock_prices / rolling_max) - 1

                # Underwater Duration Calculation
                underwater_x = []
                underwater_y = []
                cumulative_duration = 0
                for date, value in drawdown.items():
                    if value < 0:
                        cumulative_duration += 1 / 30.0
                    else:
                        cumulative_duration = 0
                    underwater_x.append(date)
                    underwater_y.append(cumulative_duration)

                # Calendar Year Returns
                annual_returns = stock_returns.resample('Y').sum()
                monthly_returns = stock_returns.resample('M').sum()

                st.subheader(f"{ticker} Analysis")
                plot_price_chart(ticker, stock_prices)
                plot_annual_returns(ticker, annual_returns)
                plot_drawdown_and_underwater(ticker, drawdown, underwater_x, underwater_y)
                plot_seasonality_and_table(ticker, monthly_returns)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
