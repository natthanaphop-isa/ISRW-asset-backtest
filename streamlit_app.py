import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
import streamlit as st
import calendar

def process_ticker_input(input_text):
    try:
        # Convert to string and strip whitespace
        input_text = str(input_text).strip()

        # Check for commas (multiple tickers)
        if "," in input_text:
            # Split by commas, clean, and convert to uppercase
            tickers = [ticker.strip().upper() for ticker in input_text.split(",")]
        else:
            # Single ticker, ensure uppercase
            tickers = [input_text.upper()]

        # Validate tickers (optional: ensure valid characters)
        valid_tickers = []
        for ticker in tickers:
            if all(char.isalnum() or char in {"^", "=", ".", "-"} for char in ticker):
                valid_tickers.append(ticker)
            else:
                st.warning(f"Invalid ticker format: {ticker}")
        
        return valid_tickers

    except Exception as e:
        st.error(f"An error occurred while processing tickers: {e}")
        return []
        
# Function to plot the price chart
def plot_price_chart(ticker, stock_prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_prices.index, y=stock_prices, mode='lines', name='Price', line=dict(color='#ffde59'), hovertemplate="$%{y:.2f}"))
    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white"
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
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
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=underwater_x,
        y=underwater_y,
        mode='lines',
        name='Underwater Duration',
        line=dict(color='#ffde59'),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Duration: %{y:.1f} Months"))
    fig.update_layout(
        title=f"{ticker} Underwater Period Duration",
        xaxis_title="Date",
        yaxis_title="Duration (Months)",
        template="plotly_white"
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    st.plotly_chart(fig)
# Function to plot annual returns chart
def plot_annual_returns(ticker, annual_return, bin_size = 10):
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
        marker=dict(color=["red" if v <= 0 else "green" for v in all_values]),
                    hovertemplate="Year: %{x}<br>Annual Return: %{y:.2f}%"
    ))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(
        title=f"{ticker} Annual Returns by Calendar Year",
        xaxis_title="Year",
        yaxis_title="Annual Returns (%)",
        template="plotly_white"
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    st.plotly_chart(fig)
    
    positive_returns = [v for v in values if v > 0]
    negative_returns = [v for v in values if v < 0]

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
            hovertemplate="Count: %{y}"
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
            hovertemplate="Count: %{y}"
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
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
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
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    st.plotly_chart(fig)
# Main Streamlit App
def main():
    st.title("Asset Analysis By Isara Wealth")
    st.markdown(f"""
    ETF RESEARCH TOOL
    โดย นพ.ณัฐธนภพ อิศรเดช (หมอเฟ้น)
    - แพทย์ และนักวิจัย: Machine Learning, Clinical NLP, Clinical Epidemiology
    - เจ้าของบล็อก www.isarawealth.com
    - IP License No. 132355 ใบอนุญาตผู้วางแผนการลงทุน โดย กลต.
    - ที่ปรึกษาทางการเงิน บลน. Finnomena
    - ตัวแทนประกันชีวิตและ Unitlink AIA รหัส 692246
    """)
    # User selects month and year
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, 
        "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, 
        "November": 11, "December": 12
    }
    tickers = st.text_input("Enter ticker symbol:", "SPY")
    # Dropdowns for month and year selection
    selected_start_month = st.selectbox("Start month:", list(months.keys()), index=0)
    selected_start_year = st.number_input("Start year:", min_value=1990, max_value=dt.date.today().year, value= (dt.date.today() - dt.timedelta(days = 365*20)).year)
    
    selected_end_month = st.selectbox("End month:", list(months.keys()), index=0)
    selected_end_year = st.number_input("End year:", min_value=1990, max_value=dt.date.today().year, value=dt.date.today().year)
    
    # Calculate the start date (first day of the month)
    start_month_number = months[selected_start_month]
    start_date = dt.date(selected_start_year, start_month_number, 1)
    
    # Calculate the end date (last day of the month)
    end_month_number = months[selected_end_month]
    last_day = calendar.monthrange(selected_end_year, end_month_number)[1]  # Get the last day of the selected month
    end_date = dt.date(selected_end_year, end_month_number, last_day)

    # User inputs
    # tickers = st.text_input("Enter ticker symbol:", "SPY")
    # start_date = st.date_input("Start date:", dt.date.today() - dt.timedelta(days=365 * 20))
    # end_date = st.date_input("End date:", dt.date.today())

    if st.button("Run Backtest"):
        # Process ticker input
        tickers = process_ticker_input(tickers)

        try:
            if not tickers:
                st.error("No valid tickers provided.")
                return

            for ticker in tickers:
                stock_prices = yf.Ticker(ticker).history(start=start_date, end=end_date)['Close']

                if stock_prices.empty:
                    st.warning(f"No data found for {ticker} in the specified date range.")
                    continue
                
                stock_returns = stock_prices.pct_change()

                # stock_prices = stock_data[ticker]
                # stock_returns = returns[ticker]

                results = {}
                
                # Calculate years
                difference = end_date - start_date
                years = difference.total_seconds() / (365.25 * 24 * 3600)
                
                # CAGR Calculation
                start_price = stock_prices.iloc[0]
                end_price = stock_prices.iloc[-1]
                cagr = ((end_price / start_price) ** (1 / years)) - 1
                # Annualized Volatility
                annual_volatility = np.std(stock_returns) * np.sqrt(252)
            
                # Drawdown Calculation
                rolling_max = stock_prices.cummax()
                drawdown = (stock_prices / rolling_max) - 1

                # Maximum Drawdown and Recovery Period Calculation
                max_drawdown = drawdown.min()
                max_drawdown_start = drawdown.idxmin()  # Date of max drawdown

                # Identify the recovery date
                recovery_date = None
                for date, price in stock_prices[max_drawdown_start:].items():
                    if price >= rolling_max[max_drawdown_start]:
                        recovery_date = date
                        break
                # Calculate recovery period
                recovery_period = None
                if recovery_date:
                    recovery_period = (recovery_date - max_drawdown_start).days

                # Underwater Duration Calculation
                underwater_x = []
                underwater_y = []
                cumulative_duration = 0
                for date, value in drawdown.items():
                    if value < 0:
                        cumulative_duration += 1 / 30
                    else:
                        cumulative_duration = 0
                    underwater_x.append(date)
                    underwater_y.append(cumulative_duration)

                # Calendar Year Returns
                annual_returns = stock_returns.resample('YE').sum()
                monthly_returns = stock_returns.resample('ME').sum()

                results[ticker] = {
                    'Compound Annual Growth Rate (CAGR)': cagr,
                    'Annual Volatility':  annual_volatility,
                    'Max Drawdown': max_drawdown,
                    'Recovery Period (Months)': recovery_period / 30.22
                    }
                
                if results:
                    st.subheader(f"{ticker} Analysis between {start_date.strftime('%b')}-{start_date.year} to {end_date.strftime('%b')}-{end_date.year}")
                    for ticker, metrics in results.items():
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                if 'Months' in metric:
                                    st.write(f"{metric}: {value:.2f} Months")
                                else:
                                    st.write(f"{metric}: {value * 100:.2f}%")
                            else:
                                st.write(f"{metric}: {value}")
                plot_price_chart(ticker, stock_prices)
                plot_annual_returns(ticker, annual_returns)
                plot_drawdown_and_underwater(ticker, drawdown, underwater_x, underwater_y)
                plot_seasonality_and_table(ticker, monthly_returns)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
