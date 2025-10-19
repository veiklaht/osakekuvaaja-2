import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

# Streamlit app details
st.set_page_config(page_title="Financial Analysis", layout="wide")
with st.sidebar:
    st.title("Financial Analysis")
    ticker = st.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL")
    period = st.selectbox("Enter a time frame", ("1D", "5D", "1M", "6M", "YTD", "1Y", "5Y"), index=2)
    submit = st.button("Submit")

# Format market cap and enterprise value into something readable
def format_value(value):
    try:
        if value is None:
            return "N/A"
        value = float(value)
        suffixes = ["", "K", "M", "B", "T"]
        suffix_index = 0
        while value >= 1000 and suffix_index < len(suffixes) - 1:
            value /= 1000
            suffix_index += 1
        return f"${value:.1f}{suffixes[suffix_index]}"
    except Exception:
        return "N/A"

def safe_format(value, fmt="{:.2f}", fallback="N/A"):
    try:
        return fmt.format(value) if value is not None else fallback
    except (ValueError, TypeError):
        return fallback

# Get next trading date based on earnings date
def get_next_trading_day(df, date):
    after = df[df.index > date]
    return after.index[0] if not after.empty else None

def get_same_or_next_trading_day(df, date):
    if date in df.index:
        return date
    return get_next_trading_day(df, date)

# If Submit button is clicked
if submit:
    if not ticker.strip():
        st.error("Please provide a valid stock ticker.")
    else:
        try:
            with st.spinner('Fetching data...', show_time=True):
                # Retrieve stock data
                stock = yf.Ticker(ticker.upper())
                info = stock.info

                st.subheader(f"{ticker} - {info.get('longName', 'N/A')}")

                # Plot historical stock price data
                period_map = {
                    "1D": ("1d", "1h"),
                    "5D": ("5d", "1d"),
                    "1M": ("1mo", "1d"),
                    "6M": ("6mo", "1wk"),
                    "YTD": ("ytd", "1mo"),
                    "1Y": ("1y", "1mo"),
                    "5Y": ("5y", "3mo"),
                }
                selected_period, interval = period_map.get(period, ("1mo", "1d"))

                # --- HAKEE DATAN JA PIIRTÄÄ: HINTAVIIVA + MEAN + ±1/±2/±3 σ -TASOT ---
                history = stock.history(period=selected_period, interval=interval)

                series = history["Close"].dropna()
                if series.empty:
                    st.warning("No price data for this period/interval.")
                else:
                    price_df = series.reset_index()
                    price_df.columns = ["Date", "Close"]

                    mean = float(series.mean())
                    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0

                    # Rakennetaan tasot
                    bands = [("Mean", mean)]
                    if std > 0.0:
                        bands += [
                            ("+1σ", mean + std), ("-1σ", mean - std),
                            ("+2σ", mean + 2*std), ("-2σ", mean - 2*std),
                            ("+3σ", mean + 3*std), ("-3σ", mean - 3*std),
                        ]

                    # Hintaviiva
                    price_line = (
                        alt.Chart(price_df)
                        .mark_line()
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Close:Q", title="Price", scale=alt.Scale(zero=False)),
                            tooltip=[
                                alt.Tooltip("Date:T", title="Date"),
                                alt.Tooltip("Close:Q", title="Close", format=",.2f"),
                            ],
                        )
                    )

                    # Sääntöviivat (μ ja ±σ:t)
                    rules_df = pd.DataFrame(
                        {"y": [v for _, v in bands], "label": [lbl for lbl, _ in bands]}
                    )
                    rules = (
                        alt.Chart(rules_df)
                        .mark_rule(strokeDash=[6, 6])
                        .encode(
                            y="y:Q",
                            color=alt.Color(
                                "label:N",
                                title="Levels",
                                scale=alt.Scale(scheme="category10"),
                            ),
                            tooltip=[
                                alt.Tooltip("label:N", title="Level"),
                                alt.Tooltip("y:Q", title="Value", format=",.2f"),
                            ],
                        )
                    )

                    chart = (price_line + rules).properties(height=320)
                    st.altair_chart(chart, use_container_width=True)

                    # Näytä arvot taulukkona
                    def _fmt(v): return safe_format(v, fmt="${:,.2f}")
                    stats_rows = [("Mean (μ)", _fmt(mean))]
                    if std > 0.0:
                        stats_rows += [
                            ("+1σ", _fmt(mean + std)),  ("-1σ", _fmt(mean - std)),
                            ("+2σ", _fmt(mean + 2*std)),("-2σ", _fmt(mean - 2*std)),
                            ("+3σ", _fmt(mean + 3*std)),("-3σ", _fmt(mean - 3*std)),
                        ]
                    st.dataframe(pd.DataFrame(stats_rows, columns=["Level", "Value"]),
                                 hide_index=True, width=420)

                col1, col2, col3 = st.columns(3)

                # Display stock information as a dataframe
                stock_info = [
                    ("Stock Info", "Value"),
                    ("Country", info.get('country', 'N/A')),
                    ("Sector", info.get('sector', 'N/A')),
                    ("Industry", info.get('industry', 'N/A')),
                    ("Market Cap", format_value(info.get('marketCap'))),
                    ("Enterprise Value", format_value(info.get('enterpriseValue'))),
                    ("Employees", info.get('fullTimeEmployees', 'N/A'))
                ]

                df = pd.DataFrame(stock_info[1:], columns=stock_info[0]).astype(str)
                col1.dataframe(df, width=400, hide_index=True)

                # Display price information as a dataframe
                price_info = [
                    ("Price Info", "Value"),
                    ("Current Price", safe_format(info.get('currentPrice'), fmt="${:.2f}")),
                    ("Previous Close", safe_format(info.get('previousClose'), fmt="${:.2f}")),
                    ("Day High", safe_format(info.get('dayHigh'), fmt="${:.2f}")),
                    ("Day Low", safe_format(info.get('dayLow'), fmt="${:.2f}")),
                    ("52 Week High", safe_format(info.get('fiftyTwoWeekHigh'), fmt="${:.2f}")),
                    ("52 Week Low", safe_format(info.get('fiftyTwoWeekLow'), fmt="${:.2f}"))
                ]

                df = pd.DataFrame(price_info[1:], columns=price_info[0]).astype(str)
                col2.dataframe(df, width=400, hide_index=True)

                # Display business metrics as a dataframe
                biz_metrics = [
                    ("Business Metrics", "Value"),
                    ("EPS (FWD)", safe_format(info.get('forwardEps'))),
                    ("P/E (FWD)", safe_format(info.get('forwardPE'))),
                    ("PEG Ratio", safe_format(info.get('pegRatio'))),
                    ("Div Rate (FWD)", safe_format(info.get('dividendRate'), fmt="${:.2f}")),
                    ("Div Yield (FWD)", safe_format(info.get('dividendYield'), fmt="{:.2f}%") if info.get('dividendYield') else 'N/A'),
                    ("Recommendation", info.get('recommendationKey', 'N/A').capitalize() if isinstance(info.get('recommendationKey'), str) else 'N/A')
                ]

                df = pd.DataFrame(biz_metrics[1:], columns=biz_metrics[0]).astype(str)
                col3.dataframe(df, width=400, hide_index=True)

                # Display earnings moves for last 12 quarters
                earnings = stock.get_earnings_dates(limit=12)
                history = stock.history(period="3y")

                results = []
                for idx, row in earnings.iterrows():
                    earnings_date = pd.to_datetime(idx).date()
                    raw_time = row.get("Time", "")
                    time_of_day = raw_time.lower() if isinstance(raw_time, str) else "pm"  # default to pm

                    try:
                        if time_of_day == "am":
                            trading_day = get_same_or_next_trading_day(history, idx)
                            prev_day = history.index[history.index < trading_day][-1]
                        else:
                            trading_day = get_next_trading_day(history, idx)
                            prev_day = history.index[history.index < idx][-1]

                        prev_close = history.loc[prev_day]["Close"]
                        next_close = history.loc[trading_day]["Close"]
                        pct_change = ((next_close - prev_close) / prev_close) * 100

                        results.append({
                            "Earnings Date": earnings_date,
                            "Price Date": trading_day.date(),
                            "Close % Change": f"{pct_change:.2f}%"
                        })

                    except Exception:
                        results.append({
                            "Earnings Date": earnings_date,
                            "Price Date": None,
                            "Close % Change": None
                        })

                df = pd.DataFrame(results)
                df = df.dropna()

                col1, col2 = st.columns([1, 2])
                with col1:
                    df_display = df.copy()
                    df_display["Close % Change"] = df_display["Close % Change"].apply(
                        lambda x: f"{float(str(x).replace('%', '')):.2f}%" if pd.notnull(x) else "N/A"
                    )
                    st.dataframe(df_display, width=400, height=450, hide_index=True)

                with col2:
                    chart_data = df.copy()
                    chart_data["Earnings Date"] = chart_data["Earnings Date"].astype(str)
                    chart_data = chart_data[chart_data["Close % Change"] != "N/A"].copy()
                    chart_data["Close % Change"] = (
                        chart_data["Close % Change"].str.replace("%","").astype(float)
                    )

                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("Earnings Date:N", sort="ascending"),
                        y=alt.Y("Close % Change:Q"),
                        color=alt.condition(
                            alt.datum["Close % Change"] > 0,
                            alt.value("green"),
                            alt.value("red")
                        ),
                        tooltip=["Earnings Date", "Price Date", alt.Tooltip("Close % Change", format=".2f")]
                    ).properties(width="container", height=450)

                    st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.exception(f"An error occurred: {e}")
