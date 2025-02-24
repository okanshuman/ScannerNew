import io
import calendar
import time
import requests
import pandas as pd
import yfinance as yf
import logging
import datetime

from fastapi import FastAPI, Form
from starlette.responses import StreamingResponse, HTMLResponse
import uvicorn

# Reduce yfinance logging output
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# URL for the official NSE stock list CSV
NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

app = FastAPI(title="Indian Stocks Scanner API with Real-time UI")

def get_stock_list():
    """
    Fetch the NSE equity list from the official CSV file and return a list of symbols.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(NSE_CSV_URL, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch stock list. HTTP {response.status_code}")
    df = pd.read_csv(io.StringIO(response.text))
    symbols = df["SYMBOL"].dropna().unique().tolist()
    problematic = {"KALYANI", "MGL", "PHOENIXLTD", "PILANIINVS"}
    symbols = [sym for sym in symbols if sym not in problematic]
    return symbols

def chunker(seq, size):
    """Yield successive chunks of list `seq` of size `size`."""
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

def get_complete_week_data(df):
    """
    Resample daily data into weeks ending on Friday.
    """
    weekly_df = df.resample("W-FRI").agg({"Open": "first", "Close": "last"})
    if len(weekly_df) == 0:
        return None
    last_week_date = weekly_df.index[-1]
    latest_date = df.index[-1]
    if latest_date < last_week_date:
        return weekly_df.iloc[-2] if len(weekly_df) >= 2 else None
    return weekly_df.iloc[-1]

def get_complete_month_data(df):
    """
    Resample daily data into months.
    """
    monthly_df = df.resample("ME").agg({"Open": "first", "Close": "last"})
    if len(monthly_df) == 0:
        return None
    latest_date = df.index[-1]
    last_day_of_month = calendar.monthrange(latest_date.year, latest_date.month)[1]
    if latest_date.day < last_day_of_month:
        return monthly_df.iloc[-2] if len(monthly_df) >= 2 else None
    return monthly_df.iloc[-1]

def scan_stocks(target_date_str: str):
    """
    Scans stocks based on conditions, using data up to the specified target date.
    """
    start_time = time.time()
    yield "data: Starting scan of Indian stocks...\n\n"

    try:
        original_symbols = get_stock_list()
        yield f"data: Fetched stock list with {len(original_symbols)} symbols.\n\n"
    except Exception as e:
        yield f"data: ERROR fetching stock list: {e}\n\n"
        return

    ticker_mapping = {symbol: f"{symbol}.NS" for symbol in original_symbols}
    inverse_mapping = {v: k for k, v in ticker_mapping.items()}
    tickers = list(ticker_mapping.values())

    result = []
    batch_size = 5
    batches = list(chunker(tickers, batch_size))
    total_batches = len(batches)

    try:
        target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        yield "data: Invalid date format. Please use YYYY-MM-DD.\n\n"
        return

    for i, batch in enumerate(batches, start=1):
        yield f"data: Processing batch {i} of {total_batches}...\n\n"

        while True:
            try:
                data = yf.download(
                    tickers=batch,
                    period="90d",
                    interval="1d",
                    group_by="ticker",
                    threads=True,
                    auto_adjust=False,
                    progress=False,
                )
                break
            except Exception as e:
                if "Rate limited" in str(e):
                    yield f"data: Rate limited on batch {i}. Retrying in 20 seconds...\n\n"
                    time.sleep(20)
                else:
                    yield f"data: Error downloading batch {i}: {e}\n\n"
                    data = None
                    break

        if data is None:
            yield f"data: Skipping batch {i} due to download errors.\n\n"
            continue

        if len(batch) == 1:
            ticker = batch[0]
            df = data.copy()
            data_dict = {ticker: df}
        else:
            data_dict = {}
            for ticker in batch:
                try:
                    df = data[ticker].dropna(how="all")
                except Exception:
                    continue
                data_dict[ticker] = df

        for ticker, df in data_dict.items():
            original_symbol = inverse_mapping.get(ticker, ticker)
            yield f"data: Scanning stock: {original_symbol}\n\n"

            if df.empty or len(df) < 10:
                yield f"data: Skipping {original_symbol} (not enough data).\n\n"
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Filter the DataFrame to include only data up to the target date
            df = df[df.index <= pd.to_datetime(target_date)]

            if df.empty:
                yield f"data: Skipping {original_symbol} (no data up to {target_date_str}).\n\n"
                continue

            latest_date = df.index[-1]
            try:
                latest_data = df.iloc[-1]
                latest_open = latest_data["Open"]
                latest_close = latest_data["Close"]
                latest_high = latest_data["High"]
                latest_volume = latest_data["Volume"]
            except Exception:
                yield f"data: Error processing latest data for {original_symbol}\n\n"
                continue

            one_week_date = latest_date - pd.Timedelta(days=7)
            df_week = df[df.index <= one_week_date]
            if df_week.empty:
                yield f"data: Skipping {original_symbol} (no data from one week ago).\n\n"
                continue
            one_week_data = df_week.iloc[-1]
            one_week_open = one_week_data["Open"]

            weekly_data = get_complete_week_data(df)
            if weekly_data is None:
                yield f"data: Skipping {original_symbol} (insufficient weekly data).\n\n"
                continue
            weekly_open = weekly_data["Open"]
            weekly_close = weekly_data["Close"]

            monthly_data = get_complete_month_data(df)
            if monthly_data is None:
                yield f"data: Skipping {original_symbol} (insufficient monthly data).\n\n"
                continue
            monthly_open = monthly_data["Open"]
            monthly_close = monthly_data["Close"]

            cond1 = 100 < latest_close < 1000
            cond2 = latest_volume >= 100000
            cond3 = latest_close > latest_open
            cond4 = weekly_close > max(3, weekly_open)
            cond5 = monthly_close > max(3, monthly_open)
            cond6 = latest_close > max(3, latest_open)
            cond7 = latest_close <= one_week_open * 1.04
            cond8 = latest_close >= one_week_open * 1.01
            cond9 = latest_close >= latest_high * 0.98

            if all([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9]):
                result.append(original_symbol)
                yield f"data: {original_symbol} meets conditions.\n\n"
                yield f"data: MATCH: {original_symbol}\n\n" # Display as soon as a match is found

        yield f"data: Completed batch {i}.\n\n"

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=elapsed_time))

    yield f"data: Finished scanning. Total matching stocks: {len(result)}\n\n"
    yield f"data: Final Result: {result}\n\n"
    yield f"data: Total Time taken: {formatted_time}\n\n"


@app.get("/", response_class=HTMLResponse)
def get_ui():
    """Serves the HTML page with a date input and start button."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Indian Stocks Scanner - Real-time UI</title>
      <style>
         body { font-family: Arial, sans-serif; margin: 20px; }
         #log { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: scroll; background: #f9f9f9; }
         #matches { border: 1px solid green; padding: 10px; height: 200px; overflow-y: scroll; background: #e9f9e9; margin-top: 20px; }
         button { padding: 10px 20px; font-size: 16px; }
         input[type="date"] { padding: 8px; font-size: 14px; }
      </style>
    </head>
    <body>
      <h1>Indian Stocks Scanner - Real-time Processing</h1>
      <label for="targetDate">Select Target Date:</label>
      <input type="date" id="targetDate" name="targetDate">
      <button onclick="startScan()">Start Scan</button>
      <h2>Processing Log:</h2>
      <div id="log"></div>
      <h2>Matching Stocks:</h2>
      <div id="matches"></div>
      <script>
        var evtSource;
        function startScan() {
          var targetDate = document.getElementById("targetDate").value;
          if (!targetDate) {
            alert("Please select a target date.");
            return;
          }
          document.getElementById("log").innerHTML = "";
          document.getElementById("matches").innerHTML = ""; // Clear previous matches
          evtSource = new EventSource("/stream?target_date=" + targetDate);
          evtSource.onmessage = function(event) {
            var logDiv = document.getElementById("log");
            var matchesDiv = document.getElementById("matches");
            logDiv.innerHTML += event.data + "<br>";
            logDiv.scrollTop = logDiv.scrollHeight;

            if (event.data.startsWith("MATCH:")) {
              var stock = event.data.substring(7); // Extract stock symbol
              matchesDiv.innerHTML += stock + "<br>";
              matchesDiv.scrollTop = matchesDiv.scrollHeight;
            }
          };
          evtSource.onerror = function(err) {
            console.error("EventSource failed:", err);
            evtSource.close();
          };
        }
      </script>
    </body>
    </html>
    """
    return html_content


@app.get("/stream")
def stream(target_date: str):
    """
    Streams stock scan results based on the provided target date.
    """
    return StreamingResponse(scan_stocks(target_date), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)