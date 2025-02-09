import io
import calendar
import time
import requests
import pandas as pd
import yfinance as yf
import logging

from fastapi import FastAPI
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
    A custom User-Agent header and timeout are set to help the request go through.
    Optionally, known problematic tickers (e.g. delisted stocks) are filtered out.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(NSE_CSV_URL, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch stock list. HTTP {response.status_code}")
    df = pd.read_csv(io.StringIO(response.text))
    symbols = df["SYMBOL"].dropna().unique().tolist()
    # Optionally filter out problematic tickers known to be delisted or with missing data
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
    If the latest week is incomplete, return the previous complete week.
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
    Uses "ME" (month end) frequency to avoid deprecation warnings.
    If the current month is incomplete, return the previous complete month.
    """
    monthly_df = df.resample("ME").agg({"Open": "first", "Close": "last"})
    if len(monthly_df) == 0:
        return None
    latest_date = df.index[-1]
    last_day_of_month = calendar.monthrange(latest_date.year, latest_date.month)[1]
    if latest_date.day < last_day_of_month:
        return monthly_df.iloc[-2] if len(monthly_df) >= 2 else None
    return monthly_df.iloc[-1]

def scan_stocks():
    """
    Generator function that scans all stocks and yields Server-Sent Event (SSE) messages
    as each stock is processed.
    
    Conditions applied:
      1. latest_close >= one_week_open * 1.01  
      2. latest_close <= one_week_open * 1.04  
      3. latest_close > latest_open  
      4. weekly_close > max(3, weekly_open)  
      5. monthly_close > max(3, monthly_open)  
      6. latest_close > max(3, latest_open)  
      7. latest_volume >= 100000  
      8. 100 < latest_close < 1000  
      9. latest_close >= latest_high * 0.98  
    """
    yield "data: Starting scan of Indian stocks...\n\n"
    try:
        original_symbols = get_stock_list()
        yield f"data: Fetched stock list with {len(original_symbols)} symbols.\n\n"
    except Exception as e:
        yield f"data: ERROR fetching stock list: {e}\n\n"
        return

    # Map each NSE symbol to its yfinance ticker (append ".NS")
    ticker_mapping = {symbol: f"{symbol}.NS" for symbol in original_symbols}
    # Inverse mapping for logging
    inverse_mapping = {v: k for k, v in ticker_mapping.items()}
    tickers = list(ticker_mapping.values())

    result = []
    batch_size = 5  # A smaller batch size helps avoid rate limits.
    batches = list(chunker(tickers, batch_size))
    total_batches = len(batches)

    for i, batch in enumerate(batches, start=1):
        yield f"data: Processing batch {i} of {total_batches} (tickers: {len(batch)})...\n\n"
        
        # Retry indefinitely for rate limit errors
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
                break  # Exit loop if download is successful
            except Exception as e:
                if "Rate limited" in str(e):
                    yield f"data: Rate limited while downloading batch {i}. Retrying after 20 seconds...\n\n"
                    time.sleep(20)
                else:
                    yield f"data: ERROR downloading batch {i}: {e}\n\n"
                    data = None
                    break

        if data is None:
            yield f"data: Skipping batch {i} due to errors.\n\n"
            continue

        # Organize the downloaded data into a dictionary: ticker -> DataFrame.
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

            # Ensure the index is a DatetimeIndex and sorted.
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

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

            # Get the open price from roughly one week ago.
            one_week_date = latest_date - pd.Timedelta(days=7)
            df_week = df[df.index <= one_week_date]
            if df_week.empty:
                yield f"data: Skipping {original_symbol} (no data from one week ago).\n\n"
                continue
            one_week_data = df_week.iloc[-1]
            one_week_open = one_week_data["Open"]

            # Get weekly and monthly complete data.
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

            # Check conditions:
            cond1 = latest_close >= one_week_open * 1.01
            cond2 = latest_close <= one_week_open * 1.04
            cond3 = latest_close > latest_open
            cond4 = weekly_close > max(3, weekly_open)
            cond5 = monthly_close > max(3, monthly_open)
            cond6 = latest_close > max(3, latest_open)
            cond7 = latest_volume >= 100000
            cond8 = (latest_close > 100) and (latest_close < 1000)
            cond9 = latest_close >= latest_high * 0.98

            if all([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9]):
                result.append(original_symbol)
                yield f"data: {original_symbol} meets conditions.\n\n"

        yield f"data: Completed batch {i}.\n\n"

    yield f"data: Finished scanning. Total matching stocks: {len(result)}\n\n"
    yield f"data: Final Result: {result}\n\n"

@app.get("/", response_class=HTMLResponse)
def get_ui():
    """
    Serves a simple HTML page with a button to start the scan and a log area that displays
    real-time processing details.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Indian Stocks Scanner - Real-time UI</title>
      <style>
         body { font-family: Arial, sans-serif; margin: 20px; }
         #log { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: scroll; background: #f9f9f9; }
         button { padding: 10px 20px; font-size: 16px; }
      </style>
    </head>
    <body>
      <h1>Indian Stocks Scanner - Real-time Processing</h1>
      <button onclick="startScan()">Start Scan</button>
      <h2>Processing Log:</h2>
      <div id="log"></div>
      <script>
        var evtSource;
        function startScan() {
          document.getElementById("log").innerHTML = "";
          evtSource = new EventSource("/stream");
          evtSource.onmessage = function(event) {
            var logDiv = document.getElementById("log");
            logDiv.innerHTML += event.data + "<br>";
            logDiv.scrollTop = logDiv.scrollHeight;
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
def stream():
    """
    SSE endpoint that streams real-time processing details as the scan runs.
    """
    return StreamingResponse(scan_stocks(), media_type="text/event-stream")

if __name__ == "__main__":
      uvicorn.run(app, host="0.0.0.0", port=5005)
