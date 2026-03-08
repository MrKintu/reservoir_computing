"""
Fetch ticker CSVs and save into ./tickers directory.
- Limits to 10 tickers.
- Uses yfinance for historical OHLCV data.
- Also contains a simple scraper to extract tickers from a news page (best-effort).
"""

import os
import re
import requests
from lxml import html
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from logger import get_logger

TICKERS_DIR = os.path.join(os.getcwd(), "tickers")
os.makedirs(TICKERS_DIR, exist_ok=True)

def create_urls(days: int = 30):
    """Create PR Newswire-like URLs for the last `days` days (best-effort)."""
    urls = []
    today = datetime.now()
    for i in range(days):
        d = today - timedelta(days=i)
        urls.append(f"https://www.prnewswire.com/news-releases/news-releases-list/?month={d.month}&day={d.day}&year={d.year}&hour=00")
    return urls

def scrape_tickers_from_pages(urls, max_tickers=10):
    """Scrape ticker-like patterns from pages. Returns a list of unique tickers (symbols)."""
    logger = get_logger("ticker_scraper")
    pattern = re.compile(r'\(([A-Z]{1,5}):\s*([A-Z]{1,5})\)')  # e.g., (NYSE: ABC)
    found = set()
    logger.info(f"Scraping {len(urls)} URLs for tickers...")
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            tree = html.fromstring(resp.content)
            snippets = tree.xpath('//*[@class="row newsCards"]')
            for s in snippets:
                text = html.tostring(s, encoding='unicode', method='text')
                for m in pattern.findall(text):
                    # m is tuple like ('NYSE', 'ABC') -> symbol is second element
                    symbol = m[1].strip()
                    found.add(symbol)
                    if len(found) >= max_tickers:
                        logger.info(f"Found {len(found)} tickers, stopping early")
                        return list(found)
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            continue
    result = list(found)[:max_tickers]
    logger.info(f"Found {len(result)} tickers total")
    return result

def download_and_save(symbol: str, start: str = "2025-03-01", end: str = None):
    """Download historical data for a symbol and save CSV to tickers/."""
    logger = get_logger("ticker_downloader")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            print(f"[WARN] No data for {symbol}")
            logger.warning(f"No data for {symbol}")
            return False
        df = df.reset_index()
        # Ensure consistent columns
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        # Handle MultiIndex columns (flatten to single level)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        else:
            df.columns = [c.strip() for c in df.columns]
        
        # Validate we have the expected columns
        if not all(col in df.columns for col in expected_cols):
            print(f"[WARN] {symbol} has unexpected columns: {df.columns.tolist()}")
            logger.warning(f"{symbol} has unexpected columns: {df.columns.tolist()}")
        
        csv_path = os.path.join(TICKERS_DIR, f"{symbol}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved {symbol} -> {csv_path}")
        logger.info(f"Saved {symbol} -> {csv_path}")
        return True
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        logger.error(f"{symbol}: {e}")
        return False

def main(max_tickers=10, days=30):
    logger = get_logger("ticker_main")
    urls = create_urls(days=days)
    symbols = scrape_tickers_from_pages(urls, max_tickers=max_tickers)
    if not symbols:
        print("[INFO] No tickers scraped; using a small default list for demo.")
        logger.info("No tickers scraped; using a small default list for demo.")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"][:max_tickers]
    print(f"[INFO] Tickers to download: {symbols}")
    logger.info(f"Tickers to download: {symbols}")
    for s in symbols:
        download_and_save(s)

if __name__ == "__main__":
    main(max_tickers=10, days=30)
