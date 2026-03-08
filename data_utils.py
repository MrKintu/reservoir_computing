"""
Small utilities for loading ticker CSVs and preparing time-series matrices.
Used by the notebooks.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List
from logger import get_logger

TICKERS_DIR = os.path.join(os.getcwd(), "tickers")

def list_ticker_files() -> List[str]:
    logger = get_logger("data_utils")
    if not os.path.exists(TICKERS_DIR):
        logger.warning(f"Tickers directory {TICKERS_DIR} does not exist")
        return []
    files = [os.path.join(TICKERS_DIR, f) for f in os.listdir(TICKERS_DIR) if f.endswith(".csv")]
    logger.info(f"Found {len(files)} ticker CSV files")
    return files

def load_close_matrix() -> pd.DataFrame:
    """Load all ticker CSVs and return a DataFrame of aligned Close prices (Date index)."""
    logger = get_logger("data_utils")
    files = list_ticker_files()
    frames = []
    logger.info(f"Loading data from {len(files)} files...")
    for f in files:
        try:
            df = pd.read_csv(f)
            # normalize column names
            df.columns = [c.strip() for c in df.columns]
            logger.debug(f"Columns in {os.path.basename(f)}: {df.columns.tolist()}")
            # find date and close columns (case-insensitive, handle ticker suffixes)
            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            close_col = next((c for c in df.columns if c.lower().startswith("close")), None)
            if date_col is None or close_col is None:
                logger.warning(f"Skipping {os.path.basename(f)} - missing date or close column. Found columns: {df.columns.tolist()}")
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: os.path.splitext(os.path.basename(f))[0]})
            df.set_index("Date", inplace=True)
            frames.append(df)
        except Exception as e:
            logger.error(f"Error loading {os.path.basename(f)}: {e}")
            continue
    if not frames:
        logger.warning("No valid data frames loaded")
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1).sort_index()
    # forward/backfill small gaps
    merged = merged.ffill().bfill()
    logger.info(f"Created merged matrix with shape {merged.shape}")
    return merged

def select_random_ticker(df: pd.DataFrame, random_state: int = 42) -> pd.Series:
    """Select a random ticker column from the DataFrame."""
    logger = get_logger("data_utils")
    if df.empty:
        logger.error("DataFrame is empty - no tickers available")
        raise ValueError("DataFrame is empty - no tickers available")
    
    # Set random seed for reproducibility using modern Generator
    rng = np.random.default_rng(random_state)
    
    # Get list of available tickers (columns)
    available_tickers = df.columns.tolist()
    logger.info(f"Available tickers: {available_tickers}")
    
    # Select random ticker
    selected_ticker = rng.choice(available_tickers)
    series = df[selected_ticker].dropna()
    series = series.astype(float)
    
    logger.info(f"Randomly selected ticker: {selected_ticker}, length: {len(series)}")
    return series

def train_val_test_split(series: pd.Series, train_frac=0.7, val_frac=0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 1D series into train/val/test arrays (time-ordered)."""
    logger = get_logger("data_utils")
    n = len(series)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train = series[:i_train].values
    val = series[i_train:i_val].values
    test = series[i_val:].values
    logger.debug(f"Split series of length {n}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test

def windowed_dataset(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create X, y windows for supervised learning (predict next step)."""
    logger = get_logger("data_utils")
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X, y = np.array(X), np.array(y)
    logger.debug(f"Created windowed dataset: X shape {X.shape}, y shape {y.shape} from series length {len(series)}")
    return X, y
