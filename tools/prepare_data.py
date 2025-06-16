import os
import glob
import json
import pandas as pd
import numpy as np

setattr(np, 'NaN', np.nan)
import pandas_ta
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 1. GLOBAL CONFIGURATION ---
DATA_DIR = "data_test"
OUTPUT_FILE = "prepared_crypto_data.jsonl"
SCALER_FILE = "crypto_scaler.joblib"

CONTEXT_LENGTH = 500
PREDICTION_LENGTH = 48
MIN_BARS = 39000

USE_TA = ['rsi', 'adx', 'atr', 'sma']

# --- 2. FEATURE ENGINEERING FUNCTION ---
def add_features(df):
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] DataFrame is missing required columns. Found: {df.columns.tolist()}. Required: {list(required_cols)}")
        return None
        
    df[list(required_cols)] = df[list(required_cols)].astype(float)

    if 'rsi' in USE_TA:
        df.ta.rsi(length=14, append=True)
    if 'adx' in USE_TA:
        df.ta.adx(length=14, append=True)
    if 'atr' in USE_TA:
        df.ta.atr(length=14, append=True)
    if 'sma' in USE_TA:
        df.ta.sma(length=14, append=True)
    if 'ema' in USE_TA:
        df.ta.ema(length=14, append=True)
    if 'macd' in USE_TA:
        df.ta.macd(append=True)
    
    df.dropna(inplace=True)
    return df

# --- 2b. Multiprocessing helpers ---

def _load_for_scaler(path):
    """Read a CSV and preprocess it for scaler fitting."""
    df = pd.read_csv(path)
    if len(df) < MIN_BARS:
        return None
    df.columns = [str(col).lower().strip() for col in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df


_SCALER = None
_FEATURE_COLS = None


def _init_worker(scaler_path, feature_cols):
    global _SCALER, _FEATURE_COLS
    _SCALER = joblib.load(scaler_path)
    _FEATURE_COLS = feature_cols


def _process_file(path):
    """Process a single CSV file into jsonl lines."""
    df = pd.read_csv(path)
    if len(df) < MIN_BARS:
        return 0, []

    df.columns = [str(col).lower().strip() for col in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    df_featured = add_features(df.copy())
    if df_featured is None or df_featured.empty:
        return 0, []
    df_featured = df_featured[_FEATURE_COLS]

    normalized_data = _SCALER.transform(df_featured)

    total_length = CONTEXT_LENGTH + PREDICTION_LENGTH
    num_sequences_in_file = len(normalized_data) - total_length + 1
    if num_sequences_in_file <= 0:
        return 0, []

    lines = []
    for i in range(num_sequences_in_file):
        window = normalized_data[i : i + total_length]
        sequence = window.flatten().tolist()
        json_line = json.dumps({
            "sequence": sequence,
            "prediction_length": PREDICTION_LENGTH,
        })
        lines.append(json_line + "\n")

    return num_sequences_in_file, lines

# --- 3. MAIN DATA PREPARATION PIPELINE (REWRITTEN) ---

def run_preparation():
    """Executes the full data preparation pipeline using a two-pass approach."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{DATA_DIR}'.")
        return

    # --- PASS 1: FIT THE SCALER ON ALL DATA GLOBALLY ---
    print("Pass 1: Fitting the Scaler on all available data...")
    
    all_data_for_scaler = []
    with Pool(processes=cpu_count()) as pool:
        for df in tqdm(pool.imap(_load_for_scaler, csv_files), total=len(csv_files), desc="Reading files for scaler"):
            if df is not None:
                all_data_for_scaler.append(df)

    if not all_data_for_scaler:
        print("No files met the MIN_BARS requirement. Exiting.")
        return

    # Concatenate DataFrames with a proper DatetimeIndex
    scaler_fit_df = pd.concat(all_data_for_scaler, axis=0).sort_index()
    
    # Add features to get all columns that will be normalized
    scaler_fit_df = add_features(scaler_fit_df)

    if scaler_fit_df is None or scaler_fit_df.empty:
        print("Could not generate features for scaler fitting. Check your data.")
        return
        
    # Now, scaler.fit receives a DataFrame with ONLY numeric columns.
    scaler = MinMaxScaler()
    scaler.fit(scaler_fit_df)
    
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler has been fitted globally and saved to: {SCALER_FILE}")
    final_feature_columns = scaler_fit_df.columns.tolist()
    NUM_FEATURES = len(final_feature_columns)


    # --- PASS 2: PROCESS EACH FILE INDIVIDUALLY AND CREATE SEQUENCES ---
    print("\nPass 2: Processing each file individually to create sequences...")
    
    total_sequences_written = 0
    processed_files_count = 0

    with open(OUTPUT_FILE, 'w') as f_out, Pool(processes=cpu_count(), initializer=_init_worker, initargs=(SCALER_FILE, final_feature_columns)) as pool:
        for num_seqs, lines in tqdm(pool.imap(_process_file, csv_files), total=len(csv_files), desc="Processing individual files"):
            if num_seqs == 0:
                continue
            f_out.writelines(lines)
            total_sequences_written += num_seqs
            processed_files_count += 1
            
    print(f"\nDone! Processed {processed_files_count} files.")
    print(f"Created '{OUTPUT_FILE}' with a total of {total_sequences_written} sequences.")
    print(f"IMPORTANT: Remember to set 'patch_len = {NUM_FEATURES}' in your model's config file.")

if __name__ == "__main__":
    run_preparation()
