import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    from tqdm import tqdm

DATA_DIR = "data_test"
OUTPUT_FILE = "prepared_close_data.jsonl"
TRAIN_OUTPUT_FILE = "train_" + OUTPUT_FILE
VAL_OUTPUT_FILE = "val_" + OUTPUT_FILE

DEFAULT_TRAIN_SIZE = 30000
DEFAULT_VAL_SIZE = 2000
SCALER_FILE = "close_scaler.joblib"

CONTEXT_LENGTH = 500
PREDICTION_LENGTH = 48


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        print("[ERROR] 'close' column not found in data")
        return None
    df["close"] = df["close"].astype(float)
    df.dropna(subset=["close"], inplace=True)
    return df[["close"]]


def run_preparation(train_size: int = DEFAULT_TRAIN_SIZE, val_size: int = DEFAULT_VAL_SIZE):
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{DATA_DIR}'.")
        return

    min_required_bars = train_size + val_size

    print("Pass 1: Fitting the Scaler on all available data...")
    all_data_for_scaler = []
    for f in tqdm(csv_files, desc="Reading files for scaler"):
        df = pd.read_csv(f)
        if len(df) < min_required_bars:
            continue
        df.columns = [str(col).lower().strip() for col in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        df_feat = add_features(df)
        if df_feat is not None and not df_feat.empty:
            all_data_for_scaler.append(df_feat)

    if not all_data_for_scaler:
        print("No files contained the required minimum number of bars. Exiting.")
        return

    scaler_fit_df = pd.concat(all_data_for_scaler, axis=0).sort_index()
    scaler = MinMaxScaler()
    scaler.fit(scaler_fit_df)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler has been fitted globally and saved to: {SCALER_FILE}")
    final_feature_columns = scaler_fit_df.columns.tolist()
    num_features = len(final_feature_columns)

    print("\nPass 2: Processing each file individually to create sequences...")
    total_sequences_written = 0
    total_val_sequences_written = 0
    processed_files_count = 0

    with open(TRAIN_OUTPUT_FILE, "w") as f_train, open(VAL_OUTPUT_FILE, "w") as f_val:
        for f in tqdm(csv_files, desc="Processing individual files"):
            df = pd.read_csv(f)
            if len(df) < min_required_bars:
                continue
            df.columns = [str(col).lower().strip() for col in df.columns]
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            df_feat = add_features(df.copy())
            if df_feat is None or df_feat.empty:
                continue
            df_feat = df_feat[final_feature_columns]
            normalized_data = scaler.transform(df_feat)

            total_length = CONTEXT_LENGTH + PREDICTION_LENGTH
            split_idx = max(len(normalized_data) - val_size, 0)
            train_start = max(split_idx - train_size, 0)
            train_data = normalized_data[train_start:split_idx]
            val_data = normalized_data[split_idx:]

            num_train_sequences = len(train_data) - total_length + 1
            if num_train_sequences > 0:
                for i in tqdm(range(num_train_sequences), desc=f"Train seqs in {os.path.basename(f)}", leave=False):
                    window = train_data[i : i + total_length]
                    sequence = window.flatten().tolist()
                    json_line = json.dumps({"sequence": sequence, "prediction_length": PREDICTION_LENGTH})
                    f_train.write(json_line + "\n")
                total_sequences_written += num_train_sequences

            num_val_sequences = len(val_data) - total_length + 1
            if num_val_sequences > 0:
                for i in tqdm(range(num_val_sequences), desc=f"Val seqs in {os.path.basename(f)}", leave=False):
                    window = val_data[i : i + total_length]
                    sequence = window.flatten().tolist()
                    json_line = json.dumps({"sequence": sequence, "prediction_length": PREDICTION_LENGTH})
                    f_val.write(json_line + "\n")
                total_val_sequences_written += num_val_sequences

            processed_files_count += 1

    print(f"\nDone! Processed {processed_files_count} files.")
    print(f"Created '{TRAIN_OUTPUT_FILE}' with a total of {total_sequences_written} sequences.")
    print(f"Created '{VAL_OUTPUT_FILE}' with a total of {total_val_sequences_written} sequences.")
    print(f"IMPORTANT: Remember to set 'input_len = {num_features}' in your model's config file.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare 1D crypto dataset")
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of bars to use for training from each CSV file")
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE, help="Number of bars to use for validation from each CSV file")

    args = parser.parse_args()
    run_preparation(train_size=args.train_size, val_size=args.val_size)
