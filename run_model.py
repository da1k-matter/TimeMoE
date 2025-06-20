import pandas as pd
import numpy as np

# ``pandas_ta`` expects the constant ``np.NaN`` which was removed in
# recent versions of NumPy (>=2.0).  For compatibility we recreate this
# alias before importing ``pandas_ta``.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import joblib
import pandas_ta
import torch
from transformers import AutoModelForCausalLM
import argparse
import os
from dataclasses import dataclass


class SimpleScaler:
    """Minimal scaler for normalizing and denormalizing data."""

    def __init__(self, mean, std, feature_names):
        self.mean = mean
        self.std = std
        self.feature_names_in_ = feature_names

    def transform(self, x):
        return np.asarray(x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


@dataclass
class InferenceConfig:
    """Configuration options for running the model."""

    checkpoint: str = "checkpoint-9205"
    scaler: str = "crypto_scaler.joblib"
    csv_path: str = "data_test/BTC_30.csv"
    context_length: int = 500
    prediction_length: int = 48


config = InferenceConfig()


def add_features(df, use_ta=None):
    if use_ta is None:
        use_ta = ['rsi', 'adx', 'atr', 'sma']
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        missing = list(required_cols - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")
    df[list(required_cols)] = df[list(required_cols)].astype(float)
    if 'rsi' in use_ta:
        df.ta.rsi(length=14, append=True)
    if 'adx' in use_ta:
        df.ta.adx(length=14, append=True)
    if 'atr' in use_ta:
        df.ta.atr(length=14, append=True)
    if 'sma' in use_ta:
        df.ta.sma(length=14, append=True)
    if 'ema' in use_ta:
        df.ta.ema(length=14, append=True)
    if 'macd' in use_ta:
        df.ta.macd(append=True)
    df.dropna(inplace=True)
    return df


def load_and_preprocess(csv_path, scaler):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    df_feat = add_features(df.copy())
    feat_cols = getattr(scaler, 'feature_names_in_', None)
    if feat_cols is not None:
        missing = [c for c in feat_cols if c not in df_feat.columns]
        if missing:
            raise ValueError(f"Data is missing columns required by scaler: {missing}")
        df_feat = df_feat[feat_cols]
    data = scaler.transform(df_feat)
    return data.astype(np.float32), df_feat.index


def sliding_forecast(model, data, scaler, context_length, prediction_length, device):
    num_windows = len(data) - context_length
    preds = []
    for start in range(num_windows):
        window = data[start:start + context_length]
        inputs = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            # Pass the raw values as ``input_ids`` so the model can apply its
            # internal embedding layer. Using ``inputs_embeds`` would bypass the
            # embedding layer and lead to a shape mismatch.
            output = model.generate(input_ids=inputs, max_new_tokens=prediction_length)
        norm_pred = output[0, -prediction_length:].cpu().numpy()
        pred = scaler.inverse_transform(norm_pred)
        preds.append(pred)
    return np.array(preds)


def main():
    """Run forecasting using the parameters specified in ``config``."""

    parser = argparse.ArgumentParser(description="Run Time-MoE forecasting")
    parser.add_argument(
        "--checkpoint",
        default=config.checkpoint,
        help="Path to the model checkpoint directory",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scaler = joblib.load(config.scaler)
    data, index = load_and_preprocess(config.csv_path, scaler)

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    preds = sliding_forecast(
        model,
        data,
        scaler,
        config.context_length,
        config.prediction_length,
        device,
    )

    pred_index = index[config.context_length:config.context_length + len(preds)]
    feat_cols = getattr(
        scaler,
        'feature_names_in_',
        [f'feat_{i}' for i in range(preds.shape[-1])],
    )
    columns = []
    for step in range(config.prediction_length):
        columns.extend([f'{c}_t{step+1}' for c in feat_cols])
    preds_2d = preds.reshape(len(preds), -1)
    df_pred = pd.DataFrame(preds_2d, index=pred_index, columns=columns)
    print(df_pred)


if __name__ == '__main__':
    main()
