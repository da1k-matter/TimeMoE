from __future__ import annotations

from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from time_moe.models.modeling_time_moe import TimeMoeForPrediction
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    from tqdm import tqdm



@dataclass
class Config:
    model_path: str
    scaler_path: str
    dataset_path: str
    context_length: int = 500
    prediction_length: int = 48
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    attn_implementation: str = "eager"


class PredictionDataset(Dataset):
    """Dataset for prepared JSONL sequences."""

    def __init__(self, path: str, context_length: int, input_size: int):
        self.context_length = context_length
        self.input_size = input_size
        self.sequences: List[np.ndarray] = []

        print(f"Loading sequences from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Loading sequences"):
                obj = json.loads(line)
                seq = np.array(obj["sequence"], dtype=np.float32)
                seq = seq.reshape(-1, input_size)
                self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        inputs = seq[: self.context_length]
        return torch.tensor(inputs, dtype=torch.float32)




def run_predict(cfg: Config):
    print(f"Loading scaler from {cfg.scaler_path}...")
    scaler = joblib.load(cfg.scaler_path)
    input_size = scaler.n_features_in_
    feature_names = list(
        getattr(scaler, "feature_names_in_", [f"f{i}" for i in range(input_size)])
    )
    print(f"Loading dataset from {cfg.dataset_path}...")

    dataset = PredictionDataset(
        cfg.dataset_path,
        context_length=cfg.context_length,
        input_size=input_size,
    )

    if len(dataset) > 0:
        tail_raw = dataset.sequences[0][-cfg.context_length:]
        tail_inv = scaler.inverse_transform(tail_raw)[-5:]
        df_tail = pd.DataFrame(tail_inv, columns=feature_names)
        print("Dataset tail:")
        print(df_tail)

    print("Creating dataloader...")
    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"Loading model from {cfg.model_path}...")

    model = TimeMoeForPrediction.from_pretrained(
        cfg.model_path,
        attn_implementation=cfg.attn_implementation,
        device_map=cfg.device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    print("Generating predictions...")

    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Predicting"):
            inputs = batch.to(cfg.device)
            out = model.generate(inputs, max_new_tokens=cfg.prediction_length)
            preds = out[:, -cfg.prediction_length :].cpu().numpy()
            all_preds.append(preds)

    preds = np.concatenate(all_preds, axis=0)
    flat = preds.reshape(-1, preds.shape[-1])
    inv = scaler.inverse_transform(flat)
    inv = inv.reshape(preds.shape)

    print(f"Predictions shape: {inv.shape}")
    if len(inv) > 0:
        print("Prediction samples:")
        for idx, sample in enumerate(inv):
            df = pd.DataFrame(sample, columns=feature_names)
            print(f"Sample {idx}:")
            print(df)
    return inv


if __name__ == "__main__":
    cfg = Config(
        model_path="checkpoint",
        scaler_path="crypto_scaler.joblib",
        dataset_path="val_prepared_crypto_data.jsonl",
    )
    predictions = run_predict(cfg)
    print(predictions)
