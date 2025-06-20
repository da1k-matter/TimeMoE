import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM
import joblib


@dataclass
class InferenceConfig:
    checkpoint_path: str = "logs/time_moe"
    dataset_path: str = "val_prepared_crypto_data.jsonl"
    scaler_path: str = "crypto_scaler.joblib"
    context_length: int = 500
    prediction_length: int = 48
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    flash_attn: bool = False


def load_last_sequence(dataset_path: str) -> list:
    """Reads the last sequence from a jsonl dataset."""
    last_line = None
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        raise ValueError("Dataset is empty: %s" % dataset_path)
    obj = json.loads(last_line)
    return obj["sequence"], obj.get("prediction_length")


def main(cfg: InferenceConfig):
    print("Configuration:", asdict(cfg))

    # Load scaler
    scaler = joblib.load(cfg.scaler_path)

    # Load model
    model_args = {"device_map": cfg.device, "torch_dtype": "auto", "trust_remote_code": True}
    if cfg.flash_attn:
        model_args["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(cfg.checkpoint_path, **model_args)
    model.eval()

    # Prepare data
    seq, pred_len = load_last_sequence(cfg.dataset_path)
    if pred_len is not None:
        cfg.prediction_length = pred_len
    total_length = cfg.context_length + cfg.prediction_length
    seq = np.array(seq, dtype=np.float32)
    input_size = seq.shape[0] // total_length
    seq = seq.reshape(total_length, input_size)
    inputs = torch.tensor(seq[: cfg.context_length], dtype=torch.float32).unsqueeze(0)

    # Forecast
    with torch.no_grad():
        gen = model.generate(inputs.to(model.device), max_new_tokens=cfg.prediction_length)
    preds = gen[0, -cfg.prediction_length:].cpu().numpy()
    preds_2d = preds.reshape(cfg.prediction_length, input_size)
    preds_unscaled = scaler.inverse_transform(preds_2d)

    if input_size == 1:
        last_pred = preds_unscaled[-1, 0]
    else:
        last_pred = preds_unscaled[-1].tolist()
    print("Last prediction:", last_pred)


if __name__ == "__main__":
    cfg = InferenceConfig()
    main(cfg)
