from transformers.trainer_callback import ProgressCallback
import copy

class LossProgressCallback(ProgressCallback):
    """Progress bar callback that displays the current loss."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if logs is None:
            return
        if state.is_world_process_zero and self.training_bar is not None:
            loss = logs.get("loss")
            if loss is not None:
                self.training_bar.set_postfix(loss=f"{loss:.4f}")

