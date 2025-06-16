from transformers.trainer_callback import ProgressCallback
import copy

class LossProgressCallback(ProgressCallback):
    """Progress bar callback that displays the current loss."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if logs is None:
            return
        if state.is_world_process_zero:
            postfix = {}
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")

            if loss is not None:
                postfix["loss"] = f"{loss:.4f}"

            if eval_loss is not None:
                postfix["val_loss"] = f"{eval_loss:.4f}"

            if postfix:
                if self.training_bar is not None:
                    self.training_bar.set_postfix(**postfix)
                elif self.prediction_bar is not None:
                    self.prediction_bar.set_postfix(**postfix)

