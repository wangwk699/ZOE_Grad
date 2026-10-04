"""Training loss recording for the supported gradient experiments."""

import json
from pathlib import Path

from transformers import Trainer, TrainerCallback


class _LossRecorder(TrainerCallback):
    def __init__(self, history):
        self.history = history

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.history["train"].append({"global_step": state.global_step, **logs})
        if "eval_loss" in logs:
            self.history["eval"].append({"global_step": state.global_step, **logs})


class RoundZOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_history = {"train": [], "eval": []}
        self.add_callback(_LossRecorder(self._loss_history))

    def save_loss_history(self):
        if self.is_world_process_zero():
            path = Path(self.args.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "loss_history.json").write_text(
                json.dumps(self._loss_history, indent=2, default=str) + "\n"
            )

    def plot_loss_curve(self, output_path, show=False):
        points = self._loss_history["train"]
        if not points:
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([item["global_step"] for item in points],
                [item["loss"] for item in points])
        ax.set(xlabel="Training step", ylabel="Loss")
        fig.savefig(output_path)
        if show:
            plt.show()
        plt.close(fig)

    def print_loss_summary(self):
        points = self._loss_history["train"]
        if points:
            print(f"Recorded {len(points)} training losses; final loss {points[-1]['loss']:.6f}")
