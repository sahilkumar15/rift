# Path: src/utils/wandb_logger.py
"""W&B wrapper that no-ops cleanly when W&B is absent or disabled."""

from __future__ import annotations

import os


class WandbLogger:
    def __init__(
        self,
        project=None,
        name=None,
        config=None,
        enabled=True,
        *,
        entity=None,
        group=None,
        tags=None,
        notes=None,
        mode="online",
        save_code=False,
        log_model=False,
    ):
        self.enabled = bool(enabled)
        self.run = None
        self._wandb = None
        self.log_model = bool(log_model)

        mode = str(mode or "online").lower()

        if mode == "disabled":
            self.enabled = False

        if not self.enabled:
            return

        try:
            import wandb

            if mode in ("online", "offline", "disabled"):
                os.environ["WANDB_MODE"] = mode

            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                group=group,
                tags=list(tags or []),
                notes=notes,
                config=dict(config) if config else {},
                save_code=bool(save_code),
            )
            self._wandb = wandb

        except Exception as e:
            print(f"[wandb] disabled ({e})")
            self.enabled = False

    def log(self, d, step=None):
        if self.enabled and self.run is not None:
            self._wandb.log(d, step=step)

    def log_table(self, key, columns, rows):
        if self.enabled and self.run is not None:
            t = self._wandb.Table(columns=columns, data=rows)
            self._wandb.log({key: t})

    def log_artifact(self, path, name=None, artifact_type="model"):
        if not self.enabled or self.run is None or not path:
            return

        if not os.path.exists(path):
            return

        artifact_name = name or os.path.basename(path).replace(".", "-")
        art = self._wandb.Artifact(artifact_name, type=artifact_type)

        if os.path.isdir(path):
            art.add_dir(path)
        else:
            art.add_file(path)

        self.run.log_artifact(art)

    def finish(self):
        if self.enabled and self.run is not None:
            self._wandb.finish()
