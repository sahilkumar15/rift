# Path: src/utils/wandb_logger.py
# Status: NEW
"""W&B wrapper that no-ops cleanly when wandb is absent or disabled."""
class WandbLogger:
    def __init__(self, project=None, name=None, config=None, enabled=True):
        self.enabled = enabled; self.run = None
        if not enabled: return
        try:
            import wandb
            self.run = wandb.init(project=project, name=name,
                                  config=dict(config) if config else {})
            self._wandb = wandb
        except Exception as e:
            print(f"[wandb] disabled ({e})"); self.enabled = False
    def log(self, d, step=None):
        if self.enabled and self.run is not None:
            self._wandb.log(d, step=step)
    def log_table(self, key, columns, rows):
        if self.enabled and self.run is not None:
            t = self._wandb.Table(columns=columns, data=rows)
            self._wandb.log({key: t})
    def finish(self):
        if self.enabled and self.run is not None: self._wandb.finish()
