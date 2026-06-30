# Path: tests/test_config_checkpoint.py
# Status: NEW
import os, tempfile
from src.utils.config import Config, merge_overrides
from src.metrics.binary_metrics import roc_auc
def test_config_dotted():
    c=Config({"a":Config({"b":3})}); assert c.get_dotted("a.b")==3
    c2=merge_overrides(c, {"a.b":9}); assert c2["a"]["b"]==9
def test_auc_known(): assert abs(roc_auc([0,0,1,1],[0.1,0.2,0.8,0.9])-1.0)<1e-9
