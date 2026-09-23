"""Merge hyb_meta.json (written by t5_hybrids) into predict_meta.json of an eval dir.  Run only when no `predict` is in flight."""
import sys
from pathlib import Path
from .common import OUT, load_json, save_json

d = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "eval" / "test"
meta = load_json(d / "predict_meta.json"); hyb = load_json(d / "hyb_meta.json")
for k, v in hyb["methods"].items():
    meta["methods"][k] = v
save_json(meta, d / "predict_meta.json"); print("merged", list(hyb["methods"]))
