"""The script a Kaggle session runs (pushed as ``kernel_main.py``). It only wires attached datasets to the runner.

Attached datasets (names are ``<prefix>-...``): the code (``finfm/`` and ``config/``), the staged shard
folder, optionally earlier output bundles to resume from, optionally offline wheels and model weights.
The runner stops itself before the session cutoff and leaves ONE bundle in /kaggle/working.
"""

import glob
import json
import os
import subprocess
import sys

INPUT = "/kaggle/input"
OUT = "/kaggle/working"


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(INPUT, pattern), recursive=True))
    if not hits:
        raise SystemExit(f"nothing attached that matches {pattern}")
    return hits[0]


code_root = os.path.dirname(find_one("finfm-code*/config"))
sys.path.insert(0, code_root)
wheels = glob.glob(os.path.join(INPUT, "finfm-wheels*"))
if wheels:  # offline install of anything the image lacks
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels[0], "vllm"], check=False)
manifest = find_one("finfm-shard*/**/shard_manifest.json")
resume = [os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "finfm-bundle*"))]
print(json.dumps({"code": code_root, "manifest": manifest, "resume_from": resume}))

from finfm.compute.runner import main  # noqa: E402  (imported after the path is set)

raise SystemExit(main(["--shard", os.path.dirname(manifest), "--out", OUT, "--config", os.path.join(code_root, "config"),
                       "--inputs", *resume]))
