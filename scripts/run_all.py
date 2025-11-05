import os
import sys
import json
import yaml
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "config.yaml"

def sh(cmd: list[str]):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def ensure_parent(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def load_cfg():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_data(cfg):
    print("Step 1:…")
    data_dir = ROOT / cfg["paths"]["data_dir"]
    if not data_dir.exists():
        print(f"data directory not found：{data_dir}")
    else:
        print(f"data directory：{data_dir}")

def run_traditional(cfg):
    print("Step 2: …")
    entry = ROOT / cfg["modules"]["traditional"]["main_script"]
    if not entry.exists():
        raise FileNotFoundError(f"：{entry}")

    # make sure output directory
    ensure_parent(ROOT / cfg["paths"]["trad_pred_valid"])
    ensure_parent(ROOT / cfg["paths"]["trad_pred_test"])
    ensure_parent(ROOT / cfg["paths"]["trad_model"])

    sh([sys.executable, str(entry), "--config", str(CONFIG)])

def run_deep(cfg):
    print("Step 3:  …")
    entry = ROOT / cfg["modules"]["deep"]["main_script"]
    if not entry.exists():
        print(f"：{entry}")
        return

    ensure_parent(ROOT / cfg["paths"]["deep_pred_valid"])
    ensure_parent(ROOT / cfg["paths"]["deep_pred_test"])

    sh([sys.executable, str(entry), "--config", str(CONFIG)])

def run_eval(cfg):
    print("Step 4:  …")
    entry = ROOT / "eval" / "evaluate.py"
    if not entry.exists():
        print(f"：{entry}")
        return

    ensure_parent(ROOT / cfg["paths"]["eval_results"])
    sh([
        sys.executable, str(entry),
        "--trad_pred", str(ROOT / cfg["paths"]["trad_pred_test"]),
        "--deep_pred", str(ROOT / cfg["paths"]["deep_pred_test"]),
        "--output",    str(ROOT / cfg["paths"]["eval_results"]),
        "--label_map", str(ROOT / cfg["paths"]["label_map"])
    ])

if __name__ == "__main__":
    cfg = load_cfg()
    run_data(cfg)
    run_traditional(cfg)
    run_deep(cfg)
    run_eval(cfg)
    print("All done.")
