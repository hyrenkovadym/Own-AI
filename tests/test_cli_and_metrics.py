from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from main import _append_metrics_record


def test_cli_modules_command_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "main.py", "modules"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert proc.returncode == 0
    output = f"{proc.stdout}\n{proc.stderr}".lower()
    assert "chat" in output
    assert "snake" in output


def test_metrics_export_csv_json_jsonl(tmp_path: Path) -> None:
    record = {"run_type": "train", "episodes": 10, "avg_score_last_100": "1.23"}

    csv_path = _append_metrics_record(str(tmp_path / "metrics.csv"), record)
    json_path = _append_metrics_record(str(tmp_path / "metrics.json"), record)
    jsonl_path = _append_metrics_record(str(tmp_path / "metrics.jsonl"), record)

    assert csv_path.exists()
    assert json_path.exists()
    assert jsonl_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "timestamp_utc" in csv_text
    assert "run_type" in csv_text
    assert "train" in csv_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["run_type"] == "train"

    jsonl_lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert jsonl_lines
