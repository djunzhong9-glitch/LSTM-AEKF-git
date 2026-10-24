import json
from pathlib import Path


def save_summary(metrics: dict, history, output_path: Path) -> None:
    """Save machine-readable metrics and training history."""

    output_path.write_text(
        json.dumps({"metrics": metrics, "training_history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
