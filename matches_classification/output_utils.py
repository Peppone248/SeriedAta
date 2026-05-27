from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def save_outputs(
    outputs: dict,
    output_dir: str = "data/processed",
    save_raw: bool = False,
) -> None:
    """
    Save pipeline outputs to disk.

    Rules:
    - DataFrames -> CSV
    - Series -> CSV
    - dicts -> JSON if possible, otherwise save nested DataFrames/Series separately
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, value in outputs.items():
        if name == "raw_df" and not save_raw:
            continue

        if isinstance(value, pd.DataFrame):
            value.to_csv(out_path / f"{name}.csv", index=False)

        elif isinstance(value, pd.Series):
            value.to_csv(out_path / f"{name}.csv", index=True)

        elif isinstance(value, dict):
            _save_dict(name, value, out_path)

        else:
            with open(out_path / f"{name}.txt", "w", encoding="utf-8") as f:
                f.write(str(value))


def _save_dict(name: str, value: dict, out_path: Path) -> None:
    """
    Save dictionaries.
    - nested DataFrames/Series get their own CSV
    - simple values go to JSON
    """
    simple_items = {}

    for sub_name, sub_value in value.items():
        full_name = f"{name}__{sub_name}"

        if isinstance(sub_value, pd.DataFrame):
            sub_value.to_csv(out_path / f"{full_name}.csv", index=False)

        elif isinstance(sub_value, pd.Series):
            sub_value.to_csv(out_path / f"{full_name}.csv", index=True)

        else:
            simple_items[sub_name] = sub_value

    if simple_items:
        with open(out_path / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(simple_items, f, indent=2, default=str)