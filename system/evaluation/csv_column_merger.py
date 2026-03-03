"""Utilities to merge two CSV files on a shared column."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import pandas as pd


VALID_JOIN_TYPES = {"inner", "outer", "left", "right"}


@dataclass
class CSVColumnMergeConfig:
    left_csv: Path
    right_csv: Path
    output_csv: Path
    on: str
    how: str = "inner"
    exclude_columns: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.left_csv = Path(self.left_csv)
        self.right_csv = Path(self.right_csv)
        self.output_csv = Path(self.output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.on:
            raise ValueError("Join column (--on) must be provided.")
        self.how = self.how.lower()
        if self.how not in VALID_JOIN_TYPES:
            valid = ", ".join(sorted(VALID_JOIN_TYPES))
            raise ValueError(f"Join type must be one of: {valid}")
        # Ensure exclude columns stored as tuple for immutability downstream.
        self.exclude_columns = tuple(self.exclude_columns)


class CSVColumnMerger:
    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Cannot locate CSV file {path}.")
        return pd.read_csv(path)

    def merge(self, config: CSVColumnMergeConfig) -> pd.DataFrame:
        left_df = self._load_csv(config.left_csv)
        right_df = self._load_csv(config.right_csv)
        merged = pd.merge(
            left_df,
            right_df,
            how=config.how,
            on=config.on,
            suffixes=("_left", "_right"),
        )
        if config.exclude_columns:
            drop_cols: List[str] = [col for col in config.exclude_columns if col in merged.columns]
            if drop_cols:
                merged = merged.drop(columns=drop_cols)
        return merged

    @staticmethod
    def write(df: pd.DataFrame, output_path: Path) -> Path:
        try:
            df.to_csv(output_path, index=False, encoding="utf-8")
            return output_path
        except PermissionError:
            fallback = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
            df.to_csv(fallback, index=False, encoding="utf-8")
            return fallback

    def run(self, config: CSVColumnMergeConfig) -> Path:
        merged_df = self.merge(config)
        return self.write(merged_df, config.output_csv)


__all__ = ["CSVColumnMergeConfig", "CSVColumnMerger"]
