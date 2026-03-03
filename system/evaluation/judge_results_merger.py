"""Utilities for pivoting judge CSV rows into a single merged table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class JudgeMergeConfig:
    input_csv: Path
    output_csv: Path

    def __post_init__(self) -> None:
        self.input_csv = Path(self.input_csv)
        self.output_csv = Path(self.output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)


class JudgeResultsMerger:
    def load(self, config: JudgeMergeConfig) -> pd.DataFrame:
        if not config.input_csv.exists():
            raise FileNotFoundError(
                f"Cannot locate {config.input_csv}. Ensure the CSV exists before merging."
            )
        df = pd.read_csv(config.input_csv)
        required_cols = {"custom_id", "permutation_id", "judge_type", "judge_answer", "text"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                "Input CSV missing required columns: " + ", ".join(sorted(missing))
            )
        return df

    def pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        pivot_df = (
            df.pivot_table(
                index=["permutation_id"],
                columns="judge_type",
                values=["judge_answer", "text"],
                aggfunc="first",
            )
            .sort_index(axis=1)
            .reset_index()
        )
        pivot_df.columns = [
            f"{value}_{judge_type}" if judge_type else value
            for value, judge_type in pivot_df.columns
        ]
        return pivot_df

    def write(self, df: pd.DataFrame, config: JudgeMergeConfig) -> Path:
        try:
            df.to_csv(config.output_csv, index=False, encoding="utf-8")
            return config.output_csv
        except PermissionError:
            fallback = config.output_csv.with_name(
                f"{config.output_csv.stem}_new{config.output_csv.suffix}"
            )
            df.to_csv(fallback, index=False, encoding="utf-8")
            return fallback

    def run(self, config: JudgeMergeConfig) -> Path:
        df = self.load(config)
        merged = self.pivot(df)
        return self.write(merged, config)


__all__ = ["JudgeMergeConfig", "JudgeResultsMerger"]
