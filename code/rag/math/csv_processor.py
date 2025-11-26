from __future__ import annotations
from pathlib import Path
import pandas as pd


class CSVProcessor:
    """Handles reading/writing & validation of CSV files."""

    REQUIRED_COLUMNS = {"generated_answer", "gold_answer"}

    def load(self, src: Path) -> pd.DataFrame:
        df = pd.read_csv(src)
        if not self.REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"CSV must contain {self.REQUIRED_COLUMNS}")
        return df

    def save(self, df: pd.DataFrame, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved results → {out.resolve()}")

    def determine_output_path(self, src: Path, out):
        return out if out else src.with_name(src.stem + "_with_metrics.csv")
