from __future__ import annotations
from pathlib import Path
import pandas as pd


class CSVProcessor:
    """Handles reading/writing & validation of CSV files."""
    
    REQUIRED_COLUMNS = {"generated_answer", "gold_answer"}

    def load(self, src: Path) -> pd.DataFrame:
        """
        Loads a CSV into a dataframe and validates that
        required columns exist. If missing, raises error.
         Ensures the evaluation pipeline always receives
         structured data with generated and gold answers.
        """
        df = pd.read_csv(src)
        if not self.REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(f"CSV must contain {self.REQUIRED_COLUMNS}")
        return df

    def save(self, df: pd.DataFrame, out: Path):
        """
        Saves dataframe to disk with index suppressed
        and creates parent directories automatically.

        Provides helpful output path confirmation.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved results → {out.resolve()}")

    def determine_output_path(self, src: Path, out):
        """
        Decide where final output will be stored:
         If user manually provided output path use it
         Otherwise default to: originalName_with_metrics.csv
        """
        return out if out else src.with_name(src.stem + "_with_metrics.csv")
