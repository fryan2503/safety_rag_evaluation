"""
Analysis module converted from Jupyter notebook.

Provides a single callable entry point:

    from analysis import analyze_csv
    analyze_csv("path/to/file.csv", output_dir="results")

The function reads the CSV, performs aggregation and scoring that
roughly mirror the original notebook, writes artifacts to disk
(CSVs, charts, and text/markdown reports), and returns the key
dataframes for further use in Python (optional).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import datetime


# # ---------------------------------------------------------------------------
# # Data structures
# # ---------------------------------------------------------------------------

# @dataclass
# class AnalysisResults:
#     """Container for results returned by analyze_csv.

#     All important outputs are also written to disk in ``output_dir``.
#     """
#     df_raw: pd.DataFrame
#     df_agg: pd.DataFrame
#     output_dir: Path
#     summary_text_path: Path
#     full_report_path: Path
#     artifacts: Dict[str, Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_seconds(x: Any) -> Optional[float]:
    """Convert strings like '12 seconds' or '1.5 s' to float seconds."""
    if pd.isna(x):
        return None
    import re

    match = re.search(r"([\d\.]+)", str(x))
    return float(match.group(1)) if match else None


def _compute_price(row: pd.Series) -> float:
    """
    Compute cost (in USD) based on model type and total tokens.

    Prices per 1M tokens (rough approximation based on the notebook comments):
      - models whose name contains 'nano': 0.40 USD
      - models whose name contains 'mini': 2.00 USD
      - otherwise: 1.00 USD (fallback)
    """
    tokens = row.get("meta_total_tokens", 0) or 0
    try:
        tokens = float(tokens)
    except Exception:
        tokens = 0.0

    model = str(row.get("model", "")).lower()

    if "nano" in model:
        price_per_million = 0.40
    elif "mini" in model:
        price_per_million = 2.00
    else:
        price_per_million = 1.00

    return (tokens / 1_000_000.0) * price_per_million


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convert a DataFrame to a GitHub-flavoured markdown table."""
    if df.empty:
        return "_(no rows)_\n"

    df = df.head(max_rows)
    # Convert floats to 4 decimal places for readability
    df = df.copy()

    def _fmt(x):
        try:
            if pd.isna(x):
                return ""
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].apply(_fmt)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows]) + "\n"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw dataframe similarly to the notebook."""
    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Booleans
    bool_cols = ["judge_answer_correctness_vs_ref", "judge_answer_helpfulness"]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype != bool:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                )

    # Numerics
    numeric_cols = ["meta_total_tokens", "cosine", "rougeL", "bleu"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Total elapsed time (seconds)
    if "total_elapsed_time" in df.columns:
        df["total_elapsed_time_sec"] = df["total_elapsed_time"].apply(_parse_seconds)

    # Price estimation
    df["price_usd"] = df.apply(_compute_price, axis=1)

    return df


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by (approach, model, top_k) like in the notebook."""
    required_group_cols = ["approach", "model", "top_k"]
    group_cols = [c for c in required_group_cols if c in df.columns]
    if len(group_cols) < 2:
        raise ValueError(
            "Not enough grouping columns found in the CSV. "
            "Expected at least 'approach' and 'model', optionally 'top_k'."
        )

    agg_dict: Dict[str, Any] = {}

    # Standard numeric metrics if present
    if "meta_total_tokens" in df.columns:
        agg_dict["avg_meta_total_tokens"] = ("meta_total_tokens", "mean")
    if "cosine" in df.columns:
        agg_dict["avg_cosine"] = ("cosine", "mean")
    if "rougeL" in df.columns:
        agg_dict["avg_rougeL"] = ("rougeL", "mean")
    if "bleu" in df.columns:
        agg_dict["avg_bleu"] = ("bleu", "mean")
    if "price_usd" in df.columns:
        agg_dict["avg_price_usd"] = ("price_usd", "mean")
    if "total_elapsed_time_sec" in df.columns:
        agg_dict["avg_latency_sec"] = ("total_elapsed_time_sec", "mean")

    # Boolean success counts
    if "judge_answer_correctness_vs_ref" in df.columns:
        agg_dict["true_correctness_count"] = ("judge_answer_correctness_vs_ref", "sum")
        agg_dict["total_correctness_count"] = (
            "judge_answer_correctness_vs_ref",
            "count",
        )

    if "judge_answer_helpfulness" in df.columns:
        agg_dict["true_helpfulness_count"] = ("judge_answer_helpfulness", "sum")
        agg_dict["total_helpfulness_count"] = ("judge_answer_helpfulness", "count")

    agg_df = (
        df.groupby(group_cols)
        .agg(**agg_dict)  # type: ignore[arg-type]
        .reset_index()
    )

    # Percent metrics
    if {"true_correctness_count", "total_correctness_count"}.issubset(agg_df.columns):
        agg_df["pct_correctness_true"] = (
            agg_df["true_correctness_count"]
            / agg_df["total_correctness_count"].replace({0: pd.NA})
            * 100.0
        )

    if {"true_helpfulness_count", "total_helpfulness_count"}.issubset(agg_df.columns):
        agg_df["pct_helpfulness_true"] = (
            agg_df["true_helpfulness_count"]
            / agg_df["total_helpfulness_count"].replace({0: pd.NA})
            * 100.0
        )

    return agg_df


def _add_scores(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking scores & combined score, roughly matching the notebook."""
    agg_df = agg_df.copy()

    score_cols: list[str] = []
    if "pct_correctness_true" in agg_df.columns:
        score_cols.append("pct_correctness_true")
    if "pct_helpfulness_true" in agg_df.columns:
        score_cols.append("pct_helpfulness_true")
    if "avg_price_usd" in agg_df.columns:
        score_cols.append("avg_price_usd")
    if "avg_latency_sec" in agg_df.columns:
        score_cols.append("avg_latency_sec")

    if not score_cols:
        # Nothing to score; just return as-is
        return agg_df

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(agg_df[score_cols].fillna(agg_df[score_cols].mean()))
    scaled_df = pd.DataFrame(scaled, columns=score_cols, index=agg_df.index)

    # For correctness & helpfulness: higher is better => direct scale.
    # For price & latency: lower is better => invert scale.
    if "pct_correctness_true" in scaled_df.columns:
        agg_df["score_correctness"] = scaled_df["pct_correctness_true"]
    if "pct_helpfulness_true" in scaled_df.columns:
        agg_df["score_helpfulness"] = scaled_df["pct_helpfulness_true"]
    if "avg_price_usd" in scaled_df.columns:
        agg_df["score_price"] = 1.0 - scaled_df["avg_price_usd"]
    # if "avg_latency_sec" in scaled_df.columns:
    #     agg_df["score_latency"] = 1.0 - scaled_df["avg_latency_sec"]

    score_components = [
        col
        for col in [
            "score_correctness",
            "score_helpfulness",
            "score_price",
            "score_latency",
        ]
        if col in agg_df.columns
    ]

    # if score_components:
        # agg_df["combined_score"] = agg_df[score_components].mean(axis=1)

    # Sort & add rank (1 = best)
    if "combined_score" in agg_df.columns:
        agg_df = agg_df.sort_values("combined_score", ascending=False)
        agg_df["rank"] = range(1, len(agg_df) + 1)

    return agg_df


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _generate_figures(agg_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Generate and save several charts; return mapping name -> path."""
    artifact_paths: Dict[str, Path] = {}

    if agg_df.empty:
        return artifact_paths

    # 1) Heatmap of avg_cosine by (approach, model)
    if {"approach", "model", "avg_cosine"}.issubset(agg_df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        heatmap_df = agg_df.pivot_table(
            values="avg_cosine", index="approach", columns="model"
        )
        sns.heatmap(heatmap_df, annot=True, ax=ax)
        ax.set_title("Average Cosine Similarity by Approach & Model")
        path = output_dir / "heatmap_avg_cosine.png"
        _save_fig(fig, path)
        artifact_paths["heatmap_avg_cosine"] = path

    # 2) Bar chart of correctness
    if {"pct_correctness_true", "model"}.issubset(agg_df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=agg_df.sort_values("pct_correctness_true", ascending=False),
            x="pct_correctness_true",
            y="model",
            hue="approach" if "approach" in agg_df.columns else None,
            ax=ax,
        )
        ax.set_title("% Correctness TRUE by Combination")
        ax.set_xlabel("% Correctness TRUE")
        ax.set_ylabel("Model")
        path = output_dir / "bar_correctness.png"
        _save_fig(fig, path)
        artifact_paths["bar_correctness"] = path

    # 3) Bar chart of helpfulness
    if {"pct_helpfulness_true", "model"}.issubset(agg_df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=agg_df.sort_values("pct_helpfulness_true", ascending=False),
            x="pct_helpfulness_true",
            y="model",
            hue="approach" if "approach" in agg_df.columns else None,
            ax=ax,
        )
        ax.set_title("% Helpfulness TRUE by Combination")
        ax.set_xlabel("% Helpfulness TRUE")
        ax.set_ylabel("Model")
        path = output_dir / "bar_helpfulness.png"
        _save_fig(fig, path)
        artifact_paths["bar_helpfulness"] = path

    # 4) Bubble chart correctness vs helpfulness
    if {"pct_correctness_true", "pct_helpfulness_true"}.issubset(agg_df.columns):
        fig, ax = plt.subplots(figsize=(10, 8))
        sizes = (
            agg_df["avg_cosine"] if "avg_cosine" in agg_df.columns else None
        )
        sns.scatterplot(
            data=agg_df,
            x="pct_correctness_true",
            y="pct_helpfulness_true",
            size=sizes,
            hue="model" if "model" in agg_df.columns else None,
            style="approach" if "approach" in agg_df.columns else None,
            sizes=(100, 600),
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(
            "Correctness vs Helpfulness (Bubble size = Avg Cosine if available)"
        )
        ax.set_xlabel("% Correctness TRUE")
        ax.set_ylabel("% Helpfulness TRUE")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        path = output_dir / "bubble_correctness_helpfulness.png"
        _save_fig(fig, path)
        artifact_paths["bubble_correctness_helpfulness"] = path

    return artifact_paths


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_text_summary(
    df_raw: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write a plain-text high level summary."""
    path = output_dir / "summary.txt"

    num_rows, num_cols = df_raw.shape
    num_combos = len(agg_df)
    

    lines = []
    lines.append("Model Evaluation Summary")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Raw rows: {num_rows}")
    lines.append(f"Raw columns: {num_cols}")
    lines.append(f"Unique (approach, model, top_k) combinations: {num_combos}")
    lines.append(str(agg_df))
    lines.append("")

    if "pct_correctness_true" in agg_df.columns:
        best_corr = agg_df.sort_values("pct_correctness_true", ascending=False).head(5)
        lines.append("Top 5 combinations by % Correctness TRUE:")
        for _, row in best_corr.iterrows():
            lines.append(
                f"  - {row.get('approach', '?')} | {row.get('model', '?')} "
                f"| top_k={row.get('top_k', '?')} -> "
                f"{row['pct_correctness_true']:.2f}% correctness"
            )
        lines.append("")

    if "pct_helpfulness_true" in agg_df.columns:
        best_help = agg_df.sort_values("pct_helpfulness_true", ascending=False).head(5)
        lines.append("Top 5 combinations by % Helpfulness TRUE:")
        for _, row in best_help.iterrows():
            lines.append(
                f"  - {row.get('approach', '?')} | {row.get('model', '?')} "
                f"| top_k={row.get('top_k', '?')} -> "
                f"{row['pct_helpfulness_true']:.2f}% helpfulness"
            )
        lines.append("")

    if "combined_score" in agg_df.columns:
        best_combined = agg_df.sort_values("combined_score", ascending=False).head(5)
        lines.append("Top 5 combinations by combined score:")
        for _, row in best_combined.iterrows():
            lines.append(
                f"  - rank {int(row['rank'])}: {row.get('approach', '?')} | "
                f"{row.get('model', '?')} | top_k={row.get('top_k', '?')} -> "
                f"combined_score={row['combined_score']:.4f}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def _write_text_summary_mod(
    df_raw: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write a plain-text high level summary (NO column truncation)."""
    path = output_dir / "summary.txt"

    num_rows, num_cols = df_raw.shape
    num_combos = len(agg_df)

    # --- Disable truncation for the DataFrame string ---
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        agg_as_text = agg_df.to_string(index=False)
    
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        ranked_help = agg_df.sort_values("score_helpfulness", ascending=False).assign(help_rank=lambda d: range(1, len(d) + 1))
        agg_as_text_help = ranked_help.to_string(index=False)
        # agg_as_text_help = agg_df.to_string(index=False)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        ranked_help = agg_df.sort_values("score_correctness", ascending=False).assign(help_rank=lambda d: range(1, len(d) + 1))
        agg_as_text_correct = ranked_help.to_string(index=False)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        ranked_help = agg_df.sort_values("avg_latency_sec", ascending=True).assign(help_rank=lambda d: range(1, len(d) + 1))
        agg_as_text_latency = ranked_help.to_string(index=False)

    lines = []
    lines.append("Model Evaluation Summary")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Raw rows: {num_rows}")
    lines.append(f"Raw columns: {num_cols}")
    lines.append(f"Unique (approach, model, top_k) combinations: {num_combos}")
    lines.append("")
    lines.append("Full Aggregated Table:")
    lines.append(agg_as_text)
    lines.append("=" * 80)
    lines.append("Helpfulness Ranked Table:")
    lines.append(agg_as_text_help)
    lines.append("=" * 80)
    lines.append("Correctness Ranked Table:")
    lines.append(agg_as_text_correct)
    lines.append("=" * 80)
    lines.append("Latency Ranked Table:")
    lines.append(agg_as_text_latency)
    lines.append("=" * 80)
    lines.append("")

    if "pct_correctness_true" in agg_df.columns:
        # best_corr = agg_df.sort_values("pct_correctness_true", ascending=False).head(5)
        best_corr = agg_df.sort_values("pct_correctness_true", ascending=False)
        lines.append("Top combinations by % Correctness TRUE:")
        for _, row in best_corr.iterrows():
            lines.append(
                f"  - {row.get('approach', '?')} | {row.get('model', '?')} "
                f"| top_k={row.get('top_k', '?')} -> "
                f"{row['pct_correctness_true']:.2f}% correctness"
            )
        lines.append("")

    if "pct_helpfulness_true" in agg_df.columns:
        # best_help = agg_df.sort_values("pct_helpfulness_true", ascending=False).head(5)
        best_help = agg_df.sort_values("pct_helpfulness_true", ascending=False)
        lines.append("Top combinations by % Helpfulness TRUE:")
        for _, row in best_help.iterrows():
            lines.append(
                f"  - {row.get('approach', '?')} | {row.get('model', '?')} "
                f"| top_k={row.get('top_k', '?')} -> "
                f"{row['pct_helpfulness_true']:.2f}% helpfulness"
            )
        lines.append("")

    if "avg_latency_sec" in agg_df.columns:
        # best_help = agg_df.sort_values("pct_helpfulness_true", ascending=False).head(5)
        best_help = agg_df.sort_values("avg_latency_sec", ascending=True)
        lines.append("Top combinations by avg_latency_sec:")
        for _, row in best_help.iterrows():
            lines.append(
                f"  - {row.get('approach', '?')} | {row.get('model', '?')} "
                f"| top_k={row.get('top_k', '?')} -> "
                f"{row['avg_latency_sec']:.2f} secs latency"
            )
        lines.append("")

    if "combined_score" in agg_df.columns:
        best_combined = agg_df.sort_values("combined_score", ascending=False).head(5)
        lines.append("Top 5 combinations by combined score:")
        for _, row in best_combined.iterrows():
            lines.append(
                f"  - rank {int(row['rank'])}: {row.get('approach', '?')} | "
                f"{row.get('model', '?')} | top_k={row.get('top_k', '?')} -> "
                f"combined_score={row['combined_score']:.4f}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def _write_full_report_markdown(
    df_raw: pd.DataFrame,
    agg_df: pd.DataFrame,
    output_dir: Path,
    artifacts: Dict[str, Path],
) -> Path:
    """Write a single markdown document summarizing everything."""
    path = output_dir / "full_report.md"

    num_rows, num_cols = df_raw.shape
    num_combos = len(agg_df)

    lines: list[str] = []
    lines.append("# Model Evaluation Report")
    lines.append("")
    lines.append(
        f"_Generated: {datetime.datetime.now().isoformat(timespec='seconds')}_"
    )
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- Raw rows: **{num_rows}**")
    lines.append(f"- Raw columns: **{num_cols}**")
    lines.append(f"- Unique (approach, model, top_k) combinations: **{num_combos}**")
    lines.append("")

    # Overall aggregate snapshot
    lines.append("## Aggregated Metrics (Top 20 by Combined Score)")
    lines.append("")
    if not agg_df.empty:
        if "combined_score" in agg_df.columns:
            top = agg_df.sort_values("combined_score", ascending=False).head(20)
        else:
            top = agg_df.head(20)
        cols = [
            c
            for c in [
                "rank",
                "approach",
                "model",
                "top_k",
                "pct_correctness_true",
                "pct_helpfulness_true",
                "avg_price_usd",
                "avg_latency_sec",
                "avg_meta_total_tokens",
                "avg_cosine",
                "avg_rougeL",
                "avg_bleu",
                "combined_score",
            ]
            if c in top.columns
        ]
        lines.append(_df_to_markdown_table(top[cols]))
    else:
        lines.append("_(no aggregated data)_")
    lines.append("")

    # Best combinations by individual metrics
    def add_metric_section(col: str, title: str):
        if col not in agg_df.columns:
            return
        lines.append(f"## {title}")
        lines.append("")
        best = agg_df.sort_values(col, ascending=False).head(10)
        cols_local = [
            c
            for c in [
                "rank",
                "approach",
                "model",
                "top_k",
                col,
                "pct_correctness_true",
                "pct_helpfulness_true",
                "avg_price_usd",
                "avg_latency_sec",
            ]
            if c in best.columns
        ]
        lines.append(_df_to_markdown_table(best[cols_local]))
        lines.append("")

    add_metric_section("pct_correctness_true", "Top Combinations by Correctness")
    add_metric_section("pct_helpfulness_true", "Top Combinations by Helpfulness")
    add_metric_section("combined_score", "Top Combinations by Combined Score")

    # Artifacts section
    if artifacts:
        lines.append("## Figures")
        lines.append("")
        for name, p in artifacts.items():
            rel_name = p.name
            lines.append(f"- **{name}** – `{rel_name}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_csv(
    csv_input: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path] = "results",
) -> None:
    """
    Analyze a CSV (or DataFrame) and write results to ``output_dir``.

    Parameters
    ----------
    csv_input:
        - Path to a CSV file, or
        - A pandas DataFrame with the expected columns.
    output_dir:
        Directory where all artifacts (CSVs, images, reports) will be written.

    Returns
    -------
    AnalysisResults
        Object containing the raw dataframe, aggregated dataframe,
        output directory, and paths to the main report files.
    """
    out_dir = _ensure_output_dir(output_dir)

    # Load CSV if needed
    if isinstance(csv_input, (str, Path)):
        df_raw = pd.read_csv(csv_input)
    else:
        df_raw = csv_input.copy()

    df_prepared = _prepare_dataframe(df_raw)
    agg_df = _aggregate_metrics(df_prepared)
    agg_scored = _add_scores(agg_df)

    # Save aggregated data
    agg_path = out_dir / "aggregated_metrics.csv"
    agg_scored.to_csv(agg_path, index=False)

    # Generate figures
    artifacts = _generate_figures(agg_scored, out_dir)

    # Text summary & full report
    # summary_path = _write_text_summary(df_prepared, agg_scored, out_dir)
    summary_path = _write_text_summary_mod(df_prepared, agg_scored, out_dir)
    # full_report_path = _write_full_report_markdown(
    #     df_prepared, agg_scored, out_dir, artifacts
    # )

    # Collect all artifact paths
    artifacts["aggregated_metrics"] = agg_path
    artifacts["summary_txt"] = summary_path
    # artifacts["full_report_md"] = full_report_path

    # return AnalysisResults(
    #     df_raw=df_prepared,
    #     df_agg=agg_scored,
    #     output_dir=out_dir,
    #     summary_text_path=summary_path,
    #     # full_report_path=full_report_path,
    #     artifacts=artifacts,
    # )
