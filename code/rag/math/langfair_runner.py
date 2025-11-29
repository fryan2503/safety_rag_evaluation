from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from metrics_calculator import LangfairMetricsCalculator
from csv_processor import CSVProcessor


class LangfairRunner:
    def __init__(
        self,
        *,
        calculator: LangfairMetricsCalculator,
        processor: CSVProcessor,
        max_concurrent: int = 8,
    ):
        self.calculator = calculator
        self.processor = processor
        self.max_concurrent = max_concurrent
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_row(self, generated: str, gold: str) -> Dict[str, Any]:
        def sync_task():
            metrics = self.calculator.compute(generated or "", gold or "")
            return {**metrics, "q": gold}

        return await self.loop.run_in_executor(self.executor, sync_task)

    async def run(self, *, q_a_csv: Path, out_csv: Optional[Path] = None) -> Path:
        df = self.processor.load(q_a_csv)
        out_csv = self.processor.determine_output_path(q_a_csv, out_csv)

        print(f"Total rows to process: {len(df)}")

        results: List[Dict[str, Any]] = []
        tasks: List[Any] = []

        for idx, row in df.iterrows():
            tasks.append(
                self.process_row(
                    row.get("generated_answer", ""),
                    row.get("gold_answer", "")
                )
            )

            if len(tasks) >= self.max_concurrent:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                tasks = []
                print(f"Processed {len(results)}/{len(df)} rows...")

        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        metrics_df = pd.DataFrame(results)
        merged_df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

        self.processor.save(merged_df, out_csv)
        return out_csv
