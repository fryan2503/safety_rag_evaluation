from pathlib import Path

from metrics_calculator import LangfairMetricsCalculator
from csv_processor import CSVProcessor
from langfair_runner import LangfairRunner

if __name__ == "__main__":
    async def main():        
        runner = LangfairRunner(
            calculator=LangfairMetricsCalculator(),
            processor=CSVProcessor(),
            max_concurrent=500,
        )
        return await runner.run(q_a_csv=Path("results/gold_set_part_1.csv"), out_csv=Path("results/gold_set_part_1_with_metrics.csv"))
    main()