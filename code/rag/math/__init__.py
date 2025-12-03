"""
Initialization for Langfair evaluation module.

This mainly ensures NLTK tokenizers are installed
for token-based metrics such as BLEU and ROUGE.
"""

import nltk
from csv_processor import CSVProcessor
from langfair_runner import LangfairRunner
from metrics_calculator import LangfairMetricsCalculator

# Ensure tokenizer downloads
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)