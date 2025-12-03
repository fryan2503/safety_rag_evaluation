"""
Package initialization for the RAG experiment framework.

This module:
 Ensures environment variables are loaded from .env files
 Ensures necessary NLTK resources are available
 Imports the key experiment-running components
"""

import dotenv
import nltk

from .rag_generation import RAGExperimentRunner
from.enums.approaches import Approaches
from .enums.llm_model import LLM

from .math import CSVProcessor
from .math import LangfairRunner
from .math import LangfairMetricsCalculator

from .approach_retrievers import ApproachRetrievers

dotenv.load_dotenv(override=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")