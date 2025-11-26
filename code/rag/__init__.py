import dotenv
import nltk

from .rag_generation import RAGExperimentRunner
from.enums.approaches import Approaches
from .enums.llm_model import LLM

from .math import CSVProcessor
from .math import LangfairRunner
from .math import LangfairMetricsCalculator

dotenv.load_dotenv(override=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")