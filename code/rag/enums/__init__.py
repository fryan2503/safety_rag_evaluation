"""
Simple import forwarder for enums.

This allows:
    from enums import Approaches, LLM
instead of:
    from enums.approaches import Approaches
    from enums.llm_model import LLM
"""

from .approaches import Approaches
from .llm_model import LLM