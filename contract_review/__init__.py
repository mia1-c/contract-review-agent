"""合同 / 制度智能审查助手 —— 多步 Agent + 自研 BM25 + 规则融合。"""

from .agent import (
    ClauseReview,
    ContractReviewAgent,
    LLMScorer,
    to_json,
    to_markdown,
)
from .bm25 import BM25Index, Hit, build_index
from .law_kb import LAW_KB
from .rules import RULES, Rule, match_rules
from .splitter import split_clauses

__version__ = "0.1.0"

__all__ = [
    "ClauseReview",
    "ContractReviewAgent",
    "LLMScorer",
    "to_json",
    "to_markdown",
    "BM25Index",
    "Hit",
    "build_index",
    "LAW_KB",
    "RULES",
    "Rule",
    "match_rules",
    "split_clauses",
]
