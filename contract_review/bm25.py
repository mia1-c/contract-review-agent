"""从零实现的 BM25 检索（中文按 unigram + bigram 切分）。

不依赖向量数据库，纯标准库实现，保证全流程可离线复现。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """中文检索友好切分：字符 unigram + 相邻字符 bigram。

    bigram 能捕捉「违约金」「单方解除」这类双字词，效果接近分词但无需词典。
    """
    clean = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
    tokens: List[str] = []
    for i, ch in enumerate(clean):
        tokens.append(ch)
        if i + 1 < len(clean):
            tokens.append(clean[i : i + 2])
    return tokens


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    score: float
    risk_bias: float


class BM25Index:
    """标准 BM25（k1=1.5, b=0.75）内存倒排索引。"""

    def __init__(self, docs: Sequence[dict], text_fn=None):
        self.docs = list(docs)
        self.text_fn = text_fn or (
            lambda d: f"{d.get('name','')} {d.get('text','')} {d.get('kw','')}"
        )
        self.corpus: List[List[str]] = [tokenize(self.text_fn(d)) for d in self.docs]
        self.n = len(self.corpus)
        self.avgdl = (sum(len(t) for t in self.corpus) / self.n) if self.n else 0.0

        self.tf: List[Counter] = [Counter(toks) for toks in self.corpus]
        self.df: Counter = Counter()
        for toks in self.corpus:
            for term in set(toks):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        # 加 1 平滑，避免负 idf
        return math.log(1 + (self.n - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Hit]:
        q_tokens = tokenize(query)
        scored = []
        for idx, tf in enumerate(self.tf):
            dl = len(self.corpus[idx])
            s = 0.0
            for term in q_tokens:
                f = tf.get(term)
                if not f:
                    continue
                s += self._idf(term) * (f * (K1 + 1)) / (
                    f + K1 * (1 - B + B * dl / (self.avgdl or 1))
                )
            if s > 0:
                scored.append((s, idx))
        scored.sort(key=lambda x: x[0], reverse=True)

        hits: List[Hit] = []
        for s, idx in scored[:top_k]:
            d = self.docs[idx]
            hits.append(
                Hit(
                    doc_id=str(d.get("id", idx)),
                    title=d.get("name", ""),
                    text=d.get("text", ""),
                    score=s,
                    risk_bias=float(d.get("risk_bias", 0.5)),
                )
            )
        return hits


def build_index(law_kb: Iterable[dict]) -> BM25Index:
    return BM25Index(law_kb)
