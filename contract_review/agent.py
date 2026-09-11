"""Agent 工作流编排：Plan -> Retrieve -> Review -> Summarize。

四步流水线中只有 Review 的「语义分」可以切换为大模型打分，
接口零改动——不接 LLM 时用规则分 + BM25 语义分即可离线跑通全流程。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Protocol, Sequence

from .bm25 import BM25Index, Hit, build_index
from .law_kb import LAW_KB
from .rules import Rule, match_rules
from .splitter import split_clauses

RULE_WEIGHT_DEFAULT = 0.6
SEMANTIC_GAIN = 1.2
LEVEL_HIGH = 0.75
LEVEL_MID = 0.42

# 最终等级由规则等级与综合分共同决定，避免「规则命中但分低被判低风险」
_LEVEL_ORDER = {"低": 0, "中": 1, "高": 2}


class LLMScorer(Protocol):
    """语义风险打分器接口。返回 0~1 的风险强度，或 None 表示不可用。"""

    def score(self, clause: str, laws: Sequence[Hit]) -> Optional[float]:
        ...


@dataclass
class ClauseReview:
    index: int
    clause: str
    hits: List[Rule]
    laws: List[Hit]
    rule_score: float
    semantic_score: float
    llm_score: Optional[float]
    final_score: float
    level: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hits"] = [h.name for h in self.hits]
        d["laws"] = [{"name": h.title, "score": round(h.score, 3)} for h in self.laws]
        return d


def _clause_key(clause: str) -> str:
    return hashlib.md5(re.sub(r"\s", "", clause).encode("utf-8")).hexdigest()


class ContractReviewAgent:
    """多步推理 + 规则融合的合同/制度审查 Agent。"""

    def __init__(
        self,
        law_kb: Optional[Sequence[dict]] = None,
        rule_weight: float = RULE_WEIGHT_DEFAULT,
        top_k: int = 3,
        llm: Optional[LLMScorer] = None,
        use_cache: bool = True,
    ) -> None:
        self.law_index: BM25Index = build_index(law_kb or LAW_KB)
        self.rule_weight = rule_weight
        self.top_k = top_k
        self.llm = llm
        self.use_cache = use_cache
        self._cache: Dict[str, ClauseReview] = {}

    # ---------- Step 1: Plan ----------
    def plan(self, text: str) -> List[str]:
        return split_clauses(text)

    # ---------- Step 2: Retrieve ----------
    def retrieve(self, clause: str) -> List[Hit]:
        return self.law_index.search(clause, top_k=self.top_k)

    # ---------- Step 3: Review ----------
    def review_clause(self, index: int, clause: str) -> ClauseReview:
        key = _clause_key(clause)
        if self.use_cache and key in self._cache:
            cached = self._cache[key]
            return ClauseReview(**{**cached.__dict__, "index": index})

        hits = match_rules(clause)
        # 规则分：显性风险强度 = 最高权重 + 其余命中项的叠加惩罚
        max_w = max((h.weight for h in hits), default=0.0)
        rule_score = min(1.0, max_w + 0.08 * max(0, len(hits) - 1))

        laws = self.retrieve(clause)
        # 语义分：BM25 命中的法条按相似度加权平均其风险暴露强度
        w_sum = sum(h.score for h in laws)
        semantic_score = (
            sum(h.score / w_sum * h.risk_bias for h in laws) if w_sum > 0 else 0.0
        )

        llm_score = None
        if self.llm is not None:
            llm_score = self.llm.score(clause, laws)
        # 有 LLM 分时，语义分取两者均值（规则分权重不变）
        if llm_score is not None:
            semantic_score = (semantic_score + float(llm_score)) / 2.0

        final = min(
            1.0,
            self.rule_weight * rule_score
            + (1 - self.rule_weight) * semantic_score * SEMANTIC_GAIN,
        )
        level = "高" if final >= LEVEL_HIGH else "中" if final >= LEVEL_MID else "低"
        # 规则明确判「高」时不低于「中」，避免显性重风险被稀释
        if hits and _LEVEL_ORDER[max(hits, key=lambda h: h.weight).level] > _LEVEL_ORDER[level]:
            level = max(hits, key=lambda h: h.weight).level

        result = ClauseReview(
            index=index,
            clause=clause,
            hits=hits,
            laws=laws,
            rule_score=round(rule_score, 3),
            semantic_score=round(semantic_score, 3),
            llm_score=llm_score,
            final_score=round(final, 3),
            level=level,
        )
        if self.use_cache:
            self._cache[key] = result
        return result

    def review(self, clauses: Sequence[str]) -> List[ClauseReview]:
        return [self.review_clause(i, c) for i, c in enumerate(clauses, start=1)]

    # ---------- Step 4: Summarize ----------
    @staticmethod
    def summarize(reviews: Sequence[ClauseReview]) -> dict:
        counts = {"高": 0, "中": 0, "低": 0}
        for r in reviews:
            counts[r.level] += 1
        return {
            "total": len(reviews),
            "high_risk": counts["高"],
            "mid_risk": counts["中"],
            "low_risk": counts["低"],
            "high_risk_clauses": [r.index for r in reviews if r.level == "高"],
            "advice": sorted(
                {h.advice for r in reviews for h in r.hits},
            ),
        }

    # ---------- 全流程 ----------
    def run(self, text: str) -> dict:
        clauses = self.plan(text)
        reviews = self.review(clauses)
        summary = self.summarize(reviews)
        return {
            "summary": summary,
            "clauses": [r.to_dict() for r in reviews],
        }


def to_markdown(report: dict) -> str:
    """把审查结果渲染成可读的 Markdown 报告。"""
    s = report["summary"]
    lines = [
        "# 合同 / 制度智能审查报告",
        "",
        f"- 条款总数：**{s['total']}**",
        f"- 高风险：**{s['high_risk']}** ｜ 中风险：**{s['mid_risk']}** ｜ 低风险：**{s['low_risk']}**",
        "",
        "---",
        "",
    ]
    for c in report["clauses"]:
        lines.append(f"## 条款 {c['index']:02d} · {c['level']}风险")
        lines.append("")
        lines.append(f"> {c['clause']}")
        lines.append("")
        lines.append(
            f"- 规则分 `{c['rule_score']}` ｜ 语义分 `{c['semantic_score']}` ｜ "
            f"综合 `{c['final_score']}`"
        )
        if c["hits"]:
            lines.append(f"- 命中规则：{'、'.join(c['hits'])}")
        else:
            lines.append("- 命中规则：无（未发现显性风险特征）")
        if c["laws"]:
            lines.append("- 相关法条：")
            for law in c["laws"]:
                lines.append(f"  - {law['name']}（相似度 {law['score']}）")
        lines.append("")

    if s["advice"]:
        lines.append("---")
        lines.append("")
        lines.append("## 修改建议汇总")
        lines.append("")
        for a in s["advice"]:
            lines.append(f"- {a}")
    lines.append("")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
