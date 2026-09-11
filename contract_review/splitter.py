"""Plan 阶段：正则两级条款切分。"""
from __future__ import annotations

import re
from typing import List

# 一级：章/条/中文序号/数字序号/【标题】等结构标记
L1_SPLIT = re.compile(
    r"(?=(?:第[一二三四五六七八九十百零\d]+条|第[一二三四五六七八九十\d]+章|"
    r"[一二三四五六七八九十]+、|\d+[\.、]\s|【[^】]{2,20}】))"
)

# 二级：句末标点
L2_SPLIT = re.compile(r"(?<=[。；;])\s*\n?")

# 认定为「纯标题」的特征：无句读且足够短
_SENT_PUNCT = re.compile(r"[，。；：、,;:]")
_TITLE_MAX = 30
_MIN_LEN = 10
_MERGE_THRESHOLD = 140


def split_clauses(text: str, merge_threshold: int = _MERGE_THRESHOLD) -> List[str]:
    """两级切分：先按结构标记切块，块内再按句子切分并合并过短片段。

    Args:
        text: 合同/制度正文。
        merge_threshold: 二级切分后，相邻片段合计短于该长度则合并，
            避免「第X条 标题」与后续正文被拆散。

    Returns:
        条款列表（已过滤纯标题行）。
    """
    normalized = (text or "").replace("\r", "").strip()
    if not normalized:
        return []

    blocks = [b.strip() for b in L1_SPLIT.split(normalized) if b.strip()]

    clauses: List[str] = []
    for block in blocks:
        parts = [p.strip() for p in L2_SPLIT.split(block) if p.strip()]
        merged: List[str] = []
        for part in parts:
            if merged and len(merged[-1]) + len(part) < merge_threshold:
                merged[-1] = merged[-1] + part
            else:
                merged.append(part)
        clauses.extend(merged)

    return [c for c in clauses if _keep(c)]


def _keep(clause: str) -> bool:
    compact = re.sub(r"\s", "", clause)
    if len(compact) < _MIN_LEN:
        return False
    # 过滤纯标题行，如「技术服务采购合同（节选）」
    is_title = not _SENT_PUNCT.search(clause) and len(compact) <= _TITLE_MAX
    return not is_title
