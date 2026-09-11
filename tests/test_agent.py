"""回归测试：人工标注的条款评估集 + 核心模块单元测试。

运行方式（任选）：
    python -m pytest tests/ -v
    python tests/test_agent.py          # 无 pytest 时也可直接运行
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contract_review import ContractReviewAgent, build_index, split_clauses  # noqa: E402
from contract_review.law_kb import LAW_KB  # noqa: E402
from contract_review.rules import match_rules  # noqa: E402

SAMPLE_PATH = ROOT / "examples" / "sample_contract.txt"

# ---------------------------------------------------------------------------
# 人工标注评估集
#   条款序号（1-based）→ 人工标注的风险等级
#   标注依据：条款是否造成权责显著失衡 / 责任敞口不可控 / 合规缺失
# ---------------------------------------------------------------------------
GOLD_LABELS = {
    1: "低",   # 第一条 服务内容        范围约定笼统但无实质风险
    2: "中",   # 第二条 价款与支付      付款时间「另行协商」→ 回款不确定
    3: "中",   # 第三条 验收            「以甲方确认为准」→ 验收标准主观
    4: "高",   # 第四条 违约责任        违约金 5%/日 + 赔偿不受限制
    5: "高",   # 第五条 合同解除        甲方单方随时解除且无需补偿
    6: "中",   # 第六条 保密            保密义务仅约束乙方，权责不对等
    7: "中",   # 第七条 知识产权        成果权属单方归属，未约定许可范围
    8: "高",   # 第八条 数据与个人信息   个人信息处理仅「按国家规定」，缺合规要件
    9: "高",   # 第九条 不可抗力        不可抗力被扩大至商业风险 + 免责
    10: "中",  # 第十条 争议解决        管辖地为甲方所在地，维权成本偏高
}


def _load_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


def evaluate() -> dict:
    """在人工标注评估集上计算风险等级准确率与高风险召回率。"""
    agent = ContractReviewAgent(rule_weight=0.6)
    report = agent.run(_load_sample())
    clauses = report["clauses"]

    assert len(clauses) == len(GOLD_LABELS), (
        f"切分条款数 {len(clauses)} 与标注数 {len(GOLD_LABELS)} 不一致"
    )

    correct = 0
    high_total = high_hit = 0
    rows = []
    for c in clauses:
        gold = GOLD_LABELS[c["index"]]
        pred = c["level"]
        correct += int(gold == pred)
        if gold == "高":
            high_total += 1
            high_hit += int(pred == "高")
        rows.append((c["index"], gold, pred, c["rule_score"], c["semantic_score"], c["final_score"]))

    return {
        "rows": rows,
        "accuracy": correct / len(GOLD_LABELS),
        "high_recall": high_hit / high_total if high_total else 1.0,
        "n": len(GOLD_LABELS),
    }


# ---------------------------------------------------------------------------
# 评估集
# ---------------------------------------------------------------------------
def test_risk_level_accuracy_is_100pct():
    r = evaluate()
    print("\n条款 | 标注 | 预测 | 规则分 | 语义分 | 综合分")
    print("-" * 58)
    for idx, gold, pred, rs, ss, fs in r["rows"]:
        flag = "✓" if gold == pred else "✗"
        print(f" {idx:>2}  |  {gold} |  {pred} | {rs:.2f}   | {ss:.2f}   | {fs:.2f}  {flag}")
    print("-" * 58)
    print(f"风险等级准确率：{r['accuracy'] * 100:.1f}%  |  高风险召回率：{r['high_recall'] * 100:.1f}%")
    assert r["accuracy"] == 1.0, f"风险等级准确率未达 100%：{r['accuracy']:.2%}"


def test_high_risk_recall_is_100pct():
    r = evaluate()
    assert r["high_recall"] == 1.0, f"高风险召回率未达 100%：{r['high_recall']:.2%}"


def test_deterministic_output():
    """同一输入必须得到同一结果（可复现性）。"""
    a = evaluate()["rows"]
    b = evaluate()["rows"]
    assert a == b


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
def test_splitter_filters_title_lines():
    clauses = split_clauses(
        "技术服务采购合同（节选）\n\n第一条 服务内容\n甲方委托乙方提供服务，服务范围以附件所列清单为准。"
    )
    assert len(clauses) == 1
    assert clauses[0].startswith("第一条")


def test_splitter_two_level():
    text = (
        "第一条 服务范围\n"
        "甲方委托乙方提供系统开发与技术咨询服务，具体内容以双方确认的需求说明书为准。\n"
        "第二条 服务费用\n"
        "本合同服务费总额为人民币十万元，甲方应于验收合格后十个工作日内一次性支付。"
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2, f"期望切出 2 条，实际 {len(clauses)}：{clauses}"
    assert clauses[0].startswith("第一条")
    assert clauses[1].startswith("第二条")


def test_bm25_ranks_relevant_law_first():
    index = build_index(LAW_KB)
    hits = index.search("甲方有权单方面解除本合同，无需事先通知乙方", top_k=1)
    assert hits, "检索结果不应为空"
    assert "解除" in hits[0].text or "563" in hits[0].title


def test_rules_are_explainable():
    hits = match_rules("乙方逾期交付的，每逾期一日应按合同总价款的百分之五支付违约金。")
    names = {h.name for h in hits}
    assert "违约金过高" in names


def test_cache_reuses_result():
    agent = ContractReviewAgent(use_cache=True)
    clause = "甲方有权随时单方面解除本合同，无需事先通知乙方。"
    first = agent.review_clause(1, clause)
    second = agent.review_clause(2, clause)
    assert first.final_score == second.final_score
    assert len(agent._cache) == 1


def _main() -> int:
    failures = 0
    tests = [
        test_risk_level_accuracy_is_100pct,
        test_high_risk_recall_is_100pct,
        test_deterministic_output,
        test_splitter_filters_title_lines,
        test_splitter_two_level,
        test_bm25_ranks_relevant_law_first,
        test_rules_are_explainable,
        test_cache_reuses_result,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
