#!/usr/bin/env python3
"""命令行入口：审查一份合同 / 制度文本。

用法：
    # 审查内置示例
    python main.py --sample

    # 审查文件
    python main.py --file examples/sample_contract.txt

    # 输出 JSON / Markdown
    python main.py --file contract.txt --format json --out report.json
    python main.py --file contract.txt --format markdown --out report.md

    # 调整规则分与语义分的融合权重
    python main.py --sample --rule-weight 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contract_review import ContractReviewAgent, to_json, to_markdown

SAMPLE = """技术服务采购合同（节选）

第一条 服务内容
乙方根据甲方要求提供系统开发与技术支持服务，具体范围以甲方另行发出的任务单为准。

第二条 合同价款与支付
本合同总价款为人民币 580,000 元。甲方在收到乙方发票后，根据项目实际情况分期支付，具体付款时间由双方另行协商确定。

第三条 验收
乙方交付成果后，以甲方确认为准。甲方确认通过后视为验收合格。

第四条 违约责任
乙方逾期交付的，每逾期一日应按合同总价款的百分之五向甲方支付违约金。因乙方原因给甲方造成的一切损失，由乙方全额赔偿，赔偿范围不受限制。

第五条 合同解除
甲方有权随时单方面解除本合同，无需事先通知乙方，乙方不得就此提出任何异议或主张任何补偿。

第六条 保密
乙方应对履行本合同过程中知悉的甲方商业信息承担保密义务，保密期限为五年。

第七条 知识产权
本项目产生的全部知识产权归甲方所有，乙方不得就相关成果申请任何专利或著作权登记。

第八条 数据与个人信息
乙方在服务过程中接触到的用户个人信息，应按照国家有关规定处理。

第九条 不可抗力
因不可抗力导致不能履约的，双方互不承担责任。不可抗力包括但不限于自然灾害、政策变化、市场行情波动及乙方资金周转困难。

第十条 争议解决
因本合同产生的争议，双方应友好协商解决；协商不成的，提交甲方所在地有管辖权的人民法院管辖。
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="合同 / 制度智能审查助手（多步 Agent + 自研 BM25 + 规则融合）"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="待审查文本文件（UTF-8）")
    src.add_argument("--sample", action="store_true", help="使用内置示例合同")
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    p.add_argument("--out", type=Path, help="结果输出路径，缺省打印到标准输出")
    p.add_argument("--rule-weight", type=float, default=0.6,
                   help="规则分权重（0~1），其余权重给语义分，默认 0.6")
    p.add_argument("--top-k", type=int, default=3, help="每条款召回法条数，默认 3")
    p.add_argument("--no-cache", action="store_true", help="关闭同条款结果缓存")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    text = SAMPLE if args.sample else args.file.read_text(encoding="utf-8")

    agent = ContractReviewAgent(
        rule_weight=args.rule_weight,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    report = agent.run(text)

    if args.format == "json":
        rendered = to_json(report)
    elif args.format == "markdown":
        rendered = to_markdown(report)
    else:
        rendered = _render_text(report)

    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
        print(f"已写入 {args.out}", file=sys.stderr)
    else:
        print(rendered)
    return 0


def _render_text(report: dict) -> str:
    s = report["summary"]
    out = []
    out.append("=" * 62)
    out.append("  合同 / 制度智能审查报告")
    out.append("=" * 62)
    out.append(
        f"条款 {s['total']} 条 ｜ 高风险 {s['high_risk']} ｜ "
        f"中风险 {s['mid_risk']} ｜ 低风险 {s['low_risk']}"
    )
    out.append("-" * 62)
    for c in report["clauses"]:
        out.append(f"[条款 {c['index']:02d}] {c['level']}风险  "
                   f"(规则 {c['rule_score']} / 语义 {c['semantic_score']} / 综合 {c['final_score']})")
        snippet = c["clause"].replace("\n", " ")
        out.append(f"   原文：{snippet[:110]}{'…' if len(snippet) > 110 else ''}")
        if c["hits"]:
            out.append(f"   规则：{'、'.join(c['hits'])}")
        for law in c["laws"]:
            out.append(f"   法条：{law['name']}（相似度 {law['score']}）")
        out.append("")
    if s["advice"]:
        out.append("-" * 62)
        out.append("修改建议汇总：")
        for a in s["advice"]:
            out.append(f"  · {a}")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
