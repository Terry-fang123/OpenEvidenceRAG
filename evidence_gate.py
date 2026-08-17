from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any


LLM_REFUSAL_MARKER = "[[EVIDENCE_GATE_REFUSE]]"

TYPE_LABELS = {
    "guideline": "临床指南",
    "systematic_review": "系统综述",
    "meta_analysis": "Meta分析",
    "randomized_controlled_trial": "随机对照试验",
    "cohort_study": "队列研究",
    "case_control_study": "病例对照研究",
    "expert_consensus": "专家共识",
    "narrative_review": "叙述性综述",
    "clinical_trial_registry": "临床试验注册",
    "unknown": "类型未知",
}


@dataclass
class GateDecision:
    should_refuse: bool
    reason_code: str = ""
    judgment: str = ""
    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    safety_note: str = (
        "本结果仅用于证据检索与研究辅助，"
        "不能替代医生完成个体诊断、开具处方"
        "或调整药物剂量。"
    )


def _combined_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title", "")),
        str(item.get("publication_type", "")),
        str(item.get("publication_types", "")),
        str(item.get("source_type", "")),
    ]

    metadata = item.get("metadata")

    if isinstance(metadata, dict):
        parts.extend([
            str(metadata.get("publication_type", "")),
            str(metadata.get("publication_types", "")),
        ])

    return " ".join(parts).lower()


def classify_evidence_type(
    item: dict[str, Any],
) -> str:
    text = _combined_text(item)

    rules = [
        (
            "meta_analysis",
            r"meta[- ]analysis|meta分析|荟萃分析",
        ),
        (
            "systematic_review",
            r"systematic review|系统综述",
        ),
        (
            "expert_consensus",
            r"expert consensus|consensus "
            r"(statement|recommendation)|专家共识",
        ),
        (
            "guideline",
            r"clinical practice guideline|"
            r"practice guideline|guidelines?|指南",
        ),
        (
            "randomized_controlled_trial",
            r"randomi[sz]ed controlled|"
            r"randomi[sz]ed trial|"
            r"\brct\b|随机对照试验",
        ),
        (
            "cohort_study",
            r"cohort study|队列研究",
        ),
        (
            "case_control_study",
            r"case[- ]control|病例对照",
        ),
        (
            "clinical_trial_registry",
            r"clinical[_ ]trial|clinicaltrials",
        ),
        (
            "narrative_review",
            r"\breview\b|综述",
        ),
    ]

    for evidence_type, pattern in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return evidence_type

    return "unknown"


def detect_requested_types(
    question: str,
) -> set[str]:
    requested: set[str] = set()

    rules = {
        "meta_analysis": (
            r"meta[- ]analysis|meta分析|荟萃分析"
        ),
        "systematic_review": (
            r"systematic review|系统综述"
        ),
        "randomized_controlled_trial": (
            r"\brct\b|randomi[sz]ed controlled|"
            r"随机对照|随机试验"
        ),
        "guideline": (
            r"(只用|只要|仅|限定|只看)"
            r".{0,4}指南"
            r"|指南.{0,4}(证据|推荐)"
            r"|^指南"  # 仅"指南"二字单独成查询
        ),
        "expert_consensus": (
            r"consensus|专家共识"
        ),
        "cohort_study": (
            r"cohort|队列研究"
        ),
    }

    for evidence_type, pattern in rules.items():
        if re.search(
            pattern,
            question,
            re.IGNORECASE,
        ):
            requested.add(evidence_type)

    return requested


def exceeds_clinical_boundary(
    question: str,
) -> bool:
    patterns = [
        (
            r"(帮我|替我|直接给我|请给我)"
            r".{0,12}"
            r"(诊断|确诊|开药|开处方|停药|"
            r"换药|调整剂量)"
        ),
        # 覆盖 "帮我开个降压药" 等"开X药"句式
        (
            r"(帮|替|给)我.{0,6}开.{0,6}药"
        ),
        (
            r"我.{0,10}(是不是|是否)"
            r".{0,6}(得了|患有)"
        ),
        (
            r"(我应该|我能不能|我是否可以)"
            r".{0,10}(停药|加量|减量|换药)"
        ),
        (
            r"(我该吃什么药|我应该吃什么药|"
            r"给我推荐具体药物)"
        ),
        (
            r"(服用|吃).{0,6}"
            r"(多少毫克|多大剂量|几片)"
        ),
    ]

    return any(
        re.search(pattern, question)
        for pattern in patterns
    )


def _identifier(item: dict[str, Any]) -> str:
    for key in (
        "pmid",
        "nct_id",
        "guideline_id",
        "source_id",
    ):
        value = str(item.get(key, "")).strip()

        if value:
            return value

    return "未提供编号"


def summarize_evidence(
    evidence_list: list[dict[str, Any]],
) -> list[str]:
    if not evidence_list:
        return [
            "没有证据通过当前相关性和质量门槛。"
        ]

    evidence_types = [
        classify_evidence_type(item)
        for item in evidence_list
    ]

    counts = Counter(evidence_types)

    type_summary = "、".join(
        f"{TYPE_LABELS[key]}{value}条"
        for key, value in counts.items()
    )

    lines = [
        f"共有{len(evidence_list)}条证据通过初步门槛。",
        f"当前证据类型：{type_summary}。",
    ]

    for item in evidence_list[:3]:
        title = str(
            item.get("title", "未提供标题")
        ).strip() or "未提供标题"

        lines.append(
            f"{_identifier(item)}：{title[:120]}"
        )

    return lines


def _type_search_suggestions(
    requested_types: set[str],
) -> list[str]:
    suggestions = []

    query_words = {
        "guideline": "疾病名称 + latest clinical guideline",
        "systematic_review": (
            "疾病名称 + 干预措施 + systematic review"
        ),
        "meta_analysis": (
            "疾病名称 + 干预措施 + meta-analysis"
        ),
        "randomized_controlled_trial": (
            "疾病名称 + 干预措施 + randomized "
            "controlled trial"
        ),
        "expert_consensus": (
            "疾病名称 + expert consensus"
        ),
        "cohort_study": (
            "疾病名称 + 结局指标 + cohort study"
        ),
    }

    for evidence_type in requested_types:
        suggestion = query_words.get(evidence_type)

        if suggestion:
            suggestions.append(suggestion)

    return suggestions


def evaluate_evidence(
    question: str,
    evidence_list: list[dict[str, Any]],
    min_reliable_evidence: int = 1,
) -> GateDecision:
    found = summarize_evidence(evidence_list)

    if exceeds_clinical_boundary(question):
        return GateDecision(
            should_refuse=True,
            reason_code="OUT_OF_CLINICAL_SCOPE",
            judgment=(
                "该问题要求进行个体诊断、开具处方"
                "或调整治疗，超出本系统的证据辅助范围。"
            ),
            found=found,
            missing=[
                "缺少完整病史、体格检查、检验结果"
                "以及医生面对面评估。",
                "仅凭当前问题和文献检索不能安全完成"
                "个体诊疗决策。",
            ],
            next_steps=[
                "携带现有检查结果咨询医生或药师。",
                "如需继续使用本系统，可改问疾病的一般"
                "治疗证据、指南建议或药物比较证据。",
            ],
        )

    evidence_types = {
        classify_evidence_type(item)
        for item in evidence_list
    }

    high_quality_types = {
        "guideline",
        "systematic_review",
        "meta_analysis",
    }

    has_single_high_quality = bool(
        evidence_types & high_quality_types
    )

    if (
        len(evidence_list) < min_reliable_evidence
        and not has_single_high_quality
    ):
        return GateDecision(
            should_refuse=True,
            reason_code="INSUFFICIENT_DIRECT_EVIDENCE",
            judgment=(
                "当前可靠证据数量不足，暂不生成"
                "确定性医学结论。"
            ),
            found=found,
            missing=[
                (
                    f"至少需要{min_reliable_evidence}条"
                    "直接相关证据，或1条高质量指南、"
                    "系统综述或Meta分析。"
                ),
                "当前证据不足以稳定支持问题中的"
                "核心结论。",
            ],
            next_steps=[
                "补充更具体的人群、干预、对照和"
                "结局信息后重新检索。",
                "优先检索最新指南、系统综述、"
                "Meta分析或随机对照试验。",
            ],
        )

    requested_types = detect_requested_types(
        question
    )

    if (
        requested_types
        and not requested_types.intersection(
            evidence_types
        )
    ):
        missing_labels = "、".join(
            TYPE_LABELS[item]
            for item in sorted(requested_types)
        )

        return GateDecision(
            should_refuse=True,
            reason_code=(
                "MISSING_REQUIRED_EVIDENCE_TYPE"
            ),
            judgment=(
                "当前结果未命中用户指定的证据类型，"
                "暂不使用其他类型证据替代回答。"
            ),
            found=found,
            missing=[
                f"缺少用户要求的证据类型："
                f"{missing_labels}。",
                "现有证据不能替代所要求的研究设计。",
            ],
            next_steps=(
                _type_search_suggestions(
                    requested_types
                )
                or [
                    "使用疾病、干预和目标证据类型"
                    "重新组合检索式。"
                ]
            ),
        )

    return GateDecision(
        should_refuse=False,
    )


def build_structured_refusal(
    decision: GateDecision,
) -> str:
    if not decision.should_refuse:
        return ""

    def section(
        title: str,
        lines: list[str],
    ) -> str:
        content = "\n".join(
            f"- {line}"
            for line in lines
        )

        return f"## {title}\n\n{content}"

    judgment_lines = [
        decision.judgment,
        f"拒答原因代码：{decision.reason_code}",
    ]

    parts = [
        section("当前判断", judgment_lines),
        section(
            "已检索到的证据",
            decision.found
            or ["没有可报告的候选证据。"],
        ),
        section(
            "证据缺口",
            decision.missing
            or ["当前证据无法满足回答要求。"],
        ),
        section(
            "建议下一步检索",
            decision.next_steps
            or ["补充问题信息后重新检索。"],
        ),
        section(
            "安全提示",
            [decision.safety_note],
        ),
    ]

    return "\n\n".join(parts)


def build_model_refusal(
    evidence_list: list[dict[str, Any]],
) -> str:
    decision = GateDecision(
        should_refuse=True,
        reason_code=(
            "INSUFFICIENT_DIRECT_EVIDENCE"
        ),
        judgment=(
            "大模型复核后认为候选证据没有直接"
            "支持问题核心结论，暂不生成确定性回答。"
        ),
        found=summarize_evidence(evidence_list),
        missing=[
            "候选文献可能只是在方法、背景或参考"
            "文献中提到相关关键词。",
            "缺少能够直接回答当前问题的研究结果。",
        ],
        next_steps=[
            "细化目标人群、干预、对照和结局。",
            "优先检索指南、系统综述、Meta分析"
            "或随机对照试验。",
        ],
    )

    return build_structured_refusal(decision)
