"""
≤3步检索—核验 Agent + Trace

三步流水线：
  Step 1: 混合检索（生成检索词 → PubMed/ClinicalTrials/Guidelines 并行检索 → 混合RAG）
  Step 2: 引用核验（元数据核验 → 主题相关性 → LLM 结论支撑度判定）
  Step 3: 回答或结构化拒答（正常回答 / 部分回答 / 拒答 + 建议补检）

对现有代码零侵入 — 内部调用 evidence_agent / rag.answer_question 等现有函数。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evidence_gate import (
    LLM_REFUSAL_MARKER,
    GateDecision,
    build_structured_refusal,
    evaluate_evidence,
    exceeds_clinical_boundary,
    summarize_evidence,
)
from main import evidence_agent
from query_builder import generate_pubmed_query
from vector_db import DEVICE, model


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CitationEntry:
    """单条引用核验细节。"""
    evidence_index: int
    pmid: str = ""
    source_id: str = ""
    title: str = ""
    metadata_valid: bool = False
    topic_relevant: bool = False
    claim_supported: bool = False
    is_background: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Step 2 核验报告。"""
    total_checked: int = 0
    valid_metadata: int = 0
    topic_relevant: int = 0
    claim_supported: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    per_citation: list[CitationEntry] = field(default_factory=list)

    def summarize(self) -> str:
        if self.total_checked == 0:
            return "无可核验证据。"
        notes: list[str] = []
        if self.valid_metadata < self.total_checked:
            notes.append(
                f"{self.total_checked - self.valid_metadata}篇元数据异常"
            )
        if self.topic_relevant < self.total_checked:
            notes.append(
                f"{self.total_checked - self.topic_relevant}篇与问题语义不相关"
            )
        if self.claim_supported < self.topic_relevant:
            notes.append(
                f"{self.topic_relevant - self.claim_supported}篇仅为背景/方法引用"
            )
        if notes:
            return (
                f"{self.total_checked}篇中"
                f"{self.claim_supported}篇直接支撑，"
                + "；".join(notes)
            )
        return (
            f"{self.total_checked}篇全部有效，"
            f"其中{self.claim_supported}篇直接支撑核心结论。"
        )


@dataclass
class StepTrace:
    """单步 Trace。"""
    step: int
    action: str
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    decision_note: str = ""
    elapsed_ms: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ChallengeTrace:
    """全链路 Trace。"""
    trace_id: str = ""
    question: str = ""
    steps: list[StepTrace] = field(default_factory=list)
    final_decision: str = ""  # answer | partial_answer | refuse | suggest_retry
    total_elapsed_ms: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "trace_id": self.trace_id,
                "question": self.question,
                "steps": [
                    {
                        "step": s.step,
                        "action": s.action,
                        **(s.input_summary if s.input_summary else {}),
                        **s.output_summary,
                        "elapsed_ms": s.elapsed_ms,
                        **((
                            {"decision_note": s.decision_note}
                        ) if s.decision_note else {}),
                    }
                    for s in self.steps
                ],
                "final_decision": self.final_decision,
                "total_elapsed_ms": self.total_elapsed_ms,
            },
            ensure_ascii=False,
            indent=2,
        )


def _safe_truncate(
    text: str,
    max_chars: int = 120,
) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


# ============================================================
# Step 1: 混合检索
# ============================================================

def step1_search(
    question: str,
    query_mode: str = "auto",
    max_results: int = 5,
    max_trials: int = 3,
    max_guidelines: int = 3,
    topk: int = 5,
    retrieval_mode: str = "hybrid",
    use_rerank: bool = False,
    history: str = "",
) -> tuple[str, list[dict], str, StepTrace]:
    """
    Step 1: 调用 evidence_agent() 完成混合检索。

    内部步骤：
      1. 生成英文 PubMed 检索词
      2. PubMed + ClinicalTrials + Guidelines 检索并写入 ChromaDB
      3. 混合检索 + Rerank → 最终证据列表

    返回: (answer, evidence_list, pubmed_query, step_trace)
    """
    started = time.perf_counter()
    log_buffer = io.StringIO()
    errors: list[str] = []

    # 1. 生成检索词
    pubmed_query = generate_pubmed_query(
        question=question,
        mode=query_mode,
        history=history,
    ).strip()

    # 2. 调用 evidence_agent 完成检索并生成回答
    answer: str = ""
    evidence_list: list[dict] = []

    try:
        with contextlib.redirect_stdout(log_buffer):
            answer, evidence_list = evidence_agent(
                question=question,
                pubmed_query=pubmed_query,
                max_results=max_results,
                max_trials=max_trials,
                max_guidelines=max_guidelines,
                topk=topk,
                include_clinical_trials=True,
                include_guidelines=True,
                query_mode=query_mode,
                history=history,
                retrieval_mode=retrieval_mode,
                use_rerank=use_rerank,
            )
    except Exception as exc:
        errors.append(str(exc))

    if evidence_list is None:
        evidence_list = []

    # 从 evidence_agent 输出中解析原始 API 返回数量
    raw_logs = log_buffer.getvalue()

    pubmed_match = re.search(
        r"PubMed 返回的 PMID：\[(.*?)\]",
        raw_logs,
    )
    initial_pmids: list[str] = []

    if pubmed_match:
        initial_pmids = [
            pid.strip().strip("'\"")
            for pid in pubmed_match.group(1).split(",")
            if pid.strip()
        ]

    trials_match = re.search(
        r"成功找到 (\d+) 项临床试验",
        raw_logs,
    )
    initial_trials = int(trials_match.group(1)) if trials_match else 0

    guidelines_match = re.search(
        r"成功找到 (\d+) 条指南/共识证据",
        raw_logs,
    )
    initial_guidelines = int(guidelines_match.group(1)) if guidelines_match else 0

    initial_total = len(initial_pmids) + initial_trials + initial_guidelines

    # 提取 RAG 最终选中的 PMID 列表
    pmids: list[str] = []

    for ev in evidence_list:
        metadata = ev.get("metadata", {})

        pmid = (
            metadata.get("pmid", "") or ev.get("pmid", "")
        )

        if pmid:
            pmids.append(str(pmid))

    # 被过滤掉的 PMID
    filtered_pmids = [
        p for p in initial_pmids
        if p not in pmids
    ]

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = StepTrace(
        step=1,
        action="search",
        output_summary={
            "query": _safe_truncate(pubmed_query, 200),
            "initial_count": initial_total,
            "initial_pubmed": len(initial_pmids),
            "initial_trials": initial_trials,
            "initial_guidelines": initial_guidelines,
            "initial_pmids": initial_pmids[:20],
            "candidate_count": len(evidence_list),
            "selected_count": len(evidence_list),
            "selected_pmids": pmids[:10],
            "filtered_pmids": filtered_pmids[:10],
        },
        elapsed_ms=elapsed_ms,
        errors=errors,
    )

    return answer, evidence_list, pubmed_query, trace


# ============================================================
# Step 2: 引用核验 Agent
# ============================================================

def _build_document_block(
    evidence: dict,
    index: int,
) -> str:
    """构建单条证据的文本块，供 LLM 核验。"""
    metadata = evidence.get("metadata", {})

    title = metadata.get("title", evidence.get("title", "")) or "未知标题"
    source_id = (
        metadata.get("pmid", "")
        or metadata.get("nct_id", "")
        or metadata.get("guideline_id", "")
        or metadata.get("pdf_id", "")
    )
    doc_text = evidence.get("document", "")
    snippet = _safe_truncate(str(doc_text), 500)

    return (
        f"[Doc {index + 1}]\n"
        f"ID: {source_id}\n"
        f"Title: {_safe_truncate(title, 150)}\n"
        f"Text: {snippet}\n"
    )


def _parse_llm_verification(
    raw_text: str,
    count: int,
) -> list[dict]:
    """解析 LLM 返回的核验 JSON。"""
    raw_text = raw_text.strip()

    # 摘出第一个 JSON 数组
    match = re.search(
        r"\[[\s\S]*\]",
        raw_text,
    )

    if match:
        raw_text = match.group(0)

    try:
        parsed = json.loads(raw_text)

        if isinstance(parsed, list):
            return [
                {
                    "topic_match": bool(e.get("topic_match", True)),
                    "has_quantitative": bool(e.get("has_quantitative", False)),
                    "is_background": bool(e.get("is_background", False)),
                }
                for e in parsed[:count]
            ]
    except json.JSONDecodeError:
        print(f"[verify_citation] JSON 解析失败，原始输出: {raw_text[:200]}")

    # 兜底
    return [
        {
            "topic_match": True,
            "has_quantitative": False,
            "is_background": False,
        }
        for _ in range(count)
    ]


def step2_verify(
    question: str,
    evidence_list: list[dict],
) -> VerificationReport:
    """Step 2: 三层引用核验。"""
    report = VerificationReport()
    report.total_checked = len(evidence_list)

    if not evidence_list:
        return report

    # ---- 层次1: 元数据核验（规则） ----
    for idx, ev in enumerate(evidence_list):
        metadata = ev.get("metadata", {})
        entry = CitationEntry(
            evidence_index=idx + 1,
            pmid=str(metadata.get("pmid", ev.get("pmid", "")) or ""),
            source_id=str(
                metadata.get("pmid", "")
                or metadata.get("nct_id", "")
                or metadata.get("guideline_id", "")
                or ""
            ),
            title=str(metadata.get("title", ev.get("title", "")) or ""),
        )

        entry.metadata_valid = bool(
            entry.source_id
            and entry.source_id.lower() not in {"unknown", "未提供编号", ""}
        )

        if not entry.metadata_valid:
            entry.issues.append("来源编号缺失或无效")
        elif not entry.title:
            entry.issues.append("标题缺失")

        report.valid_metadata += int(entry.metadata_valid)
        report.per_citation.append(entry)

    # ---- 层次2: 主题相关性（向量比对） ----
    try:
        q_emb = model.encode(
            question,
            normalize_embeddings=True,
        )

        for idx, ev in enumerate(evidence_list):
            doc_text = ev.get("document", "")
            d_emb = model.encode(
                doc_text[:2000],
                normalize_embeddings=True,
            )
            sim = float(np.dot(q_emb, d_emb))

            if sim >= 0.40:
                report.topic_relevant += 1
                report.per_citation[idx].topic_relevant = True
            else:
                report.per_citation[idx].issues.append(
                    f"语义相关度低 (sim={sim:.2f})"
                )
    except Exception as exc:
        print(f"[verify_citation] 向量比对失败: {exc}")
        # 失败时全部标记为相关
        report.topic_relevant = len(evidence_list)

        for idx in range(len(evidence_list)):
            report.per_citation[idx].topic_relevant = True

    # ---- 层次3: LLM 结论支撑度判定 ----
    doc_blocks = "".join(
        _build_document_block(ev, idx)
        for idx, ev in enumerate(evidence_list)
    )

    prompt = f"""你是一个临床证据核验员。请在以下文献中逐一判断每条证据是否直接支撑问题。

问题：{question}

对每条证据回答 3 个问题：
1. topic_match: 是否直接研究问题中的疾病、药物、结局？(true/false)
2. has_quantitative: 是否包含可直接回答问题的具体数据（百分比、效应量等）？(true/false)
3. is_background: 该片段主要是背景介绍/方法描述而非研究结果？(true/false)

只返回 JSON 数组，不包含任何其他文字：
[{{"topic_match":true,"has_quantitative":true,"is_background":false}}, ...]

文献：
{doc_blocks}"""

    try:
        from rag import call_openai

        llm_raw, _ = call_openai(prompt, thinking=False)
        verification = _parse_llm_verification(
            llm_raw,
            len(evidence_list),
        )

        for idx, v in enumerate(verification):
            if idx >= len(report.per_citation):
                break

            entry = report.per_citation[idx]
            entry.claim_supported = (
                v.get("topic_match", True)
                and not v.get("is_background", True)
            )

            if v.get("is_background"):
                entry.is_background = True
                entry.issues.append("原文内容主要为背景/方法描述")

            if not v.get("topic_match", True):
                entry.issues.append(
                    "文献主题与用户问题不直接匹配"
                )

            report.claim_supported += int(entry.claim_supported)

    except Exception as exc:
        print(f"[verify_citation] LLM 核验失败: {exc}")
        # 降级：将所有 topic_relevant 的文章标为已支撑
        for idx, entry in enumerate(report.per_citation):
            if entry.topic_relevant:
                entry.claim_supported = True
                report.claim_supported += 1

    # 汇总 unsupported_claims
    for entry in report.per_citation:
        if not entry.claim_supported and entry.topic_relevant:
            report.unsupported_claims.append(
                f"[{entry.evidence_index}] {_safe_truncate(entry.title, 50)}"
                f" — {'; '.join(entry.issues)}"
            )

    return report


# ============================================================
# Step 3: 回答或结构化拒答
# ============================================================

def _extract_citation_numbers(
    text: str,
) -> list[int]:
    """从回答文本中提取 [N] 形式的引用编号。"""
    matches = re.findall(
        r"(?<!\\|\w)\[\s*(\d+)\s*\]",
        text,
    )

    unique: list[int] = []

    for num in matches:
        n = int(num)

        if n not in unique:
            unique.append(n)

    return unique


def step3_decide(
    question: str,
    step1_answer: str,
    evidence_list: list[dict],
    verification: VerificationReport,
) -> tuple[str, list[dict], StepTrace]:
    """
    Step 3: 根据核验结果决定使用/拒绝 step1 生成的回答。

    逻辑：
      - verified >= 3 → answer（直接使用 step1 回答）
      - verified >= 1 → partial_answer（使用 step1 回答，但标记为部分）
      - verified == 0 → refuse + 结构化拒答

    step1_answer: 由 evidence_agent() 在 step1 中已经生成好的完整回答。
    不再重复调用 LLM，避免二次生成时缺少 local_pdf_query 导致质量下降。

    返回: (answer_text, evidence_list, step_trace)
    """
    started = time.perf_counter()
    verified = verification.claim_supported
    errors: list[str] = []

    if verified >= 3:
        decision = "answer"
        answer = step1_answer
        final_evidence = evidence_list

    elif verified >= 1:
        decision = "partial_answer"
        # 在回答前追加证据不完整声明
        answer = (
            "> 注意：当前检索到的证据可能不完整，"
            "以下回答基于核验后可直接支撑的部分证据。\n\n"
        ) + (step1_answer or "")
        final_evidence = evidence_list

    else:
        decision = "refuse"
        extra_note = (
            "当前未检索到可直接支撑问题的证据。\n\n"
            + verification.summarize()
        )

        if evidence_list:
            extra_note += (
                "\n\n部分关键词命中以下文献，但内容与问题无直接因果关系："
            )

            for entry in verification.per_citation[:3]:
                extra_note += (
                    f"\n- [{entry.evidence_index}] "
                    f"{_safe_truncate(entry.title, 60)}"
                )

        answer = build_structured_refusal(
            GateDecision(
                should_refuse=True,
                reason_code="INSUFFICIENT_DIRECT_EVIDENCE",
                judgment=extra_note,
                found=summarize_evidence(evidence_list),
                missing=[
                    f"{verification.total_checked}篇候选文献均无法直接支撑该问题的核心结论。",
                    "可能需要调整检索词或放宽检索条件。",
                ],
                next_steps=[
                    "使用更短、更通用的英文关键词重新检索。",
                    "检查检索词是否包含疾病、药物、结局的完整 PICO 要素。",
                    "尝试不限定证据类型做宽泛检索后再人工筛选。",
                ],
            ),
        )
        final_evidence = []

    # 引用编号核验
    if final_evidence is None:
        final_evidence = []

    citations = _extract_citation_numbers(answer)
    max_idx = len(final_evidence)
    valid = [c for c in citations if 1 <= c <= max_idx]
    invalid = [c for c in citations if c > max_idx or c < 1]

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    trace = StepTrace(
        step=3,
        action=decision,
        output_summary={
            "decision": decision,
            "reason": verification.summarize(),
            "citations_found": len(citations),
            "citations_valid": len(valid),
            "citations_invalid": len(invalid),
        },
        decision_note=verification.summarize(),
        elapsed_ms=elapsed_ms,
        errors=errors,
    )

    return answer, final_evidence, trace


# ============================================================
# 三步 Agent 控制器
# ============================================================

def run_3step_agent(
    question: str,
    query_mode: str = "auto",
    max_results: int = 5,
    max_trials: int = 3,
    max_guidelines: int = 3,
    topk: int = 5,
    retrieval_mode: str = "hybrid",
    use_rerank: bool = False,
    history: str = "",
) -> tuple[str, list[dict], ChallengeTrace]:
    """
    三步检索—核验 Agent 主入口。

    返回: (answer, evidence_list, full_trace)
    """
    trace = ChallengeTrace(
        trace_id=str(uuid.uuid4())[:12],
        question=question,
    )
    overall_start = time.perf_counter()

    # ---- Step 1: 混合检索 ----
    step1_answer, evidence_list, pubmed_query, step1_trace = step1_search(
        question=question,
        query_mode=query_mode,
        max_results=max_results,
        max_trials=max_trials,
        max_guidelines=max_guidelines,
        topk=topk,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
        history=history,
    )
    trace.steps.append(step1_trace)

    # ---- Step 2: 引用核验 ----
    step2_start = time.perf_counter()

    verification = step2_verify(
        question=question,
        evidence_list=evidence_list,
    )

    step2_elapsed = int((time.perf_counter() - step2_start) * 1000)

    step2_trace = StepTrace(
        step=2,
        action="verify_citation",
        output_summary={
            "checked": verification.total_checked,
            "valid_metadata": verification.valid_metadata,
            "topic_relevant": verification.topic_relevant,
            "direct_support": verification.claim_supported,
            "unsupported_claims": len(verification.unsupported_claims),
        },
        decision_note=verification.summarize(),
        elapsed_ms=step2_elapsed,
    )

    if verification.unsupported_claims:
        step2_trace.output_summary["problems"] = (
            verification.unsupported_claims[:5]
        )

    # 构建被拒绝的 PMID 列表（主题不匹配或元数据无效）
    rejected_items: list[str] = []

    for entry in verification.per_citation:
        if not entry.topic_relevant or not entry.metadata_valid:
            short_id = (
                entry.pmid
                or entry.source_id
                or f"[{entry.evidence_index}]"
            )
            reasons = "; ".join(entry.issues) if entry.issues else "未通过核验"
            rejected_items.append(f"{short_id}: {reasons}")

    if rejected_items:
        step2_trace.output_summary["rejected"] = rejected_items

    trace.steps.append(step2_trace)

    # ---- Step 3: 回答 / 部分回答 / 拒答 ----
    answer, final_evidence, step3_trace = step3_decide(
        question=question,
        step1_answer=step1_answer,
        evidence_list=evidence_list,
        verification=verification,
    )
    trace.steps.append(step3_trace)
    trace.final_decision = step3_trace.action

    trace.total_elapsed_ms = int(
        (time.perf_counter() - overall_start) * 1000
    )

    return answer, final_evidence, trace


# ============================================================
# 批量评估（复用 eval_runner 问题集）
# ============================================================

def run_challenge_eval(
    output_csv: str | None = None,
) -> list[dict]:
    """
    使用 eval_runner 的 EVAL_QUESTIONS 跑完整三步 Agent 并输出 CSV。

    返回: records 列表。
    """
    from eval_runner import EVAL_QUESTIONS

    records: list[dict] = []

    for item in EVAL_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        category = item["category"]
        expect_refuse = item.get("expect_refuse", False)

        print(f"\n{'=' * 60}")
        print(f"[Challenge Eval] {qid}: {question[:40]}...")
        print(f"{'=' * 60}")

        started = time.perf_counter()
        answer, evidence_list, trace = run_3step_agent(
            question=question,
            use_rerank=True,
        )
        elapsed = time.perf_counter() - started

        # 核验结果
        verification = None

        for s in trace.steps:
            if s.action == "verify_citation":
                verification = s

        record = {
            "题号": qid,
            "类别": category,
            "预期拒答": "是" if expect_refuse else "否",
            "实际决策": trace.final_decision,
            "是否拒答": "是" if trace.final_decision == "refuse" else "否",
            "证据数": len(evidence_list),
            "核验数": (
                verification.output_summary.get("checked", 0)
                if verification
                else 0
            ),
            "直接支撑数": (
                verification.output_summary.get("direct_support", 0)
                if verification
                else 0
            ),
            "问题数": (
                verification.output_summary.get("unsupported_claims", 0)
                if verification
                else 0
            ),
            "耗时(秒)": round(elapsed, 1),
            "Trace_ID": trace.trace_id,
        }
        records.append(record)

        print(f"  决策: {trace.final_decision}")
        print(f"  耗时: {elapsed:.1f}s")

    # 输出 CSV
    if output_csv and records:
        import csv
        from pathlib import Path

        path = Path(output_csv)

        fieldnames = list(records[0].keys())

        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"\n评估报告已保存: {path}")

    return records


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    # 演示：跑一个单题
    demo_question = "他汀类药物对 LDL-C 降低有哪些临床证据支持？"

    print("=" * 60)
    print("三步检索-核验 Agent 演示")
    print("=" * 60)
    print(f"问题：{demo_question}")

    answer, evidence_list, trace = run_3step_agent(
        question=demo_question,
        use_rerank=True,
    )

    print("\n" + "=" * 60)
    print("Trace JSON:")
    print("=" * 60)
    print(trace.to_json())

    print("\n" + "=" * 60)
    print("回答:")
    print("=" * 60)
    print(answer[:1500])
