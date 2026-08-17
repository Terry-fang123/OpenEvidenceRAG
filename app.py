from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import streamlit as st

from challenge_agent import run_3step_agent
from main import DEFAULT_QUESTION, evidence_agent
from qa_pipeline import check_completeness, check_safety, check_evidence_relevance
from query_builder import generate_pubmed_query


SAMPLE_QUESTIONS = {
    "血脂异常": "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
    "高血压": "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    "限钠饮食": "限钠饮食对高血压是否真的有帮助？",
    "地中海饮食": "地中海饮食对心血管风险有什么证据？",
    "糖尿病": "糖尿病患者为什么需要关注 LDL-C？",
}


QUERY_MODE_LABELS = {
    "auto": "自动模式",
    "llm": "大模型生成检索词",
    "fallback": "本地兜底检索词",
}


SOURCE_GROUPS = {
    "pubmed": ("PubMed 文献", "公共医学文献数据库"),
    "clinical_trial": ("临床试验", "ClinicalTrials.gov 注册研究"),
    "guideline": ("指南/共识", "本地整理的临床指南和共识资料"),
    "unknown": ("其他证据", "未识别来源类型"),
}


@dataclass
class RunSettings:
    question: str
    query_mode: str
    pubmed_query: str
    max_pubmed: int
    max_trials: int
    max_guidelines: int
    topk: int
    include_trials: bool
    include_guidelines: bool
    quick_demo: bool
    multi_turn: bool
    retrieval_mode: str
    dense_weight: float
    use_rerank: bool
    use_3step_agent: bool = False


def setup_page() -> None:
    st.set_page_config(
        page_title="临床证据助手",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu {
            visibility: hidden;
            height: 0;
        }
        .app-header {
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.15rem;
            padding-bottom: 0.85rem;
        }
        .app-title {
            color: #111827;
            font-size: 2.25rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0 0 0.6rem 0;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 0.96rem;
            margin: 0;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }
        .chip {
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #374151;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1;
            padding: 0.42rem 0.58rem;
        }
        .section-label {
            color: #374151;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .compact-note {
            color: #6b7280;
            font-size: 0.86rem;
            margin-top: 0.35rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
        .source-line {
            color: #4b5563;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .evidence-title {
            font-weight: 650;
            margin-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">临床证据助手</div>
            <p class="app-subtitle">面向医学问题的文献、临床试验与指南证据整合。</p>
            <div class="chip-row">
                <span class="chip">PubMed</span>
                <span class="chip">ClinicalTrials.gov</span>
                <span class="chip">指南/共识</span>
                <span class="chip">RAG 回答</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> RunSettings:
    with st.sidebar:
        st.subheader("运行设置")

        sample_key = st.selectbox(
            "测试问题",
            list(SAMPLE_QUESTIONS.keys()),
            index=0,
        )

        if "question" not in st.session_state:
            st.session_state.question = DEFAULT_QUESTION

        if st.button(
            "载入测试问题",
            use_container_width=True,
        ):
            st.session_state.question = SAMPLE_QUESTIONS[
                sample_key
            ]

        quick_demo = st.toggle(
            "快速演示模式",
            value=False,
            help=(
                "减少检索数量，适合课堂展示或比赛答辩时快速跑通完整流程。"
            ),
        )

        query_mode = st.radio(
            "PubMed 检索词生成方式",
            options=list(QUERY_MODE_LABELS.keys()),
            index=0,
            horizontal=False,
            format_func=lambda value: QUERY_MODE_LABELS.get(
                value,
                value,
            ),
        )

        manual_query_enabled = st.toggle(
            "手动 PubMed query",
            value=False,
        )

        manual_query = ""

        if manual_query_enabled:
            manual_query = st.text_area(
                "英文 PubMed query",
                height=90,
                placeholder=(
                    "dyslipidemia lifestyle intervention "
                    "statin therapy cardiovascular risk"
                ),
            ).strip()

        if quick_demo:
            max_pubmed = 2
            max_trials = 1
            max_guidelines = 2
            topk = 4

            st.info(
                "快速演示：PubMed 2 篇、临床试验 1 条、指南/共识 2 条、最终证据 4 条。"
            )
        else:
            max_pubmed = st.slider(
                "PubMed 文献数",
                min_value=1,
                max_value=10,
                value=5,
            )

        include_trials = st.toggle(
            "ClinicalTrials.gov",
            value=True,
        )

        if not quick_demo:
            max_trials = st.slider(
                "临床试验数",
                min_value=0,
                max_value=10,
                value=3,
                disabled=not include_trials,
            )

        include_guidelines = st.toggle(
            "指南/共识资料",
            value=True,
        )

        if not quick_demo:
            max_guidelines = st.slider(
                "指南/共识条数",
                min_value=0,
                max_value=5,
                value=3,
                disabled=not include_guidelines,
            )

            topk = st.slider(
                "最终证据条数",
                min_value=1,
                max_value=12,
                value=5,
            )

        # ---- 多轮追问模式 ----
        st.divider()
        st.subheader("多轮追问")

        multi_turn = st.toggle(
            "多轮追问模式",
            value=False,
            help=(
                "开启后，系统会记住之前的问答，"
                "自动理解下一轮问题中的指代关系。"
            ),
        )

        # 初始化对话历史
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        if multi_turn and st.session_state.conversation_history:
            st.caption(
                f"已记录 {len(st.session_state.conversation_history)} 轮对话"
            )

        if st.button(
            "新对话",
            use_container_width=True,
            disabled=not multi_turn,
        ):
            st.session_state.conversation_history = []
            st.rerun()

        # ---- 检索模式 ----
        st.divider()
        st.subheader("检索模式")

        retrieval_mode = st.radio(
            "检索方式",
            options=["hybrid", "vector_only"],
            index=0,
            horizontal=False,
            format_func=lambda v: (
                "混合检索 (BM25+向量)"
                if v == "hybrid"
                else "纯向量检索"
            ),
            help=(
                "混合检索：关键词 + 语义双路融合，"
                "召回更全面；"
                "纯向量检索：仅语义匹配"
            ),
        )

        dense_weight = st.slider(
            "向量权重",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
            disabled=(retrieval_mode != "hybrid"),
            help=(
                "混合检索时向量路的权重。"
                "> 1.0 偏向语义，< 1.0 偏向关键词。"
                "仅混合检索模式有效。"
            ),
        )

        use_rerank = st.toggle(
            "Rerank 精排",
            value=False,
            help=(
                "开启后，LLM 会对候选文献逐篇打分（0-10），"
                "筛选出真正直接相关的文献。"
                "会额外调用一次 LLM，响应稍慢。"
            ),
        )

        # ---- 三步 Agent 模式 ----
        st.divider()
        st.subheader("评估模式")
        use_3step_agent = st.toggle(
            "三步Agent (检索-核验-回答)",
            value=False,
            help=(
                "用 ≤3 步检索—核验 Agent 替代默认六步流程。"
                "包含自动引用核验、结构化的回答/部分回答/拒答决策，"
                "并输出全链路 Trace。"
            ),
        )

    return RunSettings(
        question=st.session_state.question,
        query_mode=query_mode or "auto",
        pubmed_query=manual_query,
        max_pubmed=max_pubmed,
        max_trials=max_trials if include_trials else 0,
        max_guidelines=max_guidelines if include_guidelines else 0,
        topk=topk,
        include_trials=include_trials,
        include_guidelines=include_guidelines,
        quick_demo=quick_demo,
        multi_turn=multi_turn,
        retrieval_mode=retrieval_mode,
        dense_weight=dense_weight,
        use_rerank=use_rerank,
        use_3step_agent=use_3step_agent,
    )


def render_question_area(
    settings: RunSettings,
) -> bool:
    st.markdown(
        "<div class='section-label'>医学问题</div>",
        unsafe_allow_html=True,
    )

    st.session_state.question = st.text_area(
        "医学问题",
        value=st.session_state.get(
            "question",
            DEFAULT_QUESTION,
        ),
        height=96,
        label_visibility="collapsed",
    )

    run_clicked = st.button(
        "运行检索与回答",
        type="primary",
        use_container_width=True,
    )

    summary_parts = [
        f"PubMed {settings.max_pubmed}",
        f"临床试验 {settings.max_trials}",
        f"指南/共识 {settings.max_guidelines}",
        f"最终证据 {settings.topk}",
    ]
    st.markdown(
        (
            "<div class='compact-note'>"
            f"当前配置：{' · '.join(summary_parts)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    return run_clicked


def source_label(evidence: dict) -> str:
    metadata = evidence.get(
        "metadata",
        {},
    )
    source_type = metadata.get(
        "source_type",
        "unknown",
    )

    if source_type == "pubmed":
        return f"PubMed PMID: {metadata.get('pmid', '')}"

    if source_type == "clinical_trial":
        return f"ClinicalTrials.gov NCT ID: {metadata.get('nct_id', '')}"

    if source_type == "guideline":
        return f"Guideline ID: {metadata.get('guideline_id', '')}"

    return str(source_type)


def count_sources(
    evidence_list: list[dict],
) -> dict[str, int]:
    counts = {
        key: 0
        for key in SOURCE_GROUPS
    }

    for evidence in evidence_list:
        metadata = evidence.get("metadata", {})
        source_type = metadata.get("source_type", "unknown")

        if source_type not in counts:
            source_type = "unknown"

        counts[source_type] += 1

    return counts


def render_run_summary(
    result: dict,
) -> None:
    evidence_list = result.get("evidence", [])
    counts = count_sources(evidence_list)
    settings = result.get("settings", {})
    query_mode = settings.get("query_mode", "auto")

    col_mode, col_pubmed, col_trials, col_guidelines = st.columns(4)

    with col_mode:
        st.metric(
            "检索模式",
            QUERY_MODE_LABELS.get(query_mode, query_mode),
        )

    with col_pubmed:
        st.metric(
            "PubMed",
            counts.get("pubmed", 0),
        )

    with col_trials:
        st.metric(
            "临床试验",
            counts.get("clinical_trial", 0),
        )

    with col_guidelines:
        st.metric(
            "指南/共识",
            counts.get("guideline", 0),
        )

    if settings.get("quick_demo"):
        st.caption("当前使用快速演示模式，检索数量已自动压缩。")


def render_evidence_list(
    evidence_list: list[dict],
) -> None:
    if not evidence_list:
        st.info("当前没有可展示的检索证据。")
        return

    grouped_evidence: dict[str, list[tuple[int, dict]]] = {
        key: []
        for key in SOURCE_GROUPS
    }

    for index, evidence in enumerate(evidence_list, start=1):
        metadata = evidence.get("metadata", {})
        source_type = metadata.get("source_type", "unknown")

        if source_type not in grouped_evidence:
            source_type = "unknown"

        grouped_evidence[source_type].append((index, evidence))

    for source_type, (group_title, group_caption) in SOURCE_GROUPS.items():
        group_items = grouped_evidence.get(source_type, [])

        if not group_items:
            continue

        st.markdown(f"**{group_title}**")
        st.caption(group_caption)

        for index, evidence in group_items:
            render_evidence_item(index, evidence)


def render_evidence_item(
    index: int,
    evidence: dict,
) -> None:
    metadata = evidence.get(
        "metadata",
        {},
    )
    title = metadata.get(
        "title",
        "未命名证据",
    )
    url = metadata.get(
        "url",
        "",
    )
    document = evidence.get(
        "document",
        "",
    )

    with st.expander(
        f"[{index}] {source_label(evidence)}",
        expanded=index <= 3,
    ):
        st.markdown(
            f"<div class='evidence-title'>{title}</div>",
            unsafe_allow_html=True,
        )

        details = [
            metadata.get("journal", ""),
            metadata.get("organization", ""),
            metadata.get("status", ""),
            metadata.get("year", ""),
        ]
        details_text = " | ".join(
            str(item)
            for item in details
            if item
        )

        if details_text:
            st.markdown(
                f"<div class='source-line'>{details_text}</div>",
                unsafe_allow_html=True,
            )

        if url:
            st.markdown(f"[打开来源]({url})")

        st.write(document)


def fetch_evidence_preview(
    question: str,
    topk: int,
) -> list[dict]:
    from vector_db import search_with_details

    return search_with_details(
        query=question,
        topk=topk,
    )


def run_pipeline(
    settings: RunSettings,
) -> None:
    if not settings.question.strip():
        st.error("问题不能为空。")
        return

    # ---- 构建多轮对话历史 ----
    history = ""

    if (
        settings.multi_turn
        and st.session_state.conversation_history
    ):
        history_items = []

        for item in st.session_state.conversation_history:
            history_items.append(
                f"Q: {item['question']}\n"
                f"A: {item['summary']}"
            )

        history = "\n---\n".join(history_items)

    log_buffer = io.StringIO()

    with st.spinner("正在生成英文检索词..."):
        pubmed_query = (
            settings.pubmed_query
            or generate_pubmed_query(
                settings.question,
                mode=settings.query_mode,
                history=history,
            )
        )

    with st.spinner("正在检索证据并生成回答..."):
        with contextlib.redirect_stdout(log_buffer):
            if settings.use_3step_agent:
                # 三步 Agent 模式
                answer, evidence_list, challenge_trace = run_3step_agent(
                    question=settings.question,
                    query_mode=settings.query_mode,
                    max_results=settings.max_pubmed,
                    max_trials=settings.max_trials,
                    max_guidelines=settings.max_guidelines,
                    topk=settings.topk,
                    retrieval_mode=settings.retrieval_mode,
                    use_rerank=settings.use_rerank,
                    history=history,
                )
            else:
                # 默认六步流水线
                answer, evidence_list = evidence_agent(
                    question=settings.question,
                    pubmed_query=pubmed_query,
                    max_results=settings.max_pubmed,
                    max_trials=settings.max_trials,
                    max_guidelines=settings.max_guidelines,
                    topk=settings.topk,
                    include_clinical_trials=settings.include_trials,
                    include_guidelines=settings.include_guidelines,
                    query_mode=settings.query_mode,
                    history=history,
                    retrieval_mode=settings.retrieval_mode,
                    dense_weight=settings.dense_weight,
                    use_rerank=settings.use_rerank,
                )
                challenge_trace = None

    # 确保 evidence_list 不为 None
    if evidence_list is None:
        evidence_list = []

    # ---- 记录本轮对话到历史 ----
    if settings.multi_turn and answer:
        # 取回答的前 200 个字符作为摘要
        answer_text = answer or ""

        summary = answer_text.split("## 简短回答")[-1] if "## 简短回答" in answer_text else answer_text
        summary = summary.split("\n\n")[0] if "\n\n" in summary else summary
        summary = summary.strip()[:200]

        st.session_state.conversation_history.append(
            {
                "question": settings.question,
                "summary": summary,
            }
        )

    st.session_state.last_run = {
        "answer": answer or "",
        "query": pubmed_query,
        "logs": log_buffer.getvalue(),
        "evidence": evidence_list,
        "challenge_trace": challenge_trace,
        "settings": {
            "query_mode": settings.query_mode,
            "max_pubmed": settings.max_pubmed,
            "max_trials": settings.max_trials,
            "max_guidelines": settings.max_guidelines,
            "topk": settings.topk,
            "include_trials": settings.include_trials,
            "include_guidelines": settings.include_guidelines,
            "quick_demo": settings.quick_demo,
        },
    }


def main() -> None:
    setup_page()
    settings = render_sidebar()

    st.title("临床证据助手")

    st.session_state.question = st.text_area(
        "医学问题",
        value=st.session_state.get(
            "question",
            DEFAULT_QUESTION,
        ),
        height=110,
    )

    run_clicked = st.button(
        "运行检索与回答",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        settings.question = st.session_state.question
        run_pipeline(settings)

    result = st.session_state.get(
        "last_run",
        {},
    )

    if not result:
        st.info("输入问题后运行检索与回答。")
        return

    render_run_summary(result)

    # ---- 展示多轮对话历史 ----
    if (
        settings.multi_turn
        and st.session_state.conversation_history
    ):
        with st.expander("对话历史", expanded=False):
            for idx, item in enumerate(
                st.session_state.conversation_history,
                start=1,
            ):
                st.markdown(f"**第 {idx} 轮**")
                st.caption(f"Q: {item['question']}")
                st.caption(f"A: {item['summary']}...")
                if idx < len(st.session_state.conversation_history):
                    st.divider()

    st.caption(f"PubMed 检索词：{result.get('query', '')}")

    col_answer, col_evidence = st.columns(
        [
            1.1,
            0.9,
        ],
        gap="large",
    )

    with col_answer:
        st.subheader("回答")
        st.markdown(result.get("answer", ""))

    with col_evidence:
        st.subheader("证据")
        render_evidence_list(
            result.get(
                "evidence",
                [],
            )
        )

    # ---- 质检报告 ----
    answer = result.get("answer", "")
    evidence_list = result.get("evidence", [])
    question = st.session_state.get("question", "")

    if answer and evidence_list:
        with st.expander("质检报告", expanded=False):
            completeness = check_completeness(answer)
            safety = check_safety(answer)
            relevance = check_evidence_relevance(question, evidence_list)

            # ---- 文字版质检摘要 ----
            report_lines = []
            report_lines.append("**质检摘要**")

            # 完整性
            if completeness.coverage_rate >= 1.0:
                report_lines.append("- 完整性：回答覆盖了全部 8 个要求段落，结构完整。")
            elif completeness.coverage_rate >= 0.75:
                report_lines.append(
                    f"- 完整性：覆盖 {completeness.covered}/{completeness.total_sections} 段"
                    f"（{completeness.coverage_rate:.0%}），"
                    f"缺失：{', '.join(completeness.missing)}。"
                )
            else:
                report_lines.append(
                    f"- 完整性：仅覆盖 {completeness.covered}/{completeness.total_sections} 段"
                    f"（{completeness.coverage_rate:.0%}），"
                    f"严重缺失：{', '.join(completeness.missing)}。"
                    f"建议优化 prompt 以确保模型按要求输出所有段落。"
                )

            # 安全性
            if safety.is_safe:
                report_lines.append("- 安全性：未发现越界表述，回答中无具体剂量建议、停药指令或诊断结论。")
            else:
                violation_types = {v["type"] for v in safety.violations}
                report_lines.append(
                    f"- 安全性：发现 {len(safety.violations)} 处安全隐患"
                    f"（{', '.join(violation_types)}），"
                    f"建议在 evidence_gate 中补充对应拦截规则。"
                )

            # 证据匹配度
            if relevance.avg_similarity >= 0.6:
                report_lines.append(
                    f"- 证据匹配度：平均 {relevance.avg_similarity:.3f}，检索证据与问题主题匹配良好。"
                )
            elif relevance.avg_similarity >= 0.4:
                report_lines.append(
                    f"- 证据匹配度：平均 {relevance.avg_similarity:.3f}，尚可但存在提升空间。"
                    f"中位数 {relevance.median_similarity:.3f}，"
                    f"最低 {relevance.min_similarity:.3f}。"
                )
            else:
                report_lines.append(
                    f"- 证据匹配度：平均 {relevance.avg_similarity:.3f}，偏低。"
                    f"建议检查 PubMed query 与问题的语义关联，或调整 topk。"
                )
            if relevance.weak_indices:
                report_lines.append(
                    f"  其中证据 {', '.join(f'[{i}]' for i in relevance.weak_indices)} "
                    f"属于弱匹配，可能引入噪音。"
                )

            # 总结
            good_count = (
                int(completeness.coverage_rate >= 0.75)
                + int(safety.is_safe)
                + int(relevance.avg_similarity >= 0.5)
            )
            if good_count == 3:
                report_lines.append("\n**总结**：各项指标良好，本次回答质量可靠。")
            elif good_count >= 2:
                report_lines.append("\n**总结**：整体质量可接受，存在个别可优化项。")
            else:
                report_lines.append("\n**总结**：多项指标不理想，建议检查检索策略与门控配置。")

            st.markdown("\n".join(report_lines))

            st.divider()

            col_c, col_s, col_r = st.columns(3)

            with col_c:
                st.metric("完整性", f"{completeness.coverage_rate:.0%}")
                st.progress(completeness.coverage_rate)
                if completeness.missing:
                    for m in completeness.missing:
                        st.caption(f":red[✗] {m}")
                else:
                    st.caption(":green[✓] 8 段完整")

            with col_s:
                if safety.is_safe:
                    st.metric("安全性", "通过")
                    st.caption(":green[✓] 未发现越界表述")
                else:
                    st.metric("安全性", f"{len(safety.violations)} 处风险", delta="-")
                    for v in safety.violations:
                        st.caption(f":red[{v['type']}] {v['context']}")

            with col_r:
                st.metric("证据匹配度", f"{relevance.avg_similarity:.2f}")
                st.progress(min(relevance.avg_similarity / 0.8, 1.0))
                st.caption(f"中位: {relevance.median_similarity:.3f}  最低: {relevance.min_similarity:.3f}")
                if relevance.weak_indices:
                    st.caption(f":orange[弱匹配证据:] {' '.join(f'[{i}]' for i in relevance.weak_indices)}")

            # 逐条明细
            st.divider()
            st.caption("证据逐条匹配度")
            cols_per_row = min(len(evidence_list), 5)
            ev_cols = st.columns(cols_per_row)
            for idx, (score, col) in enumerate(zip(relevance.scores, ev_cols), 1):
                meta = evidence_list[idx - 1].get("metadata", {}) if idx - 1 < len(evidence_list) else {}
                pmid = meta.get("pmid", "") or meta.get("nct_id", "") or meta.get("guideline_id", "")
                with col:
                    color = "#16a34a" if score >= 0.6 else "#ca8a04" if score >= 0.4 else "#dc2626"
                    st.metric(
                        f"[{idx}] {pmid[:12]}" if pmid else f"[{idx}]",
                        f"{score:.3f}",
                        delta_color="off",
                    )
                    st.markdown(f"<div style='background:{color};height:4px;border-radius:2px;'></div>", unsafe_allow_html=True)

    # ---- 执行轨迹（仅三步Agent模式） ----
    challenge_trace = result.get("challenge_trace")

    if challenge_trace is not None:
        with st.expander("执行轨迹", expanded=False):
            st.caption(f"Trace ID: {challenge_trace.trace_id}")

            for s in challenge_trace.steps:
                icon_map = {
                    "search": "🔍",
                    "verify_citation": "✅",
                    "answer": "📝",
                    "partial_answer": "⚠️",
                    "refuse": "🚫",
                }
                icon = icon_map.get(s.action, "•")

                st.markdown(
                    f"**{icon} 步骤 {s.step}：{s.action}**"
                    f"　`{s.elapsed_ms}ms`"
                )

                for key, value in s.output_summary.items():
                    if key in ("problems", "rejected", "initial_pmids", "filtered_pmids"):
                        continue
                    if key == "unsupported_claims":
                        continue
                    st.caption(f"{key}: {value}")

                if s.decision_note:
                    st.caption(f"_决策说明: {s.decision_note}_")

                if s.output_summary.get("rejected"):
                    with st.expander(f"被过滤 {len(s.output_summary['rejected'])} 条", expanded=False):
                        for item in s.output_summary["rejected"]:
                            st.warning(item)

                if s.output_summary.get("problems"):
                    with st.expander("问题详情", expanded=False):
                        for problem in s.output_summary["problems"]:
                            st.warning(problem)

                if s.errors:
                    for error in s.errors:
                        st.error(f"错误: {error}")

                st.divider()

            st.metric(
                "最终决策",
                challenge_trace.final_decision,
                delta=f"总耗时 {challenge_trace.total_elapsed_ms}ms",
            )

    with st.expander("运行日志"):
        st.code(
            result.get(
                "logs",
                "",
            )
        )


if __name__ == "__main__":
    main()
