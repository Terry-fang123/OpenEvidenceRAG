from __future__ import annotations

import asyncio
import contextlib
import html
import io
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from nicegui import run, ui

from main import DEFAULT_QUESTION, evidence_agent
from query_builder import generate_pubmed_query


SAMPLE_QUESTIONS = {
    "血脂异常": "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
    "高血压": "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    "限钠饮食": "限钠饮食对高血压是否真的有帮助？",
    "地中海饮食": "地中海饮食对心血管风险有什么证据？",
    "糖尿病": "糖尿病患者为什么需要关注 LDL-C？",
}

QUERY_STRATEGIES = {
    "自动模式": "auto",
    "优先大模型": "llm",
    "本地兜底": "fallback",
}

SOURCE_GROUPS = {
    "pubmed": ("PubMed 文献", "公共医学文献数据库"),
    "clinical_trial": ("ClinicalTrials.gov", "临床试验注册研究"),
    "guideline": ("指南 / 共识", "本地整理的临床指南和共识资料"),
    "unknown": ("其他证据", "未识别来源类型"),
}

RUN_STAGES = [
    ("正在检索 PubMed…", 0.24),
    ("正在检索临床试验…", 0.46),
    ("正在整合指南证据…", 0.68),
    ("正在生成循证回答…", 0.86),
]

STYLE = """
<style>
:root {
    --bg: #f7f9fa;
    --panel: #ffffff;
    --panel-soft: #f1f5f6;
    --line: #e3e8ec;
    --text: #17202a;
    --muted: #667085;
    --muted-2: #8a95a3;
    --teal: #0f766e;
    --teal-2: #115e59;
    --teal-soft: #e3f4f1;
    --yellow-soft: #fff7df;
    --blue-soft: #edf5ff;
    --danger-soft: #fff1f2;
}
body,
.q-page {
    background: var(--bg);
    color: var(--text);
}
.page-shell {
    background: var(--bg);
    min-height: 100vh;
    width: 100%;
}
.topbar {
    background: rgba(255, 255, 255, 0.96);
    border-bottom: 1px solid var(--line);
    height: 58px;
    padding: 0 28px;
}
.topbar-brand {
    color: var(--text);
    font-size: 18px;
    font-weight: 850;
    min-width: 260px;
}
.topbar-title {
    color: #334155;
    font-size: 15px;
    font-weight: 800;
}
.status-row {
    align-items: center;
    color: #166534;
    display: flex;
    font-size: 13px;
    font-weight: 800;
    gap: 7px;
}
.status-dot {
    background: #16a34a;
    border-radius: 999px;
    box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.12);
    height: 8px;
    width: 8px;
}
.main-body {
    align-items: stretch;
    display: flex;
    min-height: calc(100vh - 58px);
    width: 100%;
}
.sidebar {
    background: #eef2f5;
    border-right: 1px solid var(--line);
    flex: 0 0 280px;
    min-height: calc(100vh - 58px);
    padding: 20px 16px;
    width: 280px;
}
.workspace {
    flex: 1 1 auto;
    min-width: 0;
    padding: 24px 30px 36px;
}
.workspace-inner {
    margin: 0 auto;
    max-width: 1320px;
    width: 100%;
}
.sidebar-section {
    border-top: 1px solid var(--line);
    margin-top: 16px;
    padding-top: 14px;
}
.sidebar-section:first-child {
    border-top: 0;
    margin-top: 0;
    padding-top: 0;
}
.section-title {
    color: #344054;
    font-size: 13px;
    font-weight: 850;
}
.section-note,
.muted {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.48;
}
.page-title {
    color: var(--text);
    font-size: 28px;
    font-weight: 880;
    line-height: 1.16;
}
.page-subtitle {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.5;
}
.ask-card,
.metric-card,
.result-card,
.evidence-card,
.detail-card,
.error-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035);
}
.ask-card {
    padding: 18px;
}
.primary-button {
    background: var(--teal) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 850 !important;
    min-height: 40px;
    padding: 0 18px !important;
}
.secondary-button {
    border-radius: 8px !important;
    font-weight: 760 !important;
}
.run-hint {
    background: var(--teal-soft);
    border: 1px solid #b9ded8;
    border-radius: 8px;
    color: #134e4a;
    font-size: 12px;
    line-height: 1.45;
    padding: 8px 10px;
}
.metric-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    width: 100%;
}
.metric-card {
    min-height: 78px;
    padding: 13px 14px;
}
.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
}
.metric-value {
    color: var(--text);
    font-size: 22px;
    font-weight: 880;
    line-height: 1.2;
    margin-top: 8px;
}
.query-strip {
    background: var(--teal-soft);
    border: 1px solid #b9ded8;
    border-radius: 9px;
    color: #134e4a;
    font-size: 13px;
    line-height: 1.5;
    padding: 11px 13px;
    width: 100%;
}
.result-card {
    min-height: 520px;
    padding: 4px 14px 18px;
    width: 100%;
}
.empty-state {
    min-height: 190px;
}
.empty-icon {
    color: #94a3b8;
    font-size: 42px;
}
.answer-section {
    border-bottom: 1px solid #eef2f6;
    padding: 16px 4px;
}
.answer-section:last-child {
    border-bottom: 0;
}
.answer-section h3 {
    color: var(--text);
    font-size: 18px;
    font-weight: 850;
    margin: 0 0 10px 0;
}
.answer-section p,
.answer-section li {
    color: #273444;
    font-size: 15px;
    line-height: 1.75;
}
.answer-section ul,
.answer-section ol {
    padding-left: 1.25rem;
}
.answer-callout {
    background: var(--yellow-soft);
    border: 1px solid #f1dc9a;
    border-radius: 10px;
    margin: 12px 0;
    padding: 14px 16px;
}
.answer-limitations {
    background: var(--blue-soft);
    border: 1px solid #c9dcf7;
    border-radius: 10px;
    margin: 12px 0;
    padding: 14px 16px;
}
.reference-section {
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 16px;
}
.citation {
    background: #e0f2f1;
    border: 1px solid #b2dfdb;
    border-radius: 999px;
    color: #0f766e;
    display: inline-block;
    font-size: 0.82em;
    font-weight: 850;
    line-height: 1.2;
    margin: 0 2px;
    padding: 1px 6px;
}
.evidence-card {
    margin-bottom: 12px;
    padding: 14px;
}
.source-heading {
    color: var(--text);
    font-size: 15px;
    font-weight: 880;
}
.source-caption {
    color: var(--muted);
    font-size: 12px;
}
.evidence-title {
    color: var(--text);
    font-size: 15px;
    font-weight: 850;
    line-height: 1.4;
}
.evidence-meta {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
}
.evidence-snippet {
    color: #344054;
    font-size: 13px;
    line-height: 1.62;
    margin-top: 8px;
    white-space: pre-wrap;
    word-break: break-word;
}
.id-chip {
    background: #f1f5f9;
    border: 1px solid #dbe4ee;
    border-radius: 999px;
    color: #475569;
    display: inline-flex;
    font-size: 12px;
    font-weight: 800;
    padding: 4px 8px;
}
.detail-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}
.detail-card {
    padding: 14px;
}
.detail-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
}
.detail-value {
    color: var(--text);
    font-size: 14px;
    line-height: 1.55;
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
}
.query-code,
.log-code {
    background: #f3f5f7;
    border: 1px solid var(--line);
    border-radius: 8px;
    color: #1f2937;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    line-height: 1.55;
    margin: 0;
    overflow: auto;
    padding: 12px;
    white-space: pre-wrap;
}
.log-code {
    max-height: 460px;
}
.error-card {
    background: var(--danger-soft);
    border-color: #fecdd3;
    color: #9f1239;
    padding: 16px;
}
.q-field--outlined .q-field__control {
    border-radius: 9px;
}
.q-tab {
    font-weight: 850;
}
@media (max-width: 1200px) {
    .sidebar {
        flex-basis: 260px;
        width: 260px;
    }
    .metric-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}
@media (max-width: 860px) {
    .topbar {
        height: auto;
        padding: 14px 16px;
    }
    .topbar-brand {
        min-width: 0;
    }
    .topbar-title {
        display: none;
    }
    .main-body {
        flex-direction: column;
    }
    .sidebar {
        min-height: auto;
        width: 100%;
    }
    .workspace {
        padding: 18px 14px 28px;
    }
    .metric-grid,
    .detail-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 560px) {
    .metric-grid,
    .detail-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


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
    query_source: str


def clean_int(
    value: object,
    default: int,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_meta(
    evidence: dict,
    key: str,
    default: str = "",
) -> str:
    metadata = evidence.get("metadata", {})

    if isinstance(metadata, dict) and metadata.get(key) not in {
        None,
        "",
    }:
        return str(metadata.get(key))

    if evidence.get(key) not in {
        None,
        "",
    }:
        return str(evidence.get(key))

    return default


def text_snippet(
    text: str,
    max_chars: int = 560,
) -> str:
    cleaned = " ".join(str(text).split())

    if len(cleaned) <= max_chars:
        return cleaned

    return f"{cleaned[:max_chars].rstrip()}..."


def source_label(
    evidence: dict,
) -> str:
    source_type = get_meta(evidence, "source_type", "unknown")

    if source_type == "pubmed":
        return f"PMID: {get_meta(evidence, 'pmid')}"

    if source_type == "clinical_trial":
        return f"NCT ID: {get_meta(evidence, 'nct_id')}"

    if source_type == "guideline":
        return f"Guideline ID: {get_meta(evidence, 'guideline_id')}"

    return source_type


def group_evidence(
    evidence_list: list[dict],
) -> dict[str, list[tuple[int, dict]]]:
    grouped = {
        key: []
        for key in SOURCE_GROUPS
    }

    for index, evidence in enumerate(evidence_list, start=1):
        source_type = get_meta(evidence, "source_type", "unknown")

        if source_type not in grouped:
            source_type = "unknown"

        grouped[source_type].append((index, evidence))

    return grouped


def evidence_counts(
    evidence_list: list[dict],
) -> dict[str, int]:
    grouped = group_evidence(evidence_list)
    counts = {
        source_type: len(items)
        for source_type, items in grouped.items()
    }
    counts["total"] = sum(counts.values())

    return counts


def format_citations(
    markdown_text: str,
) -> str:
    return re.sub(
        r"(?<!\!)\[(\d+)\]",
        r'<span class="citation">[\1]</span>',
        markdown_text,
    )


def split_markdown_sections(
    answer: str,
) -> list[tuple[str, str]]:
    matches = list(
        re.finditer(
            r"^##\s+(.+?)\s*$",
            answer,
            flags=re.MULTILINE,
        )
    )

    if not matches:
        return [
            (
                "",
                answer,
            )
        ]

    sections: list[tuple[str, str]] = []

    leading_text = answer[: matches[0].start()].strip()

    if leading_text:
        sections.append(
            (
                "",
                leading_text,
            )
        )

    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(answer)
        )
        sections.append(
            (
                match.group(1).strip(),
                answer[start:end].strip(),
            )
        )

    return sections


def execute_pipeline(
    settings: RunSettings,
) -> dict:
    started_at = time.perf_counter()
    log_buffer = io.StringIO()

    with contextlib.redirect_stdout(log_buffer):
        pubmed_query = settings.pubmed_query.strip()

        if not pubmed_query:
            print("正在生成英文 PubMed 检索词...")
            pubmed_query = generate_pubmed_query(
                settings.question,
                mode=settings.query_mode,
            )

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
        )

        evidence_preview: list[dict] = evidence_list if evidence_list else []

    elapsed = time.perf_counter() - started_at

    return {
        "answer": answer or "",
        "query": pubmed_query,
        "evidence": evidence_preview,
        "logs": log_buffer.getvalue(),
        "elapsed": elapsed,
        "settings": asdict(settings),
    }


def build_topbar(
    refs: dict[str, Any],
) -> None:
    with ui.row().classes("topbar w-full items-center justify-between no-wrap"):
        ui.label("OpenEvidence").classes("topbar-brand")
        ui.label("临床证据助手").classes("topbar-title")
        with ui.row().classes("status-row"):
            ui.html("<span class='status-dot'></span>")
            refs["status_label"] = ui.label("系统就绪")


def build_sidebar(
    refs: dict[str, Any],
) -> None:
    with ui.column().classes("sidebar gap-2"):
        with ui.column().classes("sidebar-section gap-2 w-full"):
            ui.label("快速开始").classes("section-title")
            refs["sample_select"] = ui.select(
                list(SAMPLE_QUESTIONS.keys()),
                value="血脂异常",
                label="测试问题",
            ).props("outlined dense").classes("w-full")

            def load_sample() -> None:
                question_input = refs.get("question_input")

                if question_input:
                    question_input.set_value(
                        SAMPLE_QUESTIONS.get(
                            refs["sample_select"].value,
                            DEFAULT_QUESTION,
                        )
                    )

            ui.button(
                "载入测试问题",
                icon="playlist_add",
                on_click=load_sample,
            ).props("outline no-caps").classes("secondary-button w-full")

        with ui.column().classes("sidebar-section gap-3 w-full"):
            ui.label("检索设置").classes("section-title")
            refs["quick_demo"] = ui.switch(
                "快速演示模式",
                value=True,
            ).classes("w-full")
            ui.label(
                "开启后会自动压缩检索数量，适合课堂展示和答辩。"
            ).classes("section-note")

            refs["query_source"] = ui.toggle(
                [
                    "自动",
                    "手动",
                ],
                value="自动",
            ).props("unelevated spread").classes("w-full")

            with ui.column().classes("w-full gap-2") as manual_panel:
                refs["manual_query"] = ui.textarea(
                    label="手动 PubMed Query",
                    placeholder=(
                        "例如：dyslipidemia lifestyle intervention "
                        "statin therapy cardiovascular risk"
                    ),
                ).props("outlined autogrow").classes("w-full")

            refs["manual_query_panel"] = manual_panel
            manual_panel.bind_visibility_from(
                refs["query_source"],
                "value",
                backward=lambda value: value == "手动",
            )

        with ui.column().classes("sidebar-section gap-3 w-full"):
            ui.label("证据来源").classes("section-title")
            refs["include_trials"] = ui.switch(
                "ClinicalTrials.gov",
                value=True,
            )
            refs["include_guidelines"] = ui.switch(
                "Guidelines",
                value=True,
            )

            with ui.expansion(
                "高级检索设置",
                value=False,
            ).classes("w-full"):
                ui.label("自动查询策略").classes("section-note")
                refs["query_strategy"] = ui.select(
                    list(QUERY_STRATEGIES.keys()),
                    value="自动模式",
                ).props("outlined dense").classes("w-full")

                refs["max_pubmed"] = ui.slider(
                    min=1,
                    max=10,
                    value=5,
                    step=1,
                ).props("label").classes("w-full")
                ui.label("PubMed 文献数").classes("section-note")

                refs["max_trials"] = ui.slider(
                    min=0,
                    max=10,
                    value=3,
                    step=1,
                ).props("label").classes("w-full")
                ui.label("临床试验数").classes("section-note")

                refs["max_guidelines"] = ui.slider(
                    min=0,
                    max=5,
                    value=3,
                    step=1,
                ).props("label").classes("w-full")
                ui.label("指南/共识条数").classes("section-note")

                refs["topk"] = ui.slider(
                    min=1,
                    max=12,
                    value=5,
                    step=1,
                ).props("label").classes("w-full")
                ui.label("最终证据条数").classes("section-note")


def build_question_panel(
    refs: dict[str, Any],
) -> None:
    with ui.column().classes("ask-card w-full gap-3"):
        ui.label("医学问题").classes("section-title")
        refs["question_input"] = ui.textarea(
            value=DEFAULT_QUESTION,
        ).props("outlined autogrow clearable").classes("w-full")
        refs["question_input"].style("min-height: 98px;")

        with ui.row().classes("w-full items-center justify-between gap-3"):
            refs["run_button"] = ui.button(
                "运行检索与回答",
                icon="search",
            ).props("unelevated no-caps").classes("primary-button")

            refs["run_hint"] = ui.label(
                "快速演示模式已开启，将自动减少检索数量以加快演示速度。"
            ).classes("run-hint")

        refs["progress"] = ui.linear_progress(
            value=0,
        ).classes("w-full")
        refs["progress"].visible = False


def build_metric_card(
    label: str,
    value: str,
) -> Any:
    with ui.column().classes("metric-card gap-0") as card:
        ui.label(label).classes("metric-label")
        value_label = ui.label(value).classes("metric-value")

    return value_label


def build_metrics(
    refs: dict[str, Any],
) -> None:
    with ui.element("div").classes("metric-grid"):
        refs["metric_mode"] = build_metric_card("检索模式", "自动模式")
        refs["metric_pubmed"] = build_metric_card("PubMed", "—")
        refs["metric_trials"] = build_metric_card("临床试验", "—")
        refs["metric_guidelines"] = build_metric_card("指南", "—")
        refs["metric_total"] = build_metric_card("总证据", "—")
        refs["metric_time"] = build_metric_card("耗时", "—")


def build_result_tabs(
    refs: dict[str, Any],
) -> None:
    with ui.column().classes("result-card w-full"):
        with ui.tabs().classes("w-full") as tabs:
            refs["answer_tab"] = ui.tab("回答")
            refs["evidence_tab"] = ui.tab("证据")
            refs["details_tab"] = ui.tab("检索详情")
            refs["logs_tab"] = ui.tab("运行日志")

        with ui.tab_panels(
            tabs,
            value=refs["answer_tab"],
        ).classes("w-full bg-transparent") as panels:
            refs["panels"] = panels

            with ui.tab_panel(refs["answer_tab"]).classes("q-pa-none"):
                refs["answer_container"] = ui.column().classes(
                    "w-full gap-1 q-pa-md"
                )
            with ui.tab_panel(refs["evidence_tab"]).classes("q-pa-none"):
                refs["evidence_container"] = ui.column().classes(
                    "w-full gap-2 q-pa-md"
                )
            with ui.tab_panel(refs["details_tab"]).classes("q-pa-none"):
                refs["details_container"] = ui.column().classes(
                    "w-full gap-3 q-pa-md"
                )
            with ui.tab_panel(refs["logs_tab"]).classes("q-pa-none"):
                refs["logs_container"] = ui.column().classes(
                    "w-full q-pa-md"
                )


def build_workspace(
    refs: dict[str, Any],
) -> None:
    with ui.column().classes("workspace"):
        with ui.column().classes("workspace-inner gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("临床证据工作台").classes("page-title")
                ui.label(
                    "检索 PubMed、ClinicalTrials.gov 与临床指南，"
                    "并生成带引用的循证医学回答。"
                ).classes("page-subtitle")

            build_question_panel(refs)
            build_metrics(refs)

            refs["query_strip"] = ui.label(
                "PubMed Query 将在运行后显示。"
            ).classes("query-strip")

            build_result_tabs(refs)


def render_empty_state(
    container: Any,
) -> None:
    container.clear()

    with container:
        with ui.column().classes(
            "empty-state items-center justify-center w-full gap-1"
        ):
            ui.icon("travel_explore").classes("empty-icon")
            ui.label("输入医学问题后运行检索与回答").classes(
                "text-subtitle1 text-weight-bold text-grey-8"
            )
            ui.label(
                "系统将整合 PubMed、ClinicalTrials.gov 和指南/共识证据。"
            ).classes("muted")


def section_class(
    title: str,
) -> str:
    if "参考证据" in title:
        return "reference-section"

    if "局限" in title:
        return "answer-limitations"

    if "提醒" in title or "注意" in title:
        return "answer-callout"

    return "answer-section"


def render_answer(
    container: Any,
    answer: str,
) -> None:
    container.clear()

    with container:
        if not answer.strip():
            with ui.column().classes("error-card w-full"):
                ui.label("没有生成回答").classes("text-weight-bold")
                ui.label("请查看运行日志，确认网络连接或 API 配置。")
            return

        for title, body in split_markdown_sections(answer):
            css_class = section_class(title)

            with ui.column().classes(f"{css_class} w-full"):
                markdown_parts = []

                if title:
                    markdown_parts.append(f"### {title}")

                if body:
                    markdown_parts.append(format_citations(body))

                ui.markdown(
                    "\n\n".join(markdown_parts)
                ).classes("answer-section-content")


def render_pubmed_evidence(
    index: int,
    evidence: dict,
) -> None:
    title = get_meta(evidence, "title", "未命名 PubMed 文献")
    journal = get_meta(evidence, "journal")
    year = get_meta(evidence, "year")
    pmid = get_meta(evidence, "pmid")
    url = get_meta(evidence, "url")
    details = " · ".join(
        item
        for item in [
            journal,
            year,
        ]
        if item
    )

    with ui.column().classes("evidence-card w-full gap-2"):
        ui.label(title).classes("evidence-title")

        if details:
            ui.label(details).classes("evidence-meta")

        ui.label(
            text_snippet(evidence.get("document", ""))
        ).classes("evidence-snippet")

        with ui.row().classes("items-center gap-2"):
            if pmid:
                ui.label(f"PMID: {pmid}").classes("id-chip")
            ui.label(f"证据编号 [{index}]").classes("id-chip")
            if url:
                ui.link("查看 PubMed", url, new_tab=True).classes(
                    "text-primary text-weight-bold"
                )


def render_trial_evidence(
    index: int,
    evidence: dict,
) -> None:
    title = get_meta(evidence, "title", "未命名临床试验")
    status = get_meta(evidence, "status")
    phase = get_meta(evidence, "phase")
    nct_id = get_meta(evidence, "nct_id")
    url = get_meta(evidence, "url")

    with ui.column().classes("evidence-card w-full gap-2"):
        ui.label(title).classes("evidence-title")

        for label, value in [
            ("状态", status),
            ("Phase", phase),
            ("研究类型", get_meta(evidence, "study_type")),
            ("干预", get_meta(evidence, "interventions")),
        ]:
            if value:
                ui.label(f"{label}: {value}").classes("evidence-meta")

        ui.label(
            text_snippet(evidence.get("document", ""), max_chars=460)
        ).classes("evidence-snippet")

        with ui.row().classes("items-center gap-2"):
            if nct_id:
                ui.label(f"NCT ID: {nct_id}").classes("id-chip")
            ui.label(f"证据编号 [{index}]").classes("id-chip")
            if url:
                ui.link(
                    "查看 ClinicalTrials.gov",
                    url,
                    new_tab=True,
                ).classes("text-primary text-weight-bold")


def render_guideline_evidence(
    index: int,
    evidence: dict,
) -> None:
    title = get_meta(evidence, "title", "未命名指南/共识")
    organization = get_meta(evidence, "organization")
    year = get_meta(evidence, "year")
    guideline_id = get_meta(evidence, "guideline_id")
    url = get_meta(evidence, "url")

    with ui.column().classes("evidence-card w-full gap-2"):
        ui.label(title).classes("evidence-title")

        for label, value in [
            ("组织", organization),
            ("年份", year),
            ("主题", get_meta(evidence, "topic")),
        ]:
            if value:
                ui.label(f"{label}: {value}").classes("evidence-meta")

        ui.label(
            text_snippet(evidence.get("document", ""), max_chars=500)
        ).classes("evidence-snippet")

        with ui.row().classes("items-center gap-2"):
            if guideline_id:
                ui.label(f"Guideline ID: {guideline_id}").classes("id-chip")
            ui.label(f"证据编号 [{index}]").classes("id-chip")
            if url:
                ui.link("查看来源", url, new_tab=True).classes(
                    "text-primary text-weight-bold"
                )


def render_unknown_evidence(
    index: int,
    evidence: dict,
) -> None:
    with ui.column().classes("evidence-card w-full gap-2"):
        ui.label(source_label(evidence)).classes("evidence-title")
        ui.label(
            text_snippet(evidence.get("document", ""))
        ).classes("evidence-snippet")
        ui.label(f"证据编号 [{index}]").classes("id-chip")


def render_evidence(
    container: Any,
    evidence_list: list[dict],
) -> None:
    container.clear()

    with container:
        if not evidence_list:
            ui.label("当前没有可展示的检索证据。").classes("muted")
            return

        grouped = group_evidence(evidence_list)

        for source_type, (title, caption) in SOURCE_GROUPS.items():
            items = grouped.get(source_type, [])

            if not items:
                continue

            with ui.column().classes("w-full gap-2"):
                with ui.row().classes(
                    "items-end justify-between w-full no-wrap"
                ):
                    with ui.column().classes("gap-0"):
                        ui.label(title).classes("source-heading")
                        ui.label(caption).classes("source-caption")
                    ui.label(f"{len(items)} 条").classes("id-chip")

                for index, evidence in items:
                    if source_type == "pubmed":
                        render_pubmed_evidence(index, evidence)
                    elif source_type == "clinical_trial":
                        render_trial_evidence(index, evidence)
                    elif source_type == "guideline":
                        render_guideline_evidence(index, evidence)
                    else:
                        render_unknown_evidence(index, evidence)


def render_detail_card(
    label: str,
    value: str,
) -> None:
    with ui.column().classes("detail-card"):
        ui.label(label).classes("detail-label")
        ui.label(value or "—").classes("detail-value")


def render_search_details(
    container: Any,
    result: dict | None,
) -> None:
    container.clear()

    with container:
        if not result:
            ui.label("运行后显示检索详情。").classes("muted")
            return

        settings = result.get("settings", {})
        evidence_list = result.get("evidence", [])
        counts = evidence_counts(evidence_list)
        used_sources = []

        if counts.get("pubmed", 0):
            used_sources.append("PubMed")
        if counts.get("clinical_trial", 0):
            used_sources.append("ClinicalTrials.gov")
        if counts.get("guideline", 0):
            used_sources.append("Guidelines")

        with ui.element("div").classes("detail-grid w-full"):
            render_detail_card(
                "检索模式",
                str(settings.get("query_source", "自动")),
            )
            render_detail_card(
                "自动查询策略",
                str(settings.get("query_mode", "auto")),
            )
            render_detail_card(
                "检索数量",
                (
                    f"PubMed {settings.get('max_pubmed', 0)}；"
                    f"临床试验 {settings.get('max_trials', 0)}；"
                    f"指南 {settings.get('max_guidelines', 0)}；"
                    f"最终证据 {settings.get('topk', 0)}"
                ),
            )
            render_detail_card(
                "使用的证据来源",
                "、".join(used_sources) if used_sources else "—",
            )

        ui.label("生成的 PubMed Query").classes("section-title")
        ui.html(
            f"<pre class='query-code'>{html.escape(result.get('query', ''))}</pre>"
        )


def render_logs(
    container: Any,
    logs: str,
) -> None:
    container.clear()

    with container:
        ui.html(
            f"<pre class='log-code'>{html.escape(logs or '暂无运行日志。')}</pre>"
        )


def render_error(
    refs: dict[str, Any],
    error: Exception,
) -> None:
    refs["answer_container"].clear()

    with refs["answer_container"]:
        with ui.column().classes("error-card w-full gap-1"):
            ui.label("检索失败").classes("text-weight-bold")
            ui.label("请检查网络连接或 API 配置。")

    render_logs(
        refs["logs_container"],
        f"{type(error).__name__}: {error}",
    )
    render_search_details(
        refs["details_container"],
        None,
    )


def collect_settings(
    refs: dict[str, Any],
) -> RunSettings:
    quick_demo = bool(refs["quick_demo"].value)
    include_trials = bool(refs["include_trials"].value)
    include_guidelines = bool(refs["include_guidelines"].value)
    query_source = str(refs["query_source"].value or "自动")
    query_strategy_label = str(refs["query_strategy"].value or "自动模式")

    if quick_demo:
        max_pubmed = 2
        max_trials = 1 if include_trials else 0
        max_guidelines = 2 if include_guidelines else 0
        topk = 4
    else:
        max_pubmed = clean_int(refs["max_pubmed"].value, 5)
        max_trials = (
            clean_int(refs["max_trials"].value, 3)
            if include_trials
            else 0
        )
        max_guidelines = (
            clean_int(refs["max_guidelines"].value, 3)
            if include_guidelines
            else 0
        )
        topk = clean_int(refs["topk"].value, 5)

    pubmed_query = (
        str(refs["manual_query"].value or "").strip()
        if query_source == "手动"
        else ""
    )

    return RunSettings(
        question=str(refs["question_input"].value or "").strip(),
        query_mode=QUERY_STRATEGIES.get(query_strategy_label, "auto"),
        pubmed_query=pubmed_query,
        max_pubmed=max_pubmed,
        max_trials=max_trials,
        max_guidelines=max_guidelines,
        topk=topk,
        include_trials=include_trials,
        include_guidelines=include_guidelines,
        quick_demo=quick_demo,
        query_source=query_source,
    )


def update_hint(
    refs: dict[str, Any],
) -> None:
    if refs["quick_demo"].value:
        refs["run_hint"].set_text(
            "快速演示模式已开启，将自动减少检索数量以加快演示速度。"
        )
    else:
        refs["run_hint"].set_text(
            "完整检索模式将使用左侧高级设置中的数量参数。"
        )


def update_metrics(
    refs: dict[str, Any],
    result: dict,
) -> None:
    counts = evidence_counts(result.get("evidence", []))
    settings = result.get("settings", {})
    query_source = str(settings.get("query_source", "自动"))
    elapsed = float(result.get("elapsed", 0))

    refs["metric_mode"].set_text(query_source)
    refs["metric_pubmed"].set_text(f"{counts.get('pubmed', 0)} 篇")
    refs["metric_trials"].set_text(
        f"{counts.get('clinical_trial', 0)} 项"
    )
    refs["metric_guidelines"].set_text(f"{counts.get('guideline', 0)} 篇")
    refs["metric_total"].set_text(f"{counts.get('total', 0)} 条")
    refs["metric_time"].set_text(f"{elapsed:.1f} s")
    refs["query_strip"].set_text(
        f"PubMed Query：{result.get('query', '')}"
    )


async def rotate_run_status(
    refs: dict[str, Any],
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        for label, progress_value in RUN_STAGES:
            if stop_event.is_set():
                break

            refs["status_label"].set_text(label)
            refs["progress"].value = progress_value
            await asyncio.sleep(1.2)


async def handle_run(
    refs: dict[str, Any],
) -> None:
    settings = collect_settings(refs)

    if not settings.question:
        ui.notify("医学问题不能为空。", type="negative")
        return

    if settings.query_source == "手动" and not settings.pubmed_query:
        ui.notify("手动模式需要填写英文 PubMed Query。", type="warning")
        return

    refs["run_button"].disable()
    refs["run_button"].set_text("正在检索证据…")
    refs["progress"].visible = True
    refs["progress"].value = 0.12
    refs["query_strip"].set_text("正在生成检索词并检索多来源证据…")
    refs["answer_container"].clear()
    refs["evidence_container"].clear()
    refs["details_container"].clear()
    refs["logs_container"].clear()

    with refs["answer_container"]:
        with ui.row().classes("items-center gap-2 q-pa-md"):
            ui.spinner("dots", size="lg", color="primary")
            ui.label("正在检索证据并生成回答…").classes("text-grey-8")

    stop_event = asyncio.Event()
    status_task = asyncio.create_task(
        rotate_run_status(
            refs,
            stop_event,
        )
    )

    try:
        result = await run.io_bound(
            execute_pipeline,
            settings,
        )
    except Exception as exc:
        stop_event.set()
        await status_task
        refs["status_label"].set_text("检索失败")
        refs["progress"].visible = False
        refs["run_button"].enable()
        refs["run_button"].set_text("运行检索与回答")
        render_error(refs, exc)
        ui.notify("检索失败，请查看运行日志。", type="negative")
        return

    stop_event.set()
    await status_task

    refs["progress"].value = 1
    refs["progress"].visible = False
    refs["run_button"].enable()
    refs["run_button"].set_text("运行检索与回答")
    refs["status_label"].set_text("检索完成")

    update_hint(refs)
    update_metrics(refs, result)
    render_answer(refs["answer_container"], result.get("answer", ""))
    render_evidence(refs["evidence_container"], result.get("evidence", []))
    render_search_details(refs["details_container"], result)
    render_logs(refs["logs_container"], result.get("logs", ""))
    refs["panels"].set_value(refs["answer_tab"])

    ui.notify("检索完成。", type="positive")


@ui.page("/")
def index() -> None:
    refs: dict[str, Any] = {}

    ui.add_head_html(STYLE)
    ui.colors(
        primary="#0f766e",
        secondary="#475569",
        accent="#b45309",
    )

    with ui.column().classes("page-shell gap-0"):
        build_topbar(refs)

        with ui.element("div").classes("main-body"):
            build_sidebar(refs)
            build_workspace(refs)

    render_empty_state(refs["answer_container"])
    render_evidence(refs["evidence_container"], [])
    render_search_details(refs["details_container"], None)
    render_logs(refs["logs_container"], "")

    refs["quick_demo"].on_value_change(lambda _: update_hint(refs))
    refs["run_button"].on_click(lambda: handle_run(refs))


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="临床证据助手",
        host="127.0.0.1",
        port=8080,
        reload=False,
        show=False,
    )
