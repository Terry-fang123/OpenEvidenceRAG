from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "ecnu-plus",
).strip()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "",
).strip()

LLM_THINKING = os.getenv(
    "LLM_THINKING",
    "",
).strip().lower()


TEST_CASES = [
    {
        "id": "hypertension_long_term",
        "question": "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
        "pubmed_query": (
            "hypertension antihypertensive therapy long-term treatment "
            "cardiovascular risk guideline"
        ),
    },
    {
        "id": "dyslipidemia_lifestyle_drug",
        "question": "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
        "pubmed_query": (
            "dyslipidemia lifestyle intervention statin therapy "
            "cardiovascular risk"
        ),
    },
    {
        "id": "mediterranean_diet",
        "question": "地中海饮食对心血管风险有什么证据？",
        "pubmed_query": (
            "Mediterranean diet cardiovascular risk systematic review"
        ),
    },
    {
        "id": "sodium_reduction",
        "question": "限钠饮食对高血压是否真的有帮助？",
        "pubmed_query": (
            "sodium reduction hypertension blood pressure clinical trial"
        ),
    },
    {
        "id": "diabetes_ldl",
        "question": "糖尿病患者为什么需要关注 LDL-C？",
        "pubmed_query": (
            "type 2 diabetes LDL cholesterol cardiovascular risk guideline"
        ),
    },
]


def build_pure_llm_prompt(question: str) -> str:
    return f"""
你是一名谨慎的临床证据助手。

请回答下面的医学问题。你不能访问本项目的 RAG 证据库，因此如果不确定，请明确说明不确定。

要求：
1. 不要编造 PMID、NCT ID、指南名称、统计数字或引用。
2. 不要给个人诊断、处方剂量、停药建议或替代医生判断的结论。
3. 尽量区分生活方式、药物治疗、临床证据和证据局限。
4. 使用中文回答。

问题：
{question}
""".strip()


def call_llm(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            ".env 中没有读取到 OPENAI_API_KEY。"
        )

    client_kwargs = {
        "api_key": OPENAI_API_KEY,
    }

    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL

    client = OpenAI(**client_kwargs)

    request_body = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    if LLM_THINKING in {
        "enabled",
        "disabled",
    }:
        request_body["extra_body"] = {
            "thinking": {
                "type": LLM_THINKING,
            }
        }

    response = client.chat.completions.create(
        **request_body
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "大模型返回了空回答。"
        )

    return answer


def evaluate_answer(answer: str) -> dict:
    """
    轻量启发式评估，适合课堂演示和报告表格。
    """

    reference_ids = re.findall(
        r"\[\d+\]",
        answer,
    )

    pmids = re.findall(
        r"\bPMID[:：]?\s*\d+\b",
        answer,
        flags=re.IGNORECASE,
    )

    nct_ids = re.findall(
        r"\bNCT\d{8}\b",
        answer,
        flags=re.IGNORECASE,
    )

    has_guideline_marker = bool(
        re.search(
            r"Guideline ID|指南 ID|AHA|ACC|ESC|WHO",
            answer,
            flags=re.IGNORECASE,
        )
    )

    has_uncertainty = any(
        keyword in answer
        for keyword in [
            "不足",
            "无法确定",
            "不能确定",
            "证据有限",
            "局限",
            "不完全相关",
        ]
    )

    has_safety_boundary = any(
        keyword in answer
        for keyword in [
            "医生",
            "具体情况",
            "不能替代",
            "不替代",
            "临床判断",
        ]
    )

    has_structure = answer.count("##") >= 3

    has_no_prescription_warning = not any(
        keyword in answer
        for keyword in [
            "立即服用",
            "自行停药",
            "必须停药",
            "处方为",
        ]
    )

    real_reference_count = (
        len(pmids)
        + len(nct_ids)
        + int(has_guideline_marker)
    )

    score = sum(
        [
            bool(reference_ids),
            real_reference_count > 0,
            has_uncertainty,
            has_safety_boundary,
            has_structure,
            has_no_prescription_warning,
        ]
    )

    return {
        "has_numbered_reference": bool(reference_ids),
        "numbered_reference_count": len(reference_ids),
        "pmid_count": len(pmids),
        "nct_id_count": len(nct_ids),
        "has_guideline_marker": has_guideline_marker,
        "has_uncertainty": has_uncertainty,
        "has_safety_boundary": has_safety_boundary,
        "has_structure": has_structure,
        "has_no_prescription_warning": has_no_prescription_warning,
        "length": len(answer),
        "score": score,
    }


def run_pure_case(case: dict) -> dict:
    prompt = build_pure_llm_prompt(
        case["question"]
    )
    answer = call_llm(prompt)

    return {
        "case_id": case["id"],
        "mode": "pure",
        "question": case["question"],
        "answer": answer,
        "metrics": evaluate_answer(answer),
        "logs": "",
    }


def run_rag_case(
    case: dict,
    max_results: int,
    max_trials: int,
    max_guidelines: int,
    topk: int,
) -> dict:
    from main import evidence_agent

    log_buffer = io.StringIO()

    with contextlib.redirect_stdout(log_buffer):
        answer = evidence_agent(
            question=case["question"],
            pubmed_query=case["pubmed_query"],
            max_results=max_results,
            max_trials=max_trials,
            max_guidelines=max_guidelines,
            topk=topk,
            include_clinical_trials=max_trials > 0,
            include_guidelines=max_guidelines > 0,
            query_mode="fallback",
        )

    answer = answer or ""

    return {
        "case_id": case["id"],
        "mode": "rag",
        "question": case["question"],
        "pubmed_query": case["pubmed_query"],
        "answer": answer,
        "metrics": evaluate_answer(answer),
        "logs": log_buffer.getvalue(),
    }


def build_markdown_report(
    results: list[dict],
) -> str:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    lines = [
        "# RAG 对比评估结果",
        "",
        f"生成时间：{timestamp}",
        "",
        "## 汇总表",
        "",
        "| Case | Mode | Score | Numbered Ref | PMID | NCT | Guideline | Uncertainty | Safety | Structure | Length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        metrics = result["metrics"]
        lines.append(
            "| {case} | {mode} | {score} | {numbered} | {pmid} | {nct} | {guideline} | {uncertainty} | {safety} | {structure} | {length} |".format(
                case=result["case_id"],
                mode=result["mode"],
                score=metrics["score"],
                numbered=int(metrics["has_numbered_reference"]),
                pmid=metrics["pmid_count"],
                nct=metrics["nct_id_count"],
                guideline=int(metrics["has_guideline_marker"]),
                uncertainty=int(metrics["has_uncertainty"]),
                safety=int(metrics["has_safety_boundary"]),
                structure=int(metrics["has_structure"]),
                length=metrics["length"],
            )
        )

    lines.extend(
        [
            "",
            "## 评估维度说明",
            "",
            "- Numbered Ref：是否包含 `[1]` 这类证据编号。",
            "- PMID / NCT / Guideline：是否包含可追溯证据标识。",
            "- Uncertainty：是否说明证据不足或局限。",
            "- Safety：是否提醒不能替代医生判断或需结合具体情况。",
            "- Structure：是否有结构化小标题。",
            "",
            "## 逐题结果",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"### {result['case_id']} / {result['mode']}",
                "",
                f"问题：{result['question']}",
                "",
                "```text",
                result["answer"].strip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def save_results(
    results: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = output_dir / f"evaluation_{timestamp}.json"
    markdown_path = output_dir / f"evaluation_{timestamp}.md"

    json_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_path.write_text(
        build_markdown_report(results),
        encoding="utf-8",
    )

    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG 系统与纯大模型回答对比评估",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "pure",
            "rag",
            "both",
        ],
        default="both",
        help="评估模式。",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=len(TEST_CASES),
        help="最多评估多少个固定问题。",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="RAG 模式下 PubMed 文献数。",
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=3,
        help="RAG 模式下临床试验数。",
    )

    parser.add_argument(
        "--max-guidelines",
        type=int,
        default=3,
        help="RAG 模式下指南/共识条数。",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="RAG 模式下最终证据条数。",
    )

    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs",
        help="评估结果输出目录。",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    selected_cases = TEST_CASES[: max(args.limit, 1)]
    results: list[dict] = []

    for case in selected_cases:
        print(f"正在评估：{case['id']}")

        if args.mode in {
            "pure",
            "both",
        }:
            results.append(
                run_pure_case(case)
            )

        if args.mode in {
            "rag",
            "both",
        }:
            results.append(
                run_rag_case(
                    case=case,
                    max_results=max(
                        args.max_results,
                        1,
                    ),
                    max_trials=max(
                        args.max_trials,
                        0,
                    ),
                    max_guidelines=max(
                        args.max_guidelines,
                        0,
                    ),
                    topk=max(
                        args.topk,
                        1,
                    ),
                )
            )

    json_path, markdown_path = save_results(
        results=results,
        output_dir=Path(args.output_dir),
    )

    print(f"JSON 结果：{json_path}")
    print(f"Markdown 报告：{markdown_path}")


if __name__ == "__main__":
    main()
