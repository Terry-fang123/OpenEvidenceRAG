from __future__ import annotations

import os
import re
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

PUBMED_QUERY_MODE = os.getenv(
    "PUBMED_QUERY_MODE",
    "auto",
).strip().lower()


def build_query_prompt(question: str, history: str = "") -> str:
    """
    让大模型把中文临床问题改写成 PubMed 可用的英文检索式。

    history:
        可选的多轮对话历史，用于指代消解（如「它对肾脏有保护作用吗？」中的「它」）。
    """

    history_block = ""

    if history:
        history_block = (
            "Previous conversation:\n"
            f"{history}\n"
            "---\n"
            "The current question may contain pronouns (e.g. 'it', 'they') "
            "that refer to concepts mentioned above. "
            "Use the previous conversation to resolve these references "
            "before generating the query.\n"
        )

    return f"""
You are helping build a clinical evidence retrieval system.

Convert the user's Chinese clinical question into one concise English PubMed search query.

Requirements:
1. Output the query only. Do not explain.
2. Use English biomedical terms.
3. Prefer core concepts: disease, intervention, comparator, outcome, evidence type.
4. Use simple PubMed-friendly Boolean operators only when helpful.
5. Do not include markdown, numbering, citations, or Chinese text.
6. Keep the query under 25 words.

{history_block}User question:
{question}
""".strip()


def clean_pubmed_query(raw_query: str) -> str:
    """
    清理模型输出，避免把解释、代码块或引号带进 PubMed。
    """

    cleaned = raw_query.strip()

    cleaned = re.sub(
        r"^```(?:text|bash|json)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    cleaned = cleaned.removesuffix("```").strip()

    cleaned = re.sub(
        r"^(pubmed query|query|search query)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    lines = [
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
    ]

    if lines:
        cleaned = lines[0]

    cleaned = cleaned.strip("`\"' ")

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned


def fallback_pubmed_query(question: str) -> str:
    """
    大模型不可用时的保底检索词。覆盖赛道一常见问题。
    """

    question_lower = question.lower()
    terms: list[str] = []

    keyword_map = [
        (
            ["高血压", "血压", "hypertension"],
            [
                "hypertension",
                "antihypertensive therapy",
                "long-term treatment",
                "cardiovascular risk",
            ],
        ),
        (
            ["血脂", "胆固醇", "ldl", "他汀", "脂质", "dyslipidemia"],
            [
                "dyslipidemia",
                "lifestyle intervention",
                "statin therapy",
                "cardiovascular risk",
            ],
        ),
        (
            ["糖尿病", "血糖", "diabetes"],
            [
                "type 2 diabetes",
                "cardiovascular risk",
                "LDL cholesterol",
                "clinical guideline",
            ],
        ),
        (
            ["地中海", "mediterranean"],
            [
                "Mediterranean diet",
                "cardiovascular risk",
                "systematic review",
            ],
        ),
        (
            ["限钠", "低钠", "盐", "sodium"],
            [
                "sodium reduction",
                "hypertension",
                "blood pressure",
                "clinical trial",
            ],
        ),
    ]

    for keywords, english_terms in keyword_map:
        if any(
            keyword in question_lower
            for keyword in keywords
        ):
            terms.extend(english_terms)

    if not terms:
        terms.extend(
            [
                "clinical evidence",
                "guideline",
                "systematic review",
                "cardiovascular risk",
            ]
        )

    deduped_terms = list(dict.fromkeys(terms))

    return " ".join(deduped_terms)


def call_query_model(question: str, history: str = "") -> str:
    """
    调用配置的大模型生成英文 PubMed query。

    history:
        多轮对话历史，帮助 LLM 进行指代消解。
    """

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
                "content": build_query_prompt(question, history),
            }
        ],
        "temperature": 0,
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
            "大模型没有返回 PubMed 检索词。"
        )

    return clean_pubmed_query(answer)


def generate_pubmed_query(
    question: str,
    mode: str | None = None,
    history: str = "",
) -> str:
    """
    生成英文 PubMed query。

    mode:
    - auto：优先大模型，失败后使用保底规则
    - llm：必须调用大模型
    - fallback：只使用保底规则

    history:
        多轮对话历史，用于帮助 LLM 解决指代消解问题。
    """

    mode = (
        mode
        or PUBMED_QUERY_MODE
        or "auto"
    ).strip().lower()

    if mode == "fallback":
        return fallback_pubmed_query(question)

    try:
        query = call_query_model(question, history)

        if query:
            return query

        raise RuntimeError(
            "大模型返回的 PubMed 检索词为空。"
        )
    except Exception:
        if mode == "llm":
            raise

        return fallback_pubmed_query(question)


if __name__ == "__main__":
    test_question = input(
        "请输入中文医学问题："
    ).strip()

    print(
        generate_pubmed_query(test_question)
    )
