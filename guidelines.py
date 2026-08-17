from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )


DEFAULT_GUIDELINES_PATH = (
    BASE_DIR
    / "guidelines"
    / "clinical_guidelines.json"
)

_configured_guidelines_path = Path(
    os.getenv(
        "GUIDELINES_PATH",
        str(DEFAULT_GUIDELINES_PATH),
    )
)

GUIDELINES_PATH = (
    _configured_guidelines_path
    if _configured_guidelines_path.is_absolute()
    else BASE_DIR / _configured_guidelines_path
)


def load_guidelines(
    path: Path | None = None,
) -> list[dict]:
    """
    读取本地指南/共识资料库。
    """

    guideline_path = path or GUIDELINES_PATH

    if not guideline_path.exists():
        print(
            f"提示：没有找到指南资料库：{guideline_path}"
        )
        return []

    with guideline_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "指南资料库必须是 JSON list。"
        )

    return [
        item
        for item in data
        if isinstance(item, dict)
    ]


def guideline_score(
    guideline: dict,
    query: str,
) -> int:
    """
    根据关键词做轻量相关性评分。
    """

    query_lower = query.lower()
    score = 0

    searchable_fields = [
        "title",
        "organization",
        "topic",
        "summary",
    ]

    for field_name in searchable_fields:
        field_value = str(
            guideline.get(field_name, "")
        ).lower()

        if field_value and field_value in query_lower:
            score += 2

    keywords = guideline.get(
        "keywords",
        [],
    )

    if isinstance(keywords, list):
        for keyword in keywords:
            keyword_text = str(
                keyword
            ).strip().lower()

            if keyword_text and keyword_text in query_lower:
                score += 5

    return score


def format_guideline_text(
    guideline: dict,
) -> str:
    """
    将结构化指南条目转成可切分、可检索的证据文本。
    """

    fields = [
        ("Guideline ID", "guideline_id"),
        ("Title", "title"),
        ("Organization", "organization"),
        ("Year", "year"),
        ("Topic", "topic"),
        ("PMID", "pmid"),
        ("DOI", "doi"),
        ("URL", "url"),
        ("Summary", "summary"),
        ("Limitations", "limitations"),
    ]

    lines: list[str] = []

    for label, field_name in fields:
        value = str(
            guideline.get(field_name, "")
        ).strip()

        if value:
            lines.append(f"{label}: {value}")

    return "\n".join(lines)


def normalize_guideline(
    guideline: dict,
) -> dict:
    """
    补齐入库所需字段。
    """

    normalized = {
        "source_type": "guideline",
        "guideline_id": str(
            guideline.get("guideline_id", "")
        ).strip(),
        "title": str(
            guideline.get("title", "")
        ).strip(),
        "organization": str(
            guideline.get("organization", "")
        ).strip(),
        "year": str(
            guideline.get("year", "")
        ).strip(),
        "topic": str(
            guideline.get("topic", "")
        ).strip(),
        "url": str(
            guideline.get("url", "")
        ).strip(),
        "pmid": str(
            guideline.get("pmid", "")
        ).strip(),
        "doi": str(
            guideline.get("doi", "")
        ).strip(),
        "summary": str(
            guideline.get("summary", "")
        ).strip(),
        "limitations": str(
            guideline.get("limitations", "")
        ).strip(),
    }

    normalized["text"] = format_guideline_text(
        normalized
    )

    return normalized


def fetch_guidelines(
    query: str,
    max_results: int = 3,
) -> list[dict]:
    """
    从本地指南资料库中检索相关条目。
    """

    guidelines = load_guidelines()

    scored_guidelines = [
        (
            guideline_score(
                guideline,
                query,
            ),
            guideline,
        )
        for guideline in guidelines
    ]

    matched_guidelines = [
        (
            score,
            guideline,
        )
        for score, guideline in scored_guidelines
        if score > 0
    ]

    if not matched_guidelines:
        matched_guidelines = scored_guidelines

    matched_guidelines.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        normalize_guideline(guideline)
        for _, guideline in matched_guidelines[:max_results]
    ]


if __name__ == "__main__":
    query = input(
        "请输入问题或检索词："
    ).strip()

    for guideline in fetch_guidelines(query):
        print(guideline)
