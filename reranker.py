from __future__ import annotations

import json
import os

from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")


def rerank_documents(
    question: str,
    documents: list[dict],
) -> list[dict]:
    """
    用 LLM 对文献级聚合结果逐篇打分 0-10。

    documents 中每个 dict 应已聚合到文献级，包含：
        - document: str   (文献标题 + 摘要/全文片段)
        - metadata: dict  (含 title、pmid / nct_id / guideline_id 等)

    返回带 'rerank_score' 字段的 documents 列表（保持原顺序）。
    如果 LLM 调用失败，所有文档统一赋分 5.0（中性分）。
    """

    if not documents:
        return []

    if not OPENAI_API_KEY:
        print("reranker: 未配置 OPENAI_API_KEY，统一赋分 5.0。")
        for doc in documents:
            doc["rerank_score"] = 5.0
        return documents

    # ---- 构建 Prompt ----
    doc_blocks: list[str] = []

    for idx, doc in enumerate(documents):
        title = (
            doc.get("metadata", {}).get("title")
            or doc.get("title", "Untitled")
        )

        text = doc.get("document", "")
        # 摘要截断，避免 Prompt 过长
        snippet = text[:500] if text else ""

        doc_blocks.append(
            f"Doc {idx + 1}:\n"
            f"Title: {title}\n"
            f"Content: {snippet}"
        )

    prompt = f"""
You are a clinical evidence relevance reviewer.

User question:
{question}

Score each document (0-10) on how well it directly answers the question:
- 0-2: completely irrelevant
- 3-5: topic-related but does not directly answer
- 6-8: directly relevant
- 9-10: precisely answers

Return ONLY a JSON object (no markdown, no extra text):
{{"scores": [9, 7, 3, ...]}}

One score per document, in the same order as listed above.

{chr(10).join(doc_blocks)}
""".strip()

    # ---- 调用 LLM ----
    client_kwargs: dict[str, str] = {
        "api_key": OPENAI_API_KEY,
    }

    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL

    client = OpenAI(**client_kwargs)

    try:
        print("reranker: 正在调用 LLM 对文献打分...")

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical evidence reviewer. "
                    "Always respond with valid JSON only. "
                    "No markdown, no explanation.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=500,
            temperature=0.0,
        )

        content = response.choices[0].message.content or "{}"

    except Exception as exc:
        print(f"reranker: LLM 调用失败 ({exc})，统一赋分 5.0。")
        for doc in documents:
            doc["rerank_score"] = 5.0
        return documents

    # ---- 解析 JSON ----
    try:
        content = content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("\n```", 1)[0]

        parsed = json.loads(content)

        if not isinstance(parsed, dict) or "scores" not in parsed:
            raise ValueError("Expected {{'scores': [...]}}")

        scores = parsed["scores"]

    except (json.JSONDecodeError, ValueError) as exc:
        print(f"reranker: JSON 解析失败 ({exc})，统一赋分 5.0。")
        print(f"reranker: LLM 原始输出: {content[:200]}")
        for doc in documents:
            doc["rerank_score"] = 5.0
        return documents

    # ---- 赋值 ----
    for idx, doc in enumerate(documents):
        if idx < len(scores):
            doc["rerank_score"] = float(scores[idx])
        else:
            doc["rerank_score"] = 5.0

    score_preview = ", ".join(
        f"{d.get('source_id', '?')}:{d['rerank_score']:.0f}"
        for d in documents
    )

    print(f"reranker: 打分完成 → [{score_preview}]")
    return documents
