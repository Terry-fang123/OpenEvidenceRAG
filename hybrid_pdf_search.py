from __future__ import annotations

import math
import re

import numpy as np

from pdf_knowledge_base import (
    documents,
    embeddings,
    get_statistics,
)
from vector_db import DEVICE, model


GENERIC_QUERY_TERMS = {
    "evidence",
    "study",
    "studies",
    "research",
    "systematic",
    "review",
    "reviews",
    "randomized",
    "trial",
    "trials",
    "clinical",
    "guideline",
    "guidelines",
    "treatment",
    "therapy",
    "therapies",
    "outcome",
    "outcomes",
    "effect",
    "effects",
    "long",
    "term",
    "long-term",
    "drug",
    "drugs",
    "medication",
    "medications",
    "patient",
    "patients",
    "management",
    "cardiovascular",
}


IMPORTANT_PHRASES = [
    "blood pressure",
    "blood pressure control",
    "antihypertensive therapy",
    "antihypertensive treatment",
    "long-term treatment",
    "ldl cholesterol",
    "statin therapy",
    "mediterranean diet",
    "sodium reduction",
    "type 2 diabetes",
    "cardiovascular risk",
]


SEARCH_TEXTS = [
    (
        str(record.get("title", ""))
        + "\n"
        + str(record.get("document", ""))
    ).lower()
    for record in documents
]


def tokenize_query(
    query: str,
) -> list[str]:
    tokens = re.findall(
        r"[a-z][a-z0-9-]{2,}",
        query.lower(),
    )

    unique_tokens = list(
        dict.fromkeys(tokens)
    )

    core_terms = [
        token
        for token in unique_tokens
        if token not in GENERIC_QUERY_TERMS
        and len(token) >= 4
    ]

    return core_terms


def calculate_idf(
    terms: list[str],
) -> dict[str, float]:
    document_total = len(SEARCH_TEXTS)
    idf: dict[str, float] = {}

    for term in terms:
        document_frequency = sum(
            term in text
            for text in SEARCH_TEXTS
        )

        idf[term] = (
            math.log(
                (document_total + 1)
                / (document_frequency + 1)
            )
            + 1.0
        )

    return idf


def reference_penalty(
    text: str,
) -> float:
    lowered = text.lower()

    citation_count = (
        lowered.count(" et al.")
        + len(
            re.findall(
                r"\b(?:19|20)\d{2};\s*\d+",
                lowered,
            )
        )
    )

    starts_as_references = lowered.lstrip().startswith(
        (
            "references",
            "bibliography",
            "参考文献",
        )
    )

    if starts_as_references:
        return 0.20

    if citation_count >= 10:
        return 0.16

    if citation_count >= 6:
        return 0.10

    return 0.0


def search_local_pdfs_hybrid(
    query: str,
    topk: int = 5,
    minimum_score: float = 0.65,
) -> list[dict]:
    query = query.strip()

    if not query or topk <= 0:
        return []

    query_embedding = model.encode(
        query,
        device=DEVICE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    dense_scores = embeddings @ query_embedding

    core_terms = tokenize_query(query)
    idf = calculate_idf(core_terms)

    total_idf = sum(idf.values()) or 1.0

    query_lower = query.lower()

    active_phrases = [
        phrase
        for phrase in IMPORTANT_PHRASES
        if phrase in query_lower
    ]

    combined_scores = np.empty(
        len(documents),
        dtype=np.float32,
    )

    lexical_scores = np.zeros(
        len(documents),
        dtype=np.float32,
    )

    for index, record in enumerate(documents):
        search_text = SEARCH_TEXTS[index]

        matched_weight = sum(
            idf[term]
            for term in core_terms
            if term in search_text
        )

        lexical_score = (
            matched_weight / total_idf
            if core_terms
            else 0.0
        )

        phrase_matches = sum(
            phrase in search_text
            for phrase in active_phrases
        )

        if active_phrases:
            lexical_score += (
                phrase_matches
                / len(active_phrases)
            ) * 0.20

        title = str(
            record.get("title", "")
        ).lower()

        if core_terms and any(
            term in title
            for term in core_terms
        ):
            lexical_score += 0.08

        lexical_score = min(
            lexical_score,
            1.0,
        )

        penalty = reference_penalty(
            str(record.get("document", ""))
        )

        combined_score = (
            0.65 * float(dense_scores[index])
            + 0.35 * lexical_score
            - penalty
        )

        if core_terms and matched_weight == 0:
            combined_score -= 0.25

        lexical_scores[index] = lexical_score
        combined_scores[index] = combined_score

    candidate_indices = np.argsort(
        combined_scores
    )[::-1]

    results: list[dict] = []
    used_source_ids: set[str] = set()

    for raw_index in candidate_indices:
        index = int(raw_index)
        score = float(
            combined_scores[index]
        )

        if score < minimum_score:
            break

        record = documents[index]

        source_id = str(
            record.get("source_id", "")
        )

        if not source_id:
            continue

        if source_id in used_source_ids:
            continue

        used_source_ids.add(source_id)

        metadata = {
            key: value
            for key, value in record.items()
            if key != "document"
        }

        results.append({
            "document": record.get(
                "document",
                "",
            ),
            "metadata": metadata,
            "source_id": source_id,
            "source_type": "local_pdf",
            "pmid": record.get("pmid", ""),
            "pdf_id": record.get("pdf_id", ""),
            "file_name": record.get(
                "file_name",
                "",
            ),
            "title": record.get("title", ""),
            "url": record.get("url", ""),
            "chunk_index": record.get(
                "chunk_index",
                -1,
            ),
            "score": score,
            "dense_score": float(
                dense_scores[index]
            ),
            "lexical_score": float(
                lexical_scores[index]
            ),
            "title_term_matches": sum(
                term in str(
                    record.get("title", "")
                ).lower()
                for term in core_terms
            ),
            "distance": 1.0 - score,
        })

        if len(results) >= topk:
            break

    return results


if __name__ == "__main__":
    statistics = get_statistics()

    print("混合检索本地PDF知识库")
    print("文献数：", statistics["document_count"])
    print("文本块数：", statistics["chunk_count"])

    question = input(
        "\n请输入英文检索词："
    ).strip()

    results = search_local_pdfs_hybrid(
        question,
        topk=10,
    )

    print(f"\n检索到 {len(results)} 篇文献：")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 70)
        print("结果：", index)
        print("PMID：", result["pmid"])
        print("标题：", result["title"])
        print(
            "综合得分：",
            round(result["score"], 4),
        )
        print(
            "向量得分：",
            round(result["dense_score"], 4),
        )
        print(
            "关键词得分：",
            round(result["lexical_score"], 4),
        )
        print("内容：")
        print(result["document"][:400])