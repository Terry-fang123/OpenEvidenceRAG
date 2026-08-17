from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vector_db import DEVICE, model


BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "evidence_store"

EMBEDDINGS_PATH = STORE_DIR / "embeddings.npy"
DOCUMENTS_PATH = STORE_DIR / "documents.jsonl"
MANIFEST_PATH = STORE_DIR / "manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"找不到知识库清单：{MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_documents() -> list[dict]:
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"找不到文献记录：{DOCUMENTS_PATH}"
        )

    records: list[dict] = []

    with DOCUMENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


if not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(
        f"找不到向量文件：{EMBEDDINGS_PATH}"
    )


manifest = load_manifest()

embeddings = np.load(
    EMBEDDINGS_PATH,
    mmap_mode="r",
)

documents = load_documents()


if embeddings.shape[0] != len(documents):
    raise RuntimeError(
        "向量数量与文献记录数量不一致："
        f"{embeddings.shape[0]} != {len(documents)}"
    )


def get_statistics() -> dict:
    return {
        "document_count": manifest.get(
            "document_count",
            0,
        ),
        "chunk_count": manifest.get(
            "chunk_count",
            len(documents),
        ),
        "embedding_dimension": manifest.get(
            "embedding_dimension",
            embeddings.shape[1],
        ),
        "embedding_model": manifest.get(
            "embedding_model",
            "",
        ),
        "database_path": str(STORE_DIR),
    }


def search_local_pdfs(
    query: str,
    topk: int = 5,
) -> list[dict]:
    query = query.strip()

    if not query:
        return []

    if topk <= 0:
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

    # 所有文献向量和查询向量均已归一化，
    # 点积结果就是余弦相似度。
    scores = embeddings @ query_embedding

    # 多取一些候选文本块，随后按文献去重，
    # 避免最终结果全部来自同一篇PDF。
    candidate_count = min(
        max(topk * 20, 100),
        len(documents),
    )

    candidate_indices = np.argsort(
        scores
    )[::-1][:candidate_count]

    results: list[dict] = []
    used_source_ids: set[str] = set()

    for index in candidate_indices:
        record = documents[int(index)]
        source_id = str(
            record.get("source_id", "")
        )

        if not source_id:
            continue

        if source_id in used_source_ids:
            continue

        used_source_ids.add(source_id)

        score = float(scores[int(index)])

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
            "distance": 1.0 - score,
        })

        if len(results) >= topk:
            break

    return results


if __name__ == "__main__":
    statistics = get_statistics()

    print("本地PDF知识库")
    print("文献数：", statistics["document_count"])
    print("文本块数：", statistics["chunk_count"])
    print(
        "向量维度：",
        statistics["embedding_dimension"],
    )
    print("数据库：", statistics["database_path"])

    question = input(
        "\n请输入测试问题："
    ).strip()

    results = search_local_pdfs(
        query=question,
        topk=5,
    )

    print(f"\n检索到 {len(results)} 篇文献：")

    for result_index, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        print("\n" + "=" * 70)
        print(f"结果 {result_index}")
        print("PMID：", metadata.get("pmid", ""))
        print("标题：", metadata.get("title", ""))
        print(
            "文件：",
            metadata.get("file_name", ""),
        )
        print(
            "文本块：",
            metadata.get("chunk_index", -1),
        )
        print(
            "相似度：",
            round(result["score"], 4),
        )
        print("内容：")
        print(result["document"][:500])