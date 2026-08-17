from pathlib import Path
import json

from vector_db import collection, model, DEVICE


JSON_PATH = Path("team_evidence.json")
BATCH_SIZE = 50


if not JSON_PATH.exists():
    raise FileNotFoundError(
        f"没有找到数据文件：{JSON_PATH.resolve()}"
    )

data = json.loads(
    JSON_PATH.read_text(encoding="utf-8-sig")
)

all_records = data.get("documents", [])

# 只导入500条PubMed摘要。
# 483条PubMed PDF已经存在于local_pdf_store中，不重复导入。
pubmed_records = [
    item
    for item in all_records
    if str(item.get("source_type", "")).strip().lower()
    == "pubmed"
]

prepared = {}

for item in pubmed_records:
    identifier = str(
        item.get("identifier", "")
    ).strip()

    pmid = identifier.replace("PMID:", "").strip()
    title = str(item.get("title", "")).strip()
    summary = str(item.get("summary", "")).strip()
    url = str(item.get("url", "")).strip()

    if not pmid or not summary:
        continue

    prepared[pmid] = {
        "pmid": pmid,
        "title": title,
        "summary": summary,
        "url": url,
    }

print("JSON全部记录：", len(all_records))
print("PubMed摘要记录：", len(pubmed_records))
print("有效且去重后的PubMed记录：", len(prepared))

# 读取现有Chroma数据，按PMID去重
existing_result = collection.get(
    include=["metadatas"]
)

existing_metadatas = existing_result.get(
    "metadatas",
    [],
)

existing_pmids = {
    str(metadata.get("pmid", "")).strip()
    for metadata in existing_metadatas
    if metadata
}

pending = [
    item
    for pmid, item in prepared.items()
    if pmid not in existing_pmids
]

print("当前数据库文本块数：", collection.count())
print("已有PMID数量：", len(existing_pmids))
print("本次需要新增：", len(pending))

if not pending:
    print("没有需要新增的文献，可能已经导入过。")
    raise SystemExit(0)

success_count = 0

for start in range(0, len(pending), BATCH_SIZE):
    batch = pending[start:start + BATCH_SIZE]

    documents = [
        (
            f"Title: {item['title']}\n"
            f"Abstract: {item['summary']}"
        )
        for item in batch
    ]

    embeddings = model.encode(
        documents,
        device=DEVICE,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ids = [
        f"team_pubmed_{item['pmid']}"
        for item in batch
    ]

    metadatas = [
        {
            "source_id": f"pubmed_{item['pmid']}",
            "source_type": "pubmed",
            "pmid": item["pmid"],
            "title": item["title"],
            "url": item["url"],
            "chunk_index": 0,
            "identifier": f"PMID:{item['pmid']}",
        }
        for item in batch
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    success_count += len(batch)

    print(
        f"导入进度：{success_count}/{len(pending)}"
    )

print()
print("队友文献导入完成")
print("本次新增PubMed文献：", success_count)
print("数据库当前文本块数：", collection.count())
