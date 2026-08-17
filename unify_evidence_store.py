from pathlib import Path
from datetime import datetime
import json
import shutil

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
OLD_STORE = BASE_DIR / "local_pdf_store"
NEW_STORE = BASE_DIR / "evidence_store"
PDF_KB_PATH = BASE_DIR / "pdf_knowledge_base.py"

ROOT_JSON = BASE_DIR / "team_evidence.json"
SAVED_JSON = BASE_DIR / "data_sources" / "team_evidence.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BUILD_DIR = BASE_DIR / f"evidence_store_build_{timestamp}"
BACKUP_DIR = (
    BASE_DIR
    / "migration_backups"
    / f"unify_evidence_{timestamp}"
)


# ============================================================
# 前置检查
# ============================================================

if NEW_STORE.exists():
    raise FileExistsError(
        f"目标目录已经存在：{NEW_STORE}"
    )

if not OLD_STORE.exists():
    raise FileNotFoundError(
        f"没有找到原知识库：{OLD_STORE}"
    )

if ROOT_JSON.exists():
    source_json = ROOT_JSON
elif SAVED_JSON.exists():
    source_json = SAVED_JSON
else:
    raise FileNotFoundError(
        "没有找到 team_evidence.json"
    )

embeddings_path = OLD_STORE / "embeddings.npy"
documents_path = OLD_STORE / "documents.jsonl"
manifest_path = OLD_STORE / "manifest.json"

for path in (
    embeddings_path,
    documents_path,
    manifest_path,
    PDF_KB_PATH,
):
    if not path.exists():
        raise FileNotFoundError(path)

code_text = PDF_KB_PATH.read_text(
    encoding="utf-8-sig"
)

old_code_path = 'BASE_DIR / "local_pdf_store"'
new_code_path = 'BASE_DIR / "evidence_store"'

if code_text.count(old_code_path) != 1:
    raise RuntimeError(
        "pdf_knowledge_base.py中的路径不符合预期，"
        "迁移已停止。"
    )


# ============================================================
# 加载483篇PDF知识库
# ============================================================

print("第一阶段：加载现有PDF知识库")

old_embeddings = np.load(
    embeddings_path,
    allow_pickle=False,
)

old_documents = []

with documents_path.open(
    "r",
    encoding="utf-8-sig",
) as file:
    for line in file:
        line = line.strip()

        if line:
            old_documents.append(
                json.loads(line)
            )

old_manifest = json.loads(
    manifest_path.read_text(
        encoding="utf-8-sig"
    )
)

if len(old_documents) != old_embeddings.shape[0]:
    raise RuntimeError(
        "现有documents.jsonl与embeddings.npy数量不一致。"
    )

print("现有PDF文献数：", old_manifest["document_count"])
print("现有文本块数：", len(old_documents))
print("现有向量形状：", old_embeddings.shape)


# ============================================================
# 加载并筛选500篇PubMed摘要
# ============================================================

print()
print("第二阶段：读取队友补充数据")

team_data = json.loads(
    source_json.read_text(
        encoding="utf-8-sig"
    )
)

team_records = team_data.get("documents", [])

pubmed_records = [
    item
    for item in team_records
    if str(
        item.get("source_type", "")
    ).strip().lower() == "pubmed"
]

existing_pmids = {
    str(item.get("pmid", "")).strip()
    for item in old_documents
    if str(item.get("pmid", "")).strip()
}

prepared = {}

for item in pubmed_records:
    identifier = str(
        item.get("identifier", "")
    ).strip()

    pmid = identifier.replace(
        "PMID:",
        "",
    ).strip()

    title = str(
        item.get("title", "")
    ).strip()

    summary = str(
        item.get("summary", "")
    ).strip()

    url = str(
        item.get("url", "")
    ).strip()

    if (
        not pmid
        or not summary
        or pmid in existing_pmids
    ):
        continue

    prepared[pmid] = {
        "source_id": f"team_pubmed_{pmid}",
        "source_type": "pubmed",
        "source_origin": "team_evidence_json",
        "pdf_id": "",
        "pmid": pmid,
        "title": title,
        "file_name": "",
        "file_path": "",
        "url": url,
        "page_count": 0,
        "chunk_index": 0,
        "total_chunks": 1,
        "document": (
            f"Title: {title}\n"
            f"Abstract: {summary}"
        ),
    }

new_documents = list(prepared.values())

print("队友JSON总记录：", len(team_records))
print("其中PubMed摘要：", len(pubmed_records))
print("去重后需要追加：", len(new_documents))

if not new_documents:
    raise RuntimeError(
        "没有找到需要追加的PubMed摘要。"
    )


# ============================================================
# 为500篇摘要生成向量
# ============================================================

print()
print("第三阶段：生成PubMed摘要向量")

model_name = old_manifest[
    "embedding_model"
]

device = old_manifest.get(
    "embedding_device",
    "cpu",
)

print("向量模型：", model_name)
print("运行设备：", device)

model = SentenceTransformer(
    model_name,
    device=device,
)

new_texts = [
    item["document"]
    for item in new_documents
]

new_embeddings = model.encode(
    new_texts,
    batch_size=32,
    device=device,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype(np.float32)

if (
    new_embeddings.ndim != 2
    or new_embeddings.shape[1]
    != old_embeddings.shape[1]
):
    raise RuntimeError(
        "新旧向量维度不一致。"
    )

combined_embeddings = np.vstack([
    old_embeddings.astype(np.float32),
    new_embeddings,
])

combined_documents = [
    *old_documents,
    *new_documents,
]


# ============================================================
# 保存并验证统一知识库
# ============================================================

print()
print("第四阶段：保存统一知识库")

BUILD_DIR.mkdir(parents=True)

np.save(
    BUILD_DIR / "embeddings.npy",
    combined_embeddings,
    allow_pickle=False,
)

with (
    BUILD_DIR / "documents.jsonl"
).open(
    "w",
    encoding="utf-8",
) as file:
    for item in combined_documents:
        file.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

unique_source_ids = {
    str(item.get("source_id", ""))
    for item in combined_documents
    if item.get("source_id")
}

combined_manifest = {
    **old_manifest,
    "updated_at": datetime.now().isoformat(
        timespec="seconds"
    ),
    "store_type": "unified_static_evidence",
    "document_count": len(unique_source_ids),
    "pdf_document_count": old_manifest[
        "document_count"
    ],
    "pubmed_abstract_count": len(new_documents),
    "chunk_count": len(combined_documents),
    "embedding_dimension": int(
        combined_embeddings.shape[1]
    ),
    "source_json": (
        "data_sources/team_evidence.json"
    ),
}

(
    BUILD_DIR / "manifest.json"
).write_text(
    json.dumps(
        combined_manifest,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

# 重新读取验证
verified_embeddings = np.load(
    BUILD_DIR / "embeddings.npy",
    allow_pickle=False,
)

verified_line_count = 0

with (
    BUILD_DIR / "documents.jsonl"
).open(
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        if line.strip():
            json.loads(line)
            verified_line_count += 1

if verified_line_count != verified_embeddings.shape[0]:
    raise RuntimeError(
        "统一知识库保存验证失败。"
    )

if not np.isfinite(verified_embeddings).all():
    raise RuntimeError(
        "统一知识库包含无效向量。"
    )

print("统一文献数：", len(unique_source_ids))
print("统一文本块数：", verified_line_count)
print("统一向量形状：", verified_embeddings.shape)


# ============================================================
# 迁移目录并修改运行路径
# ============================================================

print()
print("第五阶段：完成目录迁移")

BACKUP_DIR.mkdir(parents=True)

shutil.copy2(
    PDF_KB_PATH,
    BACKUP_DIR / "pdf_knowledge_base.py",
)

BUILD_DIR.rename(NEW_STORE)

shutil.move(
    str(OLD_STORE),
    str(BACKUP_DIR / "local_pdf_store"),
)

data_sources_dir = BASE_DIR / "data_sources"
data_sources_dir.mkdir(exist_ok=True)

if source_json == ROOT_JSON:
    shutil.move(
        str(ROOT_JSON),
        str(SAVED_JSON),
    )

new_code_text = code_text.replace(
    old_code_path,
    new_code_path,
    1,
)

PDF_KB_PATH.write_text(
    new_code_text,
    encoding="utf-8",
)

print()
print("统一知识库迁移完成")
print("知识库位置：", NEW_STORE)
print("原知识库备份：", BACKUP_DIR)
print("原始数据位置：", SAVED_JSON)
print("文献数：", len(unique_source_ids))
print("文本块数：", verified_line_count)
print(
    "向量维度：",
    verified_embeddings.shape[1],
)
