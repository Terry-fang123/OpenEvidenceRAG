from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import chromadb
import pymupdf

from vector_db import DEVICE, model, split_text


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "pdf_check_report.csv"
PDF_DB_DIR = BASE_DIR / "chroma_pdf_db"
COLLECTION_NAME = "local_pdf_evidence"
LOG_PATH = BASE_DIR / "pdf_ingest_log.csv"


client = chromadb.PersistentClient(
    path=str(PDF_DB_DIR),
)

pdf_collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
)


def read_targets() -> list[dict]:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"找不到PDF检查报告：{REPORT_PATH}"
        )

    with REPORT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    targets = [
        row
        for row in rows
        if row.get("status") == "text"
    ]

    return targets


def extract_pdf_text(
    pdf_path: Path,
) -> tuple[str, int]:
    page_texts: list[str] = []

    with pymupdf.open(pdf_path) as document:
        page_count = document.page_count

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text("text").strip()

            if text:
                page_texts.append(
                    f"[Page {page_number}]\n{text}"
                )

    return "\n\n".join(page_texts), page_count


def infer_title(
    text: str,
    fallback: str,
) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith("[Page ")
    ]

    candidate_lines = [
        line
        for line in lines[:20]
        if 15 <= len(line) <= 250
    ]

    if not candidate_lines:
        return fallback

    return " ".join(candidate_lines[:3])[:500]


def already_imported(
    document_id: str,
) -> bool:
    result = pdf_collection.get(
        where={
            "source_id": document_id,
        },
        limit=1,
    )

    return bool(result.get("ids"))


def delete_existing(
    document_id: str,
) -> None:
    existing = pdf_collection.get(
        where={
            "source_id": document_id,
        }
    )

    existing_ids = existing.get("ids", [])

    if existing_ids:
        pdf_collection.delete(
            ids=existing_ids,
        )


def write_document(
    pdf_path: Path,
    force: bool = False,
) -> int:
    pdf_id = pdf_path.stem
    document_id = f"local_pdf_{pdf_id}"

    if already_imported(document_id):
        if not force:
            return 0

        delete_existing(document_id)

    text, page_count = extract_pdf_text(pdf_path)

    if len(text.strip()) < 200:
        raise ValueError(
            "PDF提取出的正文少于200个字符"
        )

    chunks = split_text(
        text,
        max_chars=1800,
        overlap=200,
    )

    if not chunks:
        raise ValueError(
            "PDF清洗和切分后没有有效文本"
        )

    embeddings = model.encode(
        chunks,
        device=DEVICE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    title = infer_title(
        text=text,
        fallback=pdf_path.name,
    )

    url = (
        f"https://pubmed.ncbi.nlm.nih.gov/{pdf_id}/"
        if pdf_id.isdigit()
        else ""
    )

    chunk_ids = [
        f"{document_id}_chunk_{index:04d}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source_id": document_id,
            "source_type": "local_pdf",
            "source_origin": "local_pdf_collection",
            "pdf_id": pdf_id,
            "pmid": pdf_id if pdf_id.isdigit() else "",
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "title": title,
            "url": url,
            "page_count": page_count,
            "chunk_index": index,
            "total_chunks": len(chunks),
        }
        for index in range(len(chunks))
    ]

    pdf_collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return len(chunks)


def append_log(
    file_name: str,
    status: str,
    chunk_count: int,
    message: str,
) -> None:
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time",
                "file_name",
                "status",
                "chunk_count",
                "message",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "time": datetime.now().isoformat(
                timespec="seconds"
            ),
            "file_name": file_name,
            "status": status,
            "chunk_count": chunk_count,
            "message": message,
        })


def show_statistics() -> None:
    result = pdf_collection.get(
        include=["metadatas"],
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    document_ids = {
        metadata.get("source_id")
        for metadata in metadatas
        if metadata
        and metadata.get("source_id")
    }

    print()
    print("本地PDF知识库统计")
    print("已入库文献数：", len(document_ids))
    print("已入库文本块数：", pdf_collection.count())
    print("数据库目录：", PDF_DB_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="本次最多处理多少篇；0表示全部。",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="重新导入已经存在的文献。",
    )

    args = parser.parse_args()

    targets = read_targets()

    if args.limit > 0:
        targets = targets[:args.limit]

    print("本次待检查文献数：", len(targets))
    print("已有文本块数：", pdf_collection.count())
    print()

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for index, row in enumerate(
        targets,
        start=1,
    ):
        pdf_path = Path(row["file_path"])

        print(
            f"[{index}/{len(targets)}] "
            f"{pdf_path.name}"
        )

        try:
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"文件不存在：{pdf_path}"
                )

            document_id = (
                f"local_pdf_{pdf_path.stem}"
            )

            existed_before = already_imported(
                document_id
            )

            chunk_count = write_document(
                pdf_path=pdf_path,
                force=args.force,
            )

            if existed_before and not args.force:
                skipped_count += 1
                print("  已存在，跳过。")

                append_log(
                    pdf_path.name,
                    "skipped",
                    0,
                    "already imported",
                )
            else:
                success_count += 1
                print(
                    f"  导入成功，文本块："
                    f"{chunk_count}"
                )

                append_log(
                    pdf_path.name,
                    "success",
                    chunk_count,
                    "",
                )

        except Exception as error:
            failed_count += 1
            print(f"  导入失败：{error}")

            append_log(
                pdf_path.name,
                "failed",
                0,
                str(error),
            )

    print()
    print("本次任务完成")
    print("成功：", success_count)
    print("跳过：", skipped_count)
    print("失败：", failed_count)

    show_statistics()


if __name__ == "__main__":
    main()