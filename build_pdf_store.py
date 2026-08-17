from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pymupdf

from vector_db import (
    DEVICE,
    MODEL_NAME,
    model,
    split_text,
)


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "pdf_check_report.csv"


def load_targets() -> list[dict]:
    with REPORT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return [
        row
        for row in rows
        if row.get("status") == "text"
    ]


def extract_pdf(
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

    candidates = [
        line
        for line in lines[:20]
        if 15 <= len(line) <= 250
    ]

    if not candidates:
        return fallback

    return " ".join(candidates[:3])[:500]


def build_store(
    targets: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_chunks: list[str] = []
    all_records: list[dict] = []
    failures: list[dict] = []
    document_count = 0

    print("第一阶段：提取和切分PDF")

    for index, row in enumerate(
        targets,
        start=1,
    ):
        pdf_path = Path(row["file_path"])

        try:
            if not pdf_path.exists():
                raise FileNotFoundError(pdf_path)

            text, page_count = extract_pdf(pdf_path)

            if len(text.strip()) < 200:
                raise ValueError(
                    "提取正文少于200个字符"
                )

            chunks = split_text(
                text,
                max_chars=1800,
                overlap=200,
            )

            if not chunks:
                raise ValueError(
                    "没有生成有效文本块"
                )

            pdf_id = pdf_path.stem
            source_id = f"local_pdf_{pdf_id}"
            title = infer_title(
                text,
                pdf_path.name,
            )

            url = (
                f"https://pubmed.ncbi.nlm.nih.gov/{pdf_id}/"
                if pdf_id.isdigit()
                else ""
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):
                all_chunks.append(chunk)

                all_records.append({
                    "source_id": source_id,
                    "source_type": "local_pdf",
                    "source_origin": "local_pdf_collection",
                    "pdf_id": pdf_id,
                    "pmid": (
                        pdf_id
                        if pdf_id.isdigit()
                        else ""
                    ),
                    "title": title,
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "url": url,
                    "page_count": page_count,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "document": chunk,
                })

            document_count += 1

        except Exception as error:
            failures.append({
                "file_name": pdf_path.name,
                "error": str(error),
            })

        if index % 25 == 0 or index == len(targets):
            print(
                f"PDF进度：{index}/{len(targets)}，"
                f"当前文本块：{len(all_chunks)}"
            )

    if not all_chunks:
        raise RuntimeError(
            "没有提取到任何有效文本块"
        )

    print()
    print("第二阶段：批量生成向量")
    print("文献数：", document_count)
    print("文本块数：", len(all_chunks))

    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        device=DEVICE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.shape[0] != len(all_records):
        raise RuntimeError(
            "向量数量与文本记录数量不一致"
        )

    print()
    print("第三阶段：保存知识库")

    temporary_embeddings = (
        output_dir / "embeddings.tmp.npy"
    )
    final_embeddings = (
        output_dir / "embeddings.npy"
    )

    np.save(
        temporary_embeddings,
        embeddings,
    )
    temporary_embeddings.replace(
        final_embeddings
    )

    temporary_documents = (
        output_dir / "documents.tmp.jsonl"
    )
    final_documents = (
        output_dir / "documents.jsonl"
    )

    with temporary_documents.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in all_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_documents.replace(
        final_documents
    )

    manifest = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "embedding_model": MODEL_NAME,
        "embedding_device": DEVICE,
        "document_count": document_count,
        "chunk_count": len(all_records),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "failed_count": len(failures),
        "failures": failures,
    }

    manifest_path = (
        output_dir / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("开始验证保存结果")

    loaded_embeddings = np.load(
        final_embeddings,
        mmap_mode="r",
    )

    with final_documents.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_record_count = sum(
            1
            for _ in file
        )

    if loaded_embeddings.shape[0] != saved_record_count:
        raise RuntimeError(
            "保存后的向量数量与记录数量不一致"
        )

    print()
    print("本地PDF知识库构建完成")
    print("文献数：", document_count)
    print("文本块数：", saved_record_count)
    print("向量形状：", loaded_embeddings.shape)
    print("失败数：", len(failures))
    print("保存目录：", output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        default="local_pdf_store",
    )

    args = parser.parse_args()

    targets = load_targets()

    if args.limit > 0:
        targets = targets[:args.limit]

    build_store(
        targets=targets,
        output_dir=BASE_DIR / args.output_dir,
    )


if __name__ == "__main__":
    main()