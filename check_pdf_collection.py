from pathlib import Path
import csv
import pymupdf

PDF_DIR = Path(r"D:\论坛\线下\3\500-collection")
REPORT_PATH = Path("pdf_check_report.csv")
MIN_TEXT_CHARS = 200

pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
results = []

print(f"开始检查，共发现 {len(pdf_files)} 篇PDF。")

for index, pdf_path in enumerate(pdf_files, start=1):
    try:
        with pymupdf.open(pdf_path) as document:
            page_count = document.page_count
            text_chars = sum(
                len(page.get_text("text").strip())
                for page in document
            )

        status = (
            "text"
            if text_chars >= MIN_TEXT_CHARS
            else "possible_scan"
        )

        results.append({
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "page_count": page_count,
            "text_chars": text_chars,
            "status": status,
            "error": "",
        })

    except Exception as error:
        results.append({
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "page_count": 0,
            "text_chars": 0,
            "status": "failed",
            "error": str(error),
        })

    if index % 25 == 0 or index == len(pdf_files):
        print(f"检查进度：{index}/{len(pdf_files)}")

with REPORT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "file_name",
            "file_path",
            "page_count",
            "text_chars",
            "status",
            "error",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

text_count = sum(
    item["status"] == "text"
    for item in results
)
scan_count = sum(
    item["status"] == "possible_scan"
    for item in results
)
failed_count = sum(
    item["status"] == "failed"
    for item in results
)

print()
print("检查完成")
print("PDF总数：", len(pdf_files))
print("可直接解析：", text_count)
print("可能是扫描版：", scan_count)
print("读取失败：", failed_count)
print("检查报告：", REPORT_PATH.resolve())
