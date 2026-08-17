from __future__ import annotations

import argparse
import os

from query_builder import generate_pubmed_query


DEFAULT_QUESTION = """
体检发现血脂偏高，
生活方式干预和药物治疗分别有哪些证据？
""".strip()

CLINICAL_TRIALS_ENABLED = os.getenv(
    "CLINICAL_TRIALS_ENABLED",
    "true",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

GUIDELINES_ENABLED = os.getenv(
    "GUIDELINES_ENABLED",
    "true",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def read_question_from_terminal() -> str:
    """
    从命令行读取中文医学问题。直接回车时使用演示问题。
    """

    print("请输入中文医学问题，直接回车将使用默认演示问题：")
    question = input("> ").strip()

    if question:
        return question

    return DEFAULT_QUESTION


def evidence_agent(
    question: str,
    pubmed_query: str | None = None,
    max_results: int = 5,
    max_trials: int = 3,
    max_guidelines: int = 3,
    topk: int = 5,
    include_clinical_trials: bool = True,
    include_guidelines: bool = True,
    query_mode: str = "auto",
    history: str = "",
    retrieval_mode: str = "hybrid",
    dense_weight: float = 1.0,
    use_rerank: bool = False,
) -> str | None:
    """
    question:
        用户原始问题，用于最终生成中文答案。

    pubmed_query:
        英文 PubMed 检索词。为空时自动由大模型生成。

    max_results:
        PubMed 返回多少篇文献。

    max_trials:
        ClinicalTrials.gov 返回多少项临床试验。

    max_guidelines:
        本地指南资料库返回多少条指南/共识证据。

    topk:
        RAG 最终检索多少条证据传给大模型。

    include_clinical_trials:
        是否检索并写入 ClinicalTrials.gov 证据。

    include_guidelines:
        是否检索并写入本地指南/共识证据。

    query_mode:
        auto：优先大模型生成 PubMed query，失败后用保底规则。
        llm：必须用大模型生成 PubMed query。
        fallback：只使用本地保底规则。

    history:
        多轮对话历史，用于上下文衔接和指代消解。
    """

    question = question.strip()

    if not question:
        print("问题不能为空。")
        return None, []

    print("=" * 60)
    print("用户问题：")
    print(question)

    if not pubmed_query:
        print("\n正在根据中文问题生成英文 PubMed 检索词...")

        pubmed_query = generate_pubmed_query(
            question=question,
            mode=query_mode,
            history=history,
        )

    pubmed_query = pubmed_query.strip()

    if not pubmed_query:
        print("英文 PubMed 检索词不能为空。")
        return None, []

    print("\n正在检索 PubMed...")
    print(f"英文检索词：{pubmed_query}")

    from pubmed import fetch_pubmed_articles, search_pubmed

    # 第一步：用英文关键词检索 PubMed
    ids = search_pubmed(
        keyword=pubmed_query,
        max_results=max_results,
    )

    print(f"PubMed 返回的 PMID：{ids}")

    # 必须先判断是否为空，不能直接使用 ids[0]
    if not ids:
        print("\nPubMed 没有返回文献。")
        print("请尝试更短、更通用的英文关键词。")
        return None, []

    print(f"\n成功找到 {len(ids)} 篇文献。")

    # 第二步：获取并解析文献详情
    print("正在获取并解析论文标题和摘要...")

    articles = fetch_pubmed_articles(ids)

    if not articles:
        print("没有解析到有效的 PubMed 文献内容。")
        return None, []

    # 第三步：存入 Chroma
    print("正在把证据写入向量数据库...")

    from vector_db import add_document

    written_count = 0

    for article in articles:
        pmid = article.get("pmid", "").strip()

        if not pmid:
            continue

        document_id = f"pubmed_{pmid}"

        chunk_count = add_document(
            text=article["text"],
            doc_id=document_id,
            metadata={
                "source_type": article["source_type"],
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
                "year": article["year"],
                "doi": article["doi"],
                "url": article["url"],
                "publication_types": article["publication_types"],
            },
        )

        written_count += chunk_count

        print(
            f"已写入 PubMed 文献 PMID: {pmid}，"
            f"文本块数量：{chunk_count}"
        )

    if written_count == 0:
        print("没有成功写入任何 PubMed 证据。")
        return None, []

    print(f"PubMed 证据已写入数据库，共 {written_count} 个文本块。")

    # 第四步：检索并写入 ClinicalTrials.gov 临床试验
    if include_clinical_trials and max_trials > 0:
        print("\n正在检索 ClinicalTrials.gov...")

        try:
            from clinical_trials import fetch_clinical_trials

            trials = fetch_clinical_trials(
                keyword=pubmed_query,
                max_results=max_trials,
            )
        except Exception as exc:
            print(
                "提示：ClinicalTrials.gov 检索失败，"
                f"将继续使用 PubMed 证据生成回答：{exc}"
            )
            trials = []

        if trials:
            print(f"成功找到 {len(trials)} 项临床试验。")
            print("正在把临床试验证据写入向量数据库...")

            trial_chunk_count = 0

            for trial in trials:
                nct_id = trial.get(
                    "nct_id",
                    "",
                ).strip()

                if not nct_id:
                    continue

                document_id = f"clinical_trial_{nct_id}"

                chunk_count = add_document(
                    text=trial["text"],
                    doc_id=document_id,
                    metadata={
                        "source_type": trial["source_type"],
                        "nct_id": trial["nct_id"],
                        "title": trial["title"],
                        "status": trial["status"],
                        "study_type": trial["study_type"],
                        "phase": trial["phase"],
                        "conditions": trial["conditions"],
                        "interventions": trial["interventions"],
                        "enrollment": trial["enrollment"],
                        "start_date": trial["start_date"],
                        "completion_date": trial["completion_date"],
                        "sponsor": trial["sponsor"],
                        "location": trial["location"],
                        "url": trial["url"],
                    },
                )

                trial_chunk_count += chunk_count

                print(
                    f"已写入临床试验 NCT ID: {nct_id}，"
                    f"文本块数量：{chunk_count}"
                )

            if trial_chunk_count:
                print(
                    "ClinicalTrials.gov 证据已写入数据库，"
                    f"共 {trial_chunk_count} 个文本块。"
                )
            else:
                print("没有成功写入任何临床试验证据。")
        else:
            print("ClinicalTrials.gov 没有返回相关临床试验。")

    # 第五步：检索并写入本地指南/共识资料
    if include_guidelines and max_guidelines > 0:
        print("\n正在检索本地指南/共识资料库...")

        try:
            from guidelines import fetch_guidelines

            guidelines = fetch_guidelines(
                query=f"{question} {pubmed_query}",
                max_results=max_guidelines,
            )
        except Exception as exc:
            print(
                "提示：指南/共识资料检索失败，"
                f"将继续使用已有证据生成回答：{exc}"
            )
            guidelines = []

        if guidelines:
            print(f"成功找到 {len(guidelines)} 条指南/共识证据。")
            print("正在把指南/共识证据写入向量数据库...")

            guideline_chunk_count = 0

            for guideline in guidelines:
                guideline_id = guideline.get(
                    "guideline_id",
                    "",
                ).strip()

                if not guideline_id:
                    continue

                document_id = f"guideline_{guideline_id}"

                chunk_count = add_document(
                    text=guideline["text"],
                    doc_id=document_id,
                    metadata={
                        "source_type": guideline["source_type"],
                        "guideline_id": guideline["guideline_id"],
                        "title": guideline["title"],
                        "organization": guideline["organization"],
                        "year": guideline["year"],
                        "topic": guideline["topic"],
                        "url": guideline["url"],
                        "pmid": guideline["pmid"],
                        "doi": guideline["doi"],
                    },
                )

                guideline_chunk_count += chunk_count

                print(
                    f"已写入指南/共识证据：{guideline_id}，"
                    f"文本块数量：{chunk_count}"
                )

            if guideline_chunk_count:
                print(
                    "指南/共识证据已写入数据库，"
                    f"共 {guideline_chunk_count} 个文本块。"
                )
            else:
                print("没有成功写入任何指南/共识证据。")
        else:
            print("本地指南/共识资料库没有返回相关条目。")

    # 第六步：RAG 检索并调用大模型回答
    print("\n正在生成带证据的回答...")

    from rag import answer_question

    result, final_evidence = answer_question(
        question,
        topk=topk,
        local_pdf_query=pubmed_query,
        local_pdf_topk=min(
            4,
            max(topk - 1, 0),
        ),
        history=history,
        retrieval_mode=retrieval_mode,
        dense_weight=dense_weight,
        use_rerank=use_rerank,
    )

    print("\n" + "=" * 60)
    print("最终回答：")
    print("=" * 60)
    print(result)

    return result, final_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenEvidence 风格的临床证据助手 MVP",
    )

    parser.add_argument(
        "-q",
        "--question",
        help="中文医学问题。不提供时进入交互式输入。",
    )

    parser.add_argument(
        "--pubmed-query",
        help="手动指定英文 PubMed 检索词，跳过自动 query 生成。",
    )

    parser.add_argument(
        "--query-mode",
        choices=[
            "auto",
            "llm",
            "fallback",
        ],
        default="auto",
        help=(
            "PubMed query 生成模式：auto=优先大模型，"
            "llm=必须大模型，fallback=本地规则。"
        ),
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="PubMed 最多返回多少篇文献。",
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=3,
        help="ClinicalTrials.gov 最多返回多少项临床试验。",
    )

    parser.add_argument(
        "--no-clinical-trials",
        action="store_true",
        help="不检索 ClinicalTrials.gov 临床试验。",
    )

    parser.add_argument(
        "--max-guidelines",
        type=int,
        default=3,
        help="本地指南/共识资料库最多返回多少条证据。",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="RAG 最终传给大模型的证据条数。",
    )

    parser.add_argument(
        "--no-guidelines",
        action="store_true",
        help="不检索本地指南/共识资料库。",
    )

    parser.add_argument(
        "--show-query-only",
        action="store_true",
        help="只生成并显示 PubMed query，不检索文献。",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    question = (
        args.question.strip()
        if args.question
        else read_question_from_terminal()
    )

    if args.pubmed_query:
        pubmed_query = args.pubmed_query.strip()
    else:
        pubmed_query = generate_pubmed_query(
            question=question,
            mode=args.query_mode,
        )

    if args.show_query_only:
        print(pubmed_query)
        return

    evidence_agent(
        question=question,
        pubmed_query=pubmed_query,
        max_results=max(
            args.max_results,
            1,
        ),
        max_trials=max(
            args.max_trials,
            0,
        ),
        max_guidelines=max(
            args.max_guidelines,
            0,
        ),
        topk=max(
            args.topk,
            1,
        ),
        include_clinical_trials=(
            CLINICAL_TRIALS_ENABLED
            and not args.no_clinical_trials
        ),
        include_guidelines=(
            GUIDELINES_ENABLED
            and not args.no_guidelines
        ),
        query_mode=args.query_mode,
    )


if __name__ == "__main__":
    main()
