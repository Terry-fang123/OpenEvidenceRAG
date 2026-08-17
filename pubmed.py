from __future__ import annotations

import xml.etree.ElementTree as ET

import requests


PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _text(element: ET.Element | None) -> str:
    """
    提取 XML 节点中的全部文本，兼容标题里嵌套标签的情况。
    """

    if element is None:
        return ""

    return " ".join("".join(element.itertext()).split())


def _first_text(
    root: ET.Element,
    paths: list[str],
) -> str:
    """
    按多个候选 XPath 查找第一个非空文本。
    """

    for path in paths:
        value = _text(root.find(path))

        if value:
            return value

    return ""


def _article_id(
    article: ET.Element,
    id_type: str,
) -> str:
    """
    从 ArticleIdList 中提取 PMID、DOI 等 ID。
    """

    for article_id in article.findall(".//ArticleId"):
        if article_id.attrib.get("IdType") == id_type:
            return _text(article_id)

    return ""


def _publication_year(article: ET.Element) -> str:
    """
    提取出版年份。PubMed 有时只有 MedlineDate，所以需要兜底。
    """

    year = _first_text(
        article,
        [
            ".//JournalIssue/PubDate/Year",
            ".//ArticleDate/Year",
            ".//PubMedPubDate[@PubStatus='pubmed']/Year",
        ],
    )

    if year:
        return year

    medline_date = _first_text(
        article,
        [
            ".//JournalIssue/PubDate/MedlineDate",
        ],
    )

    if medline_date:
        return medline_date[:4]

    return ""


def _abstract_text(article: ET.Element) -> str:
    """
    合并 PubMed 摘要。结构化摘要会保留小标题。
    """

    abstract_parts: list[str] = []

    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.attrib.get("Label", "").strip()
        text = _text(abstract_text)

        if not text:
            continue

        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)

    return " ".join(abstract_parts)


def _publication_types(article: ET.Element) -> str:
    """
    提取 PubMed PublicationType，用于后续粗略判断证据类型。
    """

    values = [
        _text(publication_type)
        for publication_type in article.findall(
            ".//PublicationTypeList/PublicationType"
        )
    ]

    return "; ".join(
        value
        for value in values
        if value
    )


def search_pubmed(keyword, max_results=5):

    params = {
        "db": "pubmed",
        "term": keyword,
        "retmode": "json",
        "retmax": max_results,
    }

    r = requests.get(
        PUBMED_SEARCH_URL,
        params=params,
        timeout=30,
    )
    r.raise_for_status()

    ids = r.json()["esearchresult"]["idlist"]

    return ids


def fetch_pubmed_detail(ids):

    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    }

    r = requests.get(
        PUBMED_FETCH_URL,
        params=params,
        timeout=30,
    )
    r.raise_for_status()

    return r.text


def parse_pubmed_articles(xml_text: str) -> list[dict]:
    """
    将 PubMed XML 解析为逐篇文章的结构化证据。
    """

    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    articles: list[dict] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _first_text(
            article,
            [
                ".//MedlineCitation/PMID",
            ],
        )

        if not pmid:
            pmid = _article_id(
                article,
                "pubmed",
            )

        title = _first_text(
            article,
            [
                ".//Article/ArticleTitle",
            ],
        )

        abstract = _abstract_text(article)

        journal = _first_text(
            article,
            [
                ".//Journal/Title",
                ".//Journal/ISOAbbreviation",
            ],
        )

        year = _publication_year(article)
        doi = _article_id(article, "doi")
        url = PUBMED_ARTICLE_URL.format(pmid=pmid)

        evidence_text_parts = [
            f"Title: {title}" if title else "",
            f"Journal: {journal}" if journal else "",
            f"Year: {year}" if year else "",
            f"PMID: {pmid}" if pmid else "",
            f"DOI: {doi}" if doi else "",
            f"Abstract: {abstract}" if abstract else "",
        ]

        evidence_text = "\n".join(
            part
            for part in evidence_text_parts
            if part
        )

        if not evidence_text.strip():
            continue

        articles.append(
            {
                "source_type": "pubmed",
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "doi": doi,
                "url": url,
                "publication_types": _publication_types(article),
                "text": evidence_text,
            }
        )

    return articles


def fetch_pubmed_articles(ids: list[str]) -> list[dict]:
    """
    获取并解析 PubMed 文献。
    """

    xml_text = fetch_pubmed_detail(ids)

    return parse_pubmed_articles(xml_text)



if __name__ == "__main__":

    ids = search_pubmed(
        "hypertension long term medication guideline"
    )

    print(ids)

    for parsed_article in fetch_pubmed_articles(ids):
        print(parsed_article)
