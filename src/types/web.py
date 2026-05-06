from __future__ import annotations

from pydantic import BaseModel


class ArxivArticle(BaseModel):
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    pdf_url: str | None = None
    entry_url: str | None = None


class ArxivSearchResult(BaseModel):
    query: str
    start: int
    max_results: int
    total_results: int
    articles: list[ArxivArticle]
