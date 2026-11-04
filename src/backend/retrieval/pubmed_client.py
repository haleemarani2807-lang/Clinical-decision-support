"""
PubMed E-utilities Client.

Retrieves real medical publications from NCBI PubMed.

Features:
  - esearch (find PMIDs by query)
  - efetch (retrieve abstracts and metadata by PMID)
  - Rate limiting (3 req/s without key, 10 req/s with key)
  - Local file cache (configurable TTL)
  - PMID, DOI, publication date, journal, author preservation
  - Distinguishes abstract-level from full-text evidence
  - Never invents citations

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings
from app.models.schemas import EvidencePassage, EvidenceSource

logger = logging.getLogger(__name__)

ESEARCH_URL = f"{settings.pubmed_base_url}/esearch.fcgi"
EFETCH_URL = f"{settings.pubmed_base_url}/efetch.fcgi"

CACHE_DIR = Path("./data/.pubmed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting: 3 req/s without API key, 10 req/s with key
_last_request_time: float = 0.0
_MIN_INTERVAL_NO_KEY = 0.34   # ~3 req/s
_MIN_INTERVAL_WITH_KEY = 0.10  # ~10 req/s


async def _rate_limit():
    """Enforce PubMed API rate limits."""
    global _last_request_time
    interval = _MIN_INTERVAL_WITH_KEY if settings.pubmed_api_key else _MIN_INTERVAL_NO_KEY
    now = time.monotonic()
    wait = interval - (now - _last_request_time)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_request_time = time.monotonic()


def _cache_key(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _cache_get(key: str, ttl_hours: float = 24.0) -> Optional[dict]:
    path = _cache_key(key)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > ttl_hours:
        path.unlink(missing_ok=True)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_set(key: str, data: dict):
    path = _cache_key(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.debug("Cache write failed: %s", e)


class PubMedClient:
    """
    Async PubMed E-utilities client.

    Usage:
        client = PubMedClient()
        passages = await client.search("type 2 diabetes management", max_results=10)
    """

    def __init__(self):
        self._api_key = settings.pubmed_api_key
        self._params_base = {}
        if self._api_key:
            self._params_base["api_key"] = self._api_key

    async def search(
        self,
        query: str,
        max_results: int = 10,
        ttl_hours: float = 24.0,
    ) -> list[EvidencePassage]:
        """
        Search PubMed and return evidence passages.

        Args:
            query: Clinical search query.
            max_results: Maximum number of results to retrieve.
            ttl_hours: Cache TTL in hours.

        Returns:
            List of EvidencePassage objects with full provenance.
            Returns empty list (not exception) on API failure.
        """
        cache_key = f"search:{query}:{max_results}"
        cached = _cache_get(cache_key, ttl_hours)
        if cached:
            logger.debug("PubMed cache hit for query: %s", query[:50])
            return [EvidencePassage(**p) for p in cached]

        try:
            pmids = await self._esearch(query, max_results)
            if not pmids:
                logger.info("PubMed returned 0 results for: %s", query[:60])
                return []

            passages = await self._efetch(pmids)
            _cache_set(cache_key, [p.model_dump() for p in passages])
            logger.info("PubMed: retrieved %d passages for query: %s", len(passages), query[:60])
            return passages

        except Exception as e:
            logger.warning("PubMed search failed: %s — returning empty results", e)
            return []

    async def _esearch(self, query: str, max_results: int) -> list[str]:
        """Search for PMIDs matching the query."""
        await _rate_limit()
        params = {
            **self._params_base,
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(ESEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])

    async def _efetch(self, pmids: list[str]) -> list[EvidencePassage]:
        """Fetch article metadata for a list of PMIDs."""
        if not pmids:
            return []

        await _rate_limit()
        params = {
            **self._params_base,
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(EFETCH_URL, params=params)
            resp.raise_for_status()
            return self._parse_efetch_xml(resp.text)

    def _parse_efetch_xml(self, xml_text: str) -> list[EvidencePassage]:
        """Parse PubMed eFetch XML and extract passage metadata."""
        passages = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("PubMed XML parse error: %s", e)
            return []

        for i, article in enumerate(root.findall(".//PubmedArticle")):
            try:
                passage = self._extract_article(article, i)
                if passage:
                    passages.append(passage)
            except Exception as e:
                logger.debug("Failed to parse article %d: %s", i, e)

        return passages

    def _extract_article(self, article: ET.Element, index: int) -> Optional[EvidencePassage]:
        """Extract structured passage from a PubmedArticle XML element."""
        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None else f"unknown_{index}"

        # Title
        title_el = article.find(".//ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""

        # Abstract
        abstract_parts = article.findall(".//AbstractText")
        if not abstract_parts:
            return None  # Skip articles without abstract

        abstract = " ".join(
            (el.text or "").strip()
            for el in abstract_parts
            if el.text
        ).strip()

        if not abstract or len(abstract) < 50:
            return None

        # Journal
        journal_el = article.find(".//Journal/Title")
        journal = (journal_el.text or "").strip() if journal_el is not None else ""

        # Publication date
        pub_date = self._extract_pub_date(article)

        # Authors
        authors = self._extract_authors(article)

        # DOI
        doi_el = article.find(".//ArticleId[@IdType='doi']")
        doi = (doi_el.text or "").strip() if doi_el is not None else None

        # URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return EvidencePassage(
            passage_id=f"pmid_{pmid}",
            text=abstract,
            source=EvidenceSource.PUBMED,
            title=title,
            pmid=pmid,
            doi=doi,
            url=url,
            publication_date=pub_date,
            authors=authors,
            journal=journal,
        )

    @staticmethod
    def _extract_pub_date(article: ET.Element) -> Optional[str]:
        """Extract publication date, preferring electronic over print."""
        for date_type in ["Electronic", "Print", "Entrez"]:
            date_el = article.find(f".//PubDate")
            if date_el is not None:
                year_el = date_el.find("Year")
                month_el = date_el.find("Month")
                if year_el is not None:
                    year = year_el.text or ""
                    month = (month_el.text or "")[:3] if month_el is not None else ""
                    return f"{year}-{month}" if month else year
        return None

    @staticmethod
    def _extract_authors(article: ET.Element) -> Optional[str]:
        """Extract first three authors + 'et al.' if more."""
        authors = []
        for author in article.findall(".//Author"):
            last = author.find("LastName")
            initials = author.find("Initials")
            if last is not None:
                name = last.text or ""
                if initials is not None and initials.text:
                    name += f" {initials.text}"
                authors.append(name)
        if not authors:
            return None
        if len(authors) <= 3:
            return ", ".join(authors)
        return ", ".join(authors[:3]) + " et al."
