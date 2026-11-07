from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import quote

import httpx

ERIC_API = "https://api.ies.ed.gov/eric/"
OPENALEX_API = "https://api.openalex.org"
CROSSREF_API = "https://api.crossref.org"
MAX_RESULTS = 25

ERIC_FIELDS = (
    "id,title,author,source,publicationdateyear,description,subject,"
    "peerreviewed,publicationtype,url,identifier,doi,issn,publisher,"
    "institution,availability,language,sponsor,educationlevel,audience"
)


@dataclass(frozen=True)
class ResearchSettings:
    contact_email: str | None
    openalex_api_key: str | None
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> ResearchSettings:
        raw_timeout = os.environ.get("RESEARCH_HTTP_TIMEOUT", "30")
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("RESEARCH_HTTP_TIMEOUT must be numeric") from exc
        if not 1 <= timeout <= 120:
            raise ValueError("RESEARCH_HTTP_TIMEOUT must be between 1 and 120")
        return cls(
            contact_email=os.environ.get("SCHOLAR_CONTACT_EMAIL") or None,
            openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
            timeout_seconds=timeout,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")
    return limit


def clean_doi(value: str) -> str:
    doi = value.strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    doi = doi.strip().lower()
    if not doi.startswith("10.") or "/" not in doi or any(ch.isspace() for ch in doi):
        raise ValueError("invalid DOI")
    if len(doi) > 255:
        raise ValueError("DOI is too long")
    return doi


def clean_eric_id(value: str) -> str:
    eric_id = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"E[DJ]\d{6,}", eric_id):
        raise ValueError("ERIC ID must look like ED123456 or EJ123456")
    return eric_id


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _first(value: Any) -> Any:
    values = _as_list(value)
    return values[0] if values else None


def _date_parts(message: dict[str, Any]) -> list[int] | None:
    for key in (
        "published",
        "published-print",
        "published-online",
        "issued",
        "created",
    ):
        parts = (
            message.get(key, {}).get("date-parts", [])
            if isinstance(message.get(key), dict)
            else []
        )
        if parts and isinstance(parts[0], list):
            return parts[0]
    return None


def _crossref_authors(message: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for author in _as_list(message.get("author")):
        if not isinstance(author, dict):
            continue
        name = " ".join(
            str(part).strip()
            for part in (author.get("given"), author.get("family"))
            if part
        )
        if name:
            result.append(name)
    return result


def _openalex_authors(work: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for authorship in _as_list(work.get("authorships")):
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        if isinstance(author, dict) and author.get("display_name"):
            result.append(str(author["display_name"]))
    return result


def _abstract_from_inverted_index(index: Any) -> str | None:
    if not isinstance(index, dict) or not index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(word)))
    if not positioned:
        return None
    positioned.sort()
    return " ".join(word for _, word in positioned)


def _normalized_title(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def title_similarity(left: str | None, right: str | None) -> float | None:
    a, b = _normalized_title(left), _normalized_title(right)
    if not a or not b:
        return None
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def normalize_eric_record(
    doc: dict[str, Any], *, retrieved_at: str | None = None
) -> dict[str, Any]:
    retrieved_at = retrieved_at or utc_now()
    eric_id = str(doc.get("id") or "")
    peer_value = doc.get("peerreviewed")
    if peer_value == "T":
        peer_status = "confirmed_by_eric_metadata"
    elif peer_value == "F":
        peer_status = "not_marked_peer_reviewed_by_eric"
    else:
        peer_status = "unknown"

    doi = _first(doc.get("doi"))
    identifiers: dict[str, Any] = {"eric": eric_id or None}
    if doi:
        try:
            identifiers["doi"] = clean_doi(str(doi))
        except ValueError:
            identifiers["doi_reported_unparsed"] = str(doi)

    source_url = f"https://eric.ed.gov/?id={eric_id}" if eric_id else None
    return {
        "schema_version": "1.0",
        "provenance": {
            "database": "ERIC",
            "provider": "Institute of Education Sciences, U.S. Department of Education",
            "record_url": source_url,
            "retrieved_at": retrieved_at,
        },
        "stable_identifiers": identifiers,
        "title": doc.get("title"),
        "authors": [str(x) for x in _as_list(doc.get("author"))],
        "year": doc.get("publicationdateyear"),
        "source_or_venue": doc.get("source"),
        "publisher": doc.get("publisher"),
        "abstract_or_description": doc.get("description"),
        "subjects": [str(x) for x in _as_list(doc.get("subject"))],
        "publication_type": [str(x) for x in _as_list(doc.get("publicationtype"))],
        "peer_review": {
            "status": peer_status,
            "source_field": peer_value,
            "warning": "ERIC's peerreviewed field is bibliographic metadata, not proof that a claim is correct.",
        },
        "open_access": {
            "status": "unknown",
            "reported_availability": doc.get("availability"),
            "reported_url": doc.get("url"),
        },
        "integrity": {
            "retraction_status": "not_provided_by_eric_api",
            "correction_status": "not_provided_by_eric_api",
            "doi_integrity_check_recommended": bool(identifiers.get("doi")),
        },
        "document_version": None,
        "raw_source_fields": {
            key: doc.get(key)
            for key in (
                "identifier",
                "issn",
                "institution",
                "language",
                "sponsor",
                "educationlevel",
                "audience",
            )
            if doc.get(key) is not None
        },
        "trust_notice": (
            "This record verifies provenance and selected metadata only. It does not validate the "
            "document's conclusions and cannot substitute for local Lean verification."
        ),
    }


def _openalex_locations(work: dict[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for location in _as_list(work.get("locations")):
        if not isinstance(location, dict):
            continue
        source = location.get("source") or {}
        if not isinstance(source, dict):
            source = {}
        locations.append(
            {
                "version": location.get("version"),
                "is_open_access": location.get("is_oa"),
                "landing_page_url": location.get("landing_page_url"),
                "pdf_url": location.get("pdf_url"),
                "source_name": source.get("display_name"),
                "source_type": source.get("type"),
                "is_in_doaj": source.get("is_in_doaj"),
            }
        )
    return locations


def normalize_openalex_work(
    work: dict[str, Any], *, retrieved_at: str | None = None
) -> dict[str, Any]:
    retrieved_at = retrieved_at or utc_now()
    ids = work.get("ids") or {}
    if not isinstance(ids, dict):
        ids = {}
    doi_value = work.get("doi") or ids.get("doi")
    doi = None
    if doi_value:
        try:
            doi = clean_doi(str(doi_value))
        except ValueError:
            pass

    openalex_url = work.get("id")
    openalex_id = str(openalex_url).rsplit("/", 1)[-1] if openalex_url else None
    oa = work.get("open_access") or {}
    if not isinstance(oa, dict):
        oa = {}
    is_retracted = bool(work.get("is_retracted"))
    primary = work.get("primary_location") or {}
    if not isinstance(primary, dict):
        primary = {}
    source = primary.get("source") or {}
    if not isinstance(source, dict):
        source = {}

    return {
        "schema_version": "1.0",
        "provenance": {
            "database": "OpenAlex",
            "provider": "OurResearch",
            "record_url": openalex_url,
            "retrieved_at": retrieved_at,
        },
        "stable_identifiers": {
            "openalex": openalex_id,
            "doi": doi,
            "pmid": ids.get("pmid"),
            "pmcid": ids.get("pmcid"),
        },
        "title": work.get("display_name") or work.get("title"),
        "authors": _openalex_authors(work),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "source_or_venue": source.get("display_name"),
        "publication_type": work.get("type_crossref") or work.get("type"),
        "abstract": _abstract_from_inverted_index(work.get("abstract_inverted_index")),
        "peer_review": {
            "status": "unknown_not_inferred_from_venue",
            "warning": "OpenAlex work or venue type is not treated as direct evidence of peer review.",
        },
        "open_access": {
            "is_open_access": oa.get("is_oa"),
            "status": oa.get("oa_status") or "unknown",
            "best_url": oa.get("oa_url"),
            "any_repository_has_fulltext": oa.get("any_repository_has_fulltext"),
        },
        "integrity": {
            "retraction_status": "retracted"
            if is_retracted
            else "not_flagged_by_openalex",
            "openalex_is_retracted": is_retracted,
            "crossref_update_check_performed": False,
        },
        "document_versions_and_locations": _openalex_locations(work),
        "citation_count": work.get("cited_by_count"),
        "trust_notice": (
            "This record verifies provenance and selected metadata only. Search results require a "
            "separate DOI integrity lookup before supporting a consequential claim."
        ),
    }


def normalize_crossref_work(
    message: dict[str, Any], *, retrieved_at: str | None = None
) -> dict[str, Any]:
    retrieved_at = retrieved_at or utc_now()
    doi = clean_doi(str(message.get("DOI"))) if message.get("DOI") else None
    date_parts = _date_parts(message)
    licenses = [
        {
            "url": item.get("URL"),
            "content_version": item.get("content-version"),
            "delay_in_days": item.get("delay-in-days"),
        }
        for item in _as_list(message.get("license"))
        if isinstance(item, dict)
    ]
    return {
        "schema_version": "1.0",
        "provenance": {
            "database": "Crossref",
            "provider": "Crossref",
            "record_url": f"https://doi.org/{doi}" if doi else message.get("URL"),
            "retrieved_at": retrieved_at,
        },
        "stable_identifiers": {"doi": doi},
        "title": _first(message.get("title")),
        "authors": _crossref_authors(message),
        "year": date_parts[0] if date_parts else None,
        "publication_date_parts": date_parts,
        "source_or_venue": _first(message.get("container-title")),
        "publisher": message.get("publisher"),
        "publication_type": message.get("subtype") or message.get("type"),
        "abstract": message.get("abstract"),
        "peer_review": {
            "status": "unknown_not_inferred_from_crossref_type",
            "warning": "DOI registration and Crossref type do not by themselves prove peer review.",
        },
        "licenses": licenses,
        "relations": message.get("relation") or {},
        "registered_updates_on_this_record": message.get("update-to") or [],
        "trust_notice": (
            "Crossref confirms deposited DOI metadata. It does not validate the work's conclusions."
        ),
    }


def extract_crossref_updates(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    found_types: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        title = _first(item.get("title"))
        for relation in _as_list(item.get("update-to")):
            if not isinstance(relation, dict):
                continue
            update_type = str(relation.get("type") or "unknown").lower()
            found_types.add(update_type)
            updates.append(
                {
                    "update_doi": item.get("DOI"),
                    "update_title": title,
                    "type": update_type,
                    "label": relation.get("label"),
                    "source": relation.get("source"),
                    "updated": relation.get("updated"),
                }
            )

    if "retraction" in found_types:
        retraction_status = "retracted"
    elif "withdrawal" in found_types:
        retraction_status = "withdrawn"
    else:
        retraction_status = "no_retraction_or_withdrawal_found_in_crossref_updates"

    correction_types = {"correction", "erratum", "expression-of-concern", "update"}
    found_corrections = sorted(found_types & correction_types)
    return {
        "retraction_status": retraction_status,
        "correction_status": (
            "updates_found:" + ",".join(found_corrections)
            if found_corrections
            else "no_correction_or_expression_of_concern_found_in_crossref_updates"
        ),
        "updates": updates,
    }


def build_claim_evidence(
    *,
    stable_identifier: str,
    source_url: str,
    title: str,
    document_version: str,
    publication_type: str,
    peer_review_status: str,
    open_access_status: str,
    retraction_status: str,
    claim: str,
    exact_quote: str,
    locator: str,
) -> dict[str, Any]:
    fields = {
        "stable_identifier": (stable_identifier, 255),
        "source_url": (source_url, 2000),
        "title": (title, 1000),
        "document_version": (document_version, 255),
        "publication_type": (publication_type, 255),
        "peer_review_status": (peer_review_status, 255),
        "open_access_status": (open_access_status, 255),
        "retraction_status": (retraction_status, 255),
        "claim": (claim, 5000),
        "exact_quote": (exact_quote, 20000),
        "locator": (locator, 500),
    }
    cleaned: dict[str, str] = {}
    for name, (value, maximum) in fields.items():
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty; use 'unknown' when applicable")
        if len(stripped) > maximum:
            raise ValueError(f"{name} exceeds {maximum} characters")
        cleaned[name] = stripped
    if not re.match(r"^https://", cleaned["source_url"], flags=re.I):
        raise ValueError("source_url must use HTTPS")

    quote_hash = hashlib.sha256(cleaned["exact_quote"].encode("utf-8")).hexdigest()
    canonical = json.dumps(
        cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    evidence_id = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "okay": True,
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "created_at": utc_now(),
        "source": {
            "stable_identifier": cleaned["stable_identifier"],
            "url": cleaned["source_url"],
            "title": cleaned["title"],
            "document_version": cleaned["document_version"],
            "publication_type": cleaned["publication_type"],
            "peer_review_status": cleaned["peer_review_status"],
            "open_access_status": cleaned["open_access_status"],
            "retraction_status": cleaned["retraction_status"],
        },
        "claim": cleaned["claim"],
        "quotation": {
            "text": cleaned["exact_quote"],
            "locator": cleaned["locator"],
            "sha256": quote_hash,
            "source_match_status": "not_checked_by_metadata_gateway",
        },
        "formal_verification_status": "not_a_lean_verification_result",
        "warnings": [
            "The gateway records the supplied quotation but does not prove that it occurs in the source.",
            "Bibliographic or peer-review status does not validate the quoted claim.",
            "Any derived Lean theorem must still pass the networkless local verifier.",
        ],
    }


def merge_doi_evidence(
    doi: str,
    crossref_message: dict[str, Any] | None,
    openalex_work: dict[str, Any] | None,
    crossref_update_items: list[dict[str, Any]],
    *,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    retrieved_at = retrieved_at or utc_now()
    crossref = (
        normalize_crossref_work(crossref_message, retrieved_at=retrieved_at)
        if crossref_message
        else None
    )
    openalex = (
        normalize_openalex_work(openalex_work, retrieved_at=retrieved_at)
        if openalex_work
        else None
    )
    update_status = extract_crossref_updates(crossref_update_items)

    oa_retracted = bool(openalex_work and openalex_work.get("is_retracted"))
    if oa_retracted:
        update_status["retraction_status"] = "retracted"

    crossref_title = crossref.get("title") if crossref else None
    openalex_title = openalex.get("title") if openalex else None
    similarity = title_similarity(crossref_title, openalex_title)
    if crossref and openalex and similarity is not None:
        bibliographic_status = (
            "consistent" if similarity >= 0.9 else "manual_review_required"
        )
    elif crossref or openalex:
        bibliographic_status = "single_database_match"
    else:
        bibliographic_status = "not_found"

    return {
        "okay": bool(crossref or openalex),
        "schema_version": "1.0",
        "lookup": {
            "doi": doi,
            "retrieved_at": retrieved_at,
            "sources_checked": ["Crossref", "Crossref updates", "OpenAlex"],
        },
        "identifier_verification": {
            "crossref_record_found": bool(crossref),
            "openalex_record_found": bool(openalex),
            "status": bibliographic_status,
            "cross_database_title_similarity": similarity,
        },
        "publication": crossref or openalex,
        "crossref_record": crossref,
        "openalex_record": openalex,
        "integrity": {
            **update_status,
            "openalex_is_retracted": oa_retracted
            if openalex_work is not None
            else None,
            "checked_at": retrieved_at,
            "warning": (
                "A clean lookup means no flag was found in these sources at this time; it is not a "
                "guarantee that no correction, concern, withdrawal, or retraction exists."
            ),
        },
        "peer_review": {
            "status": "unknown",
            "warning": (
                "Neither DOI registration nor OpenAlex indexing is treated as proof of peer review. "
                "Use explicit publisher, venue, ERIC, or editorial-policy evidence."
            ),
        },
        "formal_verification_boundary": (
            "This bibliographic result may inform proof development but cannot bypass the local "
            "Lean compiler, SafeVerify replay, or axiom policy."
        ),
    }


class ResearchClient:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.from_env()

    def _headers(self) -> dict[str, str]:
        contact = self.settings.contact_email or "not-provided"
        return {"User-Agent": f"lean-local-verifier-research/0.1 (mailto:{contact})"}

    def _openalex_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        return params

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            follow_redirects=False,
            headers=self._headers(),
        )

    async def search_eric(
        self,
        query: str,
        *,
        peer_reviewed_only: bool = True,
        publication_type: str | None = None,
        start: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query or len(query) > 1000:
            raise ValueError("query must contain 1 to 1000 characters")
        limit = clamp_limit(limit)
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or start < 0
            or start > 100000
        ):
            raise ValueError("start must be an integer between 0 and 100000")

        clauses = [f"({query})"]
        if peer_reviewed_only:
            clauses.append("peerreviewed:T")
        if publication_type:
            clean_type = publication_type.strip().replace('"', r"\"")
            if not clean_type or len(clean_type) > 100:
                raise ValueError("publication_type must contain 1 to 100 characters")
            clauses.append(f'publicationtype:"{clean_type}"')
        search = " AND ".join(clauses)
        params = {
            "search": search,
            "format": "json",
            "start": start,
            "rows": limit,
            "fields": ERIC_FIELDS,
        }
        async with self._client() as client:
            response = await client.get(ERIC_API, params=params)
            response.raise_for_status()
            payload = response.json()
        eric_response = payload.get("response") or {}
        retrieved_at = utc_now()
        records = [
            normalize_eric_record(doc, retrieved_at=retrieved_at)
            for doc in eric_response.get("docs", [])
            if isinstance(doc, dict)
        ]
        return {
            "okay": True,
            "source": "ERIC official API",
            "effective_query": search,
            "peer_reviewed_filter_applied": peer_reviewed_only,
            "total_results": eric_response.get("numFound"),
            "start": eric_response.get("start", start),
            "returned": len(records),
            "records": records,
            "retrieved_at": retrieved_at,
        }

    async def lookup_eric(self, eric_id: str) -> dict[str, Any]:
        clean_id = clean_eric_id(eric_id)
        result = await self.search_eric(
            f'id:"{clean_id}"',
            peer_reviewed_only=False,
            start=0,
            limit=1,
        )
        records = result.get("records", [])
        if not records:
            return {
                "okay": False,
                "stage": "not_found",
                "eric_id": clean_id,
                "source": "ERIC official API",
            }
        record = records[0]
        doi = record.get("stable_identifiers", {}).get("doi")
        if doi:
            record["doi_integrity"] = await self.lookup_doi(str(doi))
        return {"okay": True, "record": record}

    async def search_openalex(
        self,
        query: str,
        *,
        exclude_retracted: bool = True,
        open_access_only: bool = False,
        from_year: int | None = None,
        to_year: int | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query or len(query) > 500:
            raise ValueError("query must contain 1 to 500 characters")
        limit = clamp_limit(limit)
        for label, value in (("from_year", from_year), ("to_year", to_year)):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1000 <= value <= 3000
            ):
                raise ValueError(f"{label} must be a year between 1000 and 3000")
        if from_year is not None and to_year is not None and from_year > to_year:
            raise ValueError("from_year must not exceed to_year")

        filters: list[str] = []
        if exclude_retracted:
            filters.append("is_retracted:false")
        if open_access_only:
            filters.append("open_access.is_oa:true")
        if from_year is not None:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year is not None:
            filters.append(f"to_publication_date:{to_year}-12-31")
        params: dict[str, Any] = {
            "search": query,
            "per-page": limit,
            **self._openalex_params(),
        }
        if filters:
            params["filter"] = ",".join(filters)
        async with self._client() as client:
            response = await client.get(f"{OPENALEX_API}/works", params=params)
            response.raise_for_status()
            payload = response.json()
        retrieved_at = utc_now()
        records = [
            normalize_openalex_work(work, retrieved_at=retrieved_at)
            for work in payload.get("results", [])
            if isinstance(work, dict)
        ]
        return {
            "okay": True,
            "source": "OpenAlex official API",
            "effective_filters": filters,
            "total_results": (payload.get("meta") or {}).get("count"),
            "returned": len(records),
            "records": records,
            "retrieved_at": retrieved_at,
            "integrity_notice": (
                "Search records only include the OpenAlex retraction flag. Call lookup_doi before "
                "using a DOI-bearing work to support a consequential claim."
            ),
        }

    async def lookup_doi(self, doi: str) -> dict[str, Any]:
        clean = clean_doi(doi)
        openalex_params: dict[str, Any] = {
            "filter": f"doi:{clean}",
            "per-page": 1,
            **self._openalex_params(),
        }
        update_params = {"filter": f"updates:{clean}", "rows": 20}

        async with self._client() as client:
            crossref_request = client.get(
                f"{CROSSREF_API}/works/{quote(clean, safe='')}"
            )
            updates_request = client.get(f"{CROSSREF_API}/works", params=update_params)
            openalex_request = client.get(
                f"{OPENALEX_API}/works", params=openalex_params
            )
            responses = await asyncio.gather(
                crossref_request,
                updates_request,
                openalex_request,
                return_exceptions=True,
            )

        crossref_message: dict[str, Any] | None = None
        update_items: list[dict[str, Any]] = []
        openalex_work: dict[str, Any] | None = None
        errors: list[dict[str, str]] = []

        names = ("crossref_record", "crossref_updates", "openalex_record")
        for name, response in zip(names, responses):
            if isinstance(response, BaseException):
                errors.append({"source": name, "error": str(response)})
                continue
            if response.status_code == 404:
                continue
            try:
                response.raise_for_status()
                payload = response.json()
                if name == "crossref_record":
                    candidate = payload.get("message")
                    crossref_message = (
                        candidate if isinstance(candidate, dict) else None
                    )
                elif name == "crossref_updates":
                    candidate = (payload.get("message") or {}).get("items", [])
                    update_items = [x for x in candidate if isinstance(x, dict)]
                else:
                    candidate = payload.get("results", [])
                    openalex_work = (
                        candidate[0]
                        if candidate and isinstance(candidate[0], dict)
                        else None
                    )
            except Exception as exc:
                errors.append({"source": name, "error": str(exc)})

        result = merge_doi_evidence(
            clean,
            crossref_message,
            openalex_work,
            update_items,
        )
        result["source_errors"] = errors
        if errors and not result["okay"]:
            result["stage"] = "upstream_failure_or_not_found"
        return result
