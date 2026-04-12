from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from models import Scenario
from server.scenario_repository import COMPANIES, COUNTRIES
from server.tasks import BenchmarkTask


@dataclass(frozen=True)
class WorkspaceDocument:
    doc_id: str
    title: str
    content: str
    url: str
    entity: str
    source_type: str


class TaskWorkspace:
    """Real internal data plane for ChaosEnv retrieval tools."""

    def __init__(self, *, scenario: Scenario, task: BenchmarkTask):
        self.scenario = scenario
        self.task = task
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.documents: list[WorkspaceDocument] = []
        self.url_map: dict[str, str] = {}
        self.entity_index: dict[str, dict[str, Any]] = {}
        self.initial_files: dict[str, str] = {}
        self._build()

    def close(self) -> None:
        self.connection.close()

    def lookup_entity(self, entity: str) -> dict[str, Any]:
        normalized = self._normalize(entity)
        if not normalized:
            return {"error": "entity is required"}

        if normalized in self.entity_index:
            return self.entity_index[normalized]

        best_key, best_score = self._best_entity_match(normalized)
        if best_key is None or best_score < 0.45:
            return {"error": f"No entity found for {entity!r}"}
        return self.entity_index[best_key]

    def run_sql(self, sql: str) -> dict[str, Any]:
        statement = sql.strip()
        if not statement:
            return {"error": "sql is required"}
        if not statement.lower().startswith("select"):
            return {"error": "Only SELECT statements are allowed"}

        try:
            cursor = self.connection.execute(statement)
            rows = [dict(row) for row in cursor.fetchall()]
            return {"result": rows}
        except sqlite3.Error as exc:
            return {"error": f"SQL error: {exc}"}

    def search_documents(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        ranked = self._rank_documents(query, source_type=None)[:limit]
        return {
            "result": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "excerpt": self._excerpt(doc.content, query),
                    "url": doc.url,
                    "source_type": doc.source_type,
                    "entity": doc.entity,
                }
                for doc, _score in ranked
            ]
        }

    def web_search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        ranked = self._rank_documents(query, source_type="web")[:limit]
        return {
            "result": [
                {
                    "title": doc.title,
                    "snippet": self._excerpt(doc.content, query, max_words=18),
                    "url": doc.url,
                }
                for doc, _score in ranked
            ]
        }

    def fetch_url(self, url: str) -> dict[str, Any]:
        content = self.url_map.get(url.strip())
        if content is None:
            return {"error": f"URL not found: {url}"}
        return {"content": content}

    def fact_check(self, claim: str) -> dict[str, Any]:
        text = claim.strip()
        if not text:
            return {"error": "claim is required"}

        revenue_match = re.search(
            r"(?P<company>[A-Za-z][A-Za-z0-9 .&-]+?)\s+reported revenue of \$?(?P<value>\d+(?:\.\d+)?)B",
            text,
            re.IGNORECASE,
        )
        if revenue_match:
            company = revenue_match.group("company").strip()
            claimed = float(revenue_match.group("value"))
            row = self.connection.execute(
                "SELECT company, actual_revenue_b, accurate FROM financials WHERE lower(company)=lower(?)",
                (company,),
            ).fetchone()
            if row is None:
                return {"error": f"No financial record found for {company}"}
            actual = float(row["actual_revenue_b"])
            accurate = abs(claimed - actual) < 1e-9
            return {
                "claim": text,
                "verified": accurate,
                "actual_revenue_b": actual,
                "difference_b": round(claimed - actual, 2),
            }

        country_match = re.search(r"population of (?P<country>[A-Za-z ]+)", text, re.IGNORECASE)
        if country_match:
            country = country_match.group("country").strip()
            row = self.connection.execute(
                "SELECT name, population FROM countries WHERE lower(name)=lower(?)",
                (country,),
            ).fetchone()
            if row is None:
                return {"error": f"No population record found for {country}"}
            return {"claim": text, "verified": True, "population": int(row["population"])}

        return {"error": "Unsupported claim format for fact_check"}

    def check_consistency(self, source1: str, source2: str) -> dict[str, Any]:
        payload1 = self._coerce_json(source1)
        payload2 = self._coerce_json(source2)
        return {
            "consistent": payload1 == payload2,
            "source1": payload1,
            "source2": payload2,
        }

    def api_call(self, url: str, method: str = "GET") -> dict[str, Any]:
        normalized_method = method.upper().strip() or "GET"
        if normalized_method != "GET":
            return {"error": f"Unsupported method: {normalized_method}"}
        if url.startswith("internal://entity/"):
            entity = url.removeprefix("internal://entity/")
            return {"result": self.lookup_entity(entity)}
        if url.startswith("internal://document/"):
            doc_id = url.removeprefix("internal://document/")
            for doc in self.documents:
                if doc.doc_id == doc_id:
                    return {
                        "result": {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "content": doc.content,
                            "url": doc.url,
                        }
                    }
            return {"error": f"Document not found: {doc_id}"}
        return {"error": f"Unsupported internal API URL: {url}"}

    def _build(self) -> None:
        self._create_tables()
        self._seed_countries()
        self._seed_financials()
        self._seed_documents()
        self._seed_initial_files()

    def _create_tables(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE countries (
                name TEXT PRIMARY KEY,
                capital TEXT NOT NULL,
                population INTEGER NOT NULL,
                area_km2 INTEGER NOT NULL,
                gdp_usd REAL NOT NULL,
                gdp_per_capita REAL NOT NULL
            );

            CREATE TABLE financials (
                company TEXT PRIMARY KEY,
                quarter TEXT NOT NULL,
                claim_revenue_b REAL NOT NULL,
                actual_revenue_b REAL NOT NULL,
                revenue_b REAL NOT NULL,
                accurate INTEGER NOT NULL
            );
            """
        )

    def _seed_countries(self) -> None:
        for country in COUNTRIES:
            self.connection.execute(
                """
                INSERT INTO countries (name, capital, population, area_km2, gdp_usd, gdp_per_capita)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    country["name"],
                    country["capital"],
                    country["population"],
                    country["area_km2"],
                    country["gdp_usd"],
                    country["gdp_per_capita"],
                ),
            )
            payload = {
                "type": "country",
                "name": country["name"],
                "capital": country["capital"],
                "population": country["population"],
                "area_km2": country["area_km2"],
                "gdp_usd": country["gdp_usd"],
                "gdp_per_capita": country["gdp_per_capita"],
            }
            self.entity_index[self._normalize(str(country["name"]))] = payload
            self.entity_index[self._normalize(str(country["capital"]))] = {
                "type": "city",
                "name": country["capital"],
                "country": country["name"],
            }

    def _seed_financials(self) -> None:
        for company in COMPANIES:
            self.connection.execute(
                """
                INSERT INTO financials (
                    company,
                    quarter,
                    claim_revenue_b,
                    actual_revenue_b,
                    revenue_b,
                    accurate
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    company["name"],
                    "Q3-2025",
                    company["claim"],
                    company["actual"],
                    company["actual"],
                    int(bool(company["accurate"])),
                ),
            )
            payload = {
                "type": "company",
                "name": company["name"],
                "claim_revenue_b": company["claim"],
                "actual_revenue_b": company["actual"],
                "accurate": company["accurate"],
            }
            self.entity_index[self._normalize(str(company["name"]))] = payload

    def _seed_documents(self) -> None:
        for country in COUNTRIES:
            name = str(country["name"])
            capital = str(country["capital"])
            url = f"https://chaos.local/countries/{self._slug(name)}"
            content = (
                f"{name} is a country whose capital is {capital}. "
                f"It has population {int(country['population']):,}, area {int(country['area_km2']):,} km2, "
                f"GDP {float(country['gdp_usd']) / 1e12:.2f} trillion USD, "
                f"and GDP per capita {float(country['gdp_per_capita']):,.0f} USD."
            )
            self._add_document(
                WorkspaceDocument(
                    doc_id=f"country-{self._slug(name)}",
                    title=f"{name} country profile",
                    content=content,
                    url=url,
                    entity=name,
                    source_type="web",
                )
            )
            city_url = f"https://chaos.local/capitals/{self._slug(capital)}"
            self._add_document(
                WorkspaceDocument(
                    doc_id=f"capital-{self._slug(capital)}",
                    title=f"{capital} city entry",
                    content=f"{capital} is the capital city of {name}.",
                    url=city_url,
                    entity=capital,
                    source_type="document",
                )
            )

        for company in COMPANIES:
            name = str(company["name"])
            claim = float(company["claim"])
            actual = float(company["actual"])
            filing_url = f"https://investors.example.com/{self._slug(name)}"
            web_content = (
                f"{name} last quarter investor note. "
                f"Reported revenue in the latest filing was ${actual:.2f}B. "
                f"The circulating ${claim:.2f}B claim is "
                f"{'accurate' if bool(company['accurate']) else 'not accurate'}."
            )
            self._add_document(
                WorkspaceDocument(
                    doc_id=f"filing-{self._slug(name)}",
                    title=f"{name} quarterly filing",
                    content=web_content,
                    url=filing_url,
                    entity=name,
                    source_type="web",
                )
            )
            self._add_document(
                WorkspaceDocument(
                    doc_id=f"analysis-{self._slug(name)}",
                    title=f"{name} analyst reconciliation",
                    content=(
                        f"Analyst reconciliation for {name}: actual revenue ${actual:.2f}B, "
                        f"claim revenue ${claim:.2f}B."
                    ),
                    url=f"https://chaos.local/research/{self._slug(name)}",
                    entity=name,
                    source_type="document",
                )
            )
            self.url_map[f"https://chaos.local/investors/{self._slug(name)}/q3-2025"] = web_content

    def _seed_initial_files(self) -> None:
        self.initial_files["briefing/current_task.md"] = (
            f"# {self.task.name}\n\n"
            f"Scenario: {self.scenario.id}\n\n"
            f"Question: {self.scenario.question}\n"
        )
        self.initial_files["notes/evidence.json"] = json.dumps(
            {
                "task_id": self.task.id,
                "scenario_id": self.scenario.id,
                "question": self.scenario.question,
            },
            indent=2,
        )

    def _add_document(self, doc: WorkspaceDocument) -> None:
        self.documents.append(doc)
        self.url_map[doc.url] = doc.content

    def _rank_documents(
        self, query: str, *, source_type: str | None
    ) -> list[tuple[WorkspaceDocument, float]]:
        query_tokens = self._tokens(query)
        scored: list[tuple[WorkspaceDocument, float]] = []
        for doc in self.documents:
            if source_type is not None and doc.source_type != source_type:
                continue
            doc_tokens = self._tokens(" ".join([doc.title, doc.content, doc.entity]))
            if not query_tokens or not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens | doc_tokens)
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _best_entity_match(self, normalized: str) -> tuple[str | None, float]:
        best_key: str | None = None
        best_score = 0.0
        target_tokens = set(normalized.split())
        for key in self.entity_index:
            candidate_tokens = set(key.split())
            overlap = len(target_tokens & candidate_tokens)
            score = overlap / max(len(target_tokens | candidate_tokens), 1)
            if normalized in key:
                score = max(score, 0.8)
            if score > best_score:
                best_key = key
                best_score = score
        return best_key, best_score

    @staticmethod
    def _excerpt(content: str, query: str, *, max_words: int = 24) -> str:
        words = content.split()
        if len(words) <= max_words:
            return content
        query_tokens = TaskWorkspace._tokens(query)
        for index, word in enumerate(words):
            if TaskWorkspace._normalize(word) in query_tokens:
                start = max(0, index - 5)
                end = min(len(words), start + max_words)
                return " ".join(words[start:end])
        return " ".join(words[:max_words])

    @staticmethod
    def _coerce_json(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))

    @staticmethod
    def _slug(text: str) -> str:
        return "-".join(re.findall(r"[a-z0-9]+", text.lower()))
