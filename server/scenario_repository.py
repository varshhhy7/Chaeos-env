from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from models import DifficultyTier, Fact, FactType, Scenario


CountryRecord = dict[str, str | int | float]
CompanyRecord = dict[str, str | float | bool]


COUNTRIES: list[CountryRecord] = [
    {
        "name": "Germany",
        "capital": "Berlin",
        "population": 83_200_000,
        "area_km2": 357_022,
        "gdp_usd": 4.26e12,
        "gdp_per_capita": 51_200.0,
    },
    {
        "name": "Brazil",
        "capital": "Brasilia",
        "population": 203_000_000,
        "area_km2": 8_515_767,
        "gdp_usd": 2.17e12,
        "gdp_per_capita": 10_690.0,
    },
    {
        "name": "Canada",
        "capital": "Ottawa",
        "population": 40_100_000,
        "area_km2": 9_984_670,
        "gdp_usd": 2.14e12,
        "gdp_per_capita": 53_400.0,
    },
    {
        "name": "Japan",
        "capital": "Tokyo",
        "population": 125_700_000,
        "area_km2": 377_975,
        "gdp_usd": 4.27e12,
        "gdp_per_capita": 33_950.0,
    },
    {
        "name": "France",
        "capital": "Paris",
        "population": 67_390_000,
        "area_km2": 643_801,
        "gdp_usd": 3.03e12,
        "gdp_per_capita": 44_747.0,
    },
    {
        "name": "India",
        "capital": "New Delhi",
        "population": 1_428_627_663,
        "area_km2": 3_287_263,
        "gdp_usd": 3.73e12,
        "gdp_per_capita": 2_610.0,
    },
    {
        "name": "Australia",
        "capital": "Canberra",
        "population": 26_473_055,
        "area_km2": 7_692_024,
        "gdp_usd": 1.69e12,
        "gdp_per_capita": 63_900.0,
    },
    {
        "name": "Italy",
        "capital": "Rome",
        "population": 58_850_000,
        "area_km2": 301_340,
        "gdp_usd": 2.25e12,
        "gdp_per_capita": 38_250.0,
    },
    {
        "name": "Spain",
        "capital": "Madrid",
        "population": 48_600_000,
        "area_km2": 505_990,
        "gdp_usd": 1.58e12,
        "gdp_per_capita": 32_500.0,
    },
    {
        "name": "South Korea",
        "capital": "Seoul",
        "population": 51_710_000,
        "area_km2": 100_210,
        "gdp_usd": 1.71e12,
        "gdp_per_capita": 33_070.0,
    },
    {
        "name": "Mexico",
        "capital": "Mexico City",
        "population": 129_000_000,
        "area_km2": 1_964_375,
        "gdp_usd": 1.81e12,
        "gdp_per_capita": 14_030.0,
    },
    {
        "name": "Indonesia",
        "capital": "Jakarta",
        "population": 277_500_000,
        "area_km2": 1_904_569,
        "gdp_usd": 1.42e12,
        "gdp_per_capita": 5_120.0,
    },
    {
        "name": "Norway",
        "capital": "Oslo",
        "population": 5_550_000,
        "area_km2": 385_207,
        "gdp_usd": 485e9,
        "gdp_per_capita": 87_390.0,
    },
    {
        "name": "Egypt",
        "capital": "Cairo",
        "population": 112_700_000,
        "area_km2": 1_010_408,
        "gdp_usd": 398e9,
        "gdp_per_capita": 3_530.0,
    },
    {
        "name": "Kenya",
        "capital": "Nairobi",
        "population": 55_100_000,
        "area_km2": 580_367,
        "gdp_usd": 113e9,
        "gdp_per_capita": 2_050.0,
    },
]


COMPANIES: list[CompanyRecord] = [
    {"name": "NovaTech", "claim": 2.30, "actual": 1.87, "accurate": False},
    {"name": "HelioWorks", "claim": 0.94, "actual": 0.94, "accurate": True},
    {"name": "AsterCloud", "claim": 4.80, "actual": 4.12, "accurate": False},
    {"name": "QuantumForge", "claim": 1.45, "actual": 1.32, "accurate": False},
    {"name": "Riverline Health", "claim": 3.05, "actual": 3.05, "accurate": True},
    {"name": "Cobalt Grid", "claim": 0.72, "actual": 0.61, "accurate": False},
    {"name": "Northstar Robotics", "claim": 2.76, "actual": 2.76, "accurate": True},
    {"name": "Vela Foods", "claim": 1.18, "actual": 0.98, "accurate": False},
    {"name": "Arcadia Freight", "claim": 5.20, "actual": 4.91, "accurate": False},
    {"name": "Lumen BioSystems", "claim": 0.38, "actual": 0.38, "accurate": True},
    {"name": "SignalPeak Media", "claim": 1.02, "actual": 0.84, "accurate": False},
    {"name": "Copperleaf Energy", "claim": 6.40, "actual": 6.40, "accurate": True},
    {"name": "BlueHarbor Bank", "claim": 3.90, "actual": 3.44, "accurate": False},
    {"name": "Orbital Transit", "claim": 2.11, "actual": 2.11, "accurate": True},
    {"name": "SummitWare", "claim": 0.66, "actual": 0.57, "accurate": False},
]


DENSITY_PAIRS: list[tuple[str, str]] = [
    ("India", "Australia"),
    ("Germany", "Canada"),
    ("Japan", "France"),
    ("South Korea", "Brazil"),
    ("Italy", "Spain"),
    ("Indonesia", "Mexico"),
    ("Norway", "Egypt"),
    ("Kenya", "Canada"),
    ("France", "Australia"),
    ("Japan", "Brazil"),
    ("Germany", "Norway"),
    ("India", "Canada"),
    ("Spain", "Egypt"),
    ("South Korea", "Australia"),
    ("Italy", "Mexico"),
]


def _country_by_name(name: str) -> CountryRecord:
    for country in COUNTRIES:
        if country["name"] == name:
            return country
    raise KeyError(name)


def _country_payload(country: CountryRecord) -> dict[str, str | int | float]:
    return {
        "name": country["name"],
        "capital": country["capital"],
        "population": country["population"],
        "area_km2": country["area_km2"],
        "gdp_usd": country["gdp_usd"],
        "gdp_per_capita": country["gdp_per_capita"],
    }


def _density(country: CountryRecord) -> float:
    return round(float(country["population"]) / float(country["area_km2"]), 1)


def _warmup_scenarios() -> Iterable[Scenario]:
    for index, country in enumerate(COUNTRIES[:10], start=1):
        name = str(country["name"])
        population = int(country["population"])
        yield Scenario(
            id=f"W{index:02d}",
            benchmark_task_id="task1",
            question=f"What is the population of {name}?",
            answer={"population": population},
            required_facts=[
                Fact(
                    key="population",
                    value=population,
                    type=FactType.NUMERIC,
                    tolerance=0.02,
                )
            ],
            tool_data={
                "knowledge_base_lookup": [{"entity": name, "result": _country_payload(country)}],
                "web_search": [
                    {
                        "query": f"{name} population",
                        "result": [
                            {
                                "title": f"{name} demographic profile",
                                "snippet": f"{name} has a population of {population:,}.",
                            }
                        ],
                    }
                ],
                "database_query": [
                    {
                        "query": f"SELECT population FROM countries WHERE name='{name}'",
                        "result": [{"population": population}],
                    }
                ],
            },
            difficulty=DifficultyTier.WARMUP,
            min_tools_needed=1,
            tags=["geography", "demographics"],
            cross_validation_tools=[["knowledge_base_lookup", "web_search", "database_query"]],
        )


def _beginner_scenarios() -> Iterable[Scenario]:
    for index, country in enumerate(COUNTRIES, start=1):
        name = str(country["name"])
        capital = str(country["capital"])
        gdp_per_capita = float(country["gdp_per_capita"])
        yield Scenario(
            id=f"B{index:02d}",
            benchmark_task_id="task1",
            question=(
                f"What is the GDP per capita of the country whose capital is "
                f"{capital}? Express it in USD."
            ),
            answer={"country": name, "gdp_per_capita": gdp_per_capita},
            required_facts=[
                Fact(key="country", value=name, type=FactType.TEXT),
                Fact(
                    key="gdp_per_capita",
                    value=gdp_per_capita,
                    type=FactType.NUMERIC,
                    tolerance=0.05,
                ),
            ],
            tool_data={
                "knowledge_base_lookup": [
                    {
                        "entity": capital,
                        "result": {
                            "type": "city",
                            "name": capital,
                            "country": name,
                        },
                    },
                    {"entity": name, "result": _country_payload(country)},
                ],
                "web_search": [
                    {
                        "query": f"{capital} capital country",
                        "result": [
                            {
                                "title": f"{capital} country lookup",
                                "snippet": f"{capital} is the capital of {name}.",
                            }
                        ],
                    },
                    {
                        "query": f"{name} GDP per capita USD",
                        "result": [
                            {
                                "title": f"{name} economy profile",
                                "snippet": (
                                    f"{name}'s GDP per capita is approximately "
                                    f"${gdp_per_capita:,.0f} USD."
                                ),
                            }
                        ],
                    },
                ],
                "database_query": [
                    {
                        "query": (
                            f"SELECT name, gdp_per_capita FROM countries WHERE capital='{capital}'"
                        ),
                        "result": [{"name": name, "gdp_per_capita": gdp_per_capita}],
                    }
                ],
            },
            difficulty=DifficultyTier.BEGINNER,
            min_tools_needed=2,
            tags=["geography", "economics"],
            cross_validation_tools=[["knowledge_base_lookup", "database_query"]],
        )


def _intermediate_scenarios() -> Iterable[Scenario]:
    for index, (left_name, right_name) in enumerate(DENSITY_PAIRS, start=1):
        left = _country_by_name(left_name)
        right = _country_by_name(right_name)
        left_density = _density(left)
        right_density = _density(right)
        higher = str(left["name"] if left_density >= right_density else right["name"])
        lower_density = min(left_density, right_density)
        factor = round(max(left_density, right_density) / lower_density, 1)
        yield Scenario(
            id=f"I{index:02d}",
            benchmark_task_id="task2",
            question=(
                f"Compare the population density of {left_name} and {right_name}. "
                "Which is higher and by what factor? Population density equals "
                "population divided by area in square kilometers."
            ),
            answer={
                f"{left_name.lower().replace(' ', '_')}_density": left_density,
                f"{right_name.lower().replace(' ', '_')}_density": right_density,
                "higher": higher,
                "factor": factor,
            },
            required_facts=[
                Fact(key="higher", value=higher, type=FactType.TEXT),
                Fact(
                    key="factor",
                    value=factor,
                    type=FactType.NUMERIC,
                    tolerance=0.10,
                ),
            ],
            tool_data={
                "knowledge_base_lookup": [
                    {"entity": left_name, "result": _country_payload(left)},
                    {"entity": right_name, "result": _country_payload(right)},
                ],
                "database_query": [
                    {
                        "query": (
                            "SELECT name, population, area_km2 FROM countries "
                            f"WHERE name IN ('{left_name}', '{right_name}')"
                        ),
                        "result": [
                            {
                                "name": left_name,
                                "population": left["population"],
                                "area_km2": left["area_km2"],
                            },
                            {
                                "name": right_name,
                                "population": right["population"],
                                "area_km2": right["area_km2"],
                            },
                        ],
                    }
                ],
                "calculator": "always_live",
                "python_execute": "always_live",
                "compare_values": "always_live",
            },
            difficulty=DifficultyTier.INTERMEDIATE,
            min_tools_needed=3,
            tags=["geography", "math", "comparison"],
            cross_validation_tools=[["knowledge_base_lookup", "database_query"]],
        )


def _expert_scenarios() -> Iterable[Scenario]:
    for index, company in enumerate(COMPANIES, start=1):
        name = str(company["name"])
        claim = float(company["claim"])
        actual = float(company["actual"])
        accurate = bool(company["accurate"])
        discrepancy_pct = round(abs(claim - actual) / max(abs(actual), 1e-9) * 100, 1)
        status_phrase = "matches" if accurate else "does not match"
        yield Scenario(
            id=f"E{index:02d}",
            benchmark_task_id="task3",
            question=(
                f"{name} reported revenue of ${claim:.2f}B last quarter. Verify "
                "this claim using at least two independent sources. Is the claim "
                "accurate? If not, what is the actual revenue?"
            ),
            answer={
                "claim_accurate": accurate,
                "actual_revenue_b": actual,
                "discrepancy_pct": discrepancy_pct,
            },
            required_facts=[
                Fact(
                    key="claim_accurate",
                    value=accurate,
                    type=FactType.BOOLEAN,
                    alternatives=["accurate" if accurate else "not accurate"],
                ),
                Fact(
                    key="actual_revenue_b",
                    value=actual,
                    type=FactType.NUMERIC,
                    tolerance=0.05,
                ),
            ],
            tool_data={
                "web_search": [
                    {
                        "query": f"{name} revenue last quarter",
                        "result": [
                            {
                                "title": f"{name} quarterly earnings",
                                "snippet": (
                                    f"{name} reported ${actual:.2f}B revenue last "
                                    f"quarter; the ${claim:.2f}B claim {status_phrase} "
                                    "the filing."
                                ),
                            }
                        ],
                    }
                ],
                "database_query": [
                    {
                        "query": (
                            "SELECT revenue_b, quarter FROM financials "
                            f"WHERE company='{name}' ORDER BY quarter DESC LIMIT 1"
                        ),
                        "result": [{"revenue_b": actual, "quarter": "Q3-2025"}],
                    }
                ],
                "document_search": [
                    {
                        "query": f"{name} quarterly filing revenue",
                        "result": [
                            {
                                "doc_id": f"FILING-{index:03d}-2025Q3",
                                "excerpt": (
                                    f"Total revenue for {name}: ${actual:.2f}B for "
                                    "the quarter ended September 30, 2025."
                                ),
                            }
                        ],
                    }
                ],
                "fetch_url": [
                    {
                        "url": f"https://investors.example.com/{name.lower().replace(' ', '-')}",
                        "result": {
                            "content": (
                                f"Quarterly revenue: ${actual:.2f}B. Claim checked: ${claim:.2f}B."
                            )
                        },
                    }
                ],
                "calculator": "always_live",
                "check_consistency": [
                    {
                        "query": f"{name} revenue consistency",
                        "result": {
                            "consistent": accurate,
                            "sources_checked": 4,
                            "actual_revenue_b": actual,
                            "claim_revenue_b": claim,
                        },
                    }
                ],
            },
            difficulty=DifficultyTier.EXPERT,
            min_tools_needed=3,
            tags=["finance", "verification", "cross_validation"],
            cross_validation_tools=[
                ["web_search", "database_query", "document_search", "fetch_url"]
            ],
        )


def build_default_scenarios() -> list[Scenario]:
    scenarios = [
        *_warmup_scenarios(),
        *_beginner_scenarios(),
        *_intermediate_scenarios(),
        *_expert_scenarios(),
    ]
    expected_count = 55
    if len(scenarios) != expected_count:
        raise RuntimeError(
            f"Scenario fixture expected {expected_count} scenarios, got {len(scenarios)}"
        )
    return scenarios


@dataclass(frozen=True)
class ScenarioRepository:
    """In-memory index over deterministic scenario fixtures."""

    scenarios: Sequence[Scenario]

    @classmethod
    def default(cls) -> "ScenarioRepository":
        return cls(build_default_scenarios())

    def all(self) -> list[Scenario]:
        return list(self.scenarios)

    def get(self, scenario_id: str) -> Scenario:
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    def choose(
        self,
        *,
        rng: random.Random,
        difficulty: DifficultyTier | str | None = None,
        benchmark_task_id: str | None = None,
    ) -> Scenario:
        candidates = self.all()
        if benchmark_task_id is not None:
            candidates = [
                scenario
                for scenario in candidates
                if scenario.benchmark_task_id == benchmark_task_id
            ]
        if difficulty is not None:
            tier = DifficultyTier(difficulty)
            candidates = [scenario for scenario in candidates if scenario.difficulty == tier]
        if not candidates:
            raise ValueError(
                "No scenarios available for "
                f"difficulty={difficulty!r}, benchmark_task_id={benchmark_task_id!r}"
            )
        return rng.choice(candidates)
