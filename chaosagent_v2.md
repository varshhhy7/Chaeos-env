# ChaosAgent v2: Unreliable Tools Resilience Environment

## OpenEnv Hackathon — Round 1 Submission Plan

**Name**: `chaosagent`  
**Tagline**: "Can your agent survive when every tool lies, lags, or dies?"  
**Deadline**: April 8, 2026

---

## 1. Three Problems Fixed in v2

### Problem 1: No ListToolsAction, No Tool Filtering
**Old**: Agent calls `list_tools` to discover tools. Curriculum hides tools per tier.  
**New**: ALL tools always available. No `ListToolsAction`. Agent gets tool descriptions in the task prompt. Actions are only `CallToolAction` and `SubmitAnswerAction`. The agent must choose wisely — having 30 tools and needing only 2-3 is the point.

### Problem 2: Ground Truth — The Hard Problem
**Old**: "Snapshot real APIs at reset" — fragile, non-deterministic, unverifiable.  
**New**: **Pre-computed scenario database** (like FinQA's CSV, Email Triage's labeled dataset). Each scenario is a question with a KNOWN answer baked in at design time. Tools return **deterministic mock data from the scenario**, not live API data. The fault injector corrupts the mock data. This is how EVERY winning env works:

| Env | Ground Truth |
|-----|-------------|
| FinQA | CSV of questions + answers from SEC filings |
| Email Triage | Hardcoded labeled emails |
| AEGIS | 40 labeled attack/benign prompts |
| Chess | Game rules (programmatic) |
| Reasoning Gym | Seeded dataset with answer key |
| **ChaosAgent** | **Scenario DB: question + tool outputs + correct answer** |

The tools don't call real APIs at runtime. Each **scenario** contains:
```python
{
    "id": "S042",
    "question": "What is the current population of France, and what is the GDP per capita in USD?",
    "answer": {"population": 67390000, "gdp_per_capita": 44747.0},
    "tool_data": {
        "database_query": {"sql": "SELECT population, gdp_per_capita FROM countries WHERE name='France'",
                           "result": [{"population": 67390000, "gdp_per_capita": 44747.0}]},
        "web_search": {"query": "France population GDP per capita",
                       "result": [{"title": "France - Wikipedia", "snippet": "Population: 67.39M, GDP/capita: $44,747"}]},
        "knowledge_base_lookup": {"entity": "France",
                                  "result": {"population": 67390000, "gdp_per_capita": 44747.0, "capital": "Paris"}}
    },
    "grading": {
        "required_facts": [
            {"key": "population", "value": 67390000, "type": "numeric", "tolerance": 0.02},
            {"key": "gdp_per_capita", "value": 44747.0, "type": "numeric", "tolerance": 0.05}
        ]
    },
    "metadata": {
        "tier": "beginner",
        "domains": ["geography", "economics"],
        "min_tools_needed": 1,
        "cross_validation_possible": true
    }
}
```

**Key insight**: The tools are **scenario-aware**. When agent calls `web_search("France population")`, we return the scenario's pre-computed result (possibly corrupted by fault injector). This is deterministic, reproducible, and gradable.

### Problem 3: Tool Quality Over Quantity
**Old**: 65 tools padded with 8 weather variants, 8 finance clones, SpaceX API, astronauts, IFSC codes — useless niche garbage.  
**New**: ~30 tools that a **real generalist AI agent** would actually use. Every tool must pass the test: "Would a production AI assistant need this?" If you can't imagine asking ChatGPT/Claude to do it, the tool doesn't belong.

Design principle from the research: **Tools must create information asymmetry** — the agent CANNOT answer the question without using the tools. The answer lives in the tool data, not in the question text.

---

## 2. Redesigned Tool Catalog (30 Tools)

### Design Principles
1. **Every tool serves a distinct purpose** — no two tools overlap in function
2. **Tools mirror real agent capabilities** — search, compute, query, read, write, communicate, validate
3. **Progressive detail levels** — broad search → specific lookup → detailed query (like FinQA's get_descriptions → get_table_info → sql_query)
4. **Mix of retrieval, computation, mutation, and validation** — forces varied agent strategies

### 2.1 Information Retrieval (8 tools)

| # | Tool | Purpose | Why It Exists |
|---|------|---------|---------------|
| 1 | `web_search` | Search the web for information | Core retrieval — every agent needs this |
| 2 | `fetch_url` | Fetch and extract text content from a URL | Follow-up to search results, read full pages |
| 3 | `knowledge_base_lookup` | Look up structured facts about an entity (person, place, company, concept) | Quick factual lookup without search noise |
| 4 | `database_query` | Execute SQL against a relational database | Structured data retrieval — core agent skill |
| 5 | `document_search` | Search through a document collection (like a filing cabinet) | Find specific info in a corpus |
| 6 | `get_current_datetime` | Get current date, time, timezone | Temporal grounding — needed for freshness checks |
| 7 | `read_file` | Read contents of a file by path | File system access — fundamental capability |
| 8 | `api_call` | Make a generic HTTP API request (GET/POST) to a specified endpoint | Flexible integration point — covers any REST API |

### 2.2 Computation & Transformation (7 tools)

| # | Tool | Purpose | Why It Exists |
|---|------|---------|---------------|
| 9 | `calculator` | Evaluate mathematical expressions | Arithmetic, unit conversions, percentages |
| 10 | `python_execute` | Run Python code in a sandbox | General computation — Turing-complete fallback |
| 11 | `text_extract` | Extract structured data from unstructured text (regex, patterns) | Parse tool outputs, extract numbers/dates/names |
| 12 | `data_transform` | Filter, sort, aggregate tabular data | Process query results, find max/min/avg |
| 13 | `compare_values` | Compare two values with tolerance and report match/diff | Cross-validation helper |
| 14 | `json_query` | Query JSON data with JMESPath/JSONPath | Navigate nested API responses |
| 15 | `translate` | Translate text between languages | Multi-language data handling |

### 2.3 Storage & State (5 tools)

| # | Tool | Purpose | Why It Exists |
|---|------|---------|---------------|
| 16 | `scratchpad_write` | Store a key-value note for later use | Working memory across steps |
| 17 | `scratchpad_read` | Read a previously stored note | Recall earlier findings |
| 18 | `write_file` | Write content to a file | Persist computed results |
| 19 | `database_insert` | Insert/update a row in the database | Mutate state |
| 20 | `create_report` | Compile findings into a structured report before submitting | Pre-submission synthesis |

### 2.4 Validation & Verification (5 tools)

| # | Tool | Purpose | Why It Exists |
|---|------|---------|---------------|
| 21 | `validate_url` | Check if a URL is valid and reachable | Verify sources |
| 22 | `validate_data` | Check if data matches an expected schema/type | Catch corrupted fields |
| 23 | `check_consistency` | Compare two data sources and flag discrepancies | Cross-validation between tools |
| 24 | `fact_check` | Look up a specific claim and verify against authoritative source | Detect stale/wrong data |
| 25 | `hash_verify` | Compute hash of data to check integrity | Detect silent corruption |

### 2.5 Communication & Action (5 tools)

| # | Tool | Purpose | Why It Exists |
|---|------|---------|---------------|
| 26 | `send_notification` | Send a notification/alert message | Real agents take actions, not just retrieve |
| 27 | `schedule_task` | Schedule a follow-up action for later | Time-dependent workflows |
| 28 | `update_ticket` | Update a support/issue ticket with findings | State mutation in a ticketing system |
| 29 | `request_human_review` | Escalate to human when confidence is low | Know-your-limits signal |
| 30 | `submit_answer` | Submit final answer (SPECIAL — always works, never fails) | Termination signal |

### Tool Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Information Retrieval | 8 | Get data from various sources |
| Computation & Transform | 7 | Process and transform data |
| Storage & State | 5 | Remember and persist |
| Validation & Verification | 5 | Check data quality |
| Communication & Action | 5 | Act on findings |
| **TOTAL** | **30** | |

**Why 30, not 65?** Because every tool is distinct and necessary. An agent facing a question about a company's financials would naturally: `web_search` → `fetch_url` → `database_query` → `calculator` → `check_consistency` → `submit_answer`. No two tools do the same thing.

**Why these tools?** They match what real AI assistants (ChatGPT, Claude, Gemini) actually expose or would need: search, browse, code execution, database access, file I/O, structured data extraction, notification, and validation.

---

## 3. Ground Truth System

### 3.1 Scenario Database Structure

Each scenario is a self-contained question with pre-computed tool outputs and a known answer.

```python
@dataclass
class Scenario:
    """A single evaluation scenario."""
    id: str                          # Unique identifier
    question: str                    # Natural language question for the agent
    answer: dict                     # Ground truth answer (key-value facts)
    required_facts: list[Fact]       # What the agent must include in the answer
    tool_data: dict[str, dict]       # Pre-computed clean outputs for each relevant tool
    difficulty: str                  # warmup / beginner / intermediate / expert
    min_tools_needed: int            # Minimum tools to answer correctly
    tags: list[str]                  # Domain tags for curriculum tracking
    cross_validation_tools: list[list[str]]  # Groups of tools that can verify each other

@dataclass
class Fact:
    """A single fact that must appear in the answer."""
    key: str                         # e.g., "population"
    value: Any                       # e.g., 67390000
    type: str                        # "numeric", "text", "boolean", "date"
    tolerance: float = 0.0           # For numeric: relative tolerance (0.05 = 5%)
    alternatives: list[str] = None   # Acceptable alternative text values
```

### 3.2 How Tool Calls Work at Runtime

```
Agent calls: web_search("France population")
                    │
                    ▼
    ┌───────────────────────────────┐
    │   Tool Router                  │
    │                                │
    │   1. Look up scenario.tool_data│
    │      for "web_search"          │
    │   2. Fuzzy match query against │
    │      pre-computed queries      │
    │   3. Get clean result          │
    │                                │
    └──────────────┬────────────────┘
                   │
                   ▼
    ┌───────────────────────────────┐
    │   Fault Injector               │
    │                                │
    │   Based on seed + step + tier: │
    │   - 70% chance: return clean   │
    │   - 30% chance: inject fault   │
    │     (timeout/stale/corrupt/    │
    │      rate_limit/silent/partial)│
    └──────────────┬────────────────┘
                   │
                   ▼
        Return to agent (clean or corrupted)
```

### 3.3 Fuzzy Tool Routing

When the agent calls a tool, we need to match it against the scenario's pre-computed data. This handles the case where the agent asks slightly different queries than what we pre-computed:

```python
class ToolRouter:
    """Routes tool calls to pre-computed scenario data."""
    
    def route(self, tool_name: str, arguments: dict, scenario: Scenario) -> dict | None:
        """
        Look up the pre-computed tool output for this call.
        Returns None if the tool has no relevant data for this scenario
        (which is itself a valid signal — not every tool is useful for every question).
        """
        tool_entries = scenario.tool_data.get(tool_name, [])
        
        if not tool_entries:
            # Tool has no data for this scenario — return empty result
            return {"results": [], "message": "No relevant data found"}
        
        # For tools with query params, fuzzy match against pre-computed queries
        if "query" in arguments or "sql" in arguments:
            query = arguments.get("query") or arguments.get("sql", "")
            best_match = self._fuzzy_match(query, tool_entries)
            if best_match:
                return best_match["result"]
        
        # For non-query tools, return the default entry
        return tool_entries[0]["result"] if isinstance(tool_entries, list) else tool_entries
    
    def _fuzzy_match(self, query: str, entries: list) -> dict | None:
        """Find best matching pre-computed entry for the query."""
        # Keyword overlap matching
        query_words = set(query.lower().split())
        best_score = 0
        best_entry = None
        for entry in entries:
            entry_words = set(entry.get("query", "").lower().split())
            overlap = len(query_words & entry_words) / max(len(query_words | entry_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_entry = entry
        return best_entry if best_score > 0.3 else None
```

### 3.4 Handling "Off-Script" Tool Calls

What if the agent calls a tool that has no pre-computed data for this scenario?

- **`web_search("unrelated topic")`** → Returns `{"results": [], "message": "No relevant results"}` — realistic and non-punishing
- **`calculator("2+2")`** → Calculator, `python_execute`, and pure-computation tools ALWAYS work normally (they don't need scenario data). They compute from their inputs.
- **`database_query("SELECT * FROM nonexistent")`** → Returns `{"error": "Table not found"}` — realistic database error
- **`scratchpad_write/read`** → Always works (agent's own memory, not scenario-dependent)

### 3.5 Grading (Programmatic, No LLM)

```python
class Grader:
    """100% programmatic grading. No LLM judge."""
    
    def grade(self, submitted_answer: str, scenario: Scenario) -> float:
        """
        Returns 0.0 to 1.0 correctness score.
        Uses the same approach as FinQA (fuzzy numeric matching) 
        and Email Triage (weighted partial credit).
        """
        total_weight = 0.0
        earned_weight = 0.0
        
        for fact in scenario.required_facts:
            weight = 1.0  # Equal weight per fact
            total_weight += weight
            
            if fact.type == "numeric":
                extracted = self._extract_number(submitted_answer, fact.key)
                if extracted is not None:
                    relative_error = abs(extracted - fact.value) / max(abs(fact.value), 1e-9)
                    if relative_error <= fact.tolerance:
                        earned_weight += weight  # Full credit
                    elif relative_error <= fact.tolerance * 3:
                        earned_weight += weight * 0.5  # Partial credit
                        
            elif fact.type == "text":
                if self._text_match(submitted_answer, fact.value, fact.alternatives):
                    earned_weight += weight
                elif self._partial_text_match(submitted_answer, fact.value):
                    earned_weight += weight * 0.5
                    
            elif fact.type == "boolean":
                if self._boolean_match(submitted_answer, fact.value):
                    earned_weight += weight
        
        return earned_weight / max(total_weight, 1e-9)
```

---

## 4. Action Space

### Only Two Action Types

```python
class CallToolAction(BaseModel):
    """Call any of the 30 available tools."""
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")

class SubmitAnswerAction(BaseModel):
    """Submit the final answer. Terminates the episode."""
    answer: str = Field(..., description="The agent's final answer")
    reasoning: str = Field(default="", description="How the agent arrived at this answer")

# Union type
ChaosAgentAction = CallToolAction | SubmitAnswerAction
```

**No ListToolsAction**. Tool descriptions are included in the initial observation at reset. The agent sees all 30 tools from the start and must decide which to use.

### Initial Observation (at reset)

```python
class ChaosAgentObservation(BaseModel):
    """Returned after each step."""
    task_question: str                         # The question to answer
    tool_result: Optional[ToolResult] = None   # Result of last tool call (if any)
    available_tools: Optional[list[ToolDesc]] = None  # Full tool list — only in first obs
    warning: Optional[str] = None              # Repeat/circuit breaker warnings  
    steps_taken: int
    max_steps: int
```

At `reset()`, the observation includes `available_tools` (descriptions of all 30 tools). After that, each `step()` returns only the `tool_result`.

---

## 5. Fault Injection (Unchanged from v1)

Six failure modes, seed-based deterministic injection. Same as before but tuned:

| Mode | Probability by Tier |
|------|-------------------|
| `TIMEOUT` | warmup: 10%, beginner: 15%, intermediate: 20%, expert: 25% |
| `RATE_LIMIT` | warmup: 5%, beginner: 10%, intermediate: 15%, expert: 15% |
| `STALE_DATA` | warmup: 0%, beginner: 10%, intermediate: 15%, expert: 20% |
| `SILENT_FAIL` | warmup: 0%, beginner: 0%, intermediate: 10%, expert: 15% |
| `PARTIAL_RESPONSE` | warmup: 0%, beginner: 0%, intermediate: 10%, expert: 15% |
| `CORRUPT_FIELD` | warmup: 0%, beginner: 0%, intermediate: 0%, expert: 10% |
| **Total per call** | warmup: ~15%, beginner: ~30%, intermediate: ~50%, expert: ~70% |

Special rule: **`submit_answer`, `calculator`, `python_execute`, `scratchpad_read/write`** NEVER fail. These are agent-internal tools — making them unreliable would just be frustrating, not educational.

---

## 6. Scenario Design (50+ Scenarios)

### 6.1 Scenario Tiers

| Tier | # Scenarios | Tools Needed | Failure Complexity | Example |
|------|-------------|-------------|-------------------|---------|
| Warmup | 10 | 1-2 | Only timeout/rate_limit | "Look up the population of Germany" |
| Beginner | 15 | 2-3 | + stale data | "Find the CEO of Apple and the company's revenue" |
| Intermediate | 15 | 3-4 | + silent/partial fails | "Compare GDP of US and China, compute the ratio" |
| Expert | 15 | 4-6 | All 6 modes, cascading | "Cross-validate a company's market cap using 3 different data sources" |

### 6.2 Example Scenarios (Full Detail)

#### Warmup: W01 — Simple Fact Lookup
```python
Scenario(
    id="W01",
    question="What is the population of Germany?",
    answer={"population": 83200000},
    required_facts=[
        Fact(key="population", value=83200000, type="numeric", tolerance=0.02)
    ],
    tool_data={
        "knowledge_base_lookup": {
            "entity": "Germany",
            "result": {"name": "Germany", "population": 83200000, "capital": "Berlin", 
                       "continent": "Europe", "gdp_usd": 4.26e12}
        },
        "web_search": [
            {"query": "Germany population", 
             "result": [{"title": "Germany - Wikipedia", 
                         "snippet": "Germany has a population of approximately 83.2 million people."}]}
        ],
        "database_query": [
            {"query": "SELECT population FROM countries WHERE name='Germany'",
             "result": [{"population": 83200000}]}
        ]
    },
    difficulty="warmup",
    min_tools_needed=1,
    tags=["geography", "demographics"],
    cross_validation_tools=[["knowledge_base_lookup", "web_search", "database_query"]]
)
```

#### Beginner: B05 — Two-Step Chain
```python
Scenario(
    id="B05",
    question="What is the GDP per capita of the country whose capital is Tokyo? Express in USD.",
    answer={"country": "Japan", "gdp_per_capita": 33950.0},
    required_facts=[
        Fact(key="country", value="Japan", type="text"),
        Fact(key="gdp_per_capita", value=33950.0, type="numeric", tolerance=0.05)
    ],
    tool_data={
        "knowledge_base_lookup": [
            {"entity": "Tokyo", 
             "result": {"type": "city", "country": "Japan", "population": 13960000}},
            {"entity": "Japan",
             "result": {"name": "Japan", "population": 125700000, "gdp_usd": 4.27e12, 
                        "gdp_per_capita": 33950.0, "capital": "Tokyo"}}
        ],
        "web_search": [
            {"query": "capital Tokyo which country",
             "result": [{"title": "Tokyo - Capital of Japan", "snippet": "Tokyo is the capital of Japan."}]},
            {"query": "Japan GDP per capita USD",
             "result": [{"title": "Japan Economy", "snippet": "Japan's GDP per capita is approximately $33,950 USD."}]}
        ],
        "database_query": [
            {"query": "SELECT name, gdp_per_capita FROM countries WHERE capital='Tokyo'",
             "result": [{"name": "Japan", "gdp_per_capita": 33950.0}]}
        ]
    },
    difficulty="beginner",
    min_tools_needed=2,
    tags=["geography", "economics"],
    cross_validation_tools=[["knowledge_base_lookup", "database_query"]]
)
```

#### Intermediate: I03 — Computation + Cross-Validation
```python
Scenario(
    id="I03",
    question="Compare the population density of India and Australia. Which is higher and by what factor? (Population density = population / area in km²)",
    answer={
        "india_density": 473.4, "australia_density": 3.3, 
        "higher": "India", "factor": 143.5
    },
    required_facts=[
        Fact(key="higher", value="India", type="text"),
        Fact(key="factor", value=143.5, type="numeric", tolerance=0.10)
    ],
    tool_data={
        "knowledge_base_lookup": [
            {"entity": "India",
             "result": {"population": 1428627663, "area_km2": 3287263, "capital": "New Delhi"}},
            {"entity": "Australia",
             "result": {"population": 26473055, "area_km2": 7692024, "capital": "Canberra"}}
        ],
        "database_query": [
            {"query": "SELECT name, population, area_km2 FROM countries WHERE name IN ('India', 'Australia')",
             "result": [
                 {"name": "India", "population": 1428627663, "area_km2": 3287263},
                 {"name": "Australia", "population": 26473055, "area_km2": 7692024}
             ]}
        ],
        "calculator": "always_live",  # Calculator computes from inputs, no pre-computed data needed
        "python_execute": "always_live"
    },
    difficulty="intermediate",
    min_tools_needed=3,
    tags=["geography", "math", "comparison"],
    cross_validation_tools=[["knowledge_base_lookup", "database_query"]]
)
```

#### Expert: E02 — Cross-Validate with Corrupt Data
```python
Scenario(
    id="E02",
    question="A company called 'NovaTech' reported revenue of $2.3B last quarter. Verify this claim using at least 2 independent sources. Is the claim accurate? If not, what is the actual revenue?",
    answer={"claim_accurate": False, "actual_revenue_b": 1.87, "discrepancy_pct": 23.0},
    required_facts=[
        Fact(key="claim_accurate", value=False, type="boolean"),
        Fact(key="actual_revenue_b", value=1.87, type="numeric", tolerance=0.05)
    ],
    tool_data={
        "web_search": [
            {"query": "NovaTech revenue last quarter",
             "result": [
                 {"title": "NovaTech Q3 Earnings", "snippet": "NovaTech reported $1.87B revenue, missing analyst estimates."},
                 {"title": "NovaTech PR", "snippet": "NovaTech announces strong quarter with continued growth."}
             ]}
        ],
        "database_query": [
            {"query": "SELECT revenue_b, quarter FROM financials WHERE company='NovaTech' ORDER BY quarter DESC LIMIT 1",
             "result": [{"revenue_b": 1.87, "quarter": "Q3-2025"}]}
        ],
        "document_search": [
            {"query": "NovaTech financial filing",
             "result": [{"doc_id": "SEC-10Q-NT-2025Q3", "excerpt": "Total revenue: $1,870,000,000 for the three months ended Sep 30, 2025."}]}
        ],
        "fetch_url": [
            {"url": "https://novatech.example.com/investors",
             "result": {"content": "Quarterly Revenue: $1.87B | YoY Growth: 12% | Net Income: $340M"}}
        ],
        "calculator": "always_live"
    },
    difficulty="expert",
    min_tools_needed=2,
    tags=["finance", "verification", "cross_validation"],
    cross_validation_tools=[["web_search", "database_query", "document_search", "fetch_url"]]
)
```

### 6.3 Scenario Domains

| Domain | Warmup | Beginner | Intermediate | Expert | Total |
|--------|--------|----------|-------------|--------|-------|
| Geography/Demographics | 3 | 3 | 2 | 2 | 10 |
| Finance/Business | 2 | 3 | 3 | 4 | 12 |
| Science/Technology | 2 | 3 | 3 | 3 | 11 |
| General Knowledge | 2 | 3 | 3 | 3 | 11 |
| Multi-Domain | 1 | 3 | 4 | 3 | 11 |
| **Total** | **10** | **15** | **15** | **15** | **55** |

---

## 7. Reward Function

### 7.1 Simple, Clean, Deterministic

```python
def compute_reward(self) -> float:
    """
    Reward range: -1.0 to +1.0
    Normalized for GRPO training stability.
    
    Components:
    1. Answer correctness:   0.0 to 1.0  (THE main signal, 70% weight)
    2. Resilience bonus:     0.0 to 0.2  (answered correctly despite failures)
    3. Efficiency:          -0.1 to 0.1  (step economy)
    4. Repeat penalty:      -0.2 to 0.0  (punish blind repetition)
    """
    
    # 1. CORRECTNESS (0.0 to 1.0) — 70% of reward
    correctness = self.grader.grade(self.submitted_answer, self.scenario)
    
    # 2. RESILIENCE BONUS (0.0 to 0.2)
    # +0.2 if agent got correct answer despite tool failures occurring
    resilience = 0.0
    if correctness > 0.7 and self.faults_injected > 0:
        resilience = 0.2
    
    # 3. EFFICIENCY (-0.1 to 0.1)
    # Good: used close to min_tools_needed steps
    # Bad: used way more steps than needed
    step_ratio = self.steps_taken / max(self.scenario.min_tools_needed * 2, 1)
    if step_ratio <= 1.5:
        efficiency = 0.1   # Efficient
    elif step_ratio <= 3.0:
        efficiency = 0.0   # Acceptable
    else:
        efficiency = -0.1  # Wasteful
    
    # 4. REPEAT PENALTY (-0.2 to 0.0)
    # -0.05 per repeated identical call (max -0.2)
    repeat_penalty = max(-0.2, -0.05 * self.repeat_tracker.total_repeats)
    
    # TOTAL: weighted sum, clamped to [-1.0, 1.0]
    total = (correctness * 0.7) + resilience + efficiency + repeat_penalty
    return max(-1.0, min(1.0, total))
```

### 7.2 Step Penalty for Timeout

If agent hits max_steps without submitting: `reward = -0.5`

### 7.3 Why This Is Better Than v1

| v1 Problem | v2 Fix |
|-----------|--------|
| Reward range -2 to +8 (unstable for GRPO) | Normalized to -1.0 to +1.0 |
| 6 reward components (overengineered) | 4 components, correctness dominates |
| Phase tracking (complex, fragile) | Removed — correctness IS the signal |
| LLM-detectable failure handling | Simple: did you get it right despite failures? |
| "Did agent cross-validate" heuristic | Rewarded implicitly through correctness |

---

## 8. Curriculum System (Simplified)

The curriculum controls ONLY which failure modes are active and at what probability. **All tools are always available.** Tier advancement is based purely on episode success rate.

```python
class CurriculumController:
    TIERS = ["warmup", "beginner", "intermediate", "expert"]
    
    ADVANCEMENT = {
        "warmup":       {"min_episodes": 5,  "min_success_rate": 0.6},
        "beginner":     {"min_episodes": 8,  "min_success_rate": 0.5},
        "intermediate": {"min_episodes": 10, "min_success_rate": 0.4},
        "expert":       None,  # Final tier
    }
    
    def __init__(self):
        self.tier_index = 0
        self.episodes_in_tier = 0
        self.recent_scores = deque(maxlen=15)
    
    @property
    def current_tier(self) -> str:
        return self.TIERS[self.tier_index]
    
    def record_episode(self, correctness: float):
        self.episodes_in_tier += 1
        self.recent_scores.append(correctness)
        
        req = self.ADVANCEMENT.get(self.current_tier)
        if req is None:
            return  # Already at expert
        
        if (self.episodes_in_tier >= req["min_episodes"] and 
            self.success_rate >= req["min_success_rate"]):
            self.tier_index = min(self.tier_index + 1, len(self.TIERS) - 1)
            self.episodes_in_tier = 0
    
    @property
    def success_rate(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(1 for s in self.recent_scores if s > 0.5) / len(self.recent_scores)
```

---

## 9. Models (Pydantic)

```python
# === Actions ===
class CallToolAction(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)

class SubmitAnswerAction(BaseModel):
    answer: str
    reasoning: str = ""

ChaosAgentAction = CallToolAction | SubmitAnswerAction

# === Observations ===
class ToolDesc(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema

class ToolResult(BaseModel):
    tool_name: str
    result: Optional[dict] = None
    error: Optional[str] = None
    message: Optional[str] = None

class ChaosAgentObservation(BaseModel):
    task_question: str
    tool_result: Optional[ToolResult] = None
    available_tools: Optional[list[ToolDesc]] = None  # Only in first obs
    warning: Optional[str] = None
    steps_taken: int
    max_steps: int

# === State ===
class ChaosAgentState(BaseModel):
    scenario_id: str
    task_question: str
    difficulty_tier: str
    steps_taken: int
    max_steps: int
    tools_called: list[str]
    faults_injected: int
    is_done: bool
    cumulative_reward: float
    curriculum_tier: str
    episodes_completed: int
```

---

## 10. File Structure

```
chaosagent/
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── README.md
├── client.py                       # MCPToolClient (pass)
├── models.py                       # Actions, Observations, State
├── inference.py                    # Baseline agent
├── scenarios/
│   ├── warmup.json                 # 10 warmup scenarios
│   ├── beginner.json               # 15 beginner scenarios
│   ├── intermediate.json           # 15 intermediate scenarios
│   └── expert.json                 # 15 expert scenarios
├── server/
│   ├── __init__.py
│   ├── app.py                      # create_app()
│   ├── environment.py              # ChaosAgentEnvironment
│   ├── tool_router.py              # Routes tool calls to scenario data
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py             # All 30 tool definitions
│   │   └── live_tools.py           # calculator, python_execute, scratchpad (always-live)
│   ├── fault_injector.py
│   ├── curriculum.py
│   ├── grader.py
│   └── repeat_tracker.py
└── tests/
    ├── test_grader.py
    ├── test_fault_injector.py
    └── test_tool_router.py
```

---

## 11. Implementation Priority

### Day 1 (Today): Core Infrastructure (~6 hours)
1. `models.py` — All Pydantic models
2. `server/environment.py` — Reset/step/state skeleton
3. `server/fault_injector.py` — 6 failure modes
4. `server/grader.py` — Programmatic scoring
5. `server/tool_router.py` — Scenario-based tool routing
6. `server/tools/registry.py` — Tool definitions (names, descriptions, schemas)
7. `server/tools/live_tools.py` — Calculator, Python exec, scratchpad
8. `openenv.yaml` + `pyproject.toml` + `Dockerfile`

### Day 2: Scenarios & Curriculum (~5 hours)
9. Write 10 warmup scenarios (JSON)
10. Write 15 beginner scenarios
11. Write 15 intermediate scenarios  
12. Write 10 expert scenarios (50 total minimum)
13. `server/curriculum.py`
14. `server/repeat_tracker.py`
15. Integration testing — run through all scenarios

### Day 3: Polish & Submit (~4 hours)
16. `inference.py` — Baseline agent
17. `README.md` — Full documentation
18. `client.py` — MCPToolClient
19. `openenv validate` pass
20. Deploy to HF Space
21. GRPO training script (if time)

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pre-computed scenarios, not live APIs** | Every winning env uses pre-computed ground truth. Live APIs break reproducibility. |
| **30 quality tools, not 65 padded** | Real agents need diverse capabilities, not 8 weather variants. Every tool is distinct. |
| **No ListToolsAction** | All tools always visible. Agent must choose wisely. This tests tool selection intelligence. |
| **Programmatic grading only** | LLM judges add variance that hurts GRPO training. FinQA and Email Triage prove programmatic works. |
| **Reward normalized to [-1, 1]** | Stable for GRPO. Correctness is 70% of signal. |
| **calculator/python_execute never fail** | These are agent-internal computation — making them unreliable isn't educational. |
| **Scenario DB as JSON files** | Easy to add scenarios, version control, review. No database dependency. |
| **Fuzzy tool routing** | Agents won't use exact pre-computed queries. Keyword overlap matching handles variation. |

---

*ChaosAgent v2: Pre-computed truth, real diversity, all tools always available.*
