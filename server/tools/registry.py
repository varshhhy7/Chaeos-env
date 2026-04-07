from typing import Dict, Any

# This registry contains the definitions of all 30 tools for the environment.
# These match the design from section 2 of chaosagent_v2.md

TOOL_REGISTRY = [
    # 2.1 Information Retrieval
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch and extract text content from a URL",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    },
    {
        "name": "knowledge_base_lookup",
        "description": "Look up structured facts about an entity",
        "parameters": {
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"]
        }
    },
    {
        "name": "database_query",
        "description": "Execute SQL against a relational database",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"]
        }
    },
    {
        "name": "document_search",
        "description": "Search through a document collection",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Get current date, time, timezone",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "default": "UTC"}}
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a file by path",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "api_call",
        "description": "Make a generic HTTP API request",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "method": {"type": "string", "default": "GET"}},
            "required": ["url"]
        }
    },

    # 2.2 Computation & Transformation
    {
        "name": "calculator",
        "description": "Evaluate mathematical expressions",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    },
    {
        "name": "python_execute",
        "description": "Run Python code in a sandbox",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"]
        }
    },
    {
        "name": "text_extract",
        "description": "Extract structured data from unstructured text",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "pattern": {"type": "string"}},
            "required": ["text", "pattern"]
        }
    },
    {
        "name": "data_transform",
        "description": "Filter, sort, aggregate tabular data",
        "parameters": {
            "type": "object",
            "properties": {"data": {"type": "string"}, "operation": {"type": "string"}},
            "required": ["data", "operation"]
        }
    },
    {
        "name": "compare_values",
        "description": "Compare two values with tolerance",
        "parameters": {
            "type": "object",
            "properties": {"value1": {"type": "number"}, "value2": {"type": "number"}, "tolerance": {"type": "number", "default": 0.0}},
            "required": ["value1", "value2"]
        }
    },
    {
        "name": "json_query",
        "description": "Query JSON data",
        "parameters": {
            "type": "object",
            "properties": {"data": {"type": "string"}, "query": {"type": "string"}},
            "required": ["data", "query"]
        }
    },
    {
        "name": "translate",
        "description": "Translate text between languages",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"]
        }
    },

    # 2.3 Storage & State
    {
        "name": "scratchpad_write",
        "description": "Store a key-value note",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"]
        }
    },
    {
        "name": "scratchpad_read",
        "description": "Read a previously stored note",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]
        }
    },
    {
        "name": "database_insert",
        "description": "Insert/update a row in the database",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}, "data": {"type": "string"}},
            "required": ["table", "data"]
        }
    },
    {
        "name": "create_report",
        "description": "Compile findings into a structured report",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"]
        }
    },

    # 2.4 Validation & Verification
    {
        "name": "validate_url",
        "description": "Check if a URL is valid and reachable",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"]
        }
    },
    {
        "name": "validate_data",
        "description": "Check if data matches an expected schema",
        "parameters": {
            "type": "object",
            "properties": {"data": {"type": "string"}, "schema": {"type": "string"}},
            "required": ["data", "schema"]
        }
    },
    {
        "name": "check_consistency",
        "description": "Compare two data sources and flag discrepancies",
        "parameters": {
            "type": "object",
            "properties": {"source1": {"type": "string"}, "source2": {"type": "string"}},
            "required": ["source1", "source2"]
        }
    },
    {
        "name": "fact_check",
        "description": "Look up a specific claim and verify against authoritative source",
        "parameters": {
            "type": "object",
            "properties": {"claim": {"type": "string"}},
            "required": ["claim"]
        }
    },
    {
        "name": "hash_verify",
        "description": "Compute hash of data to check integrity",
        "parameters": {
            "type": "object",
            "properties": {"data": {"type": "string"}, "expected_hash": {"type": "string"}},
            "required": ["data", "expected_hash"]
        }
    },

    # 2.5 Communication & Action
    {
        "name": "send_notification",
        "description": "Send a notification/alert message",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"]
        }
    },
    {
        "name": "schedule_task",
        "description": "Schedule a follow-up action for later",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}, "time": {"type": "string"}},
            "required": ["task", "time"]
        }
    },
    {
        "name": "update_ticket",
        "description": "Update a support/issue ticket with findings",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "update": {"type": "string"}},
            "required": ["ticket_id", "update"]
        }
    },
    {
        "name": "request_human_review",
        "description": "Escalate to human when confidence is low",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"]
        }
    },
    # The submit_answer tool is technically an Action, but we can list it for completeness or handle it structurally.
    {
        "name": "submit_answer",
        "description": "Submit final answer. Terminates the episode.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "reasoning": {"type": "string"}},
            "required": ["answer"]
        }
    }
]

def get_all_tools() -> list[Dict[str, Any]]:
    return TOOL_REGISTRY
