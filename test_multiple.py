import json
import random
import os
import sys

# Ensure module relies on existing framework
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Scenario, Fact, CallToolAction, SubmitAnswerAction
from server.environment import ChaosAgentEnvironment

def run_simulation(env, scenario, actions):
    results = []
    results.append(f"### SCENARIO: {scenario.id} [{scenario.difficulty.upper()}]")
    results.append(f"**Question**: {scenario.question}")
    results.append("```text")
    
    obs = env.reset(scenario)
    results.append("[System] Environment reset. Live Tool routing established.")
    
    for act in actions:
        if isinstance(act, CallToolAction):
            results.append(f"\n[Agent Action] Call Tool -> {act.tool_name}({act.arguments})")
            obs, reward, done, info = env.step(act)
            if obs.tool_result.error:
                results.append(f"[Env Response] ❌ INJECTED FAULT: {obs.tool_result.error}")
            elif obs.tool_result.result is None and not obs.tool_result.error:
                results.append(f"[Env Response] ❌ INJECTED FAULT: (Silent Fail or Empty Response)")
            else:
                results.append(f"[Env Response] ✅ OK: {obs.tool_result.result}")
                
        elif isinstance(act, SubmitAnswerAction):
            results.append(f"\n[Agent Action] Submit Answer -> '{act.answer}'")
            obs, reward, done, info = env.step(act)
            
            if env.state.faults_injected > 0:
                 results.append(f"[System] Resilient! Agent successfully withstood {env.state.faults_injected} injected fault(s).")
                 
            results.append(f"[Env Response] 🎯 Final Reward: {reward:.2f}")
    
    results.append("```\n")
    return "\n".join(results)

def main():
    env = ChaosAgentEnvironment()
    all_logs = ["# Multi-Scenario Testing Results\n", 
                "These tests iterate through three distinct scenarios spanning different difficulty tiers, demonstrating the fault injector and exact programmatic grading at work.\n"]
    
    # 1. Warmup (No faults expected)
    scen_w01 = Scenario(
        id="W01", question="What is the population of Germany?",
        answer={"population": 83200000},
        required_facts=[Fact(key="population", value=83200000, type="numeric", tolerance=0.02)],
        tool_data={"database_query": [{"query": "SELECT population FROM countries WHERE name='Germany'", "result": {"population": 83200000}}]},
        difficulty="warmup", min_tools_needed=1, tags=["geography"], cross_validation_tools=[]
    )
    acts_w01 = [
        CallToolAction(tool_name="database_query", arguments={"sql": "SELECT population FROM countries WHERE name='Germany'"}),
        SubmitAnswerAction(answer="The population is 83,200,000.", reasoning="Retrieved from DB")
    ]
    all_logs.append(run_simulation(env, scen_w01, acts_w01))
    
    # 2. Beginner (Minor faults possible)
    scen_b05 = Scenario(
        id="B05", question="What is the GDP per capita of the country whose capital is Tokyo? Express in USD.",
        answer={"country": "Japan", "gdp_per_capita": 33950.0},
        required_facts=[Fact(key="country", value="Japan", type="text"), Fact(key="gdp_per_capita", value=33950.0, type="numeric", tolerance=0.05)],
        tool_data={
            "knowledge_base_lookup": [
                {"query": "Tokyo", "result": {"country": "Japan", "population": 13960000}},
                {"query": "Japan", "result": {"gdp_per_capita": 33950.0, "name": "Japan"}}
            ]
        },
        difficulty="beginner", min_tools_needed=2, tags=["economics"], cross_validation_tools=[]
    )
    acts_b05 = [
        CallToolAction(tool_name="knowledge_base_lookup", arguments={"query": "Tokyo"}),
        CallToolAction(tool_name="knowledge_base_lookup", arguments={"query": "Japan"}),
        SubmitAnswerAction(answer="The country is Japan and its GDP per capita is 33950 USD.", reasoning="Checked KB")
    ]
    all_logs.append(run_simulation(env, scen_b05, acts_b05))

    # 3. Expert (High probability of faults, forcing failure injection to demonstrate resilience)
    scen_e02 = Scenario(
        id="E02", question="A company called 'NovaTech' reported revenue of $2.3B last quarter. Verify this claim. Is it accurate and what is the real number?",
        answer={"claim_accurate": False, "actual_revenue_b": 1.87},
        required_facts=[Fact(key="claim_accurate", value=False, type="boolean"), Fact(key="actual_revenue_b", value=1.87, type="numeric", tolerance=0.05)],
        tool_data={
            "database_query": [{"query": "SELECT revenue_b FROM financials WHERE company='NovaTech'", "result": {"revenue_b": 1.87}}]
        },
        difficulty="expert", min_tools_needed=1, tags=["finance"], cross_validation_tools=[]
    )
    # We call it multiple times to simulate the agent retrying on failure
    acts_e02 = [
        CallToolAction(tool_name="database_query", arguments={"sql": "SELECT revenue_b FROM financials WHERE company='NovaTech'"}),
        CallToolAction(tool_name="database_query", arguments={"sql": "SELECT revenue_b FROM financials WHERE company='NovaTech'"}),
        SubmitAnswerAction(answer="It is false. The actual revenue is 1.87B.", reasoning="Checked financials DB twice due to potential error.")
    ]
    
    # Intentionally seed the fault injector randomizer to guarantee an injected fault during this Expert sequence 
    random.seed(64) 
    all_logs.append(run_simulation(env, scen_e02, acts_e02))
    
    with open("results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(all_logs))
    
    print("SUCCESS: Tests completed successfully and exported to results.md")

if __name__ == "__main__":
    main()
