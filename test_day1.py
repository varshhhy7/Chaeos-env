import os
import sys

# Ensure module is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Scenario, Fact, CallToolAction, SubmitAnswerAction
from server.environment import ChaosAgentEnvironment

def main():
    print("Testing module imports and basic instantiation...")
    
    env = ChaosAgentEnvironment()
    print("Environment created successfully.")
    
    mock_scenario = Scenario(
        id="W01",
        question="What is the population of Germany?",
        answer={"population": 83200000},
        required_facts=[
            Fact(key="population", value=83200000, type="numeric", tolerance=0.02)
        ],
        tool_data={
            "database_query": {
                "query": "SELECT population FROM countries WHERE name='Germany'",
                "result": {"population": 83200000}
            }
        },
        difficulty="warmup",
        min_tools_needed=1,
        tags=["geography"],
        cross_validation_tools=[]
    )
    
    obs = env.reset(mock_scenario)
    print(f"Observation task question: {obs.task_question}")
    print(f"Available tools count: {len(obs.available_tools)}")
    
    action1 = CallToolAction(tool_name="database_query", arguments={"sql": "SELECT population FROM countries WHERE name='Germany'"})
    obs1, r1, d1, info1 = env.step(action1)
    
    print(f"Tool Result: {obs1.tool_result.result}")
    
    action2 = SubmitAnswerAction(answer="The population is 83,200,000.", reasoning="From database.")
    obs2, r2, d2, info2 = env.step(action2)
    
    print(f"Final Reward: {r2}")
    print(f"Done: {d2}")

if __name__ == "__main__":
    main()
