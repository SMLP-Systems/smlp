# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from smlp_agent import SmlpAgent  # Assumes your agent is correctly implemented

"""
LangGraph-based multi-step execution pipeline for SMLP tasks.

This module allows chaining SMLP task nodes into a directed execution graph
(e.g., training → evaluation → reporting), based on a user’s natural language query.
"""


# Define shared state
class SmlpState(TypedDict):
    input: str
    messages: List[HumanMessage]
    smlp_result: str
    evaluation: str

# Initialize SMLP Agent
smlp_agent = SmlpAgent()

# Node: Execute the user's text as an SMLP command
def smlp_task_node(state: SmlpState) -> SmlpState:
    user_input = state["input"]
    result = smlp_agent.run_text_command(user_input)
    state["smlp_result"] = result
    state["messages"].append(HumanMessage(content=result))
    return state

# Node: Evaluate the result
def evaluation_node(state: SmlpState) -> SmlpState:
    result = state["smlp_result"]
    evaluation = f"SMLP Task executed. Summary:\n{result[:500]}..."
    state["evaluation"] = evaluation
    state["messages"].append(HumanMessage(content=evaluation))
    return state

# Build the graph
builder = StateGraph(SmlpState)
builder.add_node("smlp_task", smlp_task_node)
builder.add_node("evaluate", evaluation_node)

builder.set_entry_point("smlp_task")
builder.add_edge("smlp_task", "evaluate")

graph = builder.compile()

# Function to run
def run_mcp(query: str):
    initial_state = {"input": query, "messages": [], "smlp_result": "", "evaluation": ""}
    final_state = graph.invoke(initial_state)
    return final_state

# Test
if __name__ == "__main__":
    #query = "Fine-tune TinyLlama on AG News and summarize results"
    query = "Train DT model on data file '../regr_smlp/data/smlp_toy_num_resp_mult.csv', responses y1,y2, features x,p1,p2; save model as model_test99. Then pareto optimize the model configuration using lazy strategy. Set epsilon = 0.04 and delta_rel = 0.02. Use spec file '../regr_smlp/specs/smlp_toy_num_resp_mult_free_inps_beta_objv.spec', output directory '../regr_smlp/code', prefix 'regr_test99'. Then summaarize SMLP results."
    result = run_mcp(query)
    print("FINAL EVALUATION:\n", result["evaluation"])
