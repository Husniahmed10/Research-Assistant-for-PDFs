"""
tracing/langsmith_setup.py — LangSmith Agentic Workflow Tracing Configuration
===============================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import os and settings

Step 2:  Write setup_langsmith() → None:
            - Called once in main.py during startup
            - Set environment variables that LangChain/LangGraph reads automatically:
                os.environ["LANGCHAIN_TRACING_V2"]  = "true"
                os.environ["LANGCHAIN_API_KEY"]     = settings.LANGCHAIN_API_KEY
                os.environ["LANGCHAIN_PROJECT"]     = settings.LANGCHAIN_PROJECT
                os.environ["LANGCHAIN_ENDPOINT"]    = settings.LANGCHAIN_ENDPOINT
            - Once set, EVERY LangGraph agent run is automatically traced
            - No additional code needed in agent files!

Step 3:  (Optional) Add custom metadata to runs:
            from langsmith import traceable

            @traceable(
                name="research_qa_agent",
                tags=["production", "rag"],
                metadata={"version": "1.0"}
            )
            async def run_agent(state: AgentState):
                return await agent.ainvoke(state)

WHAT LANGSMITH DASHBOARD SHOWS:
    - Full agent execution graph (which nodes ran, in what order)
    - Each node's input and output state
    - LLM calls: prompt, response, token count, latency, cost
    - Retrieval results: what chunks were retrieved and their scores
    - Overall run latency breakdown
    - Side-by-side comparison of different runs (great for prompt debugging)
    - Error traces with full stack

TIP FOR DEMO:
    Open LangSmith dashboard during demo → run a query → show the interviewer
    the entire agentic reasoning trace live in real-time!
"""

