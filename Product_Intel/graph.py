# product_intel/graph.py
# ----------------------
# The LangGraph "Sidekick" class:
# - State includes messages and a "mode" described through success_criteria
# - worker() composes the system prompt from agents.py based on mode
# - ToolNode executes any required tool calls
# - evaluator() checks if success criteria is met or more input is needed
#
# The public API exposes:
#   await sidekick.setup()        -> loads tools, builds graph
#   await sidekick.run_step(...)  -> runs one graph pass (worker->tools->eval)
#   sidekick.cleanup()            -> closes Playwright etc.

from typing import Annotated, List, Any, Optional, Dict, Tuple
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents import select_prompt
from tools import load_all_tools, graceful_close


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user or the assistant is stuck"
    )


class Sidekick:
    """
    Sidekick: a graph-driven assistant.
    - worker: crafts response using selected analyst prompt and tools
    - tools: executes pending tool calls (if any)
    - evaluator: checks result vs success_criteria to continue or stop
    """

    def __init__(self, model_name_worker: str = "gpt-4o-mini", model_name_eval: str = "gpt-4o-mini"):
        self._worker_llm_with_tools = None
        self._evaluator_llm_structured = None
        self._tools = None
        self._graph = None

        self._browser = None
        self._playwright = None

        self._memory = MemorySaver()
        self._thread_id = "product-intel-sidekick"

        # Model names can be overridden by caller
        self._model_name_worker = model_name_worker
        self._model_name_eval = model_name_eval

    async def setup(self) -> None:
        """
        Initialize tools, bind models to tools, and compile the graph.
        """
        self._tools, self._browser, self._playwright = await load_all_tools()

        worker_llm = ChatOpenAI(model=self._model_name_worker)
        self._worker_llm_with_tools = worker_llm.bind_tools(self._tools)

        evaluator_llm = ChatOpenAI(model=self._model_name_eval)
        self._evaluator_llm_structured = evaluator_llm.with_structured_output(EvaluatorOutput)

        await self._build_graph()

    def _compose_system_message(self, state: State) -> str:
        """
        Compose the system message dynamically based on success_criteria content.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = state.get("success_criteria", "")
        analyst_prompt = select_prompt(mode)

        base = f"""Current date/time: {now}

General rules:
- Use tools when they can improve freshness, accuracy, or provide URLs.
- If you run Python REPL, remember to print() results to see output.
- Prefer concise, well-structured Markdown.
- ALWAYS end with a '## Sources' section listing raw URLs you consulted.
"""

        if state.get("feedback_on_work"):
            base += f"""
Important: Your last attempt did not meet the success criteria.
Feedback to address:
{state['feedback_on_work']}
"""

        return analyst_prompt + "\n\n" + base

    def worker(self, state: State) -> Dict[str, Any]:
        """
        Worker composes system prompt + user messages, then invokes the bound LLM-with-tools.
        """
        system_message = self._compose_system_message(state)

        # Ensure we have exactly one SystemMessage at the top.
        messages = state["messages"]
        found_system = False
        for m in messages:
            if isinstance(m, SystemMessage):
                m.content = system_message
                found_system = True
                break
        if not found_system:
            messages = [SystemMessage(content=system_message)] + messages

        response = self._worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def worker_router(self, state: State) -> str:
        """
        Route to tools if the last message queued tool calls; else to evaluator.
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "evaluator"

    def _format_history(self, messages: List[Any]) -> str:
        """
        Render concise history for evaluator.
        """
        out = []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append(f"User: {m.content}")
            elif isinstance(m, AIMessage):
                out.append(f"Assistant: {m.content or '[Tool call]'}")
        return "\n".join(out)

    def evaluator(self, state: State) -> State:
        """
        Evaluate whether success criteria is met or more input is needed.
        """
        last_text = state["messages"][-1].content
        sys = "You evaluate whether the Assistant satisfied the user's success criteria."

        user = f"""Conversation:
{self._format_history(state["messages"])}

Success criteria:
{state["success_criteria"]}

Assistant's last message:
{last_text}

Score whether the criteria is met, provide feedback, and decide if more input is needed.
"""
        eval_messages = [SystemMessage(content=sys), HumanMessage(content=user)]
        result = self._evaluator_llm_structured.invoke(eval_messages)

        new_state: State = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback: {result.feedback}",
                }
            ],
            "feedback_on_work": result.feedback,
            "success_criteria_met": result.success_criteria_met,
            "user_input_needed": result.user_input_needed,
        }
        return new_state

    def route_post_eval(self, state: State) -> str:
        """
        If done or needs user input -> END; else loop back to worker.
        """
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        return "worker"

    async def _build_graph(self) -> None:
        """
        Compile the StateGraph with nodes and edges.
        """
        sg = StateGraph(State)
        sg.add_node("worker", self.worker)
        sg.add_node("tools", ToolNode(tools=self._tools))
        sg.add_node("evaluator", self.evaluator)

        sg.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        sg.add_edge("tools", "worker")
        sg.add_conditional_edges("evaluator", self.route_post_eval, {"worker": "worker", "END": END})
        sg.add_edge(START, "worker")

        self._graph = sg.compile(checkpointer=self._memory)

    async def run_step(
        self,
        user_message: str,
        success_criteria: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Run one super-step of the graph. Returns updated chat history.
        """
        # Convert simple history into LC messages
        msgs: List[Any] = []
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))

        # Append the new user message
        msgs.append(HumanMessage(content=user_message))

        # Build initial state
        state: State = {
            "messages": msgs,
            "success_criteria": success_criteria or "Provide a clear, accurate analysis.",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        config = {"configurable": {"thread_id": self._thread_id}}
        result = await self._graph.ainvoke(state, config=config)

        # The ToolNode + Evaluator append their own assistant messages.
        # For convenience, we return the last assistant response + evaluator feedback.
        out = history.copy()
        out.append({"role": "user", "content": user_message})

        # Depending on routes, result["messages"] can vary; we add anything AI said at the end:
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                out.append({"role": "assistant", "content": m.content})
            elif isinstance(m, dict) and m.get("role") == "assistant":
                out.append({"role": "assistant", "content": m.get("content", "")})

        return out

    def cleanup(self) -> None:
        """
        Close browser + playwright gracefully.
        """
        # This function is sync (Gradio callback friendly) and delegates to async closer.
        try:
            import anyio
            anyio.run(graceful_close, self._browser, self._playwright)
        except Exception:
            # Fallback if anyio isn't present:
            import asyncio
            asyncio.run(graceful_close(self._browser, self._playwright))
