import argparse
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import (
    extract_video_id,
    fetch_transcript,
    get_full_metadata,
    get_thumbnails,
    search_youtube,
)

load_dotenv()

TOOL_LIST = [
    extract_video_id,
    fetch_transcript,
    search_youtube,
    get_full_metadata,
    get_thumbnails,
]
TOOL_MAPPING = {tool.name: tool for tool in TOOL_LIST}
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "8"))


def build_llm() -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the agent.")

    return ChatOpenAI(model=DEFAULT_MODEL, api_key=api_key).bind_tools(TOOL_LIST)


def execute_tool(tool_call: Dict) -> ToolMessage:
    """Execute a single tool call and wrap the result as a ToolMessage."""
    tool_name = tool_call.get("name")
    tool = TOOL_MAPPING.get(tool_name)

    if tool is None:
        content = f"Error: Unknown tool '{tool_name}'"
    else:
        try:
            tool_response = tool.invoke(tool_call.get("args") or {})
            content = json.dumps(tool_response) if isinstance(tool_response, (dict, list)) else str(tool_response)
        except Exception as e:
            content = f"Error: {e}"

    return ToolMessage(content=content, tool_call_id=tool_call.get("id", "unknown-tool-call"))


def run_agent(query: str, llm: Optional[Any] = None, max_tool_rounds: int = MAX_TOOL_ROUNDS) -> List:
    """Run the agent until it stops requesting tool calls or hits the loop limit."""
    llm = llm or build_llm()
    messages: List = [HumanMessage(content=query)]
    messages.append(llm.invoke(messages))

    tool_rounds = 0
    while getattr(messages[-1], "tool_calls", None):
        if tool_rounds >= max_tool_rounds:
            break

        messages.extend(execute_tool(tc) for tc in messages[-1].tool_calls)
        messages.append(llm.invoke(messages))
        tool_rounds += 1

    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YouTube agent.")
    parser.add_argument("query", help="Question or task to send to the agent")
    parser.add_argument("--max-tool-rounds", type=int, default=MAX_TOOL_ROUNDS, help="Maximum number of tool rounds before stopping")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_agent(args.query, max_tool_rounds=args.max_tool_rounds)
    print(result[-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
