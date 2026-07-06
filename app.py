import os
import json
from dotenv import load_dotenv
from typing import List, Dict
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools import *

load_dotenv()

tools = [extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_thumbnails]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")).bind_tools(tools)
tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}

def execute_tool(tool_call: Dict) -> ToolMessage:
    """Execute a single tool from tool_call data and return a ToolMessage"""
    try:
        tool_response = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        content = json.dumps(tool_response) if isinstance(tool_response, (dict, list)) else str(tool_response)
    except Exception as e:
        content = f"An error occured: {e}"
    return ToolMessage(content=content, tool_call_id=tool_call["id"])

# llm will decide the tool to use and return an AIMessage with tool_calls array
# response_1 = llm.invoke(messages)

# # Add the returned AIMessage to the messages history
# messages.append(response_1)

# # Create a tool message with this data
# tool_message = execute_tool(response_1.tool_calls[0])

# # Add the tool response to our messages list
# messages.append(tool_message)

# # Now invoke the llm again with the new message
# response_2 = llm.invoke(messages)
# # print(response_2.content)

initial_setup = RunnablePassthrough.assign(
    messages = lambda mapper: [HumanMessage(content=mapper["query"])]
)

first_llm_call = RunnablePassthrough.assign(
    ai_response = lambda mapper: llm.invoke(mapper["messages"])
)

first_tool_processing = RunnablePassthrough.assign(
    tool_messages = lambda mapper: [
        execute_tool(tc) for tc in mapper["ai_response"].tool_calls
    ]
).assign(
    messages = lambda mapper: mapper["messages"] + [mapper["ai_response"]] + mapper["tool_messages"]
)

second_llm_call = RunnablePassthrough.assign(
    ai_response_2 = lambda mapper: llm.invoke(mapper["messages"])
)

second_tool_processing = RunnablePassthrough.assign(
    tool_messages_2 = lambda mapper: [
        execute_tool(tc) for tc in mapper["ai_response_2"].tool_calls
    ]
).assign(
    messages = lambda mapper: mapper["messages"] + [mapper["ai_response_2"]] + mapper["tool_messages_2"]
)

final_summary = RunnablePassthrough.assign(
    summary = lambda mapper: llm.invoke(mapper["messages"]).content
) | RunnableLambda(lambda mapper: mapper["summary"])

chain = (
    initial_setup
    | first_llm_call
    | first_tool_processing
    | second_llm_call
    | second_tool_processing
    | final_summary
)

def process_tool_calls(messages: List) -> List:
    """Recursively call tool processor"""
    last_message = messages[-1]
    tool_messages = [
        execute_tool(tc) for tc in getattr(last_message, "tool_calls", [])
    ]

    updated_messages = messages + tool_messages

    next_ai_message = llm.invoke(updated_messages)

    return updated_messages + [next_ai_message]

def should_continue(messages: List) -> bool:
    """Determine if there are more tools to call"""
    last_message = messages[-1]

    return bool(getattr(last_message, "tool_calls", None))

def _recursive_chain(messages: List) -> List:
    if should_continue(messages):
        new_messages = process_tool_calls(messages)
        return _recursive_chain(new_messages)
    return messages

recursive_chain = RunnableLambda(_recursive_chain)

full_chain = (
    RunnableLambda(lambda mapper: [HumanMessage(content=mapper["query"])])
    | RunnableLambda(lambda messages: messages + [llm.invoke(messages)])
    | recursive_chain
)

# prompt = "I want to get thumbnails for this video https://www.youtube.com/watch?v=hfIUstzHs9A"
try:
    query = {
        "query": "Get me the top trending video in the world. Tell me it's id, give me the metadata and generate a short summary. Finally get me it's available thumbnails. Play the song briefly"
    }
    result = full_chain.invoke(query)
    print(f"{result[-1].content}\n")
    print("*" * 50)
    print(result)
except Exception as e:
    print(e)
