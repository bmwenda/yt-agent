import unittest

from langchain_core.messages import AIMessage

from app import execute_tool, run_agent


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


class AppTests(unittest.TestCase):
    def test_execute_tool_handles_unknown_tool(self):
        message = execute_tool({"name": "missing_tool", "args": {}, "id": "call-1"})
        self.assertIn("Unknown tool", message.content)

    def test_run_agent_returns_final_assistant_message(self):
        fake_llm = FakeLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "extract_video_id", "args": {"url": "https://youtu.be/dQw4w9WgXcQ"}, "id": "call-1"}],
                ),
                AIMessage(content="done"),
            ]
        )

        messages = run_agent("find the video id", llm=fake_llm, max_tool_rounds=2)

        self.assertEqual(messages[-1].content, "done")
        self.assertGreaterEqual(len(messages), 3)

    def test_run_agent_stops_at_tool_limit(self):
        fake_llm = FakeLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "extract_video_id", "args": {"url": "https://youtu.be/dQw4w9WgXcQ"}, "id": "call-1"}],
                )
            ]
        )

        messages = run_agent("find the video id", llm=fake_llm, max_tool_rounds=0)

        self.assertEqual(len(messages), 2)
        self.assertEqual(len(fake_llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
