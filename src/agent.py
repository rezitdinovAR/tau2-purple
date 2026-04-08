"""Advanced purple agent for tau2-bench (AgentBeats).

The green (evaluator) agent calls this purple agent over A2A. On the first call
it sends:

    <domain policy>

    Here's a list of tools you can use (you can use at most one tool at a time):
    [<openai tool schemas>]

    and

    {<respond function schema>}

    Please respond in JSON format.
    ...

    Now here are the user messages:
    <initial user turn>

On every subsequent call it sends either the next user turn or a tool result.
The agent must reply with a single JSON object describing one action:

    {"name": "<tool_name>", "arguments": {<tool args>}}

This implementation:
  - Parses the policy and tool schemas out of the first message and feeds them
    to the LLM as native function-calling tools (much more reliable than
    free-form JSON output).
  - Uses a strong "elite customer service agent" system prompt that emphasizes
    policy compliance, careful reasoning, and one-action-per-turn discipline.
  - Retries on malformed responses with an explicit error message.
  - Falls back to a `respond` action if everything fails so that the
    benchmark run never crashes.
  - Is provider-agnostic via litellm — set TAU2_AGENT_LLM to any model id
    (openai/gpt-4.1, anthropic/claude-sonnet-4-5, openai/gpt-4o, ...).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import litellm
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()


ADVANCED_SYSTEM_PROMPT = """You are an elite customer-service agent operating under a strict company policy.
Your single objective: help the user complete their task while STRICTLY following the policy.

Operating principles:
1. POLICY FIRST. Re-read the relevant policy section before any action. Never violate it, even if the user insists.
2. THINK BEFORE YOU ACT. Reason silently about what the user needs, what the policy allows, and which tool (if any) is the right next step.
3. ONE ACTION PER TURN. Make at most a single tool call per response. Never chain actions in one turn.
4. NEVER INVENT FACTS. All IDs, names, prices, dates, account numbers, and policy clauses must come from the user or from a previous tool result. If you don't have a value, ask the user for it via the `respond` tool.
5. VERIFY ARGUMENTS. Double-check every argument against the tool schema and against what the user actually said. Use exact strings, not paraphrases.
6. ASK CLARIFYING QUESTIONS when information is missing or ambiguous. It is much better to ask than to guess.
7. CONFIRM IRREVERSIBLE ACTIONS. Before any cancellation, refund, modification, purchase, or other state change, summarize what you're about to do and explicitly ask the user to confirm.
8. STAY ON SCRIPT. If the user requests something the policy forbids, politely decline and explain the relevant policy briefly.
9. BE CONCISE. Keep messages to the user short, professional, and free of filler.
10. CLOSE THE LOOP. Once the task is fully resolved and the user has nothing more to ask, send a brief closing confirmation via `respond`.

You MUST always reply with exactly one tool call (function call). If you want to talk to the user instead of using a domain tool, call the `respond` tool with your message in the `content` argument. Never output free-form text outside of a tool call.
"""


RESPOND_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Respond directly to the user with a message instead of calling a domain tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send to the user.",
                },
            },
            "required": ["content"],
        },
    },
}


def _extract_balanced_array(text: str, search_from: int = 0) -> str | None:
    """Find the first balanced ``[...]`` JSON array at or after ``search_from``."""
    i = search_from
    while i < len(text) and text[i] != "[":
        i += 1
    if i >= len(text):
        return None

    begin = i
    depth = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[begin : i + 1]
        i += 1
    return None


def _parse_first_message(text: str) -> tuple[str, list[dict[str, Any]] | None, str]:
    """Split the first green-agent message into (policy, tool schemas, user turn)."""
    tools: list[dict[str, Any]] | None = None
    policy = text

    tools_marker = "Here's a list of tools you can use"
    t_idx = text.find(tools_marker)
    if t_idx >= 0:
        policy = text[:t_idx].rstrip()
        array_text = _extract_balanced_array(text, t_idx)
        if array_text:
            try:
                parsed = json.loads(array_text)
                if isinstance(parsed, list):
                    tools = parsed
            except json.JSONDecodeError:
                tools = None

    user_marker = "Now here are the user messages:"
    u_idx = text.find(user_marker)
    if u_idx >= 0:
        user_msgs = text[u_idx + len(user_marker) :].strip()
    else:
        # Fallback: feed the whole message as user content so we never lose data.
        user_msgs = text

    return policy, tools, user_msgs


def _normalize_tool_schemas(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make sure each tool is wrapped in the OpenAI ``{"type": "function", ...}`` shape."""
    out: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "function" and "function" in entry:
            out.append(entry)
        elif "name" in entry and "parameters" in entry:
            # Bare function schema → wrap it.
            out.append({"type": "function", "function": entry})
        elif "function" in entry:
            out.append({"type": "function", "function": entry["function"]})
    return out


class Agent:
    """Stateful per-conversation purple agent."""

    def __init__(self) -> None:
        self.model = os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1")
        self.temperature = float(os.getenv("TAU2_AGENT_TEMPERATURE", "0.0"))
        self.max_retries = int(os.getenv("TAU2_AGENT_MAX_RETRIES", "2"))
        self.use_native_tools = os.getenv("TAU2_AGENT_USE_NATIVE_TOOLS", "1").lower() not in ("0", "false", "no")

        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] | None = None
        self._initialized = False

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Thinking..."),
        )

        if not self._initialized:
            self._initialize_from_first_message(input_text)
        else:
            self.messages.append({"role": "user", "content": input_text})

        action = self._get_next_action()

        # Store the assistant turn as plain JSON content so we never get tangled
        # up in OpenAI's tool_call/tool_message round-trip rules.
        self.messages.append({"role": "assistant", "content": json.dumps(action)})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=action))],
            name="Action",
        )

    # ------------------------------------------------------------------ helpers

    def _initialize_from_first_message(self, text: str) -> None:
        policy, parsed_tools, user_msgs = _parse_first_message(text)

        system_content = (
            f"{ADVANCED_SYSTEM_PROMPT}\n\n"
            f"=== DOMAIN POLICY ===\n{policy.strip()}"
        )

        normalized_tools: list[dict[str, Any]] | None = None
        if parsed_tools:
            normalized_tools = _normalize_tool_schemas(parsed_tools)
            tool_names = {t["function"]["name"] for t in normalized_tools if "function" in t}
            if "respond" not in tool_names:
                normalized_tools.append(RESPOND_TOOL)

        if normalized_tools and self.use_native_tools:
            self.tools = normalized_tools
        else:
            # Fall back to JSON-mode prompting and embed the tool list in the
            # system prompt so the model still sees the schemas.
            self.tools = None
            if normalized_tools:
                system_content += (
                    "\n\n=== AVAILABLE TOOLS ===\n"
                    f"{json.dumps(normalized_tools, indent=2)}"
                )
            else:
                system_content += (
                    "\n\n=== RESPOND TOOL ===\n"
                    f"{json.dumps(RESPOND_TOOL, indent=2)}"
                )
            system_content += (
                '\n\nReply with EXACTLY ONE JSON object of the form '
                '{"name": "<tool_name>", "arguments": {<args>}}. '
                "No prose, no markdown, no code fences."
            )

        self.messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_msgs or text},
        ]
        self._initialized = True

    def _get_next_action(self) -> dict[str, Any]:
        last_error: str | None = None

        for _ in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": self.temperature,
                }
                if self.tools:
                    kwargs["tools"] = self.tools
                    kwargs["tool_choice"] = "required"
                else:
                    kwargs["response_format"] = {"type": "json_object"}

                completion = litellm.completion(**kwargs)
                msg = completion.choices[0].message

                # 1) Native tool-call path.
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    tc = tool_calls[0]
                    raw_args = tc.function.arguments
                    if isinstance(raw_args, str):
                        args = json.loads(raw_args) if raw_args.strip() else {}
                    else:
                        args = raw_args or {}
                    return {"name": tc.function.name, "arguments": args}

                # 2) JSON-mode path.
                content = msg.content or ""
                content = self._strip_code_fences(content)
                if not content:
                    raise ValueError("empty response from model")
                action = json.loads(content)
                if not isinstance(action, dict) or "name" not in action or "arguments" not in action:
                    raise ValueError(f"missing 'name'/'arguments' in {action!r}")
                return action

            except Exception as exc:  # noqa: BLE001 — we want to retry on any error
                last_error = str(exc)
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid ({last_error}). "
                            "Reply with EXACTLY ONE tool call. If you don't need a "
                            "domain tool, call the `respond` tool with your message "
                            "in `content`. Do not output anything outside the tool call."
                        ),
                    }
                )

        # Final safety net so the run never crashes.
        return {
            "name": "respond",
            "arguments": {
                "content": (
                    "I'm having trouble formulating a proper response right now. "
                    "Could you please rephrase or clarify your request?"
                )
            },
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\n?\s*```\s*$", "", text)
        return text.strip()
