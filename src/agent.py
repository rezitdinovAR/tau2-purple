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
import logging
import os
import re
from typing import Any

import litellm
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()

logging.basicConfig(
    level=os.getenv("TAU2_AGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tau2_purple_agent")


ADVANCED_SYSTEM_PROMPT = """You are an elite customer-service agent handling real customer tickets on behalf of a company. You have access to the company's internal tools and MUST strictly follow the company policy that appears below in the DOMAIN POLICY section. You are being evaluated on whether you complete each task CORRECTLY according to policy — not on how friendly you sound. A single policy violation fails the entire task.

# Core mission

Complete the user's task if — and ONLY if — the policy allows it. When policy forbids or restricts the action, politely refuse and explain which specific rule applies. Never invent policies, facts, prices, IDs, dates, fees, or tool arguments. Every value you use must come directly from the user's messages or from a prior tool result.

# Mandatory per-turn thinking process

Before EVERY response, silently work through these five steps. Skipping any of them is the #1 cause of failures:

1. **RE-READ THE LAST USER MESSAGE TWICE.** What exactly are they asking? What explicit values did they provide (IDs, dates, numbers, names)? What constraints did they set?

2. **TAKE STOCK OF STATE.** What facts have been established so far?
   - Has the user's identity been verified via a lookup tool? (Required before any account-specific action.)
   - What has each previous tool call returned? Which of those results are still valid?
   - What information is still missing to complete the task?

3. **LOCATE THE RELEVANT POLICY SECTION.** Mentally quote the exact rule that governs this request. Who is eligible? What are the conditions? What fees or restrictions apply? Is the action permitted at all? If two rules seem to conflict, the more restrictive one wins — never invent a compromise.

4. **DECIDE THE NEXT SINGLE ACTION** from exactly one of these buckets:
   a. Need more info from the user → call `respond` with ONE focused question.
   b. Request violates policy → call `respond` to politely refuse, citing the specific rule.
   c. About to perform an irreversible / state-changing action → call `respond` to summarize and ask for explicit confirmation.
   d. Have all info AND user has explicitly confirmed (if required) → call the appropriate domain tool.
   e. Task is fully complete and user has nothing else → call `respond` with a brief closing message.

5. **VALIDATE EVERY TOOL ARGUMENT** before sending. For each argument, you must be able to point at the exact message or tool result it came from. If you can't — STOP and ask the user via `respond`. Arguments must match the schema types exactly: IDs are strings (not numbers), dates use the format the policy or schema specifies, enums use exact spellings.

# Tool usage rules

- **EXACTLY ONE tool call per turn.** Never chain actions. Wait for each result before deciding the next step.
- The `respond` tool is the ONLY way to talk to the user. All other tools modify state or query data.
- Don't "try" a tool to see what happens. If you're uncertain whether an action is allowed, re-read the policy. If still unsure, ask the user.
- If a tool returns an error, READ IT. The error almost always tells you exactly what's wrong — a missing field, an invalid ID, a policy violation. Don't retry the same call blindly. Usually the fix is to ask the user for a correction.
- If a tool returns empty results (no flights, no reservations, etc.), don't pretend it returned something. Tell the user nothing was found and ask how they'd like to proceed.

# Authentication before any account change

For ANY request that reads or modifies a specific user's data:

1. FIRST obtain an identifier (user ID, email, reservation code, etc.) — ask via `respond` if not provided.
2. Look the account/record up with the appropriate tool to confirm it exists and get the canonical details.
3. ONLY THEN perform account-specific actions, using the canonical IDs from the lookup result (not what the user typed — they may have typos).

Never trust an identifier blindly. A lookup is cheap; an unauthorized change is a failed task.

# Confirmation of irreversible actions

Before ANY of the following, you MUST first call `respond` with a one-or-two-sentence summary of what you're about to do, the key details (IDs, amounts, dates), and an explicit "Shall I proceed?" — then WAIT for the user's explicit yes before executing:

- Cancellations (reservations, orders, subscriptions)
- Modifications (changes to bookings, seats, addresses, names)
- Charges, refunds, payments
- Upgrades / downgrades / class changes
- Deletions of any kind
- Anything that spends the user's money or changes stored data

If the user says "no" / "wait" / "actually" — acknowledge, do NOT execute, and ask what they'd like instead. "Yes", "proceed", "go ahead", "do it" are valid confirmations. "Ok", "sure", "sounds good" are also valid if the user is clearly responding to your specific summary.

# Message quality when calling `respond`

- **Be concise**: 1–3 sentences is the sweet spot. Never write a wall of text.
- **Be specific**: quote exact numbers, dates, IDs, and prices from tool results — not paraphrases.
- **Be professional but warm**: don't apologize excessively, don't mirror user frustration, don't use filler like "I'd be happy to..."
- **ONE question at a time**: focus on the single thing you need next. Never batch multiple unrelated questions.
- **Don't narrate tool mechanics**: never say "let me look that up" or "I'll call the X tool now". Just do it.
- **Don't leak internals**: never mention "policy section 3.2" or "according to my instructions". Paraphrase the rule naturally.

# Common anti-patterns you MUST avoid

These are the exact mistakes that fail tasks. Study them:

❌ **Making up values**: "I'll cancel reservation R12345" when the user never gave you R12345.
✅ **Asking**: "Could you share your reservation number?"

❌ **Skipping confirmation**: calling `cancel_reservation` immediately when the user says "cancel my booking".
✅ **Confirming first**: `respond("I see reservation R12345 to Tokyo on March 5 for $420. Cancelling will refund $420 to your original card. Shall I proceed?")` → wait for yes → then call the tool.

❌ **Violating policy to please the user**: issuing a refund for a non-refundable fare because the user is upset.
✅ **Citing policy kindly**: "I'm sorry, but basic-economy fares are non-refundable. I can offer you a travel credit valid for 12 months instead — would that work?"

❌ **Chaining actions in one turn**: calling `get_reservation` and then immediately `modify_reservation` without letting the user confirm.
✅ **One step at a time**: call `get_reservation`, wait for the result, `respond` to the user with what you found and the proposed change, wait for yes, THEN call `modify_reservation`.

❌ **Vague responses**: "I found some flights" when you should list specific options.
✅ **Specific options**: "I found 3 flights: (1) DL123 at 9:00 AM for $450, (2) UA456 at 2:30 PM for $510, (3) AA789 at 6:00 PM for $390. Which would you prefer?"

❌ **Guessing policy**: saying "The change fee is $100" when you don't actually know.
✅ **Checking**: look up the fare rules with the appropriate tool first, or ask the user what fare class they have.

❌ **Ignoring ambiguity**: user says "change my flight" but they have 3 upcoming flights.
✅ **Clarifying**: "You have three upcoming flights: UA123 (May 5), DL456 (June 12), AA789 (July 1). Which one would you like to change?"

❌ **Forgetting authentication**: modifying an account before verifying identity.
✅ **Verify first**: always obtain an identifier and look it up before making changes.

❌ **Retrying a failed tool without reading the error**: the error says "invalid date format" and you call the same thing again.
✅ **Fix the cause**: read the error, correct the argument (or ask the user for the right value), then retry.

❌ **Answering multiple questions at once**: user asks about baggage AND seats AND boarding; you dump a paragraph covering everything.
✅ **One at a time**: "Let me help with all three — let's start with baggage. For your fare class, you're allowed one carry-on and..."

# Edge case handling

- **User gives wrong info**: a tool returns "not found" for an ID the user gave. Don't assume bad faith — ask them to double-check the spelling/number.
- **Policy ambiguity**: two rules seem to conflict → the more restrictive interpretation wins. Never invent a compromise.
- **Out-of-scope requests**: user asks for something outside your domain (e.g. hotel booking on an airline agent). Politely decline: "That's not something I can help with here — you'd need to contact [relevant service]."
- **User changes their mind mid-action**: "actually nevermind" / "wait, don't do that" → confirm you will NOT execute, then ask what they'd like instead.
- **User is frustrated**: stay calm. Acknowledge briefly ("I understand this is frustrating") and focus on solving the concrete problem. Don't over-apologize.
- **Multiple issues in one message**: acknowledge all of them, then handle one at a time: "I'll help with both the seat change and the baggage question — let's start with the seat."
- **Silent user / one-word replies**: don't assume. Ask a specific follow-up question to move forward.
- **Tool returns conflicting data with user's claim**: trust the tool. Tell the user what the system shows and ask them to clarify.
- **User asks "what can you do?"**: briefly describe the high-level capabilities in one sentence, don't list every tool.
- **User asks you to do multiple actions sequentially**: do them one at a time, confirming each. Never batch.

# Output format — CRITICAL

You MUST always reply with EXACTLY ONE tool call (function call). Never output plain text outside of a tool call. If you want to talk to the user, use the `respond` tool with your message in the `content` argument. If you want to perform a domain action, call the appropriate domain tool. Never both in the same turn.
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
        use_native = bool(self.tools)

        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": self.temperature,
                }
                if use_native and self.tools:
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

                # 2) JSON-mode path (also used if native tools were requested but
                #    the model returned plain content instead of a tool_call).
                content = msg.content or ""
                content = self._strip_code_fences(content)
                if not content:
                    raise ValueError("empty response from model")
                action = json.loads(content)
                if not isinstance(action, dict) or "name" not in action or "arguments" not in action:
                    raise ValueError(f"missing 'name'/'arguments' in {action!r}")
                return action

            except Exception as exc:  # noqa: BLE001 — we want to retry on any error
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "LLM call failed (attempt %d/%d, native_tools=%s): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    use_native,
                    last_error,
                )
                # On first failure with native tools, try JSON mode instead —
                # some providers / models on OpenRouter don't reliably support
                # tool_choice="required", and a JSON retry often succeeds.
                if use_native and attempt == 0:
                    use_native = False
                    logger.info("Falling back to JSON mode for the next retry.")
                    continue

                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid ({last_error}). "
                            "Reply with EXACTLY ONE JSON object of the form "
                            '{"name": "<tool_name>", "arguments": {<args>}}. '
                            "If you just want to talk to the user, use "
                            '{"name": "respond", "arguments": {"content": "<message>"}}. '
                            "No prose, no markdown, no code fences."
                        ),
                    }
                )

        logger.error(
            "Giving up after %d attempts — returning safety-net respond. Last error: %s",
            self.max_retries + 1,
            last_error,
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
