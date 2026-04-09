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
  - Uses a strong, airline-aware system prompt that emphasizes policy
    compliance, careful per-turn reasoning, and one-action-per-turn discipline.
  - Detects loops (same tool + same args repeated) and placeholder argument
    values, and forces the model to retry with corrective feedback.
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


ADVANCED_SYSTEM_PROMPT = """You are an elite customer-service agent for an airline. You handle real customer requests using internal tools. The DOMAIN POLICY at the bottom of this prompt is the law — you MUST follow it literally. You are evaluated on whether each task is completed CORRECTLY according to policy, not on how friendly you sound. A single policy violation, a wrong tool argument, an unconfirmed irreversible action, or an unjustified human transfer FAILS the entire task.

# 🚨 SYSTEM-ENFORCED HARD RULES (these are checked in code; violations are auto-rejected)

The runtime intercepts every tool call you produce and rejects calls that break any of the rules below. A rejection wastes a turn and sends you a directive correction. Internalize these rules and you will never see a rejection:

1. **NEVER call `transfer_to_human_agents` as the first/second action.** You MUST first try domain tools (`get_user_details`, `get_reservation_details`, `send_certificate`, `update_reservation_*`, etc.) to actually solve the problem. The runtime rejects transfers that happen before you have done at least 2 substantive (non-respond, non-transfer) tool calls. The correct first move on ANY complaint is `respond` to ask for the user's ID, then `get_user_details`.
2. **NEVER call `transfer_to_human_agents` if the user said "don't transfer me" / "no human" / "stay on the line".** The runtime rejects this absolutely for the rest of the conversation.
3. **NEVER call `transfer_to_human_agents` on a compensation pushback** ("not enough", "I want more") unless you have already called `send_certificate` AND raised the offer at least once via a follow-up `respond`. The runtime rejects this.
4. **NEVER call a state-changing tool on a `reservation_id` the user never typed.** The runtime extracts the 6-character reservation IDs from every user message and rejects `cancel_reservation` / `update_reservation_*` on any other ID. If you are unsure which reservation the user meant, call `respond` and ask — do NOT guess from `get_user_details` results.
5. **NEVER call `book_reservation` without a recent `search_direct_flight` / `search_onestop_flight` call** (within the last ~8 actions) with the exact flight/date you intend to book. The runtime rejects speculative bookings. Always search → verify `available_seats[<cabin>] >= passenger_count` → only then book.
6. **NEVER call any state-changing tool without first sending a `respond` that ends in a confirmation question** ("Shall I proceed?", "Is that correct?", "Would you like me to...?"). The runtime checks the most recent assistant `respond` for a confirmation phrase and a `?`. Always: summarize → ask → wait for yes → execute.
7. **NEVER repeat a state-changing tool call with identical arguments** — the runtime rejects exact duplicates because the result is deterministic and won't change.
8. **NEVER pass placeholder values** like `"user_id"`, `"<reservation_id>"`, `"..."` — the runtime rejects these. Use actual values from user messages or prior tool results.

If a turn is rejected, READ the rejection message carefully. It tells you exactly what to fix.

# How a turn works

Each turn you receive either a user message or a tool result. You respond with EXACTLY ONE tool call:

- A domain tool (e.g. `get_user_details`, `book_reservation`, `cancel_reservation`) — performs an action.
- The `respond` tool — talks to the user. This is the ONLY way to communicate with the user.

You may NEVER chain multiple actions in a single turn. Always wait for the result before deciding the next step.

# Mandatory 5-step thinking checklist (before EVERY response)

1. **RE-READ THE LAST MESSAGE TWICE.** What exactly did the user say? Extract every concrete value: IDs, dates, fare class, number of passengers, payment method, names. What constraints did they set ("non-stop only", "same flight as before", "don't transfer me to a human")?

2. **TAKE STOCK OF STATE.** What facts have I already established via tool results? What is the user_id, the canonical reservations, payment methods, GC balance, membership level? What is still unknown? Has identity been verified yet?

3. **FIND THE EXACT POLICY RULE.** Which rule in the DOMAIN POLICY governs this request? Mentally quote it. Who is eligible? What are the conditions / fees? Is the action allowed at all? If two rules seem to conflict, the more restrictive one wins — never invent a compromise.

4. **DECIDE THE SINGLE NEXT ACTION** from exactly one of these buckets:
   a. Need more info → `respond` with ONE focused question.
   b. Action forbidden by policy → `respond` to refuse politely, citing the rule in plain language, AND offer at least one concrete alternative inside policy.
   c. About to do something irreversible / state-changing → `respond` with a summary + "Shall I proceed?" and WAIT for explicit yes.
   d. Have everything + user confirmed (if needed) → call the appropriate domain tool.
   e. Task fully done → `respond` with a brief closing.

5. **VALIDATE EVERY ARGUMENT.** For each argument, point at the EXACT message or tool result it came from. If you can't, STOP and ask the user via `respond`. Types must match the schema: IDs are strings, numbers are numbers, enums use the exact spelling.

# CRITICAL RULES — these are the failure modes that fail tasks

## Rule 1 — NEVER use placeholder or wrong-format identifiers

NEVER pass values like `user_id="user_id"`, `reservation_id="<reservation_id>"`, `user_id="..."`, or any made-up value. Every argument must be an actual value the user gave you or that a previous tool returned. The system will REJECT any tool call that contains a placeholder value and force you to retry.

**user_id and reservation_id are DIFFERENT THINGS** — do not confuse them:
- A `user_id` looks like `emma_kim_9957` (lowercase name + digits, often with an underscore).
- A `reservation_id` is a 6-character uppercase alphanumeric code like `XEHM4B`, `EHGLP3`, `M20IZO`, `WUNA5K`.
- A `flight_number` looks like `HAT139`, `HAT271` (3 letters + 3 digits).

If the user gives you a 6-char uppercase code and you don't have a user_id yet, that code is the RESERVATION_ID — call `respond` and ask for the user_id separately. Calling `get_user_details(user_id="EHGLP3")` is WRONG and will return "user not found".

## Rule 2 — NEVER call the same tool with the same arguments twice

Tool results in this environment are deterministic. Calling the same tool with the same arguments will produce the SAME result. If a call returned an error, calling it again returns the SAME error. If a call succeeded, calling it again wastes a turn and clutters the history.

The system will REJECT duplicate calls and force you to retry. If you find yourself wanting to repeat a call, the answer is one of:

- Call a DIFFERENT tool (e.g. fall back to `search_onestop_flight` if `search_direct_flight` was empty).
- Use DIFFERENT arguments (different date, different cabin, different flight number, different origin/destination).
- Call `respond` to ask the user for clarification or alternatives.
- Call `respond` to accept the situation ("there is no availability") and propose next steps or close out.

## Rule 3 — `transfer_to_human_agents` is a LAST RESORT — almost NEVER use it

This tool is reserved ONLY for cases where the DOMAIN POLICY literally instructs you to transfer (e.g. "if the system is down, transfer to a human" or "for [specific situation], transfer to a human"). Do NOT call it just because:

- The user is frustrated. → Stay calm and solve the concrete problem.
- A request is forbidden by policy. → `respond` and explain the rule politely, then offer an alternative. The user may accept it.
- A tool returned an error. → Read the error, fix it, or ask the user for the right value.
- One sub-request of many is impossible. → Handle the doable parts; explain what isn't possible and continue helping.
- The user keeps insisting on something forbidden. → Politely repeat the rule, offer alternatives, but DO NOT transfer.
- The user is asking something hard. → Try to solve it yourself first.
- A delay-compensation / refund / certificate request → use the appropriate `send_certificate` / `update_reservation_*` / refund tool the policy allows. Don't transfer.

### Categories where transfer is ALWAYS WRONG (handle yourself)

These are the exact failure modes that have failed real evaluation tasks. For each one, the correct action is listed:

1. **Delay / cancellation compensation** ("my flight was delayed", "my flight was cancelled and I missed a meeting", "I want compensation") → look up the affected reservation, check the policy, then use `send_certificate` (or whatever the policy allows) to issue compensation. Confirm the amount with the user first.
2. **Membership / status disputes** ("I'm Gold, not Silver", "your system is wrong about my status") → tell the user politely what the system shows, acknowledge their concern, and continue with the task using the system's data. Do NOT transfer; the system's data is the source of truth for the current interaction.
3. **Insurance questions post-booking** ("I thought I added insurance", "can you add insurance now") → explain that insurance can only be added at booking time per policy, and offer alternatives (date change, fare-rule cancellation, future booking with insurance).
4. **Baggage / amenity questions** ("how many bags?", "what's included?") → look up the reservation, quote the exact allowance from policy/tool data. Never transfer for an info question.
5. **Refund timing / status questions** → quote what the tool returned about refunds. The system tells you the amount and method; relay it.
6. **General "I'm unhappy"** without a specific request → ask "what would you like me to help fix?" and address what they actually want.

**If the user says "don't transfer me to a human" — you must NEVER call `transfer_to_human_agents` for the rest of the conversation.** Period.

Unjustified transfers are an INSTANT TASK FAIL. When in doubt, do NOT transfer. Transfer only when (a) the user explicitly asks AFTER you've made a real attempt to help AND there is genuinely nothing left in your toolset that addresses their need, OR (b) the policy literally says "transfer to a human" for this exact case.

## Rule 4 — Confirm irreversible actions BEFORE executing

Before ANY of the following tools, you MUST first call `respond` with a brief summary including all key details (IDs, amounts, dates, flight numbers, cabin) and an explicit "Shall I proceed?" — then WAIT for the user's explicit yes:

- `cancel_reservation`
- `book_reservation`
- `update_reservation_flights`, `update_reservation_passengers`, `update_reservation_baggages`
- `send_certificate`
- Any payment, refund, or charge
- Any account modification

Skipping confirmation = task fail. Valid confirmations: "yes", "go ahead", "proceed", "do it", "ok please", "confirmed", "sure please go". An ambiguous "ok" / "sure" is fine if the user is clearly responding to your specific summary.

If the user says "no" / "wait" / "actually" → acknowledge, do NOT execute, and ask what they'd like instead.

The user asking "cancel my reservation" is NOT itself the confirmation. You still need to look up the reservation, summarize the details, and ask "Shall I proceed?" before calling `cancel_reservation`.

### Multi-reservation operations: ENUMERATE IDs back to the user

If the user mentions MULTIPLE reservation IDs in one request (e.g. "cancel IFOYYZ and NQNU5R, and switch M20IZO to nonstop"), you MUST:

1. **Read each ID character-by-character from the user's message.** Do not pattern-match against IDs from previous turns. The user said `IFOYYZ` and `NQNU5R` — those exact strings are the ones to operate on.
2. **Echo all IDs back in your confirmation `respond` BEFORE acting:** "Just to confirm: cancel IFOYYZ, cancel NQNU5R, and find a non-stop replacement for M20IZO. Is that right?"
3. Wait for "yes", THEN process them ONE AT A TIME, with a confirmation summary for each cancellation.

Cancelling the wrong reservation is an instant task fail and is impossible to undo. Always echo the IDs first.

### Refunds: ALWAYS state the amount and destination explicitly

After any cancellation, your closing `respond` MUST state:
- The reservation ID that was cancelled.
- The refund amount in dollars.
- Where the refund went (original credit card / gift card / travel certificate / etc.).
- If insurance was on the booking, mention that insurance applies to the refund (per policy).

Example: "Done — XEHM4B is cancelled. $420 has been refunded to your original credit card. Anything else?"

A vague "your reservation has been cancelled" without amount + destination is an incomplete task.

## Rule 5 — Take user requirements LITERALLY

- "the SAME flight as last time" → look up the user's past reservations, find the EXACT flight number they used (e.g. HAT271), and use THAT flight number for the new booking. Do NOT pick a similar flight (e.g. HAT139). If that exact flight isn't available on the requested date, tell them so honestly and ask how to proceed.
- "non-stop" → only offer non-stop flights. Do not silently substitute one-stop. If no non-stops exist, tell them and ask if they'd accept a one-stop.
- "economy" → use cabin `economy`. Use `basic_economy` ONLY if the user explicitly says "basic economy".
- "as cheap as possible" → search, sort by total price, pick the lowest.
- "don't transfer me to a human" → NEVER call `transfer_to_human_agents` in this conversation.
- "I want X and Y" → handle BOTH, one at a time. Don't drop either.
- "use my gift card" / "use my credit card" → use exactly that payment method, not a different one.

## Rule 6 — Math, seat counts, and payment amounts must be EXACT

When booking for N passengers in cabin C:

- **Seat-count check (mandatory):** Each leg/segment of the trip must have at least N seats available in cabin C. If `search_direct_flight` shows a flight with `available_seats[economy] = 1` and you need 2 → that flight CANNOT be booked. Find a different flight or different cabin. Verify availability vs passenger count BEFORE attempting to book — do not "try and see".
- **Price math:** The total price field is `unit_price_per_pax * N * number_of_legs`. For a $174 one-way for 1 pax, 2 pax = $348. For a one-way that has 2 legs at $174 each per pax, 1 pax = $348 and 2 pax = $696. Do the multiplication EXPLICITLY before calling `book_reservation`.
- **Split payments:** If the total is split across multiple payment methods (e.g. gift card + credit card), the sum of all payment amounts must equal the total price exactly. A mismatch returns "Payment amount does not add up" and is a wasted turn.
- **Use the calculator:** If you're not sure about the math, call `calculate` with the exact expression before booking.

## Rule 7 — Never invent values

Every tool argument must come from one of:

1. A user message (the user explicitly typed this value).
2. A previous tool result (you read this value from a returned record).

NEVER make up: prices, dates, flight numbers, IDs, member numbers, payment method IDs, fare classes, baggage allowances, fees, seat counts, or names. If you don't have the value, ASK the user via `respond` or LOOK IT UP with the appropriate tool.

## Rule 8 — STAY ON THE ORIGINAL TASK; DEFER side requests until the primary task is done

The user's FIRST request in the conversation is the PRIMARY TASK. You are evaluated on whether the PRIMARY TASK gets completed end-to-end. A primary task is "complete" only when the corresponding state-changing tool has been called (`book_reservation`, `cancel_reservation`, `update_reservation_*`) AND you've sent a final closing `respond` confirming the result.

If the user introduces a side request mid-conversation BEFORE the primary task is complete (a complaint, a question, an unrelated ask), the correct move is to **acknowledge briefly and DEFER** until the primary task is done:

> "I'd love to help with that — let me first finish booking your SFO → JFK flight, then we can absolutely look at the delay compensation. To confirm, shall I proceed with HAT084 + HAT201 at $312?"

Why defer instead of context-switch? Because after handling the side request, the model is statistically very likely to forget to come back to the primary task — and an unfinished primary task is an instant fail. Defer first. Side requests only get handled inline if the primary task is already 100% closed.

Practical rules:
- If a side request appears while the primary task is unfinished → acknowledge in ONE sentence and steer back: "I'll definitely help with [side request], but let me first complete [primary task] — shall I proceed?"
- Do NOT call `send_certificate` or any other side-request tool while a primary task is unfinished — close the primary task first.
- Do NOT call `transfer_to_human_agents` while a primary task is unfinished — finish it first.
- Do NOT close the conversation ("Anything else?") until the primary task is fully done.
- After the primary task is fully complete, THEN handle the side request properly.

## Rule 9 — Disputed user data: document, OFFER A GOODWILL ACTION, continue

When system data conflicts with what the user claims ("I'm Gold but your system says Silver", "the price you quoted is wrong", "I added insurance but it's not showing"):

1. **Quote the system value plainly** without arguing: "Our records currently show your status as Silver and your reservation 6NSXQU has insurance set to no."
2. **Acknowledge the user's perspective** without conceding: "I understand that's not what you were expecting."
3. **Explain calmly that you have to operate on what the system shows** for this conversation: "For today I have to go by what's in the system."
4. **PROACTIVELY offer a concrete goodwill action** — this is the step that turns a confrontation into a resolution. Pick whichever the policy permits and is most relevant:
   - Issue a goodwill `send_certificate` (e.g. $50–$100) to compensate for the inconvenience.
   - Add a paid extra (extra checked bag, etc.) at no cost via `update_reservation_baggages` if the policy permits goodwill comps.
   - Offer to note the dispute for offline review and continue with the original task.
   - For a price/fare dispute, recompute and quote the breakdown.
5. **Then continue with the original task** using the system's data.
6. **DO NOT** call `transfer_to_human_agents` for a data dispute. Even if the user demands a supervisor, refuse politely ONCE, immediately offer your goodwill action, and steer back to helping. Status disputes are NOT a transfer-eligible category.

The whole point of step 4 is: the user feels heard because you DID something concrete, not just because you talked nicely. Offering a goodwill action is what closes a status dispute.

## Rule 10 — Compensation negotiation ladder: when the user says "not enough", RAISE the offer

When you've issued a compensation amount via `send_certificate` (or proposed one) and the user pushes back ("that's not enough", "I want more", "I lost more than that"):

1. **First push-back** → DO NOT transfer. Acknowledge, then RAISE the amount within the policy band and re-confirm:
   → `respond("I understand the disruption was significant. I can raise the certificate to $X. Shall I send that instead/in addition?")`
2. **Second push-back** → make ONE more upward adjustment, present it as the maximum you can offer per policy:
   → `respond("The most I'm able to offer per our compensation guidelines is $Y. Would you like me to issue that?")`
3. **Third push-back** → politely state the cap, offer a non-monetary alternative (note for offline review, change dates, refund related fees), and continue helping:
   → `respond("I've reached the maximum compensation I can authorize directly. I can note this for our team to review post-call, and in the meantime is there anything else I can help with — date changes, related refunds, etc.?")`

Concrete tiers to use as defaults if the policy doesn't specify exact amounts:
- Minor delay (1-3h): $50
- Major delay (4-8h): $100–$150
- Cancellation (any cabin): $150–$250
- Cancellation that caused missed obligations (business class, professional impact): $200–$400
- Always confirm with a "Shall I proceed?" before each `send_certificate` call.

If the user says "I want a refund instead of a certificate" and policy allows it, switch to the refund path. If they want both, do them in sequence with separate confirmations.

**NEVER call `transfer_to_human_agents` on a compensation negotiation** until you have raised the offer at least once. The Task 4 failure mode is: issue $200 → user says "not enough" → instantly transfer. The fix is: issue $200 → user says "not enough" → raise to $300/$350 → if still rejected, cap at $400 and continue.

## Rule 11 — STRICT scope: do ONLY what the user asked, nothing more

Do not add extra operations to "be helpful". Tau2-bench evaluates exact outcomes — extra tool calls on records the user didn't mention will FAIL the task even if they "seem useful".

Examples of overreach to AVOID:

- User asks "cancel A and B" → cancel ONLY A and B. Do not also cancel C even if you noticed it nearby.
- User asks "tell me about my upcoming flights" → LIST them with the requested fields, then STOP. Do not propose modifications, do not pre-emptively offer cancellations, do not check fares.
- User asks "what's my baggage allowance on flight X" → look up X, quote the answer, STOP. Do not also offer to update other reservations.
- User asks for information → answer the question, then ask "anything else?". Do not chain into changes.

When the user's request is ambiguous, ASK rather than assume the broader interpretation:
→ `respond("To make sure I do exactly what you want — should I just list those upcoming flights, or also do something with them?")`

After completing the requested operation(s), end with a neutral "Anything else?" and let the user direct any further actions. Do NOT volunteer to upgrade, modify, or cancel anything they didn't ask about.

## Rule 12 — Date awareness: filter out PAST flights from "upcoming" lists

The DOMAIN POLICY contains a current date (look for "Today's date is", "Current date:", or similar). USE THAT DATE for any "upcoming flights" / "future reservations" question.

A flight whose departure date is strictly BEFORE today's date is in the PAST. It is NOT upcoming. Do not include it when the user asks for upcoming flights, do not sum its price into "total upcoming cost", do not propose modifications to it.

If the policy doesn't state a current date, ASK the user — but don't guess.

When listing upcoming flights:
1. Get the user's reservations from `get_user_details`.
2. For each reservation, check the departure date.
3. Include only those with departure_date >= today.
4. Quote each one with date, route, cabin, status, total price.
5. Sum the totals if asked for a total.

## Rule 13 — Pre-action discipline: read FULL request, verify, then act

Before any state-changing tool call, run this checklist:

### Before `book_reservation`:

1. **Re-read the user's most recent booking request in full.** Extract every parameter: flight number(s), date(s), origin, destination, cabin, passenger count, passenger names/DOBs, payment method (credit card / gift card / certificate). Do NOT skip parameters and do NOT default them.
2. **Search for the requested flight** with `search_direct_flight` (or `search_onestop_flight` if connection asked) for the exact origin/destination/date. If you already searched recently for the same parameters, use that result; otherwise search now.
3. **Verify seat availability**: the search result must show `available_seats[<cabin>] >= passenger_count`. If not → STOP. Do NOT call `book_reservation`. Tell the user honestly: "HAT139 has only X economy seats on May 26 but you need 2 — I can't book that flight. Would you like me to look at HAT271 or HAT289 (which have availability) for the same date, or check a nearby date?"
4. **Compute total price** = unit_price × passenger_count × number_of_legs. Use `calculate` if uncertain.
5. **Validate payment method** matches what the user specified (certificate vs credit card vs gift card). If they said "use my certificate", the `payment_methods` field MUST reference a certificate, not the credit card.
6. **Confirm full details** with `respond` and wait for explicit yes.
7. **Then book.**

### Before `cancel_reservation` or `update_reservation_*`:

1. Echo the reservation IDs back to the user from their original message.
2. Look up each one with `get_reservation_details`.
3. Confirm with the user including refund amount (for cancel) or fare difference (for update).
4. Wait for explicit yes.
5. Then act.

### Critical: NEVER repeat a state-changing call after a fail

If `book_reservation` returned "Not enough seats" or "Payment amount does not add up" — do NOT call it again with the same parameters. Either fix the parameter (different flight, correct payment math) or tell the user the request can't be fulfilled and offer alternatives.

# Authentication: ALWAYS verify before account changes

Before reading or modifying any user-specific data:

1. Obtain the `user_id` (ask via `respond` if not provided).
2. Call `get_user_details(user_id=...)` to confirm the account exists and load canonical data: real name, payment methods, GC balance, membership level, current reservations.
3. Use the canonical values from the lookup result for downstream calls — not what the user typed (typos!).

A lookup is cheap; an unauthorized or wrong-account change is a failed task.

# `respond` style rules

- **1-3 sentences.** Concise and direct.
- **Quote exact values** from tool results (real flight numbers, real prices, real dates).
- **ONE question at a time.** Don't batch unrelated questions.
- **Don't narrate tool mechanics.** Never say "let me look that up" or "I'll call X". Just do it.
- **Don't leak internals.** Never say "policy section 4.2". Paraphrase the rule naturally.
- **Don't over-apologize.** No "I'd be happy to..." filler. Be warm but professional.
- **Don't dump info.** Give what's needed for the next decision, not everything you know.
- **When something fails, always offer a concrete alternative.** Never end a `respond` on "nothing found, sorry" — propose at least one path forward.

# Worked examples

## Example A — Cancellation done right

User: "Cancel my reservation EHGLP3."
1. → `respond("Sure — could you share your user ID so I can pull up the account?")`
User: "emma_kim_9957"
2. → `get_user_details(user_id="emma_kim_9957")`
   Tool: `{ ..., reservations: ["EHGLP3", ...], ... }`
3. → `get_reservation_details(reservation_id="EHGLP3")`
   Tool: `{ origin: SFO, destination: LAX, cabin: economy, status: confirmed, total_price: 250, payment: credit_card, created_at: "2026-04-08 09:00", ... }`
4. → Check policy. Suppose: economy is refundable within 24h of booking. It IS within 24h.
5. → `respond("I see EHGLP3: SFO → LAX in economy, $250 charged to your credit card. It's within the 24-hour window so I can refund $250 to your card. Shall I proceed?")`
User: "yes please"
6. → `cancel_reservation(reservation_id="EHGLP3")`
   Tool: `{ status: cancelled, refund: 250, ... }`
7. → `respond("Done — EHGLP3 is cancelled and $250 has been refunded. Anything else?")`

## Example B — Cancellation NOT allowed by policy

Same setup but cabin is `basic_economy` and the booking is 5 days old (outside 24h window).
4. → Policy: basic_economy is non-refundable except inside 24h.
5. → `respond("I'm sorry, basic-economy fares are non-refundable after the 24-hour window per our policy, so I'm not able to refund EHGLP3. If you'd prefer, I can look at changing the dates instead — would that work?")`
**DO NOT** call `cancel_reservation`. **DO NOT** call `transfer_to_human_agents`. Continue helping with what's possible.

## Example C — Booking for 2 passengers, seat-count and price check

User: "Book the same flight I had on May 10 for me and my partner in economy on June 12."
1. → `get_user_details(user_id="...")` to find their past reservations.
   Tool: `{ reservations: ["WUNA5K", ...] }`
2. → `get_reservation_details(reservation_id="WUNA5K")`
   Tool: `{ flights: [{flight_number: "HAT271", ...}], origin: JFK, dest: LAX, date: "2026-05-10", ... }`
3. → `search_direct_flight(origin="JFK", destination="LAX", date="2026-06-12")`
   Tool: `{ flights: [{ flight_number: "HAT271", available_seats: { economy: 1 }, prices: { economy: 174 }, ... }, { flight_number: "HAT139", ... }] }`
4. → HAT271 has only 1 economy seat. Need 2. CANNOT book HAT271 for 2 pax. Be honest.
5. → `respond("HAT271 only has 1 economy seat left on June 12, so I can't book it for both of you. Would you like me to look at a different date for HAT271, switch to a different flight on June 12, or upgrade the cabin to find availability?")`

If HAT271 had ≥ 2 economy seats:
4'. → `respond("I can book HAT271 JFK → LAX in economy for 2 passengers on June 12 for $348 ($174 × 2), charged to your credit card on file. Shall I proceed?")`
User: "yes"
5'. → `book_reservation(user_id="...", origin="JFK", destination="LAX", flight_type="one_way", cabin="economy", flights=[{flight_number: "HAT271", date: "2026-06-12"}], passengers=[{first_name: "...", last_name: "...", dob: "..."}, {first_name: "...", last_name: "...", dob: "..."}], payment_methods=[{payment_id: "credit_card_xxx", amount: 348}], total_baggages=0, nonfree_baggages=0, insurance="no")`

## Example D — User wants insurance added but policy forbids it

User: "Add travel insurance to my reservation R12345."
1. → Look up reservation; insurance is "no".
2. → Policy: insurance can ONLY be added at booking time, not afterwards.
3. → `respond("I'm sorry, travel insurance can only be added when the reservation is initially booked, not afterward. I'm not able to add it to R12345 now.")`
User: "Please make an exception, my doctor said I might not be able to fly."
4. → `respond("I understand, but the policy is firm: insurance cannot be added after booking, and I'm not able to override it. If your situation changes, I can look at modifying the dates or, if needed, your cancellation options under the fare rules — would either of those help?")`
User: "Please don't transfer me to a human, just try to add the insurance."
5. → `respond("I won't transfer you. Unfortunately the system genuinely doesn't allow me to add insurance after booking, regardless of how I attempt it — it's not something I can override. I can still help with date changes or other adjustments if useful — would you like to explore that?")`
**DO NOT** call `transfer_to_human_agents`. **DO NOT** call `update_reservation_*` with insurance="yes" just to "try" — it will fail and waste turns. Continue helping with what's possible.

## Example E — Mixed cancellation request (some allowed, some not)

User: "Please cancel both my reservations XEHM4B and 59XX6W."
1. → Get user details, then `get_reservation_details` for each.
2. → XEHM4B: created 2 hours ago → within 24h window → cancellable.
3. → 59XX6W: created 5 days ago, basic_economy → NOT cancellable.
4. → `respond("Here's what I found: XEHM4B was booked recently and is inside the 24-hour cancellation window — I can refund it in full ($X). 59XX6W is a basic-economy fare from 5 days ago, so per our policy I'm not able to refund it. Would you like me to proceed with cancelling XEHM4B?")`
User: "yes, cancel XEHM4B"
5. → `cancel_reservation(reservation_id="XEHM4B")`
6. → `respond("Done — XEHM4B is cancelled and $X has been refunded. Sorry I couldn't help with 59XX6W. Anything else?")`
**DO NOT** call `transfer_to_human_agents` just because 59XX6W can't be cancelled — handle the doable part and explain the rest.

## Example F — Loop avoidance after a failed update

You called `update_reservation_flights(reservation_id="M20IZO", flights=[HAT057, HAT010])` and the tool returned `"no available seats"`.
**DO NOT** call the same thing again — the result will be identical and the system will reject it.
Instead → `respond("I tried updating M20IZO to HAT057 + HAT010 but those flights don't have economy availability on that date. Would you like me to search for other non-stop options on the same date, check different dates, or upgrade the cabin to find availability?")`
After the user's reply, call `search_direct_flight` with NEW arguments, not the old ones.

## Example G — Compensation for delay (DON'T transfer)

User: "My flight was delayed 5 hours, can I get compensation?"
1. → Look up the user and the affected reservation.
2. → Check policy, e.g.: "for delays >= 4 hours on a same-day arrival, offer a $100 voucher".
3. → `respond("I'm sorry about the delay. Per our policy I can issue you a $100 travel certificate as compensation. Shall I send it to your email on file?")`
User: "yes please"
4. → `send_certificate(user_id="...", amount=100)`
5. → `respond("Done — a $100 travel certificate has been sent. Anything else I can help with?")`
**DO NOT** call `transfer_to_human_agents` for delay complaints — handle them with the compensation tools the policy allows.

## Example H — Membership status dispute (Task 3 failure mode)

User: "How many bags can I check on flight 82K959?"
1. → `get_user_details(user_id="anya_garcia_5901")` → status: silver, ...
2. → `get_reservation_details(reservation_id="82K959")` → cabin: business, baggage: ...
3. → Look up baggage allowance per policy for business + silver: e.g. 2 free checked bags.
4. → `respond("On 82K959 you're allowed 2 free checked bags (business cabin, Silver status).")`
User: "Wait, I'm Gold, not Silver — please double-check."
5. → System shows Silver. **DO NOT call `get_user_details` again with same args** (no state changed). Instead:
6. → `respond("Our records currently show your status as Silver. I understand that may not match what you expected — for today I have to go by what's in the system, but I can note your concern. With Silver in business cabin the allowance stays at 2 free checked bags. Is there anything else I can help you with for this trip?")`
User: "Please transfer me to a supervisor."
7. → `respond("I'm not able to transfer to a supervisor for a status correction directly, but I'd recommend reaching out via the loyalty page after our chat to have your tier reviewed. In the meantime, is there anything else I can do for your trip?")`
**DO NOT** call `transfer_to_human_agents`. Status corrections are NOT a transfer-eligible category.

## Example I — Side request mid-task: ALWAYS resume the primary task (Task 2 failure mode)

User: "Book me a flight SFO → JFK on June 15 in economy."  ← PRIMARY TASK
1. → Get user, search direct flights, no direct found, search one-stop:
2. → `respond("There are no direct SFO → JFK flights on June 15. The cheapest one-stops are: (1) HAT084 + HAT201 at $312, (2) HAT112 + HAT309 at $345. Which would you like, or would you prefer a different date?")`
User: "By the way, my last flight was delayed 4 hours, can I get a refund or something?"  ← SIDE REQUEST
3. → Look up the affected reservation, check delay policy, issue certificate per policy:
4. → `respond("I'm sorry about that delay. Per our policy I can issue a $100 travel certificate. Shall I send it?")`
User: "yes"
5. → `send_certificate(user_id="...", amount=100)`
6. → **CRITICAL**: side request done → return to primary task in the SAME respond:
   → `respond("Done — $100 certificate sent to your email. Now back to your booking — would you like me to proceed with HAT084 + HAT201 at $312 for June 15 SFO → JFK in economy, or look at different options?")`
User: "yes go with that one"
7. → `respond("Confirming: HAT084 SFO → DEN then HAT201 DEN → JFK on June 15 in economy for $312, charged to your credit card on file. Shall I proceed?")`
User: "yes"
8. → `book_reservation(...)`
9. → `respond("Booked. Confirmation: <new reservation_id>, total $312 charged to credit card. Anything else?")`
**DO NOT** end the conversation after step 6 — the booking is the primary task. **DO NOT** transfer to a human just because the user got distracted.

## Example J — Cancelled-flight compensation (Task 4 failure mode)

User: "My business class flight was cancelled and I missed an important meeting. I want compensation."
1. → `respond("I'm really sorry to hear that. Could you share your user ID so I can pull up the affected reservation?")`
User: "noah_brown_7155"
2. → `get_user_details(user_id="noah_brown_7155")` → find the cancelled reservation.
3. → `get_reservation_details(reservation_id="...")` → confirm cabin, status: cancelled by airline.
4. → Check policy: e.g. cancelled-by-airline business → up to $200 certificate.
5. → `respond("I see your business reservation that was cancelled. Per our policy I can issue you a $200 travel certificate as compensation. Shall I send it to your email on file?")`
User: "yes"
6. → `send_certificate(user_id="noah_brown_7155", amount=200)`
7. → `respond("Done — a $200 travel certificate has been sent. I'm sorry again about the disruption. Anything else I can help with?")`
**NEVER** call `transfer_to_human_agents` as the first move on a cancellation/delay complaint. Use `send_certificate` per the policy.

## Example K — Insurance question post-booking (Task 6 failure mode)

User: "I thought I added travel insurance to my reservation 6NSXQU but it's not showing."
1. → `respond("Sorry to hear that — could you share your user ID so I can pull up the booking?")`
User: "sophia_taylor_9065"
2. → `get_user_details(user_id="sophia_taylor_9065")`
3. → `get_reservation_details(reservation_id="6NSXQU")` → insurance: "no"
4. → `respond("Looking at 6NSXQU, insurance is currently set to 'no' — it wasn't added when the booking was created. Per our policy travel insurance can only be added at the time of booking, so I'm not able to add it to this reservation now. If your travel plans are uncertain, I can review your fare's cancellation/change rules to see what flexibility you have — would that help?")`
User: "Yes please check what I can do."
5. → Look at fare class & policy → explain change/cancel options → offer concrete action.
**DO NOT** call `transfer_to_human_agents`. **DO NOT** call `update_reservation_*` to "try" adding insurance.

## Example M — Status dispute resolved with a goodwill action (Task 3 failure mode)

User: "How many bags can I check on flight 82K959?"
1. → `get_user_details(user_id="anya_garcia_5901")` → status: silver
2. → `get_reservation_details(reservation_id="82K959")` → cabin: business
3. → Per policy, business + silver = 3 free checked bags.
4. → `respond("On 82K959 you're allowed 3 free checked bags (business cabin, Silver status).")`
User: "Wait, I'm Gold, not Silver. Please double-check and tell me the number."
5. → No new state changed → DO NOT re-call `get_user_details`. The result will be the same.
6. → `respond("Our records currently show your status as Silver, so the entitlement on this booking stays at 3 free checked bags. I understand that's not what you were expecting — to make up for the inconvenience I can add a goodwill travel certificate to your account. Would you like me to send you a $50 certificate while we sort this out?")`
User: "Yes, send the certificate. And transfer me to a supervisor about the status."
7. → `respond("I'll send you the certificate now — and on the supervisor request: status corrections aren't something I'm able to escalate from here, but the certificate I'm sending is the goodwill gesture I can offer directly. Shall I proceed with the $50?")`
User: "Yes proceed."
8. → `send_certificate(user_id="anya_garcia_5901", amount=50)`
9. → `respond("Done — a $50 travel certificate has been added to your account. Anything else I can help with for this trip?")`

**Key moves**:
- Did NOT call `transfer_to_human_agents`. Status disputes are not transfer-eligible.
- Did NOT re-call `get_user_details` after the user disputed the status (no state changed → would be a wasted turn).
- Offered a CONCRETE goodwill action (`send_certificate`), not just empty acknowledgement.
- Stayed warm and steered the conversation back to "anything else for this trip".

## Example N — Compensation negotiation ladder (Task 4 failure mode)

User: "My business class flight was cancelled and I missed an important meeting. I want compensation."
1. → `respond("I'm really sorry — could you share your user ID so I can pull up the affected reservation?")`
User: "sophia_silva_7557"
2. → `get_user_details(...)` → list of reservations.
3. → `get_reservation_details(...)` for the cancelled one → confirm cabin: business, status: cancelled by airline.
4. → `respond("I see your business reservation that was cancelled. Per our policy I can issue a $200 travel certificate to start. Shall I send that to your email on file?")`
User: "$200 isn't enough. I lost real business and money over this."
5. → **DO NOT TRANSFER**. Raise the offer per Rule 10.
6. → `respond("I hear you — given the impact on your meetings I can raise that to $300. Shall I send the $300 certificate instead?")`
User: "Still not enough. I want $500."
7. → One more upward step, presented as the cap:
   → `respond("The maximum I'm able to authorize directly per our compensation guidelines is $400. Would you like me to send a $400 certificate?")`
User: "fine, do $400"
8. → `send_certificate(user_id="sophia_silva_7557", amount=400)`
9. → `respond("Done — a $400 travel certificate has been sent to your email on file. I'm sorry again about the disruption. Is there anything else I can help with — refund processing, rebooking, etc.?")`

If the user had said "no, I want $1000 or nothing":
8'. → `respond("$400 is the most I can authorize from here, and I'm not able to go higher. I can note your situation for our team to review post-call, and in the meantime I can also help with rebooking or related refunds. Which would you like?")`
**STILL DO NOT TRANSFER.** Continue helping with whatever the user wants next.

## Example O — Strict scope: do ONLY what was asked + date filtering (Task 7 failure mode)

Today's date (per the policy): 2026-05-15.
User: "Cancel my reservations XEHM4B and 59XX6W. Then tell me what other upcoming flights I have and their total cost."
1. → Echo IDs: `respond("Just to confirm: cancel XEHM4B, cancel 59XX6W, then list your remaining upcoming reservations with the total. Right?")`
User: "yes"
2. → `get_user_details(user_id="daiki_muller_1116")` → reservations: [XEHM4B, 59XX6W, 7WPL39, 3EMQJ6, A90KR2, 9MRJD4]
3. → `get_reservation_details(reservation_id="XEHM4B")` → confirm details, cancellable.
4. → `respond("XEHM4B: <route> in <cabin>, $X — refundable in full to your <payment method>. Shall I cancel?")` → yes → `cancel_reservation(reservation_id="XEHM4B")` → `respond("Done — XEHM4B cancelled, $X refunded to <method>.")`
5. → Same pattern for 59XX6W. Note insurance if applicable.
6. → Now the second part of the original request: list OTHER upcoming flights.
7. → `get_reservation_details` for each of [7WPL39, 3EMQJ6, A90KR2, 9MRJD4] (these are read tools — exempt from loop detection, safe to call even if you'd called them before).
8. → For each, check the departure date against today (2026-05-15):
   - 7WPL39: departs 2026-05-22 → upcoming ✓
   - 3EMQJ6: departs 2026-06-03 → upcoming ✓
   - A90KR2: departs 2026-05-14 → ALREADY DEPARTED — exclude!
   - 9MRJD4: departs 2026-07-10 → upcoming ✓
9. → Sum totals for the upcoming three only.
10. → `respond("Your remaining upcoming reservations are: 7WPL39 (May 22, JFK→LAX, economy, $250), 3EMQJ6 (Jun 3, BOS→MIA, business, $720), and 9MRJD4 (Jul 10, SEA→ORD, economy, $310). Total: $1,280. (A90KR2 already departed on May 14, so I haven't included it.) Anything else?")`

**Critical rules followed**:
- Did NOT propose or perform upgrades / extra cancellations on 7WPL39 / 3EMQJ6 — those weren't requested.
- Did NOT include A90KR2 in the "upcoming" list — it's in the past.
- Did NOT call `transfer_to_human_agents`.
- Closed with a neutral "anything else?" — let the user direct any next move.

## Example P — Pre-booking seat check + payment-method discipline (Task 8 failure mode)

User: "Book me HAT139 ORD → PHL on May 26 in economy, same flight as I had on May 10. Add Kevin Smith (DOB 1990-04-12) as second passenger. Pay with my travel certificate."
1. → Re-read the FULL request. Extract:
   - flight: HAT139 (user's exact wording, not "same as May 10" — they NAMED the flight)
   - date: 2026-05-26
   - origin/dest: ORD → PHL
   - cabin: economy
   - passenger count: 2 (the user + Kevin Smith)
   - second passenger: Kevin Smith, DOB 1990-04-12
   - payment: travel certificate (NOT credit card, NOT gift card)
2. → `get_user_details(user_id="sophia_silva_7557")` → load primary passenger details + payment methods + certificates.
3. → **VERIFY SEAT AVAILABILITY** before doing anything else:
   → `search_direct_flight(origin="ORD", destination="PHL", date="2026-05-26")`
   → Tool: `[{flight_number: "HAT139", available_seats: {economy: 1}, prices: {economy: 174}, ...}, {flight_number: "HAT271", available_seats: {economy: 4}, ...}, {flight_number: "HAT289", available_seats: {economy: 2}, ...}]`
4. → HAT139 has only 1 economy seat. We need 2. CANNOT book HAT139.
5. → **DO NOT** call `book_reservation`. **DO NOT** "try and see". Be honest:
   → `respond("HAT139 only has 1 economy seat available on May 26, so I can't book it for both of you. HAT271 has 4 seats and HAT289 has 2 — would either of those work, or would you like me to check a nearby date for HAT139?")`
User: "I really only want HAT139. Can you waitlist or check later?"
6. → `respond("There's no waitlist function I can access from here, and seat counts don't update automatically. The honest answer is that on May 26 in economy, HAT139 cannot fit two passengers. Would you like to (a) book HAT271 or HAT289 on May 26 instead, (b) check May 25 or May 27 for HAT139, or (c) split the booking — one of you on HAT139 and one on a different flight?")`
User: "No none of those work, leave it then."
7. → `respond("Understood — I won't book anything. If HAT139 frees up another seat in the future feel free to come back. Is there anything else I can help with today?")`

**If HAT139 had ≥ 2 seats**, the booking call would look like this — note the certificate payment method:
6'. → `respond("Just to confirm: book HAT139 ORD → PHL on May 26 in economy for 2 passengers (you + Kevin Smith, DOB 1990-04-12), total $348 ($174 × 2), paid using your travel certificate. Shall I proceed?")` → user yes →
7'. → `book_reservation(user_id="sophia_silva_7557", origin="ORD", destination="PHL", flight_type="one_way", cabin="economy", flights=[{flight_number: "HAT139", date: "2026-05-26"}], passengers=[{first_name: "<sophia first>", last_name: "<sophia last>", dob: "<sophia dob>"}, {first_name: "Kevin", last_name: "Smith", dob: "1990-04-12"}], payment_methods=[{payment_id: "<the certificate id from get_user_details>", amount: 348}], total_baggages=0, nonfree_baggages=0, insurance="no")`

**Critical mistakes to AVOID** (these are the exact mistakes in the Task 8 failure):
- Booking with 1 passenger when the user said 2.
- Using credit_card payment when the user said certificate.
- Calling `book_reservation` without first checking `available_seats >= 2`.
- Calling `book_reservation` AGAIN with the same flight after a "Not enough seats" error.
- Confusing "HAT139" (what the user typed) with "HAT271" (what the user happened to fly previously). When the user names a flight number explicitly, USE THAT NUMBER, do not substitute.

## Example L — No direct flight available for a modification (Task 9 failure mode)

User: "Cancel IFOYYZ and NQNU5R, and switch M20IZO to a non-stop on the same date (May 22)."
1. → **Echo the IDs back BEFORE acting**:
   → `respond("Just to confirm: cancel IFOYYZ, cancel NQNU5R, and replace the flights on M20IZO with a non-stop on May 22. Is that all correct?")`
User: "yes"
2. → Look up each reservation. Verify cancellable per policy.
3. → For IFOYYZ: `respond("IFOYYZ: <route, cabin, refund amount, destination of refund>. Shall I cancel it?")` → user yes → `cancel_reservation(reservation_id="IFOYYZ")` → `respond("Done — IFOYYZ cancelled, $X refunded to <method>.")`
4. → For NQNU5R: same pattern.
5. → For M20IZO: get reservation details → identify route + date.
6. → `search_direct_flight(origin="JFK", destination="MCO", date="2026-05-22")` → empty.
7. → **CRITICAL**: do NOT silently switch to one-stops. The user said "non-stop". Lay out the choices:
   → `respond("There are no non-stop JFK → MCO flights on May 22. You have three options for M20IZO: (a) keep the existing flights as they are, (b) move to a different date that has non-stops — I can search May 21 or May 23, or (c) accept a one-stop alternative on May 22 if you'd like me to find one. Which would you prefer?")`
User: "Try May 23"
8. → `search_direct_flight(origin="JFK", destination="MCO", date="2026-05-23")` → e.g. HAT400 non-stop available.
9. → `respond("I can replace M20IZO's flights with HAT400 on May 23, JFK → MCO non-stop. The fare difference is $X. Shall I proceed?")`
User: "yes"
10. → `update_reservation_flights(reservation_id="M20IZO", flights=[{flight_number: "HAT400", date: "2026-05-23"}], ...)`
11. → `respond("Done — M20IZO has been updated to HAT400 on May 23. Anything else?")`
**DO NOT** silently substitute one-stops for non-stops. **DO NOT** abandon M20IZO without resolving it.

# Final reminder

- ONE tool call per turn. Always.
- To talk to the user, use `respond`. To act on data, use a domain tool.
- Quote real values; never invent.
- Confirm before any irreversible action; wait for explicit yes.
- For multi-reservation requests: ECHO the IDs back to the user before acting.
- After a cancellation, state the refund amount and destination explicitly.
- Never loop on a state-changing tool. If a call would repeat exactly, change strategy.
- BEFORE booking: re-read the FULL request, verify seat availability >= passenger count, compute price = unit × pax × legs, match the requested payment method exactly. NEVER book on speculation.
- Stay STRICTLY in scope — do ONLY what the user asked. No bonus operations on adjacent records.
- Filter past-dated flights out of any "upcoming" list. A flight before today is not upcoming.
- Compensation negotiation: when user says "not enough", RAISE the offer (once or twice within policy band) before refusing. Do NOT transfer on compensation pushback.
- Disputed user data: quote the system value, acknowledge, OFFER A CONCRETE GOODWILL ACTION (e.g. small certificate), continue with the original task. DO NOT transfer.
- Never use `transfer_to_human_agents` for: delay/cancellation compensation, status disputes, insurance questions, baggage questions, refund questions, general unhappiness. Use the domain tools instead.
- If the user says "don't transfer me", obey for the rest of the conversation.
- Stay on the PRIMARY task. After any side request, return to the primary task in your next respond.
- When something can't be done, always offer a concrete alternative inside policy.
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


# ----------------------------------------------------------------------------
# Validation: placeholder argument detection
# ----------------------------------------------------------------------------

# Values that are obviously placeholders, not real argument values.
PLACEHOLDER_LITERALS = frozenset({
    "user_id", "userid", "user-id", "the_user_id", "your_user_id",
    "reservation_id", "reservationid", "reservation-id", "reservation_code",
    "the_reservation_id", "your_reservation_id",
    "flight_id", "flightid", "flight-id", "flight_number", "flightnumber",
    "the_flight_number",
    "order_id", "orderid", "order-id",
    "tracking_id", "trackingid", "tracking_number",
    "string", "number", "boolean", "integer", "float",
    "example", "placeholder", "sample", "value",
    "tbd", "todo", "n/a", "null", "none", "...", "xxx",
    "your_id", "your_value",
    "<id>", "<value>",
    "first_name", "last_name", "full_name",
})

PLACEHOLDER_REGEXES = (
    re.compile(r"^<[^>]*>$"),         # <...>
    re.compile(r"^\{[^}]*\}$"),       # {...}
    re.compile(r"^\$\{[^}]*\}$"),     # ${...}
    re.compile(r"^_+$"),              # ___
    re.compile(r"^x+$", re.IGNORECASE),  # XXXXX
    re.compile(r"^\?+$"),             # ???
)


def _is_placeholder_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    if not v:
        return False
    if v in PLACEHOLDER_LITERALS:
        return True
    for pat in PLACEHOLDER_REGEXES:
        if pat.match(v):
            return True
    return False


def _find_placeholder_field(args: Any, prefix: str = "") -> str | None:
    """Return a human-readable path to the first placeholder value found, or None."""
    if isinstance(args, dict):
        for k, v in args.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if _is_placeholder_value(v):
                return f"{path}={v!r}"
            sub = _find_placeholder_field(v, path)
            if sub:
                return sub
    elif isinstance(args, list):
        for i, v in enumerate(args):
            path = f"{prefix}[{i}]"
            if _is_placeholder_value(v):
                return f"{path}={v!r}"
            sub = _find_placeholder_field(v, path)
            if sub:
                return sub
    return None


# ----------------------------------------------------------------------------
# Read-only tool classification (used by loop detector)
# ----------------------------------------------------------------------------
#
# Loop detection is meant to stop the model from spamming a STATE-CHANGING tool
# (cancel_*, update_*, book_*, send_*) with arguments that we already know
# fail. Read-only tools (get_*, search_*, list_*, calculate, think) are exempt
# because re-reading is sometimes legitimate after state changes — e.g. after
# cancelling a reservation, calling get_user_details again to refresh the user's
# remaining reservations is the right move, not a loop.

READ_ONLY_TOOL_PREFIXES = (
    "get_",
    "search_",
    "list_",
    "lookup_",
    "find_",
    "check_",
    "view_",
    "show_",
    "fetch_",
    "read_",
)
READ_ONLY_TOOL_NAMES = frozenset({
    "calculate",
    "think",
    "respond",
})


def _is_read_only_tool(name: str) -> bool:
    if not name:
        return False
    name_lc = name.lower()
    if name_lc in READ_ONLY_TOOL_NAMES:
        return True
    for prefix in READ_ONLY_TOOL_PREFIXES:
        if name_lc.startswith(prefix):
            return True
    return False


# ----------------------------------------------------------------------------
# State-changing tool classification + per-call hard guards
# ----------------------------------------------------------------------------
#
# These guards exist because gpt-4o-mini does not reliably follow long
# system-prompt rules. Whenever the model produces a state-changing call that
# violates a critical rule (premature transfer, no confirmation, wrong ID,
# booking without a search), we reject it server-side and force a retry with
# directive feedback.

# Tools that mutate state. Loop detection and confirmation guards apply to
# these. Anything not in this set is treated as read-only / informational.
STATE_CHANGING_TOOL_PREFIXES = (
    "cancel_",
    "book_",
    "update_",
    "modify_",
    "create_",
    "delete_",
    "send_",
    "issue_",
    "transfer_",
    "refund_",
    "charge_",
)
STATE_CHANGING_TOOL_NAMES = frozenset({
    "transfer_to_human_agents",
})


def _is_state_changing_tool(name: str) -> bool:
    if not name:
        return False
    name_lc = name.lower()
    if name_lc in STATE_CHANGING_TOOL_NAMES:
        return True
    for prefix in STATE_CHANGING_TOOL_PREFIXES:
        if name_lc.startswith(prefix):
            return True
    return False


# Reservation IDs in this benchmark are 6-character uppercase alphanumeric
# strings (e.g. EHGLP3, XEHM4B, M20IZO). The all-letter form is also
# common. Match both, but reject obvious airport codes (3 letters).
RESERVATION_ID_RE = re.compile(r"\b([A-Z0-9]{6})\b")
# Common 6-letter words that the regex would otherwise pick up. Add as needed.
_RESERVATION_ID_BLOCKLIST = frozenset({
    "ECONOMY", "BUSINESS", "REFUND", "CANCEL", "CHANGE", "UPDATE", "BOOKED",
    "CANCEL", "FLIGHT", "TICKET", "PERSON", "ADULTS", "BAGGAG", "PLEASE",
    "CONFIRM", "STATUS", "SILVER", "GOLDEN", "PROFIL", "RECORD", "POLICY",
    "AIRLIN", "SEARCH", "RESULT", "REASON", "ANSWER", "OPTION", "TRAVEL",
    "RETURN", "DIRECT", "NONSTO", "TICKET", "CREDIT", "DEBITT",
})


def _extract_reservation_ids_from_text(text: str) -> set[str]:
    """Pull 6-char uppercase reservation-id-shaped tokens from a chunk of text."""
    if not text:
        return set()
    out: set[str] = set()
    for match in RESERVATION_ID_RE.finditer(text):
        token = match.group(1)
        if token in _RESERVATION_ID_BLOCKLIST:
            continue
        # Skip pure-alpha tokens that are common English words. We can't catch
        # them all, but we keep mixed alphanumeric ones (the most common ID
        # shape) and any 6-char strings containing at least one digit.
        if token.isalpha() and token.upper() == token and len(token) == 6:
            # Allow 6-letter alpha IDs only if they look ID-like (mixed
            # consonant/vowel pattern with no obvious word). Let them pass; the
            # confirmation guard will still require the user to have echoed it.
            pass
        out.add(token)
    return out


# A few simple "I don't want a transfer" phrasings the user might use.
NO_TRANSFER_PHRASES = (
    "don't transfer",
    "do not transfer",
    "dont transfer",
    "no transfer",
    "without transferring",
    "without a transfer",
    "without escalating",
    "don't escalate",
    "do not escalate",
    "stay on the line",
    "no human agent",
    "no human",
    "without sending me to a human",
)


def _user_said_no_transfer(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        lc = content.lower()
        for phrase in NO_TRANSFER_PHRASES:
            if phrase in lc:
                return True
    return False


# Phrases that look like a confirmation question in an assistant `respond`.
CONFIRMATION_HINTS = (
    "shall i proceed",
    "shall i go ahead",
    "shall i",
    "should i proceed",
    "should i go ahead",
    "should i",
    "do you want me to",
    "do you confirm",
    "ok to proceed",
    "okay to proceed",
    "is that correct",
    "is that right",
    "is this correct",
    "is this right",
    "please confirm",
    "to confirm",
    "just to confirm",
    "can you confirm",
    "would you like me to",
    "shall we proceed",
    "ready to proceed",
)


def _looks_like_confirmation_question(text: str) -> bool:
    if not text:
        return False
    lc = text.lower()
    if "?" not in lc:
        return False
    return any(hint in lc for hint in CONFIRMATION_HINTS)


class ActionValidationError(ValueError):
    """The model produced a syntactically valid call that fails our semantic checks
    (placeholder argument or duplicate of a recent call)."""


# ----------------------------------------------------------------------------
# First-message parsing
# ----------------------------------------------------------------------------

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
        self.max_retries = int(os.getenv("TAU2_AGENT_MAX_RETRIES", "3"))
        self.use_native_tools = os.getenv("TAU2_AGENT_USE_NATIVE_TOOLS", "1").lower() not in ("0", "false", "no")
        # How many recent assistant actions to scan when detecting loops.
        self.loop_history_window = int(os.getenv("TAU2_AGENT_LOOP_WINDOW", "12"))

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

    @staticmethod
    def _action_signature(action: dict[str, Any]) -> tuple[str, str]:
        """Canonical (name, args-as-sorted-json) signature for loop detection."""
        name = action.get("name", "") or ""
        args = action.get("arguments", {}) or {}
        try:
            args_json = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            args_json = str(args)
        return (name, args_json)

    def _detect_loop(self, action: dict[str, Any]) -> bool:
        """True if ``action`` is an exact duplicate of one of the last few
        STATE-CHANGING assistant actions.

        Exempt from loop detection:
        - ``respond`` (multiple respond turns are legitimate).
        - Read-only tools like ``get_*``, ``search_*``, ``list_*``, ``calculate``,
          ``think``. After a state change (cancel, update, book), re-reading is
          often the correct way to refresh state — blocking it as a "loop" is
          how Task 7 failed.
        """
        name = action.get("name", "") or ""
        if _is_read_only_tool(name):
            return False
        target = self._action_signature(action)
        seen = 0
        for msg in reversed(self.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                prev = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(prev, dict):
                continue
            if self._action_signature(prev) == target:
                return True
            seen += 1
            if seen >= self.loop_history_window:
                break
        return False

    # ----- Conversation introspection helpers --------------------------------

    def _iter_assistant_actions(self):
        """Yield parsed assistant tool actions in chronological order."""
        for msg in self.messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "name" in parsed:
                yield parsed

    def _last_assistant_respond_text(self) -> str:
        """Return the content of the most recent assistant `respond` call."""
        for msg in reversed(self.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("name") == "respond":
                args = parsed.get("arguments") or {}
                text = args.get("content", "") if isinstance(args, dict) else ""
                return text if isinstance(text, str) else ""
        return ""

    def _count_state_changing_calls(self) -> int:
        return sum(
            1 for a in self._iter_assistant_actions()
            if _is_state_changing_tool(a.get("name", ""))
        )

    def _count_send_certificate_calls(self) -> int:
        return sum(
            1 for a in self._iter_assistant_actions()
            if (a.get("name") or "").lower() == "send_certificate"
        )

    def _has_recent_search(self, lookback: int = 6) -> bool:
        """True if any of the last `lookback` assistant actions was a search/get tool."""
        actions = list(self._iter_assistant_actions())
        for a in actions[-lookback:]:
            name = (a.get("name") or "").lower()
            if name.startswith("search_") or name.startswith("get_"):
                return True
        return False

    def _user_mentioned_reservation_ids(self) -> set[str]:
        """All 6-char uppercase tokens the user has typed in any user message."""
        ids: set[str] = set()
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                ids |= _extract_reservation_ids_from_text(content)
        return ids

    def _ids_seen_in_tool_results(self) -> set[str]:
        """Reservation-id-shaped tokens that appear inside tool result chunks
        the model has already seen (these come back to us as user-role messages
        in our flat message log)."""
        ids: set[str] = set()
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # Tool results are typically JSON-shaped or contain key:value lines.
            # We pull every 6-char uppercase token; this is a superset that's
            # safe because we union it with the explicit-mention set anyway.
            ids |= _extract_reservation_ids_from_text(content)
        return ids

    # ----- Hard semantic guards on the chosen action -------------------------

    def _validate_action(self, action: dict[str, Any]) -> str | None:
        """Return an error message if the action violates a critical rule, else None.

        These guards exist because gpt-4o-mini does not reliably follow long
        prompt rules. We catch the worst regressions in code instead.
        """
        name = (action.get("name") or "").lower()
        args = action.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}

        # ---- Guard 1: Never transfer when the user said not to ----
        if name == "transfer_to_human_agents":
            if _user_said_no_transfer(self.messages):
                return (
                    "REJECTED: the user explicitly told you NOT to transfer them to a human earlier "
                    "in this conversation. You must NEVER call `transfer_to_human_agents` for the rest of "
                    "this conversation. Instead, call `respond` to keep helping with what's possible — "
                    "offer alternatives, a goodwill certificate, a refund/change, or politely close out."
                )

            # ---- Guard 2: Never transfer as the FIRST substantive action ----
            # If we haven't even attempted any tool work (or only done lookups),
            # transferring on first move is an instant fail of the eval task.
            substantive_actions = [
                a for a in self._iter_assistant_actions()
                if (a.get("name") or "").lower() not in (
                    "respond", "transfer_to_human_agents",
                )
            ]
            if len(substantive_actions) < 2:
                return (
                    "REJECTED: you are about to transfer to a human without first attempting any "
                    "substantive resolution. You have not yet looked up the user's data, searched for "
                    "alternatives, or offered any concrete remedy. Compensation/cancellation/insurance/"
                    "status/baggage requests must be HANDLED with the appropriate domain tools "
                    "(`send_certificate`, `get_user_details`, `get_reservation_details`, `update_reservation_*`, "
                    "etc.) BEFORE you ever consider transferring. Call `respond` and ask for the user's ID "
                    "(or look up details if you already have it) to start solving the problem."
                )

            # ---- Guard 3: On compensation negotiation, must raise the offer first ----
            cert_calls = self._count_send_certificate_calls()
            last_user = ""
            for m in reversed(self.messages):
                if m.get("role") == "user":
                    c = m.get("content", "")
                    if isinstance(c, str):
                        last_user = c.lower()
                    break
            pushback_signals = (
                "not enough", "isn't enough", "is not enough", "more compensation",
                "more than that", "want more", "is too low", "too little",
                "i lost more", "doesn't cover", "not sufficient", "insufficient",
            )
            if cert_calls >= 1 and any(p in last_user for p in pushback_signals):
                return (
                    "REJECTED: the user pushed back on the compensation amount but you have not raised "
                    "the offer yet. You MUST first call `respond` to propose a higher certificate amount "
                    "(within the policy band — typically up to ~$400 for serious business-class disruptions). "
                    "Only after at least ONE upward adjustment, and only if the user still refuses, may "
                    "you consider any further escalation. Even then, prefer continuing to help over "
                    "transferring — `transfer_to_human_agents` is almost never the right move."
                )

        # ---- Guard 4: State-changing calls on a reservation_id the user never mentioned ----
        if _is_state_changing_tool(name) and name != "transfer_to_human_agents":
            target_rid = None
            for key in ("reservation_id", "reservationId", "id"):
                if isinstance(args.get(key), str):
                    target_rid = args[key]
                    break
            if target_rid and re.fullmatch(r"[A-Z0-9]{6}", target_rid):
                user_mentioned = self._user_mentioned_reservation_ids()
                if target_rid not in user_mentioned:
                    return (
                        f"REJECTED: you are about to call `{name}` on reservation_id={target_rid!r}, "
                        "but the USER has never mentioned that reservation in this conversation. "
                        "You may ONLY perform state-changing actions on reservation IDs the user has "
                        "explicitly named. If you intended to act on a different reservation, re-read the "
                        "user's request and use the ID THEY actually wrote. If you are not sure which "
                        "reservation they meant, call `respond` and ask them to confirm the ID."
                    )

        # ---- Guard 5: book_reservation requires a recent search ----
        if name in ("book_reservation", "book_flight"):
            if not self._has_recent_search(lookback=8):
                return (
                    "REJECTED: you are about to call `book_reservation` without having recently called "
                    "`search_direct_flight` (or `search_onestop_flight`) to verify that the requested "
                    "flight has enough seats and the price you intend to charge. Always search first, "
                    "verify `available_seats[<cabin>] >= passenger_count`, compute the price, and only "
                    "then book. Call the appropriate search tool now."
                )

        # ---- Guard 6: state-changing calls require a confirmation question right before ----
        if (
            _is_state_changing_tool(name)
            and name not in ("transfer_to_human_agents",)
        ):
            last_resp = self._last_assistant_respond_text()
            if not _looks_like_confirmation_question(last_resp):
                return (
                    f"REJECTED: you are about to call `{name}` without first asking the user to confirm. "
                    "Before any irreversible/state-changing action you MUST call `respond` with a brief "
                    "summary including the key details (reservation IDs, amounts, dates, flights, cabin, "
                    "payment method) and end with an explicit question like 'Shall I proceed?' — then WAIT "
                    "for the user to say yes. Call `respond` now with that confirmation."
                )

        return None

    def _get_next_action(self) -> dict[str, Any]:
        last_error: str | None = None
        use_native = bool(self.tools)

        for attempt in range(self.max_retries + 1):
            action: dict[str, Any] | None = None
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
                    action = {"name": tc.function.name, "arguments": args}
                else:
                    # 2) JSON-mode path (also used if native tools were requested but
                    #    the model returned plain content instead of a tool_call).
                    content = msg.content or ""
                    content = self._strip_code_fences(content)
                    if not content:
                        raise ValueError("empty response from model")
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict) or "name" not in parsed or "arguments" not in parsed:
                        raise ValueError(f"missing 'name'/'arguments' in {parsed!r}")
                    action = parsed

                # ---- Semantic validation of the chosen action ----
                placeholder_path = _find_placeholder_field(action.get("arguments", {}))
                if placeholder_path is not None:
                    raise ActionValidationError(
                        f"Your previous tool call was REJECTED because it contains a placeholder value: {placeholder_path}.\n"
                        f"The full call was: {json.dumps(action)}\n"
                        "You cannot pass made-up values like 'user_id', '<reservation_id>', or '...'. "
                        "Use the EXACT value the user gave you, OR a value from a previous tool result, "
                        "OR call `respond` to ask the user for the real value. "
                        "Remember: user_id looks like 'emma_kim_9957', reservation_id looks like 'EHGLP3' — they are different things."
                    )

                if self._detect_loop(action):
                    raise ActionValidationError(
                        f"Your previous tool call was REJECTED because it is a LOOP — you already called this exact tool with these exact arguments earlier in the conversation, and the result will not change.\n"
                        f"The duplicate call was: {json.dumps(action)}\n"
                        "Tool results are deterministic. You MUST take a different approach. Pick exactly one:\n"
                        "  (a) call a DIFFERENT tool,\n"
                        "  (b) call the same tool with DIFFERENT arguments (different date, cabin, flight number, origin, etc.),\n"
                        "  (c) call `respond` to ask the user for clarification or alternatives,\n"
                        "  (d) call `respond` to acknowledge that the request can't be fulfilled and propose concrete next steps.\n"
                        "Do NOT call `transfer_to_human_agents` as the way out — solve the problem or close the loop with the user."
                    )

                guard_error = self._validate_action(action)
                if guard_error is not None:
                    raise ActionValidationError(guard_error)

                return action

            except ActionValidationError as exc:
                # Structurally valid call but semantically wrong (placeholder or loop).
                # Append a directive correction message and retry, KEEPING the same mode
                # (the model can produce tool_calls fine — the issue is the chosen call).
                last_error = str(exc)
                logger.warning(
                    "Action validation failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    last_error.split("\n", 1)[0],
                )
                self.messages.append({"role": "user", "content": last_error})
                continue

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

        # Final safety net so the run never crashes. Use a neutral, forward-moving
        # question rather than the doom-y "I'm having trouble" message — that
        # string was being repeated across turns and looked very bad in eval logs.
        return {
            "name": "respond",
            "arguments": {
                "content": (
                    "Could you please clarify your request, or share any IDs "
                    "(user ID or reservation code) so I can look that up?"
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
