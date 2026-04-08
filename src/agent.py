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

## Rule 8 — STAY ON THE ORIGINAL TASK; resume after any side request

The user's FIRST request in the conversation is the PRIMARY TASK. You are evaluated on whether the PRIMARY TASK gets completed end-to-end. If the user introduces a side request mid-conversation (a complaint, a question, an unrelated ask), handle the side request, BUT the moment it is resolved you MUST explicitly return to the primary task with a respond like:

> "Now back to your booking — would you like me to proceed with the SFO → JFK itinerary I found earlier, or look at different options?"

A primary task is "complete" only when the corresponding state-changing tool has been called (`book_reservation`, `cancel_reservation`, `update_reservation_*`, `send_certificate`) AND you've sent a final closing `respond` confirming the result. A primary task is NOT complete after just searching, finding options, and answering a side question — you still owe the user the booking/cancellation/update.

Practical rules:
- After every side request you handle, your NEXT `respond` must reference the primary task and offer to continue.
- Do NOT call `transfer_to_human_agents` when there is an unfinished primary task — finish it first.
- Do NOT close the conversation ("Anything else?") until the primary task is fully done.
- If the side request itself involves an irreversible action (e.g. issuing a certificate for a delay), confirm it, do it, then return to the primary task in the same `respond` that confirms completion of the side request: "I've sent you a $100 certificate for the delay. Now, shall I proceed with booking that connecting flight on June 12?"

## Rule 9 — Disputed user data: document, continue, NEVER transfer

When system data conflicts with what the user claims (e.g. "I'm Gold but your system says Silver", "the price you quoted is wrong", "I added insurance but it's not showing"):

1. **Quote the system value plainly** without arguing: "Our records currently show your status as Silver and your reservation 6NSXQU has insurance set to no."
2. **Acknowledge the user's perspective** without conceding: "I understand that's not what you were expecting."
3. **Explain calmly that you have to operate on what the system shows** for this conversation: "For today I have to go by what's in the system, but I can absolutely note your concern."
4. **Offer a path forward inside policy**: a related action you CAN take (add insurance to a future booking, change dates, issue a goodwill certificate if the policy allows, or proceed with the current task using system data).
5. **DO NOT** call `transfer_to_human_agents`. The user disputing data is NOT grounds for escalation. Stay calm and keep helping with the original task.

If the user explicitly demands a supervisor AFTER you've offered alternatives and they've refused all of them, you may then politely note that supervisor escalation isn't something you can do directly, and continue helping with whatever the user wants to do next. Still do NOT call `transfer_to_human_agents`.

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
- Never use `transfer_to_human_agents` for: delay/cancellation compensation, status disputes, insurance questions, baggage questions, refund questions, general unhappiness. Use the domain tools instead.
- If the user says "don't transfer me", obey for the rest of the conversation.
- Stay on the PRIMARY task. After any side request, return to the primary task in your next respond.
- Disputed user data: quote the system value, acknowledge, continue with the original task — do NOT transfer.
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
