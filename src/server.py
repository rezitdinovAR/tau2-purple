import argparse
import os

import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the tau2-bench purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=None,
        help="LLM model id passed to litellm (e.g. openai/gpt-4.1, anthropic/claude-sonnet-4-5)",
    )
    args = parser.parse_args()

    if args.agent_llm:
        os.environ["TAU2_AGENT_LLM"] = args.agent_llm

    skill = AgentSkill(
        id="tau2_task_fulfillment",
        name="Tau2 Customer Service Task Fulfillment",
        description=(
            "Solves customer service tasks across the tau2-bench airline, retail, and "
            "telecom domains. Reads the domain policy and tool schemas from the first "
            "message and replies with one structured tool call per turn."
        ),
        tags=["benchmark", "tau2", "tau2-bench", "customer-service", "agentbeats"],
        examples=[
            "Run a tau2-bench airline task",
            "Cancel a reservation following the company policy",
            "Troubleshoot a telecom connectivity issue with the user",
        ],
    )

    agent_card = AgentCard(
        name="tau2-purple-agent",
        description=(
            "Advanced purple agent for the tau2-bench AgentBeats scenario. Uses native "
            "function calling, a strict policy-first system prompt, and robust JSON "
            "parsing with retries to maximize pass rate on the airline, retail, and "
            "telecom domains."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(
        server.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
