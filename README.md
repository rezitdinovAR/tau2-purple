# tau2-purple-agent

An advanced **purple agent** for the [tau2-bench](https://github.com/sierra-research/tau2-bench)
scenario on [AgentBeats](https://agentbeats.dev/agentbeater/tau2-bench).

It is built on the [RDI-Foundation/agent-template](https://github.com/RDI-Foundation/agent-template)
A2A scaffold and is designed for **maximum pass rate** on the airline, retail,
and telecom domains while remaining cost-efficient.

## What's inside

- **`src/agent.py`** — the brains.
  - Parses the policy and tool schemas out of the green agent's first message.
  - Feeds them to the LLM as **native function-calling tools** (much more
    reliable than free-form JSON output) with `tool_choice="required"` so the
    model is forced to emit a single, structured action per turn.
  - Uses a strong "elite customer-service agent" system prompt that emphasizes
    policy compliance, careful reasoning, one-action-per-turn discipline, and
    explicit confirmation of irreversible actions.
  - Retries on malformed responses with an explicit error message and falls
    back to a safe `respond` action so a benchmark run never crashes.
  - Provider-agnostic via [litellm](https://docs.litellm.ai/docs/providers) —
    set `TAU2_AGENT_LLM` to any model id (`openai/gpt-4.1`,
    `anthropic/claude-sonnet-4-5`, `openai/gpt-4o`, …).
- **`src/server.py`** — A2A server with the agent card filled in for tau2-bench.
- **`src/executor.py`** — A2A executor; reuses an `Agent` instance per
  `context_id` so the conversation state survives across turns.
- **`src/messenger.py`** — A2A messaging helpers (only needed if the agent ever
  needs to call out to other agents).
- **`amber-manifest.json5`** — declares the config schema (API keys, model id,
  temperature) and the runtime command for AgentBeats / Amber.
- **`.env.example`** — copy to `.env` and fill in your provider key.

## Configuration

All behaviour is controlled via environment variables (loaded from `.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `TAU2_AGENT_LLM` | `openai/gpt-4.1` | litellm model id |
| `TAU2_AGENT_TEMPERATURE` | `0.0` | sampling temperature |
| `TAU2_AGENT_MAX_RETRIES` | `2` | retries on malformed LLM response |
| `TAU2_AGENT_USE_NATIVE_TOOLS` | `1` | set to `0` to fall back to JSON mode |
| `OPENAI_API_KEY` | — | required if you use an OpenAI model |
| `ANTHROPIC_API_KEY` | — | required if you use an Anthropic model |

## Running locally

```bash
# 1. Install dependencies
uv sync

# 2. Create your .env from the template and fill in a provider key
cp .env.example .env
# edit .env and set OPENAI_API_KEY (or ANTHROPIC_API_KEY etc.)

# 3. Run the agent
uv run src/server.py --host 127.0.0.1 --port 9009
```

Verify it's up:

```bash
curl http://127.0.0.1:9009/.well-known/agent-card.json
```

## Running with Docker

```bash
docker build -t tau2-purple-agent .
docker run -p 9009:9009 --env-file .env tau2-purple-agent --host 0.0.0.0 --port 9009
```

## Tests

```bash
uv sync --extra test
# (start the agent in another terminal — see "Running locally")
uv run pytest --agent-url http://localhost:9009
```

## Submitting to AgentBeats

See the **"How to submit"** section at the bottom of this file (or ask Claude
for the step-by-step instructions). In short:

1. Push this repo to GitHub. The included GitHub Actions workflow will build,
   test, and publish a Docker image to `ghcr.io/<you>/<repo>:latest`.
2. Edit `amber-manifest.json5` and replace the placeholder image reference with
   your real GHCR image.
3. Register the agent on https://agentbeats.dev as a **purple** agent, point it
   at the GHCR image, and supply your API key as the `openai_api_key` (or
   `anthropic_api_key`) config value.
4. Submit it against the `tau2-bench` green agent.
