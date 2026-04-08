import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Agent must be running before tests start."""
    url = request.config.getoption("--agent-url")

    # trust_env=False so an ambient HTTP_PROXY / HTTPS_PROXY / ALL_PROXY in the
    # shell doesn't make httpx route the loopback request through a proxy that
    # then returns 5xx (very common on Windows with VPNs / corporate proxies).
    try:
        with httpx.Client(trust_env=False, timeout=2) as client:
            response = client.get(f"{url}/.well-known/agent-card.json")
        if response.status_code != 200:
            pytest.exit(f"Agent at {url} returned status {response.status_code}", returncode=1)
    except Exception as e:
        pytest.exit(f"Could not connect to agent at {url}: {e}", returncode=1)

    return url
