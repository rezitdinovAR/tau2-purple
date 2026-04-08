FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ENV UV_HTTP_TIMEOUT=300

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

# Note: we don't pass `--locked` here so the build also works on first push
# before the user has regenerated uv.lock for the new dependencies. Once you
# run `uv sync` locally and commit the refreshed lock, you can re-add --locked
# for fully reproducible builds.
RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
