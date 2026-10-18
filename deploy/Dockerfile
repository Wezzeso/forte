# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12.15-slim-trixie
ARG NODE_IMAGE=node:24.22.0-bookworm-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1

FROM ${NODE_IMAGE} AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY web/ ./
COPY brand/ /brand/
RUN npm run build

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS app-build
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
# ffprobe handles bounded metadata reads; inference stays in the separate GPU image.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 forte \
    && useradd --uid 10001 --gid 10001 --home-dir /app --no-create-home forte \
    && mkdir -p /var/lib/forte/uploads /var/lib/forte/results /var/lib/forte/work \
    && chown -R 10001:10001 /var/lib/forte
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FORTE_DATA_DIR=/var/lib/forte FORTE_WEB_DIST=/app/web/dist \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
COPY --from=app-build /app/.venv /app/.venv
# Root-level generic catalogs are needed by pipeline imports. Private reviews and
# source media are excluded from the context and are never copied here.
COPY rules/ /app/rules/
COPY annotations/__init__.py annotations/catalog.py annotations/taxonomy.v1.json /app/annotations/
COPY configs/ /app/configs/
COPY serving/ /app/serving/
COPY models/manifest.json /app/models/manifest.json
COPY deploy/entrypoint.py deploy/worker_health.py /app/deploy/
COPY --from=web-build /web/dist /app/web/dist
USER 10001:10001
ENTRYPOINT ["python", "/app/deploy/entrypoint.py"]
CMD ["uvicorn", "forte.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
