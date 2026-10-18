# syntax=docker/dockerfile:1.7
# Separate training image; does not modify the running inference container.
# This exact cu129 producer stack passed BF16 on the authorized A100/565 host.
FROM vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb
USER root
WORKDIR /app
# The inherited optional torchcodec binary needs CUDA13. Frames are decoded by
# the approved video pipeline; training consumes reviewed JPEGs, never audio.
RUN uv pip uninstall --system torchcodec \
    && uv venv --system-site-packages /opt/forte-training
COPY training/requirements-addons.lock /app/training/requirements-addons.lock
# Only add PEFT/accelerate to an isolated venv. No dependencies are upgraded.
RUN env -u UV_OVERRIDE uv pip install --python /opt/forte-training/bin/python --no-deps --require-hashes \
    --requirement /app/training/requirements-addons.lock
COPY training/*.py /app/training/
COPY training/temporal/*.py /app/training/temporal/
COPY training/temporal/fact-axis.v1.json /app/training/temporal/fact-axis.v1.json
COPY training/temporal/w2-protocol.v1.json /app/training/temporal/w2-protocol.v1.json
COPY annotations/taxonomy.v1.json /app/annotations/taxonomy.v1.json
COPY training/runtime-lock.json /app/training/runtime-lock.json
COPY training/e1.example.json /app/training/e1.example.json
COPY src/forte/ /app/src/forte/
COPY scripts/audit_data.py /app/scripts/audit_data.py
COPY scripts/build_reviewed_split.py /app/scripts/build_reviewed_split.py
COPY scripts/exposure_registry.py /app/scripts/exposure_registry.py
COPY scripts/split_gate.py /app/scripts/split_gate.py
COPY scripts/build_automated_split.py /app/scripts/build_automated_split.py
COPY evaluation/*.py /app/evaluation/
COPY models/manifest.json /app/models/manifest.json
COPY configs/decision/default.json /app/configs/decision/default.json
ENV PATH="/opt/forte-training/bin:${PATH}" \
    PYTHONPATH=/app:/app/src \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
# This check reads package metadata and never initializes CUDA during a build.
RUN python -m training.runtime_check --lock /app/training/runtime-lock.json
ENTRYPOINT ["python", "-m", "training.train"]
CMD ["--config", "/private/e1.local.json", "--preflight"]
