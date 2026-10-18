# Dependency versions and verification

Registry verification: **17 October 2026**. These versions were returned by each project's public PyPI JSON endpoint. Exact pins and transitive hashes are in `pyproject.toml` and `uv.lock`; registry freshness is not a claim of complete application or GPU compatibility.

| Dependency | Pinned version | Primary registry source | Purpose |
|---|---|---|---|
| FastAPI | 0.143.3 | https://pypi.org/pypi/fastapi/json | CPU API |
| Pydantic | 2.14.1 | https://pypi.org/pypi/pydantic/json | Canonical contracts |
| Uvicorn | 0.54.0 | https://pypi.org/pypi/uvicorn/json | ASGI server |
| python-multipart | 0.0.32 | https://pypi.org/pypi/python-multipart/json | Upload parser |
| SQLAlchemy | 2.1.5 | https://pypi.org/pypi/sqlalchemy/json | Durable leased jobs |
| psycopg + binary | 3.3.6 | https://pypi.org/pypi/psycopg/json | PostgreSQL driver |
| HTTPX | 0.28.1 | https://pypi.org/pypi/httpx/json | Private model client and HTTP tests |
| Pillow | 13.0.0 | https://pypi.org/pypi/pillow/json | JPEG validation and evidence |
| PyAV | 19.0.1 | https://pypi.org/pypi/av/json | Timestamp-aware visual decoding |
| jsonschema | 4.26.0 | https://pypi.org/pypi/jsonschema/json | Model output/schema validation |
| pytest | 9.1.1 | https://pypi.org/pypi/pytest/json | Tests |
| pytest-asyncio | 1.4.0 | https://pypi.org/pypi/pytest-asyncio/json | Async tests |
| openpyxl | 3.1.5 | https://pypi.org/pypi/openpyxl/json | Local data audit only |
| Hatchling | 1.32.4 | https://pypi.org/pypi/hatchling/json | Package build |

The optional Linux/Python 3.12 training extra now selects **Torch 2.13.0+cu129**, **torchvision 0.28.0+cu129**, **Transformers 5.17.0**, [PEFT 0.21.2](https://pypi.org/project/peft/), and [accelerate 1.15.0](https://pypi.org/project/accelerate/). Torch/vision use the explicit [official CUDA 12.9 index](https://download.pytorch.org/whl/cu129/torch/) ([vision listing](https://download.pytorch.org/whl/cu129/torchvision/)). The current listings provide these Linux/Python 3.12 wheels; no 2.14.1/0.29.1 CUDA 12.9 wheel was found. The newer generic PyPI candidates were replaced after the actual CUDA 13 image failed on host driver 565.57.01, while CUDA 12.9 passed. Transformers matches the selected serving processor version. PEFT/accelerate remain current stable pins.

The compatible optional lock resolves 85 packages; CPU/dev still installs 39 packages with unchanged API pins. `uv sync --locked` succeeded and targeted training/contract checks passed (37 tests, 39 subtests). This does not install or test training on the GPU. Root Pillow stays 13.0.0 for the CPU application; A09's fully isolated training image can independently pin Pillow 12.3.0 for exact serving-library parity. No real training is authorized until reviewed targets and the scenario split are locked.

The frontend lock pins React/React DOM **19.3.1**, TypeScript **7.0.2**, Vite **8.3.4**, React Vite plugin **6.1.2**, Playwright **1.64.1** and lucide-react **1.59.0** in `web/package.json` / `web/package-lock.json`. A16's TypeScript/production build and eight synthetic browser cases pass on that stack. The deployed Node builder is pinned separately in `deploy/Dockerfile`.

When deploying offline, prepare wheels and model snapshots on an authorized host, preserve hashes/licenses, and install the locked dependencies before disabling egress. A successful CPU install does not establish offline model readiness.

## Actual private GPU image

A15 inspected the selected stable official vLLM `0.31.0-cu129` image on the authorized A100. Its producer-pinned packages are vLLM distribution `0.31.0+cu129` (import version `0.31.0`), Torch `2.13.0+cu129`, torchvision `0.28.0+cu129`, Transformers processor `5.17.0`, and Pillow `12.3.0`. This environment is isolated from CPU API dependencies. A09's exact training image inherits that tested stack and adds only the hashed PEFT/accelerate requirements; the root optional extra keeps CPU Pillow independent. Preserve the tested image stack; independently upgrading bundled dependencies changes the runtime and requires validation. Actual CUDA/BF16, model warmup/readiness and isolated synthetic-marker inference pass; real-video quality is a separate gate. See the selected digest, CUDA/startup and synthetic concurrency proofs under `configs/serving/`.
