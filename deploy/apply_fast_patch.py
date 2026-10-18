"""Idempotently enable FORTE_ANALYZER=fast in an installed forte package (used by Dockerfile.fastpatch)."""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])  # .../site-packages/forte


def patch(rel, old, new, guard, alternatives=()):
    path = root / rel
    text = path.read_text(encoding="utf-8")
    if guard in text:
        print(f"{rel}: already patched")
        return
    for o, n in ((old, new), *alternatives):
        if o in text:
            text = text.replace(o, n, 1)
            break
    else:
        raise SystemExit(f"{rel}: anchor not found")
    if "\nimport os\n" not in text:
        text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport os\n", 1)
    path.write_text(text, encoding="utf-8")
    print(f"{rel}: patched")


patch("worker.py",
      "        if pipeline is not None:\n            self.pipeline = pipeline\n",
      "        if pipeline is not None:\n            self.pipeline = pipeline\n"
      "        elif os.environ.get(\"FORTE_ANALYZER\", \"\").strip().lower() == \"fast\":\n"
      "            from forte.fast.service import FastPipeline\n"
      "            self.pipeline = FastPipeline(settings)\n",
      "forte.fast.service import FastPipeline",
      alternatives=[("        self.pipeline = pipeline or AnalysisPipeline(settings)\n",
                     "        if pipeline is None and os.environ.get(\"FORTE_ANALYZER\", \"\").strip().lower() == \"fast\":\n"
                     "            from forte.fast.service import FastPipeline\n"
                     "            pipeline = FastPipeline(settings)\n"
                     "        self.pipeline = pipeline or AnalysisPipeline(settings)\n")])

patch("api/main.py",
      "    if config.model_native_launch_config_path is not None:\n        from forte.native_dispatch import native_model_alive\n",
      "    if os.environ.get(\"FORTE_ANALYZER\", \"\").strip().lower() == \"fast\":\n"
      "        base = (os.environ.get(\"FORTE_FAST_MODEL_BASE_URL\") or config.model_base_url).rstrip(\"/\")\n"
      "        try:\n"
      "            async with httpx.AsyncClient(timeout=config.readiness_timeout_seconds, trust_env=False,\n"
      "                                         follow_redirects=False) as client:\n"
      "                response = await client.get(base + \"/models\")\n"
      "            return response.status_code == 200\n"
      "        except httpx.HTTPError:\n"
      "            return False\n"
      "    if config.model_native_launch_config_path is not None:\n        from forte.native_dispatch import native_model_alive\n",
      "FORTE_FAST_MODEL_BASE_URL",
      alternatives=[("    def expected_pins() -> dict[str, Any] | None:\n",
                     "    if os.environ.get(\"FORTE_ANALYZER\", \"\").strip().lower() == \"fast\":\n"
                     "        base = (os.environ.get(\"FORTE_FAST_MODEL_BASE_URL\") or config.model_base_url).rstrip(\"/\")\n"
                     "        try:\n"
                     "            async with httpx.AsyncClient(timeout=config.readiness_timeout_seconds, trust_env=False,\n"
                     "                                         follow_redirects=False) as client:\n"
                     "                response = await client.get(base + \"/models\")\n"
                     "            return response.status_code == 200\n"
                     "        except httpx.HTTPError:\n"
                     "            return False\n"
                     "    def expected_pins() -> dict[str, Any] | None:\n")])

patch("api/metadata.py",
      "    root = config.data_dir.resolve()\n",
      "    root = config.data_dir.resolve()\n"
      "    try:\n"
      "        from forte.fast.service import ui_metadata as _fast_ui_metadata\n"
      "        fast = _fast_ui_metadata(root, job_id, attempt)\n"
      "        if fast is not None:\n"
      "            return fast\n"
      "    except Exception:\n"
      "        pass\n",
      "_fast_ui_metadata")
