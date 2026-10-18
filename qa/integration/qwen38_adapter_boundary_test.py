"""Independent stdlib and hostile-metadata boundaries for serving-only lineage."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from serving.qwen38_adapter_contract import AdapterContractError, read_tensor_header, verify_adapter_contract
from serving.qwen38_source_lineage import SourceLineageError, archive_inventory, verify_source_lineage
from serving.qwen38_identity import file_sha256
from serving.tests.qwen38_adapter_fixtures import (adapter_config_fixture, archive_fixture,
    inventory_fixture, tensor_header_fixture, write_adapter_fixture)

ROOT = Path(__file__).resolve().parents[2]


def test_new_contract_and_lineage_import_without_any_site_packages():
    child = subprocess.run([sys.executable, "-S", "-c",
        "import sys,json; import serving.qwen38_adapter_contract, serving.qwen38_source_lineage; "
        "print(json.dumps({name:name in sys.modules for name in ['torch','peft','safetensors','numpy']}))"],
        cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert child.returncode == 0 and json.loads(child.stdout) == dict.fromkeys(["torch", "peft", "safetensors", "numpy"], False)


@pytest.mark.parametrize("raw", [b'{"x":{},"x":{}}', b'{"x":NaN}', b'{"x":', b'[]'])
def test_invalid_or_ambiguous_safetensors_header_fails_closed(tmp_path, raw):
    path = tmp_path / "generated.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)
    with pytest.raises(AdapterContractError):
        read_tensor_header(path)


@pytest.mark.parametrize("selector", [0, False, [], "", [0]])
def test_explicit_peft_layer_selector_cannot_hide_behind_falsey_values(tmp_path, selector):
    path = tmp_path / "generated.safetensors"
    write_adapter_fixture(path)
    config = {**adapter_config_fixture(), "layers_to_transform": selector}
    with pytest.raises(AdapterContractError):
        verify_adapter_contract(inventory_fixture(), config, path)


@pytest.mark.parametrize("name,kind", [("../outside.py", "file"), ("/absolute.py", "file"),
    ("C:/outside.py", "file"), ("a\\b.py", "file"), ("serving/link.py", "link"), ("same.py", "duplicate")])
def test_lineage_rejects_path_aliases_links_and_duplicate_members(tmp_path, name, kind):
    path = tmp_path / "generated.tar"
    with tarfile.open(path, "w") as bundle:
        item = tarfile.TarInfo(name)
        if kind == "link":
            item.type, item.linkname = tarfile.SYMTYPE, "../outside.py"
            bundle.addfile(item)
        else:
            item.size = 1
            bundle.addfile(item, BytesIO(b"x"))
            if kind == "duplicate":
                bundle.addfile(item, BytesIO(b"x"))
    with pytest.raises(SourceLineageError, match="member_invalid"):
        archive_inventory(path, file_sha256(path))


@pytest.mark.parametrize("name", ["evaluation/qwen38_pipeline.py", "serving/qwen38_grammar.py",
    "src/forte/decision/structured.py", "training/qwen38_silver.py"])
def test_new_validator_lineage_does_not_authorize_quality_prompt_or_training_changes(tmp_path, name):
    training, runtime = tmp_path / "original.tar", tmp_path / "new-serving.tar"
    archive_fixture(training, {name: b"original"})
    archive_fixture(runtime, {name: b"changed", "serving/qwen38_adapter_contract.py": b"reviewed helper"})
    with pytest.raises(SourceLineageError, match="source_difference"):
        verify_source_lineage(training_archive=training, training_sha256=file_sha256(training),
            recorded_training_sha256=file_sha256(training), runtime_archive=runtime, runtime_sha256=file_sha256(runtime))
