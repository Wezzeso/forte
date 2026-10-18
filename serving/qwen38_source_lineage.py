"""Verify distinct immutable training and serving source archives on the CPU."""

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


class SourceLineageError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise SourceLineageError(code)


# These are the only runtime implementation deltas reviewed for this release.
# Tests may change freely because they are not imported by the serving path;
# training and common inference sources require exact equality, except the
# single source- and AST-proven candidate timeout insertion audited below.
ALLOWED_RUNTIME_DELTAS = frozenset({"deploy/qwen38_bind_runtime.py", "serving/qwen38_identity.py",
    "serving/qwen38_adapter_contract.py", "serving/qwen38_source_lineage.py",
    "evaluation/qwen38_quality_equivalence.py"})


def verify_candidate_timeout_forwarding(old_bytes, new_bytes):
    """Prove one audited candidate-only timeout keyword, preserving all else.

    Both whole-module AST identity and removal of exactly one isolated source
    line are required. A default timeout, different expression, changed branch,
    other keyword, statement, comment or whitespace change is not authorized.
    """
    try:
        original = ast.parse(old_bytes)
        updated = ast.parse(new_bytes)
    except (SyntaxError, ValueError, TypeError, UnicodeError):
        raise SourceLineageError("lineage_timeout_source_invalid") from None

    expected_test = ast.parse('getattr(settings, "model_candidate_launch_config_path", None) is not None', mode="eval").body
    def call(tree):
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_provider_configuration"]
        require(len(functions) == 1, "lineage_timeout_provider_function_mismatch")
        branches = [node for node in functions[0].body if isinstance(node, ast.If)
                    and ast.dump(node.test) == ast.dump(expected_test)]
        require(len(branches) == 1, "lineage_timeout_candidate_branch_mismatch")
        calls = [node for node in ast.walk(branches[0]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "values" and node.func.attr == "update"]
        require(len(calls) == 1 and not calls[0].args, "lineage_timeout_values_update_mismatch")
        return calls[0]

    old_call, new_call = call(original), call(updated)
    require(not any(keyword.arg == "request_timeout_sec" for keyword in old_call.keywords),
            "lineage_timeout_original_already_configured")
    keywords = [keyword for keyword in new_call.keywords if keyword.arg == "request_timeout_sec"]
    require(len(keywords) == 1, "lineage_timeout_added_keyword_required")
    keyword = keywords[0]
    expected_value = ast.parse('config["limits"]["request_timeout_seconds"]', mode="eval").body
    require(ast.dump(keyword.value) == ast.dump(expected_value), "lineage_timeout_value_mismatch")
    new_call.keywords.remove(keyword)
    require(ast.dump(original) == ast.dump(updated), "lineage_timeout_other_ast_changed")
    lines = new_bytes.splitlines(keepends=True)
    require(keyword.lineno == keyword.end_lineno and 0 < keyword.lineno <= len(lines),
            "lineage_timeout_added_source_line_required")
    line = lines[keyword.lineno - 1]
    # AST columns are UTF-8 byte offsets. Only indent and the trailing separator
    # may share the added line, so removing it cannot hide another statement.
    require(not line[:keyword.col_offset].strip()
            and line[keyword.end_col_offset:].strip() == b","
            and b"".join(lines[:keyword.lineno-1] + lines[keyword.lineno:]) == old_bytes,
            "lineage_timeout_other_source_bytes_changed")
    return {"schema_version": "forte.qwen38.candidate-http-timeout-delta.v1", "verified": True,
        "old_source_sha256": hashlib.sha256(old_bytes).hexdigest(),
        "new_source_sha256": hashlib.sha256(new_bytes).hexdigest(),
        "unchanged_ast_sha256": hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        "added_keyword": "request_timeout_sec", "added_value": 'config["limits"]["request_timeout_seconds"]',
        "candidate_branch_only": True, "other_ast_identical": True,
        "other_source_bytes_identical": True, "baseline_default_unchanged": True,
        "comparison_condition_difference": "candidate HTTP timeout now inherits the configured request budget"}


def verify_candidate_inference_mode(old_bytes, new_bytes):
    """Prove only PEFT's explicit inference-mode adapter activation changed.

    The exact keyword bytes are removed from the unique CandidateRuntime.load
    activation call; both the entire old source and its AST must be recovered.
    The existing loaded-tensor and no-trainable-parameters guards stay intact.
    """
    try:
        original, updated = ast.parse(old_bytes), ast.parse(new_bytes)
    except (SyntaxError, ValueError, TypeError, UnicodeError):
        raise SourceLineageError("lineage_inference_source_invalid") from None

    def call(tree):
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "CandidateRuntime"]
        require(len(classes) == 1, "lineage_inference_runtime_class_mismatch")
        methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
                   and node.name == "load"]
        require(len(methods) == 1, "lineage_inference_load_method_mismatch")
        calls = [node for node in ast.walk(methods[0]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "set_adapter"
                 and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "model"
                 and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "self"]
        require(len(calls) == 1 and len(calls[0].args) == 1
                and isinstance(calls[0].args[0], ast.Constant)
                and calls[0].args[0].value == "candidate",
                "lineage_inference_activation_call_mismatch")
        return calls[0]

    old_call, new_call = call(original), call(updated)
    require(not old_call.keywords, "lineage_inference_original_keywords_present")
    require(len(new_call.keywords) == 1 and new_call.keywords[0].arg == "inference_mode",
            "lineage_inference_keyword_required")
    keyword = new_call.keywords[0]
    require(isinstance(keyword.value, ast.Constant) and keyword.value.value is True,
            "lineage_inference_value_mismatch")
    new_call.keywords.remove(keyword)
    require(ast.dump(original) == ast.dump(updated), "lineage_inference_other_ast_changed")
    lines = new_bytes.splitlines(keepends=True)
    require(keyword.lineno == keyword.end_lineno and 0 < keyword.lineno <= len(lines),
            "lineage_inference_inline_keyword_required")
    offset = sum(len(line) for line in lines[:keyword.lineno-1]) + keyword.col_offset
    end = sum(len(line) for line in lines[:keyword.end_lineno-1]) + keyword.end_col_offset
    require(new_bytes[offset:end] == b"inference_mode=True" and new_bytes[offset-2:offset] == b", "
            and new_bytes[:offset-2] + new_bytes[end:] == old_bytes,
            "lineage_inference_other_source_bytes_changed")
    return {"schema_version": "forte.qwen38.candidate-inference-mode-delta.v1", "verified": True,
        "old_source_sha256": hashlib.sha256(old_bytes).hexdigest(),
        "new_source_sha256": hashlib.sha256(new_bytes).hexdigest(),
        "unchanged_ast_sha256": hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        "added_keyword": "inference_mode", "added_value": "True",
        "candidate_runtime_load_only": True, "other_ast_identical": True,
        "other_source_bytes_identical": True, "serving_only": True,
        "comparison_condition_difference": "PEFT candidate activation explicitly preserves inference mode"}


def archive_source_member(path, name, expected_digest):
    with tarfile.open(path, "r:*") as archive:
        stream = archive.extractfile(name)
        require(stream is not None, "lineage_audited_source_member_missing")
        raw = stream.read(16*1024*1024 + 1)
    require(len(raw) <= 16*1024*1024 and hashlib.sha256(raw).hexdigest() == expected_digest,
            "lineage_audited_source_member_mismatch")
    return raw


def archive_inventory(path, expected_digest):
    require(isinstance(expected_digest, str) and re.fullmatch(r"[a-f0-9]{64}", expected_digest),
            "lineage_archive_pin_required")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    require(digest.hexdigest() == expected_digest, "lineage_archive_bytes_mismatch")
    result, total = {}, 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            name = member.name
            relative = PurePosixPath(name)
            require(isinstance(name, str) and name and "\\" not in name and ":" not in name
                    and not relative.is_absolute() and relative.as_posix() == name
                    and all(part not in {".", ".."} for part in relative.parts)
                    and member.isfile() and not member.issym() and not member.islnk()
                    and name not in result and 0 <= member.size <= 16*1024*1024,
                    "lineage_archive_member_invalid")
            total += member.size
            require(len(result) < 5000 and total <= 128*1024*1024, "lineage_archive_budget_exceeded")
            stream = archive.extractfile(member)
            require(stream is not None, "lineage_archive_member_unreadable")
            result[name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def verify_source_lineage(*, runtime_archive, runtime_sha256, training_archive,
                          training_sha256, recorded_training_sha256):
    require(training_sha256 == recorded_training_sha256, "lineage_training_run_source_mismatch")
    runtime = archive_inventory(runtime_archive, runtime_sha256)
    training = archive_inventory(training_archive, training_sha256)
    changed = {name for name in set(runtime) | set(training) if runtime.get(name) != training.get(name)}
    timeout = None
    inference_mode = None
    pipeline = "src/forte/pipeline.py"
    if pipeline in changed:
        require(pipeline in training and pipeline in runtime, "lineage_timeout_pipeline_member_missing")
        timeout = verify_candidate_timeout_forwarding(
            archive_source_member(training_archive, pipeline, training[pipeline]),
            archive_source_member(runtime_archive, pipeline, runtime[pipeline]))
    service = "serving/qwen38_service.py"
    if service in changed:
        require(service in training and service in runtime, "lineage_inference_service_member_missing")
        inference_mode = verify_candidate_inference_mode(
            archive_source_member(training_archive, service, training[service]),
            archive_source_member(runtime_archive, service, runtime[service]))
    forbidden = {name for name in changed if name not in ALLOWED_RUNTIME_DELTAS
        and not (name.startswith("serving/tests/") and name.endswith(".py"))
        and not (name == pipeline and timeout is not None)
        and not (name == service and inference_mode is not None)}
    require(not forbidden, "lineage_unaudited_source_difference")
    protected = {name for name in training if name.startswith(("training/", "src/forte/", "rules/", "annotations/"))}
    require(any(name.startswith("training/") and name.endswith(".py") for name in protected)
            and "training/architectures.py" in protected and "training/data.py" in protected
            and "src/forte/pipeline.py" in protected and "rules/catalog.json" in protected
            and "annotations/taxonomy.v1.json" in protected, "lineage_protected_source_missing")
    require(all(runtime.get(name) == training[name] for name in protected
                if not (name == pipeline and timeout is not None)), "lineage_training_or_common_source_changed")
    return {"schema_version": "forte.qwen38.training-serving-lineage.v1", "verified": True,
        "training_source_archive_sha256": training_sha256, "runtime_source_archive_sha256": runtime_sha256,
        "same_archive": runtime_sha256 == training_sha256, "protected_files_verified": len(protected),
        "protected_files_byte_identical": len(protected) - int(timeout is not None),
        "protected_files_with_audited_timeout_delta": int(timeout is not None),
        "candidate_http_timeout_delta": timeout,
        "serving_files_with_audited_inference_mode_delta": int(inference_mode is not None),
        "candidate_inference_mode_delta": inference_mode,
        "protected_file_inventory_sha256": hashlib.sha256(json.dumps({name: training[name] for name in sorted(protected)},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "audited_runtime_deltas": sorted(changed), "run_bytes_changed": False, "adapter_bytes_changed": False}
