# Terminal artifact handoff to the local reviewer

`fetch-proposal` downloads only a completed attempt's private audit and the sampled JPEGs referenced by that audit. It does not download the source video or a public jury result. It does not inspect visual content, call an external AI API, label a clip, authorize training, or restart either review server.

Run it only after the benchmark owner has explicitly handed off a terminal job and verified that its private artifact exists. A processing job, a guessed path, a provider readiness response or a planned benchmark is insufficient.

The owner saves a private JSON handoff inside an excluded local directory:

```json
{
  "schema_version": "forte.proposal-artifact-handoff.v1",
  "terminal_verified": true,
  "artifact_available": true,
  "terminal_status": "failed",
  "job_id": "OPAQUE_JOB_ID",
  "remote_work_root": "/var/lib/forte/work",
  "remote_frame_root": "/var/lib/forte/work",
  "remote_audit_path": "/var/lib/forte/work/OPAQUE_JOB_ID/attempt-1/analysis.private.json",
  "ssh_target": "OBSERVED_EXISTING_SSH_ALIAS",
  "ssh_wsl_distribution": "Ubuntu",
  "remote_container": "OBSERVED_WORKER_CONTAINER",
  "remote_python": "python3"
}
```

These values are placeholders. Use only observed connection and artifact references supplied by the deployment/benchmark owner. `terminal_status` is `done` or `failed`; both can provide useful **unreviewed** observations. Optional fields: `audit_sha256`, `ssh_port`, existing `ssh_identity_path`, `ssh_config_path`, `ssh_known_hosts_path`, and `remote_docker_sudo: true` only where the established authorized route already requires `sudo -n docker`. Omit `remote_container` when artifact files are actually accessible directly on the SSH host. Omit `ssh_wsl_distribution` when using the native local SSH client. SSH file references belong to the selected client environment: absolute POSIX references for WSL, absolute local file references for native SSH. Never put private key contents, passwords or tokens in the handoff.

```powershell
python -m annotations.cli fetch-proposal --handoff annotations/private/proposal-handoff.json
```

There is no need to guess a local source filename or manually prepare source/path mappings: the helper binds the original content SHA-256 to existing seeded local inputs. Optional repeatable `--source-root` flags can restrict those input roots further. The default roots are the already seeded input parent directories, and the exact local source is still checked by SHA-256.

The SSH connection uses batch mode, existing authentication and strict host-key verification. It never auto-accepts a new host, prompts for credentials or copies SSH secrets between Windows and WSL. A fixed read-only Python exporter runs on the authorized team host or observed worker container. Audit paths travel as JSON on stdin rather than shell text. Binary tar stdout and SSH stderr are captured privately, not returned as agent output.

The exporter verifies job ownership and original frame hashes before streaming only the original audit, its referenced JPEGs and a transfer manifest. Local extraction rejects traversal, symbolic/hard links, duplicate members, unexpected/unreferenced files and oversized payloads. Default bounds are 32 MiB for an audit, 16 MiB per JPEG, 8,192 frames and 1 GiB total payload. Local hashes, job/window bindings, source identity and rebasing are checked again before proposal import.

Verified artifacts are kept at `annotations/private/proposals/<opaque-job>-<audit-hash-prefix>/`, with the original audit bytes/hash intact. Their location remains stable for the workbench. Transport diagnostics are private `*.transport.stderr.private.txt` files in that excluded tree. Do not paste their contents into a cloud chat; report only aggregate counters or static error codes.

Successful terminal output contains only `transfer_status`, downloaded frame/payload counts, `proposals_imported`, and `already_imported`. The review record remains `in_progress` / `local_model`, with no verdict, human review mark, negative label or training permission. If a human has already saved review work, the helper retains the verified private download and returns `artifact_verified_private_download_retained_import_requires_local_action` instead of replacing their work.

The updated workbench can be opened separately on port 8766 while an older port-8765 tab preserves its unsaved draft. Both use the same SQLite payload schema and optimistic revision checks. Importing through the CLI never closes or reloads those tabs. If two tabs edit the same saved record, stale revisions are rejected.

For the W2 teacher, `remote_audit_path` may end in `teacher.private.json` under
the same exact job-root constraints. `terminal_status:done` means the bounded
teacher process finished; its audit may honestly remain `phase:partial` with
incomplete video coverage. Importing it preserves that partial state and never
creates a clean verdict. The observed exporter environment must resolve the
actual audit/frame paths, including any read-only mounted worker volume.

Synthetic checks:

```powershell
python -m pytest annotations/test_workflow.py annotations/test_proposals.py annotations/test_rebase.py annotations/test_fetch.py -q
```

62 tests and 11 path/archive subtests pass on this host. One symlink-creation test is explicitly skipped because the Windows account lacks permission. Transport tests use generated JPEGs and a synthetic transport, never the team SSH server or supplied clips; they verify that the network hook cannot run before a terminal handoff. The fixed exporter's checksum/archive logic is also exercised with a synthetic filesystem adapter. Actual transfer success must be reported separately after a real terminal artifact handoff.
