# Private remote audit download and local review

The network transfer remains with the authorized deployment operator. This package does not fetch media, contact a model, display private content to a coding assistant, or perform human review.

Download the team's private `analysis.private.json` and its original JPEGs through the authorized SSH/file-transfer route. Preserve the directory structure below one explicit remote frame root. For example, if the remote root is `/var/lib/forte/work`, a reference to `/var/lib/forte/work/OPAQUE_JOB/attempt-1/frames/FRAME.jpg` must exist locally at `LOCAL_DOWNLOAD_ROOT/OPAQUE_JOB/attempt-1/frames/FRAME.jpg`. The job and frame names here are fictional placeholders.

Keep the download directory stable after import, because proposals retain immutable local frame references. Keep all artifacts inside `annotations/private/proposals/` or another explicitly private, excluded directory. Original videos remain at their existing read-only input location; no rename or copy is needed to match a remote upload name.

Prepare two private mapping files without printing their contents:

```json
{
  "64_CHARACTER_ORIGINAL_SOURCE_SHA256": "C:\\PRIVATE\\INPUTS\\ORIGINAL.mov"
}
```

```json
{
  "remote_frame_root": "/var/lib/forte/work",
  "local_frame_root": "C:\\PRIVATE\\DOWNLOADS"
}
```

The first object maps original source content hashes to local inputs, independent of remote filenames. The example hash is a placeholder; actual keys must be source SHA-256 values. The second object identifies the exact preserved remote/local frame-root relationship. Choose the smallest applicable roots, not the disk root. Paths in both files are private metadata and must not be pasted into a cloud chat.

```powershell
python -m annotations.cli import-proposal --input C:\PRIVATE\DOWNLOADS\analysis.private.json --proposal-root C:\PRIVATE\DOWNLOADS --source-root C:\PRIVATE\INPUTS --frame-root C:\PRIVATE\DOWNLOADS --source-map C:\PRIVATE\source-map.json --path-map C:\PRIVATE\path-map.json
```

The existing allowed-root checks still apply after rebasing. Both mapping flags are required together; supplying either alone fails. Source/frame roots are repeatable when more than one specifically approved directory is needed.

Programmatic interface:

```python
from annotations.rebase import rebase_proposal

rebased = rebase_proposal(
    audit,                       # locally loaded private JSON object
    local_source_by_hash,         # {original_sha256: absolute_local_input_path}
    remote_frame_root,            # absolute POSIX root used by team deployment
    local_frame_root,             # absolute local root containing unchanged JPEGs
)
```

The function returns a deep copy. It reads files to validate them and writes nothing. It replaces `asset.owned_path`, registered frame paths and matching embedded request frame paths. It preserves source content hashes, frame IDs, PTS, job bindings, original JPEG hashes, observed facts, candidates and model provenance. A separate `local_rebase` record describes the path mapping and verification; it does not claim review or inference completion.

Every original frame must carry its declared `jpeg_sha256`; the helper refuses to invent missing transfer integrity metadata. It verifies local source SHA-256, each mapped JPEG SHA-256, JPEG format/dimensions and source timeline/job ownership. Remote paths must be absolute POSIX paths below the explicit root. Traversal components, Windows drive/device names, separator confusion, prefix siblings, symlink escapes, duplicate frame IDs and multiple frame IDs aliasing one local file are rejected. Embedded request references must match the registered remote frame before paths change.

`import-proposal` performs rebasing in memory and then runs all usual proposal validation. The downloaded original audit stays byte-for-byte unchanged, and its original SHA-256 remains the proposal's audit identity. Raw predictions remain private `local_model_proposals`, state `in_progress`, origin `local_model`. Importing does not create a human annotation, negative label, split lock or training authorization, and it cannot overwrite saved human work.

The workbench process does not need to stop for CLI imports. If its server code predates the proposal UI, restart it only when the human reviewer is idle and has saved their edits. This helper never restarts the server.

Synthetic checks:

```powershell
python -m pytest annotations/test_workflow.py annotations/test_proposals.py annotations/test_rebase.py -q
```

On this Windows host: 52 passed, 8 path-traversal subtests passed, and one symlink test was explicitly skipped because the local account lacks symlink-creation permission. Root-containment checks remain active; that skip is a test-environment limitation. These checks measure neither real model quality nor actual human review progress.
