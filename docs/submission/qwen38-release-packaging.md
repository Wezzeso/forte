# Qwen3.8 release packaging

The deployment and isolated ML packagers now capture source with two matching
reads and verify unchanged tree membership. Archives use normalized tar metadata
and a gzip timestamp of zero, so identical captured bytes produce identical
archive hashes. Existing archive or sidecar paths are rejected; select a new
versioned output path for every release or backup.

Each `<archive>.manifest.json` binds every member's SHA-256 as well as the whole
archive's SHA-256, byte size and file count. `deploy.package_policy.verify_package`
independently reads and checks those bytes without extracting. A source archive
is separate evidence from a built image, a loaded model, an adapter, training
completion, quality evaluation or deployment. The manifests record no invented
Git identity; this workspace has no verified Git commit.

Public source selections reject private directories, labels, splits, selections,
machine observations, raw outputs, backups, databases, keys, credentials, footage,
weights and links or junctions. Public branding images may enter from dedicated
asset directories. Existing Python evidence assembly code remains included.
The public model snapshot and upstream research stash remain outside runtime
contexts; the model/processor/tokenizer snapshot is mounted separately and
verified against its public pinned manifest. Filename exclusions do not
de-identify arbitrary new content: source must continue to live in public roots,
and private inputs and generated artifacts must continue to live in private
roots.

Actual local BuildKit scratch exports checked the root deployment, existing
training, feature runtime and new candidate training contexts. Every exported
regular file passed the shared source policy and none contained links. The
training context now includes the taxonomy and two temporal JSON definitions
which its Dockerfile requires. Dockerfile-specific allowlists re-exclude the
contents of each reopened parent directory before allowing individual files;
without that step an apparently narrow `!deploy/` or `!training/` exception can
admit unrelated descendants. The feature and candidate contexts each exported
exactly four intended public files.

Focused validation: 77 packaging tests passed, with one host limitation skip for
creating Windows symlinks; the dataset audit's 15 synthetic checks also passed.
A separate synthetic Windows ancestor junction was created and rejected by the
same source policy. The dated JSON audit
records the source hashes and actual context exports. It predates the final
integrated freeze and does not itself select or deploy a candidate.

To create a release after integration is frozen:

```powershell
.\.venv\Scripts\python.exe deploy/package_code.py --output tmp/<unique-release-source>.tar.gz
.\.venv\Scripts\python.exe deploy/package_ml_code.py --output tmp/<unique-release-ml-source>.tar.gz
```

Retain the archive and its sidecar together. Inspect `files` to confirm new
candidate source and public examples are present. Bind actual image/model/
adapter/runtime identities separately when those artifacts exist.
