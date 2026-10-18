# Source-bound fact transitions

A generated eight-second decision proves a representational recall defect.
The handler is explicitly present in source frames at seconds 0 and 1, then
explicitly absent in the candidate's source frames at seconds 5, 6 and 7.
All other required facts for `CASH_UNATTENDED` are supplied as supported
synthetic observations.

Before the correction, the direct candidate was rejected because its absence
state differed from the batch presence state, despite disjoint support times.
The ordinary structured format could not express the transition at all: it
accepted only candidate references to the single batch fact. All three
regressions failed before the change: direct event assembly, raw structured
output and compact-frame output.

The optional raw candidate `local_facts` list now carries explicit scoped
observations. Existing raw candidates remain valid. Prompt version
`forte.visual-observation.v1.1.0` documents the additive contract. Structured
validation, compact-frame restoration and event assembly share the same gate:

- Both old and local states must be explicit opposite `present`/`absent` values.
- Local source supports must be nonempty, known, inside the candidate and cited
  by its `supporting_frame_ids`.
- The complete previous support span must lie strictly before or after the
  candidate. Shared boundary timestamps, overlapping evidence and a prior span
  straddling the candidate are rejected.
- The original batch fact remains unchanged. Unknown context is not promoted
  to a decisive observation, and no absence is inferred.

The legal catalogue, nineteen fact axes, required context, required action
support and public evidence anchors are unchanged. Contradictory or unsupported
observations still fail explicitly. A transition cannot satisfy a missing
departure action or an unclear visibility gate.

After correction, the generated transition reaches a grounded positive event
through both canonical and compact source IDs. **247 focused tests and 51
subtests pass**, including the frozen rule gates and earlier refinement-support
regressions. The generated proof and exact changed-source hashes are in
`qa/integration/temporal-fact-scope-synthetic.json`.

This verifies representability and validation behavior. It does not demonstrate
that a vision model will emit correct local observations or establish any
private-video accuracy improvement. No private input, GPU run, model call or
remote mutation was used.
