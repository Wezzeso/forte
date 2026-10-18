# Additional public materials for visible-fact accuracy

Checked 17 October 2026. The useful additions are narrowly selected action primitives, procedural-error examples, object grounding and hard normal negatives. They can support the accuracy objective; no public benchmark establishes 100% correctness on these bank videos. This research preserves Qwen3.8 as the requested main-model target and makes no candidate selection.

The structured catalog is `configs/serving/additional-public-materials-20261017.catalog.json`. It records eight candidates, exact public source revisions, metadata hashes, licenses, limitations and download URLs. Model weight hashes are expected publisher identities. No dataset video or model weight was acquired or executed in this research; public text metadata was read and selected large-download endpoints checked with HTTP HEAD.

## Most useful data candidates

| Candidate | Published license and useful supervision | Practical acquisition and limitation |
| --- | --- | --- |
| [CAP, Visym](https://github.com/visym/cap/tree/8a04fd865667cd80c661e552e260ea0de8ab41d5) | CC BY 4.0. Actor-centered action intervals and tracks; generic handover, countertop pickup/putdown, phone pocketing and hands in pockets. | Start with the [12-KB class list](https://raw.githubusercontent.com/visym/cap/8a04fd865667cd80c661e552e260ea0de8ab41d5/classification/cap_labels.csv). The publisher's detection-validation bundle is 901,869,561 bytes by HEAD; its classification archive is 308,736,485,473 bytes. Preserve source partitions and inspect bundle structure before obtaining media. Generic staged actions provide auxiliary examples, not cash-custody truth. |
| [HoloAssist](https://holoassist.github.io/) | CDLA-Permissive-2.0. Fine-grained action start/end times and correct/error/correction attributes. | [Annotation JSON](https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json) is 117,011,015 bytes by HEAD; the compressed-video archive is 155,285,145,600 bytes. Select small relevant sessions after annotation inspection. Device/assembly actions differ from bank work; conversation-derived facts require independent visual support. |
| [IndustReal human-video dataset](https://github.com/TimSchoonbeek/IndustReal/tree/ad86a35d4b5125739e24af677e5d7b55c74af945) | Apache-2.0, also confirmed by the [registered v2 dataset metadata](https://api.datacite.org/dois/10.4121/b008dd74-020d-4ea4-a8ba-7bb60769d224.v2). Procedure state, omissions and execution errors. | The [publisher's v2 record](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224/2) returned 403 through the research tool. Small procedure definitions remain public in the repository. Assembly errors are useful auxiliary reasoning examples; preserve published splits, especially errors withheld from training. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | CC BY 4.0 annotations; images have separately recorded licenses. Verified box classes include Coin, Human hand, Person, Countertop and Handbag. | Filter published annotations and obtain only images with independently verified compatible license/creator metadata, as the [publisher requires](https://storage.googleapis.com/openimages/web/factsfigures_v7.html). Still images support object discrimination; they provide no temporal omission label. |
| [Microsoft BankNote-Net](https://github.com/microsoft/banknote-net/tree/57a6bf549879d6c6f700b6ba26724c957f770c5d) | CDLA-Permissive-2.0 in the dataset/model license section; released banknote embeddings and encoder. | Low priority: the main release contains feature vectors, its listed currencies omit KZT, and it has no cash-action supervision. The catalog binds the publisher's encoder digest. Preserve the restriction against reconstructing high-resolution banknotes from the vectors. |

CC BY 4.0 permits commercial sharing and adaptation with attribution, a license link and modification notices. CDLA-Permissive-2.0 permits use and sharing subject to retaining its agreement when sharing data. Apply each license to its stated component scope and preserve original source/attribution records. [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/), [CDLA-Permissive-2.0 terms](https://cdla.dev/permissive-2-0/).

No ready-to-use, commercially compatible cash-counting/pocketing/abandonment **video** corpus with bank-procedure truth was verified in this pass. Larger unrelated corpora should be secondary to correctly supervised eligible project TRAIN material.

## Optional pretrained instruments

These are candidates for separately bounded local experiments. Their role is to add evidence or cross-check TRAIN annotations; none replaces the requested main model or directly emits a legal decision.

| Candidate | Official published license | Useful role and limit |
| --- | --- | --- |
| [IDEA-Research Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71) | Apache-2.0 | Text-conditioned object proposals. Tiny/occluded notes may be missed; detection scores require calibration and an empty result cannot establish absence. |
| [Meta SAM 2.1 small](https://huggingface.co/facebook/sam2.1-hiera-small/tree/ee5bba1d82bb8749febdf90f45e84b687142ba03) | Apache-2.0 | Track initialized actor/object regions across frames. It needs reliable initialization and can lose identity through occlusion. Keep masks as derived metadata and preserve original evidence JPEGs. |
| [Microsoft Florence-2 large](https://huggingface.co/microsoft/Florence-2-large/tree/21a599d414c4d928c9032694c424fb94458e3594) | MIT | Optional region/phrase grounding teacher cross-check. The current card describes continued pretraining with a quality caveat. Pin and inspect its code path; single-frame captions do not prove temporal custody or omission. |

The catalog records safetensor sizes and expected hashes for all three. The project has no measured memory/latency/quality result for this combination. Any extra model must enter through the serialized compute owner, with image/source pins, local-only processing, complete call accounting and the unchanged rental stop.

## Task-specific prompt and supervision improvements

Use the existing strict observation schema and the nineteen-axis fact IDs in `training/temporal/fact-axis.v1.json`. The following instructions describe observation requirements rather than a replacement response schema:

> Describe visible physical facts using supplied original frame IDs and source times. For each supported fact, identify the evidence and its point or interval scope. Use `unclear` when object identity, actor continuity, procedure context or required coverage is missing. Treat an empty/occluded frame as local visibility evidence. Establish complete opportunity before asserting an omitted action. Keep generated legal conclusions out of the observation output.

Prioritize these project-specific checks:

1. Establish `CASH_VISIBLE` from identifiable visual evidence, with ambiguity between banknotes, receipts, cards and personal objects recorded. Then track the same object and actor before asserting `CASH_HANDOVER_VISIBLE`, `PREVIOUSLY_ACCEPTED_CASH_TRACKED` or `CASH_PLACED_IN_CLOTHING_POCKET`.
2. For `CASH_COUNT_VISIBLE`, distinguish visible note separation/counting from sorting, stacking, stamping, typing or moving a bundle. Include a visible counting-machine path where supported. Sparse samples can miss a fast count; refine the relevant interval using original frames within the declared budget.
3. For absence of a required count, establish `ALL_COUNTING_PATHS_VISIBLE`, `PROCEDURE_CONTEXT_COMPLETE` and `WORK_AREA_CONTINUOUSLY_VISIBLE` across the relevant opportunity. Mask unsupported fields. A local missing action must retain its local scope.
4. For unattended cash, combine supported cash-on-surface continuity with actual handler departure and coverage. Record temporary occlusion and reaching away as uncertainty where continuity is incomplete.
5. Preserve receipt/issue context, loose-versus-packaged handling and catalogue exceptions in the deterministic decision layer. Bind source PTS and source frame hashes; model decimal precision cannot reduce annotation uncertainty.

Build hard normal examples from eligible TRAIN material with the same sampling policy: phone or empty hand entering a pocket; receipts/personal objects moved near clothing; temporary hand occlusion; reaching to a cabinet; cash handled through a visible counting machine; already packaged cash where the frozen rule permits it; customer/staff motion while the responsible handler remains. These cases should supervise observed facts and uncertainty. The surrounding video-level verdict must not fill all unknown action fields with `absent`.

Public auxiliary examples need a separate dataset identity and task/schema. A CAP generic transfer can teach object/actor continuity; an IndustReal omission can teach procedure-state reasoning. Neither may be relabeled as a Kazakhstan cash violation, and unrelated examples cannot silently satisfy the E1 reviewed-full-fact gate. Machine-produced project labels retain SILVER provenance, exact support joins and unknown/conflict masks.

Compare any prompt, data or auxiliary-model change on one frozen development recipe. Report all attempted files and failures, precision/recall for supported labels, and error categories for cash confusion, missed counting, object continuity, false departure, context and incomplete coverage. Keep tuning away from final-check sources. A measured improvement can justify selection; extra instruments or extra training rows alone cannot.

## Materials deferred in this scope

- [EPIC-KITCHENS-100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/license.txt) publish CC BY-NC 4.0 terms, which do not meet this commercial-compatible acquisition scope.
- [Ego4D](https://ego4d-data.org/docs/start-here/) requires an executed agreement and credential approval. This access requirement is the blocker here; this document makes no blanket noncommercial-license claim about Ego4D.
- [HaGRID](https://github.com/hukenovs/hagrid) states a variant of CC BY-SA. Its exact custom terms were not verified in this pass and its gesture domain is lower priority.
- Public cash videos found through search remain deferred until the publisher's actual reuse license and provenance are verified. Public visibility alone does not establish training rights.

The catalog is a reviewable acquisition shortlist and training guide. Download, training, accuracy gains and production integration remain separate measured actions.
