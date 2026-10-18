# Public CCTV and counter-style training materials

Checked 17 October 2026. No exact bank cash-procedure video dataset with a verified unrestricted commercial-compatible license was found in this pass. The closest verified source is **MEVA's official TRAIN footage of payment transactions and object transfers**, recorded by fixed surveillance cameras. Its payment label includes cash, card and phone transactions; cash handling in the acquired subset remains unconfirmed until a separate original-frame audit.

Four original MEVA clips were acquired on the existing Brev host at the root's explicit authorization. They total **603,857,450 bytes**, plus **31,669,459 bytes** of sixteen matching annotation/definition/license files. This acquisition used public publisher URLs only. No GPU work, frame viewing, teacher/model calls or training was performed.

Remote directory: `/data/forte-public-data/meva-counter-train-v1`. Local and remote `subset.manifest.json` both have SHA-256 `30f157ce24c586db1e91979351ae4845a5fcbefccb0ac59d07d9cfd41a16aac4`. The remote original acquisition receipt has SHA-256 `dbbd629d9cc1a7e2386a00a0224ed798bdf7f62485f0461566e7e03369740b8b`.

## Actual source and acquired subset

[MEVA's publisher page](https://mevadata.org/) and [full dataset license](https://mevadata.org/resources/MEVA-data-license.txt) publish CC BY 4.0 terms. The freshly served license bytes matched the frozen SHA-256 `bdeedfb765049c87f92a2450369ad70882fca3371190b2a6b7e560e103c922e8` before media acquisition. Attribution to Kitware, IARPA and the dataset authors is saved in `ATTRIBUTION.txt` alongside the originals.

The [official annotation repository](https://gitlab.kitware.com/meva/meva-data-repo) is frozen at `421841a75577b697c314e952e585aecbb1b99e17`. Only its `kitware-meva-training` partition was used. The publisher describes this partition as training annotations with a reduced audit stage. Its published status table reports 107 purchase instances, 630 object transfers, one package abandonment and three theft instances across the entire partition. Those are publisher annotation counts, not newly reviewed source truth.

The bounded metadata search read all 44 activity files from TRAIN's `2018-03-07/11` directory, totaling 450,920 bytes. Three clips contain sixteen purchase annotations. The fourth adds another camera's generic transfer examples. Each original was downloaded unchanged from the publisher's anonymous S3 bucket and hashed locally on the host:

| Original clip | Bytes | Published instances | SHA-256 |
| --- | ---: | --- | --- |
| [G421, 10:55–11:00](https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/11/2018-03-07.10-55-01.11-00-01.school.G421.r13.avi) | 162,927,962 | 8 purchases; 15 transfers | `836545d81c6c1d38a2d304919b3405013e4da61d3d376a5efa46cbe6499cf722` |
| [G421, 11:05–11:10](https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/11/2018-03-07.11-05-01.11-10-01.school.G421.r13.avi) | 148,976,200 | 3 purchases; 8 transfers | `d7336269cadd3e6dfdffe171c48d78adfb061832a4473b3789e6f354e84e6434` |
| [G331, 11:10–11:15](https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/11/2018-03-07.11-10-00.11-15-00.bus.G331.r13.avi) | 131,269,982 | 5 purchases | `9a94dd1ab762d0229ccb01e828a7c476b971b4632d3b58c28ec7d335aae7d970` |
| [G424, 11:05–11:10](https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/11/2018-03-07.11-05-05.11-10-05.school.G424.r13.avi) | 160,683,306 | 2 transfers | `337177f25de6c32d3bbc4a9d4d181f395603ed0022f50521ee57699e27ed1ecc` |

The first three clips total 443,174,144 bytes; all four total about 576 MiB. Three camera IDs provide candidate viewpoint diversity, which still needs visual suitability checking. Instances can overlap and are not independent scenarios. All four clips share one split group: `MEVA-KF1-20180307T1055-1115-all-selected-cameras`. Any future overlapping or adjacent camera segments must retain that role. These clips are public auxiliary TRAIN; they are never project validation, final-check or bank-procedure gold.

Matching activities, geometry and track-type annotations were downloaded from the same pinned TRAIN revision. Definitions, class names, TRAIN README and license were also preserved. S3 multipart ETags were recorded as object metadata and were not treated as SHA-256. The manifest contains measured full-byte SHA-256 for every acquired artifact.

## What the annotation definitions actually mean

The publisher's complete relevant definition pages were inspected. The local definition PDF is preserved at `configs/datasets/cctv-cash-public-20261017/MEVA-Annotation-Definitions.pdf` with SHA-256 `6a32c2d27f5aab6418c2a0d35cc5880425af48b729e99387fd074987d5d1792b`.

- `person_purchases` includes cash, credit card and phone payment. Its interval starts one second before payment presentation and ends one second after receipt of the goods or returned payment, whichever occurs later. Its label does not establish cash identity, counting, receipt/issue context or exact contact onset.
- Generic object transfer does not establish that the object is cash. The generic pickup definition explicitly excludes money and individual paper sheets. Missing pickup labels therefore cannot supervise missing cash handling.
- Package abandonment uses a package at least small-purse size and a no-return or two-minute unattended criterion. It cannot be relabeled as this project's cash-left-unattended rule.
- The theft label embeds ownership/permission assumptions. It cannot directly supervise a Kazakhstan legal violation from appearance alone.

The acquired annotation files include transaction and actor-support opportunities. They do not establish enough cash-positive, cash-negative, pocketing, counting or omission examples to claim the bank accuracy problem has been solved.

## Other same-style candidates screened

| Source | Verified outcome |
| --- | --- |
| [CARR retail CCTV](https://api.figshare.com/v2/articles/29470079) | Historical v4 listed CC BY 4.0 and a link-only Drive folder. Fresh publisher API v5, published **17 October 2026 at 10:42:01 UTC**, says “OLD Dataset”, “Deleted” and contains no files. Current availability, classes, size and usable footage are not established. The stale search result was rejected as download evidence. |
| [RetailAction official release](https://huggingface.co/datasets/standard-cognition/RetailAction/tree/49cb590723db921a4bd5a38adea92f6abd3f7a00) | Real ceiling-mounted retail views, but its custom Standard.AI license requires a further license once relevant product/service revenue exceeds US$10,000. This is conditional commercial use; it was not treated as an unrestricted compatible source. Its labels are shelf take/put/touch, not cash procedures. Its TRAIN archive is 26,082,080,256 bytes. No media acquired. |
| [NVIDIA PhysicalAI-Event-Videos](https://huggingface.co/datasets/nvidia/PhysicalAI-Event-Videos/tree/7f79b39fcd1ef1dc39f23f227cd8a9480d6077db) | Has synthetic retail/bank scenarios and normal examples, but the card adds Google terms for Veo videos, including restrictions relevant to similar/competing models. The public provenance file does not map a generator to every video, so a clearly usable Cosmos-only Qwen-training subset was not established. Only public metadata was acquired. Three checkout-text records were found, all normal; that is not cash-action ground truth. |
| [RPC](https://rpc-dataset.github.io/) | Official CC BY-NC-SA 4.0 dataset terms. Excluded from commercial-compatible acquisition. |
| [MERL Shopping](https://www.merl.com/research/highlights/merl-shopping-dataset) | Official page permits download for research; no commercial grant verified. No media acquired. |
| [AI City 2022 retail challenge](https://www.aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf) | Dataset agreement restricts use after the challenge, including derived models, to noncommercial academic purposes. The generator repository's MIT software license does not relicense challenge footage. |
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) and derived shoplifting mirrors | No explicit commercial video grant verified on the primary source. A mirror's license tag or implementation code license is insufficient. Excluded. |
| [ShanghaiTech Campus](https://svip-lab.github.io/dataset/campus_dataset.html) | Official page provides download/citation information, but no explicit commercial video grant was verified; campus anomalies also differ from cash-counter procedures. Excluded. |

Frozen metadata and license/readme hashes are listed in `configs/datasets/cctv-cash-public-20261017/metadata.inventory.json`. `acquisition-plan.json` records the screened sources, selected public originals, limits and adapter policy. The broader earlier shortlist remains in `docs/integration/additional-public-materials-20261017.md`.

## Bounded auxiliary teacher and adapter plan

The next work is a separate future gated experiment; acquisition did not authorize or execute teacher calls. The user's own eligible TRAIN videos retain priority for actual training. Public auxiliary processing must not delay their GPU work.

1. Verify source decode clocks, native PTS/frame correspondence and exact alignment of the original KPF annotations to transcoded `r13` media. Annotation frame indices must not be assumed to equal decoded indices. Inspect a few original frames locally for camera/transaction suitability, without altering source files or evidence pixels.
2. Predeclare a maximum **12 windows / 24 local teacher calls**, with at most eight original frames per window. A possible mix is four purchase contexts, two generic transfers, two phone/document interactions, two idle/occluded contexts and two actor-departure contexts. Available eligible examples may be fewer; record that shortfall without inventing labels. Avoid repeated near-identical moments and preserve overlap grouping.
3. Ask neutral physical questions about object visibility, hand/object continuity, transfer direction, actor presence and occlusion. Preserve the original public action labels as a separate `forte.public-cctv-primitive.v1` auxiliary task. New machine-supported cash observations, if any, are separate partial SILVER records with exact source frames, support joins, uncertainty and conflict masks.
4. Keep unobserved cash identity, cash counting, receipt/issue context and complete-opportunity coverage unknown. A generic purchase/transfer label cannot directly populate the project's cash or legal axes. Context windows can supply candidate hard normals, but unlabeled intervals are not automatically negative examples.
5. Retain source license/revision, original file hash, published TRAIN role, scenario group, camera, native clocks, original annotation ID/actor tracks/frame bounds, supporting frames, teacher/image/prompt identity, exact call accounting and target masks. Do not weaken E1 gold gates or include the project's validation/final-check sources.
6. Only after these gates, run a small predeclared auxiliary-data ablation against the frozen development recipe, recording all attempted files and failures. Compare false positives for ordinary payment, phone/document handling and departure, plus recall on genuinely supported cash facts. Keep the auxiliary data only if measured evidence supports it.

This acquisition demonstrates available, license-bound public TRAIN bytes in a closer CCTV/payment domain. Cash coverage, annotation-clock alignment, labels, training benefit and production integration remain unverified.
