# Similar datasets for Forte cash video review

Checked 18 October 2026, Asia/Qyzylorda.

Start with **KZ-BD for Kazakh currency recognition and the existing MEVA subset for surveillance action examples**, then add selected CAP actions. New complementary sources include multi-banknote tenge photographs, checkout bag-grabbing videos and HO-Cap hand-object sequences. These sources cover different parts of the visual problem; none of the sources checked supplies ready-made Forte cash-procedure violation labels.

The accompanying [research catalog](../../configs/datasets/similar-datasets-20261018.catalog.json) separates published labels, possible project uses, access and licensing. It is a research artifact, not an input accepted by the training runners.

## Recommended sources

The order reflects practical usefulness for this project, rather than measured accuracy gains.

| Priority | Dataset and source | Available supervision | Application to Forte | Main limitation |
| --- | --- | --- | --- | --- |
| First | [KZ-BD](https://data.mendeley.com/datasets/dny3dgvvw8/1) | Publisher reports 4,200 tenge images, 14 denomination classes, YOLO annotations and a 70:30 training/validation split. CC BY 4.0. | Initialize or adapt a cash detector with an explicit currency-object mapping. | Mobile-phone photographs, not surveillance video. No counting, custody, handover or negative cash labels. |
| First | [MEVA](https://mevadata.org/) | Surveillance activity intervals and actor geometry; payment and generic object-transfer classes. Dataset [CC BY 4.0](https://mevadata.org/resources/MEVA-data-license.txt). | Learn transfers, actor presence and surveillance appearance; audit the previously acquired TRAIN clips before further acquisition. | A purchase can involve cash, card or phone. Generic transfer does not establish a cash transfer. |
| Next | [CAP](https://github.com/visym/cap) | Actor tracks and temporal action boundaries, captured from an elevated third-person perspective. CC BY 4.0. | Separate handover, countertop pickup/putdown, phone pocketing and hands entering pockets. | Staged atomic actions and generic objects; no cash-specific or complete-procedure labels. |
| Targeted addition | [Plastic bag grabbing at checkout](https://data.mendeley.com/datasets/h3hs3nnvbc/1) | Binary action clips from simulated checkout. CC BY 4.0. | Contrast taking an object with touching, packing or other similar counter movements. | Bags are not cash. The binary labels do not provide exact action boundaries. |
| Targeted addition | [Kazakh Banknote Image Dataset](https://data.mendeley.com/datasets/gyw87k93yb/1) | Six denomination categories plus a mixed category, including multiple banknotes and old/new 5000-tenge designs. CC BY 4.0. | Add clutter and multiple-note appearance after KZ-BD; annotate instances if training a detector. | Folder labels only; the publisher explicitly supplies no bounding boxes. |
| Later | [HO-Cap](https://irvlutd.github.io/HOCap/) | Multiple RGB-D views, hand/object poses and pick/place or hand-to-hand sequences. Dataset CC BY 4.0. | Develop hand-object continuity and occlusion handling using RGB and projected labels. | Tabletop manipulation differs from distant cash CCTV; no money or bank procedure labels. |

### KZ-BD and multi-banknote tenge images

KZ-BD is the strongest newly identified source for local currency appearance. Begin with its existing detection annotations, check image/annotation joins, and preserve its public split. Group duplicate or near-identical captures before constructing any additional evaluation subset. Small-object, blur, scale and occlusion augmentation are proposed adaptation steps; their usefulness must be measured on project footage.

The landing page calls all classes banknotes, while the [associated article](https://www.sciencedirect.com/science/article/pii/S2352340924000490) describes coins and paper notes. Confirm object types before creating a banknote-only mapping. Its 2023 publication also means coverage of newer tenge designs cannot be assumed.

The newer companion dataset is described in the [authors' 2026 article](https://www.sciencedirect.com/science/article/pii/S2352340926001873), which reports 1,846 images. Its mixed category has no per-image denomination list or instance counts. Treat the number as publisher-reported until the release archive is inventoried. Neither image source provides cash-absent examples: a missing box or label cannot automatically become a negative cash target.

KZ-BD's [published ZIP](https://data.mendeley.com/public-files/datasets/dny3dgvvw8/files/be2c791b-e7b5-4686-802b-345090f369eb/file_downloaded) is about 6.19 GB; its file endpoint answered HTTP HEAD successfully. The companion release's download-all metadata lists about 8.89 GB, but a direct cache URL returned HTTP 403. Verify a working publisher download route before planning that acquisition. Archive contents were not inspected.

### MEVA and CAP action labels

The [existing acquisition record](same-style-public-cctv-materials-20261017.md) records four MEVA TRAIN clips, about 20 minutes and 576 MiB, with matching annotations. That receipt was not rechecked on the remote host in this pass. Use the existing bytes before downloading more. Confirm cash visibility and annotation-to-native-frame alignment locally; payment labels alone are insufficient.

CAP's pinned [class list](https://raw.githubusercontent.com/visym/cap/8a04fd865667cd80c661e552e260ea0de8ab41d5/classification/cap_labels.csv) contains `person_transfers_object_to_person`, `person_picks_up_object_from_countertop`, `person_puts_down_object_on_countertop`, `person_puts_phone_into_pocket`, `person_takes_phone_from_pocket` and `person_puts_hands_in_back_pockets`. Object insertion into a bag, purse or backpack supplies additional container-motion primitives. Retain these as auxiliary action classes. A phone entering a pocket is useful as a contrasting action; it does not prove cash is absent elsewhere in the scene. Preserve public partition labels. The small advertised detection bundle is validation data, while the classification archives are hundreds of gigabytes; do not silently repurpose validation as a training pool.

### Checkout bag videos and HO-Cap

The checkout [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723439/) reports 989 five-second clips: 318 taking-bag positives and 671 negatives. Adjacent clips start two seconds apart and overlap for three seconds. Keep clips from the same original recording together; random clip splitting would leak nearly identical footage. The collection uses red bags and simulated checkout, limiting appearance diversity. Its labels can supervise bag-taking only, with cash facts left unknown.

The publisher's [file metadata](https://data.mendeley.com/public-api/datasets/h3hs3nnvbc/files?folder_id=root&version=1&%24start=0&%24limit=1000) exposes a 2,748,741,127-byte ZIP. Its [download endpoint](https://data.mendeley.com/public-files/datasets/h3hs3nnvbc/files/2ffda810-3cc3-43d9-b54f-4db5530b472b/file_downloaded) answered HTTP HEAD with the same size. The catalog records the publisher's digest separately from a future locally measured checksum.

HO-Cap provides eight RealSense views plus HoloLens capture. Its poses and masks are generated semi-automatically; they are not human-reviewed bank-action boundaries. Its [pose pipeline](https://arxiv.org/abs/2406.06843) assumes rigid objects, whereas loose banknotes deform. Its site offers poses separately at about 23.8 MB, while subject archives range from 3.2 to 20.6 GB. Inspect sequence metadata before choosing a small RGB subset. Keep synchronized views of one sequence in one group. The dataset's CC BY 4.0 license is separate from toolkit software licensing.

## Mapping to the current fact model

These are proposed transfer routes. External labels do not automatically populate the project's nineteen fact axes.

| Project target | Useful external supervision | Extra evidence needed |
| --- | --- | --- |
| `CASH_VISIBLE` | KZ-BD currency-object boxes; newly annotated companion images | Verified note/coin types, visibility at CCTV scale and discrimination from receipts, cards and paper |
| `CASH_ON_WORK_SURFACE` | Cash detection plus surface proposals | Verified spatial relation in the actual image |
| `CASH_HANDOVER_VISIBLE` | MEVA/CAP transfer actions; HO-Cap continuity | Cash identity and a continuous transfer between people; hand-to-hand manipulation alone is insufficient |
| `CASH_PLACED_IN_CLOTHING_POCKET` and `CASH_PLACED_IN_PERSONAL_CONTAINER` | CAP pocket/bag movement primitives | The same visible cash tracked into the actual destination; destination/context uncertainty preserved |
| `CASH_HANDLER_PRESENT` and `CASH_HANDLER_LEAVES_WORKSTATION` | MEVA actor tracks | Cash-handler role and project-specific workstation boundary |
| `CASH_COUNT_VISIBLE` | Explicit counting actions in restricted sources below | Usable video rights, source-bound counting labels and distinction from stacking/sorting |
| `ALL_COUNTING_PATHS_VISIBLE`, `PROCEDURE_CONTEXT_COMPLETE` and omission events | No direct coverage established in the shortlist | Continuous, complete local procedure coverage; hidden or cropped steps stay unknown |

An [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) subset can add Person, Human hand, Coin, Countertop, Handbag and Mobile phone boxes. The published box vocabulary does not include Banknote or Receipt. Use it for support objects and verified contrasting examples, not as a substitute cash dataset. Google licenses annotations under CC BY 4.0 and requires users to [verify each image's license](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses).

## Explicit cash actions and sources to defer

| Source | Relevant finding | Decision for this project |
| --- | --- | --- |
| [NTU RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) | Includes A74 counting money, A56 giving an object and A118 exchanging things, with synchronized lab views. | Strong semantic match for counting, but the official terms restrict use to noncommercial academic research and require permission for commercial use. Registration and approval are required. |
| [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) | Includes a counting-money action class and class-specific archive links. | Inspect metadata as a research lead. Original YouTube videos retain [their individual licenses](https://github.com/google-deepmind/tapnet#license-and-disclaimer); annotation availability does not clear the footage. |
| [RetailAction](https://huggingface.co/datasets/standard-cognition/RetailAction) | Real elevated retail imagery and shelf interactions. | Conditional commercial terms and weak cash coverage; retain the [earlier license review](same-style-public-cctv-materials-20261017.md). |
| [RetailS](https://github.com/TeCSAR-UNCC/RetailS) | A [2026 retail surveillance study](https://arxiv.org/html/2603.04723v1) releases pose sequences for concealment research. | No RGB cash evidence or verified commercial dataset grant; preserve the test role of published concealment incidents. Use the paper's table rather than the README's swapped incident/frame columns. |
| [Assembly101](https://assembly-101.github.io/) and [IKEA ASM](https://ikeaasm.github.io/) | Fine-grained fixed-view procedural actions. | Their noncommercial dataset terms prevent treating them as cleared additions to the commercial-use track. |
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | Surveillance anomalies, including shoplifting. | Broad anomaly labels cannot identify the required cash facts; no commercial video grant was verified in the earlier review. |

HoloAssist, IndustReal and other earlier candidates remain documented in [additional public materials](additional-public-materials-20261017.md). They are lower priority than local currency and counter-view actions for this specific gap. UCI Banknote Authentication supplies numerical features, so it cannot train a visual cash detector or temporal action model.

## Applying the datasets

1. **Prepare KZ-BD as a separate object task.** Retain original classes, boxes, split, DOI and license; add a derived currency-object mapping after checking note/coin identities. Check actual archive contents and capture groups before training. Add verified noncash objects rather than interpreting unannotated regions as cash-absent.
2. **Audit the existing MEVA subset locally.** Reuse the bounded twelve-window plan in its acquisition record. Keep original activity labels separate from any newly generated, source-supported cash observations. Mask unsupported fields and retain teacher provenance for machine labels.
3. **Add only relevant public TRAIN actions.** Select CAP transfer, countertop and pocket primitives, followed by a small checkout-bag subset if parent-recording grouping can be recovered. Keep all related views/windows together. Exclude public evaluation partitions from this proposed auxiliary training pool.
4. **Use an explicit auxiliary training route.** The current W1 experiment trains a temporal head on frozen image features. Adding images cannot update that frozen encoder, and generic action labels cannot enter its loss as bank-violation labels. A cash detector, explicitly adapted visual model, or separate auxiliary action loss needs its own manifest, target masks and comparison. The research catalog does not bypass existing runner gates.
5. **Compare additions one at a time on fixed development data.** Compare baseline, cash adaptation, then action adaptation with the same inputs and budget. Report failures, false alarms, supported fact precision/recall, uncertainty and event timing where reference labels exist. Preserve the project final-check split for the final selected candidate.

The main remaining data gap is continuous bank-style counting and custody supervision. A small authorized staged collection could contrast counting versus sorting, cash versus receipt transfer, cash versus phone pocketing, actual departure versus occlusion, and complete versus clipped procedures. External sources should supplement those distinctions; more generic anomaly videos will not supply them automatically.

This pass added public-source research and a catalog. It downloaded no new dataset media, launched no training and changed no serving configuration.
