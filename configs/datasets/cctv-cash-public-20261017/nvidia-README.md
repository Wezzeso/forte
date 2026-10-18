---
license: other
license_name: nvidia-dataset-license
license_link: >-
  https://assets.ngc.nvidia.com/products/api-catalog/legal/Personas-Dataset-NVIDIA-Training-Dataset-License-Agreement-2025-11-26.pdf
pretty_name: PhysicalAI-Event-Videos
task_categories:
- video-classification
- video-text-to-text
language:
- en
size_categories:
- 1K<n<10K
tags:
- video-anomaly-detection
- safety-events
- surveillance
- synthetic
- text-to-video-retrieval
---

# PhysicalAI-Event-Videos

PhysicalAI-Event-Videos is a video-anomaly and event dataset for developing text-to-video anomaly-search and safety-event-understanding systems. Version 1.0 provides structured annotations for 1,612 parent-video records and 3,796 labeled chunks, with 71,023 captions and queries across person-attribute, general-caption, anomaly, and action search tasks.

The redistributable media payload contains 1,486 NVIDIA-generated parent videos and 2,939 packaged chunk clips. Of those chunk clips, 2,919 have retained labels and 20 remain in the media archive without labels. CHAD and VAD-R1 media are not redistributed; the annotation index includes labels and public source pointers for those datasets.

## Dataset highlights

![A 3-by-3 grid of representative NVIDIA-generated normal and nonviolent event videos.](physicalai_event_videos_highlights_3x3.gif)

The montage shows nine representative NVIDIA-generated videos spanning normal activity and nonviolent retail, workplace, and public-space events. It includes selected examples from NV-Retail-Shoplifting-v0.1 and does not include CHAD or VAD-R1 source media.

## Dataset description

The dataset covers human-activity anomalies such as falling, fighting, assault, climbing, throwing objects, shoplifting, and vaping or smoking. It also includes warehouse and industrial-safety events such as forklift incidents, forklift-person near misses, falling or damaged boxes, fire and smoke, liquid spills, and personal protective equipment violations.

Hard-negative samples depict normal or non-anomalous activity that may visually resemble an anomalous event. These samples are intended to help models distinguish true safety incidents from benign activities and reduce false-positive detections.

The videos span simulated environments including warehouses, retail spaces, offices, parking lots, streets, restaurants, parks, hospitals, schools, construction sites, stadiums, transit areas, airports, and banks.

- **Owner:** NVIDIA Corporation
- **Creation date:** June 15, 2026
- **Version:** v1.0
- **Split:** train only; validation and test splits are not included
- **Modalities:** video, audio, and text

## License and terms of use

License: NVIDIA Dataset License Agreement.

Use of this dataset is governed by the NVIDIA Dataset License Agreement. Certain videos in this dataset were generated using Google Veo 3. The Veo-generated videos also remain subject to the applicable [Google terms](https://cloud.google.com/terms/service-terms), including restrictions on using such videos to develop a similar or competing product or service or to create or improve models similar to a Google model.

VAD-R1 and CHAD source media are not distributed. Users must obtain those datasets directly from the applicable source under its respective terms. The Public Safety Sample Videos seed dataset is not distributed.

## Intended use

PhysicalAI-Event-Videos is intended for application developers working on video understanding and safety-related AI systems. Potential uses include:

- Video anomaly detection
- Safety-event classification
- Temporal anomaly localization
- Video-language model training
- Synthetic-to-real transfer research
- Hard-negative mining and false-positive reduction
- Industrial and warehouse safety monitoring research
- Retail and public-space anomaly detection research
- Multimodal audio, video, and text representation learning

## Dataset characterization

### Data collection method

The release combines annotations for NVIDIA-generated and public-source videos with a redistributable NVIDIA-generated media payload. NVIDIA-generated videos were created using Google Veo 3 and Cosmos 3 Super. Annotations also cover CHAD and VAD-R1, but those third-party source videos are not distributed; users must obtain them from their original sources. The Public Safety Sample Videos seed dataset is not distributed.

### Labeling method

Labeling is hybrid: synthetic, automated, and manually corrected. Event labels and descriptive metadata are derived from generation specifications and prompts. Derived video chunks and associated manifests were produced through automated preprocessing. A subset of annotations received human review and correction; remaining query and caption fields may be model-generated.

## Dataset format and contents

- **Video:** MP4 containers with H.264 video
- **Audio:** AAC audio streams embedded in MP4 files when available
- **Text:** JSON labels and provenance metadata, plus CSV manifests and summary statistics

```text
prepare_chunks.py                 optional reconstruction from downloaded parent videos
README.md                         this dataset card
physicalai_event_videos_highlights_3x3.gif
anomaly-category-distribution.csv
anomaly-category-mapping.json
2026/
├── train/
│   ├── videos.zip                1,486 NVIDIA-generated parent videos
│   ├── chunks.zip                2,939 NVIDIA-generated chunk clips
│   ├── labels.json               1,612 parent records with 3,796 labeled chunks
│   └── video_manifest.csv        filename-to-dataset lookup
├── stats.json                    composition and per-field query counts
└── provenance.json               public source and generation provenance
```

The 2,939 packaged chunk clips comprise 2,919 labeled clips plus 20 retained media clips whose labels are not included. CHAD and VAD-R1 rows in `labels.json` identify source media through public pointers; their media is not present in the archives.

## Label schema

`labels.json` is a list of per-video records:

```jsonc
{
  "split": "train",
  "dataset": "NV-VAD-v1.1",
  "video_id": "video_01126",
  "anomaly_type": "person running",
  "total_frames": 189,
  "video_duration_sec": 7.875,
  "fps": 24.0,
  "video_path": "videos/video_01126.mp4",
  "chunks": [
    {
      "chunk_index": 0,
      "start_time_sec": 0.0,
      "end_time_sec": 5.0,
      "start_frame": 0,
      "end_frame": 120,
      "is_anomaly": true,
      "anomaly_type": "person running",
      "dense_caption": "...",
      "scene_caption": "...",
      "pas_queries": ["..."],
      "caption_queries": ["..."],
      "anomaly_queries": ["..."],
      "action_queries": ["..."],
      "chunk_file": "chunk_000.mp4"
    }
  ]
}
```

### Joining labels to media

| Requested media | Path inside the extracted archive |
|---|---|
| Parent video | `videos/` + `basename(video_path)` |
| Chunk clip | `chunks/<dataset>/<video_id>/<chunk_file>` |

Parent-video basenames are unique across the redistributable NVIDIA-generated videos. Chunk filenames are not unique on their own, so chunk paths are nested by `dataset` and `video_id`.

The 1,486 NVIDIA-generated parent videos use a deterministic global sequence, `video_00001.mp4` through `video_01486.mp4`. The filename does not encode its source; use `dataset` in `labels.json` or `video_manifest.csv` to distinguish releases.

## Annotation examples

Each labeled chunk contains temporal bounds, an anomaly decision and free-text anomaly type, two complementary captions, and four query sets. The example below is an excerpt from an NVIDIA-generated forklift-person near-miss chunk.

| Annotation type | Example |
|---|---|
| Temporal localization | `0.0-5.0 seconds` |
| Anomaly flag | `is_anomaly: true` |
| Anomaly type | `near miss forklift person` |
| Dense caption | A forklift drives down a warehouse aisle and nearly collides with two employees standing in its path, forcing them to step aside quickly. 00:00-00:02: Two warehouse employees in safety vests stand in the aisle discussing a clipboard while a forklift approaches from the distance. 00:02-00:04: The forklift continues to drive forward toward the two employees, who remain stationary and focused on their documents. 00:04-00:07: As the forklift closes the distance rapidly, the employees notice it and quickly step out of the way to avoid a collision. |
| Scene caption | The scene takes place indoors within a large warehouse aisle, flanked by towering orange industrial shelving units filled with boxed inventory. The camera is positioned at a fixed, eye-level angle looking down the center of the aisle. The area is brightly lit with cool, overhead industrial lighting, and the general activity involves personnel in safety vests coordinating logistics and operating a forklift. |
| Person-attribute search (PAS) | `person wearing yellow vest and grey shirt`; `person wearing yellow vest and black pants` |
| General caption query | `warehouse aisle with orange industrial shelving units`; `two employees discussing a clipboard in a warehouse` |
| Anomaly query | `person in yellow safety vest nearly hit by a forklift`; `forklift almost colliding with warehouse employees` |
| Action query | `a forklift driving`; `people standing in a path` |

`anomaly_type` remains free text. The derived anomaly families below are provided for summary reporting and do not replace the original labels.

## Dataset quantification

### Annotation coverage

- **Parent records:** 1,612
- **Labeled chunks:** 3,796
- **Anomalous labeled chunks:** 2,483
- **Normal labeled chunks:** 1,313
- **Captions and queries:** 71,023 total

### Query counts

| Query field | Count |
|---|---:|
| Person-attribute search (`pas_queries`) | 18,103 |
| General caption (`caption_queries`) | 18,823 |
| Anomaly (`anomaly_queries`) | 18,913 |
| Action (`action_queries`) | 15,184 |
| **Total** | **71,023** |

### Source coverage

| Source dataset | Parent records | Labeled chunks | Media availability in this release |
|---|---:|---:|---|
| `CHAD` | 46 | 392 | Source pointers only |
| `NV-Retail-Shoplifting-v0.1` | 200 | 399 | NVIDIA-generated media included |
| `NV-VAD-v1` | 925 | 1,799 | NVIDIA-generated media included |
| `NV-VAD-v1.1` | 361 | 721 | NVIDIA-generated media included |
| `VAD-R1` | 80 | 485 | Source pointers only |
| **Total** | **1,612** | **3,796** | |

### Redistributable NVIDIA-generated media

- **Parent videos:** 1,486
- **Labeled chunk clips:** 2,919
- **Retained but unlabeled chunk clips:** 20
- **Packaged chunk clips:** 2,939

## Anomaly-category distribution

`anomaly_type` is a free-text field rather than a closed categorical label. For reporting, each of the 2,483 anomalous chunks is assigned to one mutually exclusive family using a documented, deterministic first-match mapping. These derived families do not replace the original labels.

| Derived anomaly family | Labeled chunks | Share of anomalous chunks |
|---|---:|---:|
| Fall / collapse | 379 | 15.3% |
| Fight / assault | 340 | 13.7% |
| Smoking / vaping | 282 | 11.4% |
| Climbing / jumping | 201 | 8.1% |
| Throwing / vandalism | 164 | 6.6% |
| Running | 160 | 6.4% |
| Fire / smoke / explosion | 145 | 5.8% |
| Falling / damaged objects | 137 | 5.5% |
| Traffic / parking safety | 125 | 5.0% |
| Retail theft / concealment | 111 | 4.5% |
| Forklift incident | 91 | 3.7% |
| Liquid spill / leak | 91 | 3.7% |
| Animal-related hazard | 91 | 3.7% |
| Generic warehouse-safety violation | 82 | 3.3% |
| PPE violation | 32 | 1.3% |
| Weather / environmental hazard | 27 | 1.1% |
| Other / ambiguous | 25 | 1.0% |

Percentages use anomalous labeled chunks as the denominator and may not sum to exactly 100% because values are rounded to one decimal place. See [`anomaly-category-distribution.csv`](anomaly-category-distribution.csv) for the summary table and [`anomaly-category-mapping.json`](anomaly-category-mapping.json) for the ordered mapping rules.

## Preparing chunks from downloaded videos

`chunks.zip` is ready to use. If parent videos are downloaded separately, the optional `prepare_chunks.py` script reconstructs labeled time windows without making network requests:

```bash
python3 prepare_chunks.py \
  --labels 2026/train/labels.json \
  --videos-dir /path/to/downloaded/videos \
  --output-dir 2026/train/chunks \
  --workers 4
```

The script searches the video directory recursively, reads exact start and end times from `labels.json`, and writes `chunks/<dataset>/<video_id>/<chunk_file>`. It requires Python 3, `ffmpeg`, and `ffprobe`. Use `--dry-run` to validate inputs or `--dataset NAME` to prepare a subset. Reconstructed clips match the labeled time windows but may not be byte-identical across ffmpeg versions.

## Public source pages

These are public third-party landing pages, not NVIDIA-controlled storage:

- **CHAD:** [official repository](https://github.com/TeCSAR-UNCC/CHAD) and [primary video archive](https://drive.google.com/file/d/13am4hfhicErcozAYgkmQm02_K-cCmtkQ/view?usp=sharing)
- **VAD-R1:** [public Hugging Face dataset](https://huggingface.co/datasets/wbfwonderful/Vad-R1)

CHAD and VAD-R1 media must be obtained from their original sources under the applicable terms. This release does not redistribute that media.

## Known limitations

1. **Third-party source dependency.** CHAD and VAD-R1 annotations require users to obtain the corresponding media separately and comply with each source dataset's terms.
2. **Model-generated text.** A subset of annotations received human review and correction, while remaining query and caption fields may contain model errors or inconsistencies.
3. **Free-text anomaly types.** `anomaly_type` is not a closed ontology. The supplied category mapping is a reporting aid based on deterministic text matching.
4. **Near-duplicate scenarios.** Many NVIDIA-generated videos share related generation scenarios. Split by generation scenario, not only by `video_id`, to reduce train-evaluation leakage.
5. **Synthetic-to-real gap.** Models trained on simulated media may not transfer directly to real environments, camera systems, or populations without additional validation.

## Ethical considerations

NVIDIA believes Trustworthy AI is a shared responsibility and has established policies and practices to enable development for a wide array of AI applications. Developers should work with their organizations to ensure this dataset meets the requirements of the relevant industry and use case and addresses unforeseen product misuse.

Users should evaluate performance, safety, fairness, privacy, and failure modes in the intended deployment context. Synthetic data may still encode artifacts or biases from prompts and generation systems, and third-party annotations should be assessed with their source context and terms.
