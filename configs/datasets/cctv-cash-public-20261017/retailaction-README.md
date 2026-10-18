---
task_categories:
- video-classification
language:
- en
tags:
- Retail
- Action
- Video
- ICCV
- Multi-View
- Spatio-Temporal
- Localization
- Interactions
size_categories:
- 10K<n<100K
license: other
---
# RetailAction Dataset

[![Conference](https://img.shields.io/badge/ICCV-2025-blue)](https://retailvisionworkshop.github.io/)  
**Paper**: *RetailAction: Dataset for Multi-View Spatio-Temporal Localization of Human-Object Interactions in Retail*  
**Accepted at**: ICCV 2025 – Retail Vision Workshop  
**Authors**: Davide Mazzini, Alberto Raimondi, Bruno Abbate, Daniel Fischetti, David M. Woollard  
**Organization**: [Standard AI](https://standard.ai/)  

---

## Overview

RetailAction is a large-scale dataset designed for **multi-view spatio-temporal localization of human–object interactions in real-world retail environments**.  

Unlike previous action recognition datasets, RetailAction focuses on **precisely localizing customer–product interactions** (e.g., *take*, *put*, *touch*) in both **space and time**, across **multiple synchronized ceiling-mounted cameras** in operational convenience stores.

The dataset is released to advance research in:
- Fine-grained action recognition in retail
- Multi-view action localization
- Human–object interaction understanding
- Applications such as shopper behavior analysis, autonomous checkout, and retail analytics

---

## Key Features

- **Real-world scale**: 21,000 multi-view annotated samples (≈41 hours of video)  
- **Multi-view recordings**: Each sample contains **two synchronized top-view videos** from ceiling-mounted cameras  
- **Real customers**: 10,000+ unique shoppers across **10 convenience stores in the United States**  
- **Interaction-centric labels**: Instead of bounding boxes around people, **point-based annotations** mark the exact location where an item is touched  
- **Action categories**:  
  - `Take`: picking up an item  
  - `Put`: placing back an item  
  - `Touch`: hand contact without taking/placing  
- **Efficient video clips**: Each video has ≤32 frames, selected via motion-aware frame scoring for maximum informativeness  
- **Privacy preserved**: All data anonymized with facial blurring and removal of store identifiers  

---

## Get Started

### Download the Dataset

To download the full dataset, you can use the following code. If you encounter any issues, please refer to the official Hugging Face documentation.

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/standard-cognition/RetailAction
```

### Extract the Dataset

```bash
# Navigate to the directory where the extraction script is located
cd data
# Extract the three dataset splits
tar -xf test.tar
tar -xf validation.tar
tar -xf train.tar
```

---

## Dataset Statistics

- **Total samples**: 21,000  
- **Total video hours**: ~41h  
- **Unique shoppers**: 10,000+  
- **Stores**: 10 real-world locations  

**Action distribution**:

- `Take`: 97.2%
- `Put`: ~2%
- `Touch`: <1%

**Number of actions per segment**:

- 0 actions: 9.1%  
- 1 action: 84.7%  
- 2 actions: 5.5%  
- \>2 actions: 0.7%  

**Duration distributions**:

- Segment duration: typically 10–50s  
- Action duration: mostly ≤3s  

**Store representation**:

- Store 1: 36.2%  
- Store 2: 26.1%  
- Store 3: 18.7%  
- Store 4: 9.1%  
- Remaining stores: 9.9% combined  

---

## Dataset Splits

The dataset is partitioned by **unique shopper identity** to avoid leakage:

- **Train**: 17,222 samples  
- **Validation**: 1,277 samples  
- **Test**: 2,501 samples  

Identifiers are anonymized and not released.

---

## File Structure

Each dataset sample contains:
```
sample_xxxxx/
├── rank0_video.mp4              # First camera view
├── rank1_video.mp4              # Second camera view
├── metadata.json          # Metadata including sampling scores, poses, face positions and spatio-temporal labels for actions
```

### Annotations

The `metadata.json` file contains comprehensive annotations organized into several sections:

#### Action Labels

Each human-object interaction includes:

- **Action class**: `{take, put, touch}`
- **Temporal interval**: Normalized start/end times (0.0-1.0) relative to segment duration
- **Spatial coordinates**: `(x, y)` normalized coordinates for each camera view (`rank0`, `rank1`)

#### Camera Data (`action_cam`)

For each camera view (`rank0`, `rank1`):

- **Frame timestamps**: ISO 8601 timestamps for each video frame.
- **Face positions**: Detected face locations with `(col, row)` coordinates and timestamps of the subject of interest.
- **Sampling scores**: Motion-aware frame importance scores with timestamps
- **Pose data**: Full-body pose estimation is provided only for the subject of interest (i.e., the person with labeled actions). In videos with multiple people, only this subject has associated face positions and pose data.
  - Joint coordinates for 18 body keypoints (head, shoulders, elbows, wrists, hands, waist, hips, knees, ankles, feet)
  - Confidence scores for each joint detection

#### Segment Information

- **Temporal bounds**: Start and end timestamps for the entire video segment. `sampled_at_start` for every video starts at 1970-01-01T00:00:00 for anonymization purposes

---

## Spatial Metric Normalization (meters/pixels factor)

To fairly evaluate spatial localization, we convert distances from pixels to meters.  

1. For each video, we estimate the **meters-per-pixel factor** (`m_px_factor`) based on bone lengths derived from 2D pose keypoints.  
2. We compare measured pixel bone lengths to the average real-world bone lengths (computed from >10K 3D poses across multiple stores and cameras).  
3. The ratio gives a per-video normalization factor used to compute Euclidean distances in meters between predicted and ground-truth interaction points.  

### Average Bone Lengths in Meters

Computed from our 3D retail pose dataset:

```python
BONE_LENGTH_MEANS = {
    ("neck", "nose"): 0.19354,
    ("left_shoulder", "left_elbow"): 0.27096,
    ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210,
    ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204,
    ("left_knee", "left_ankle"): 0.39530,
    ("right_hip", "right_knee"): 0.39266,
    ("right_knee", "right_ankle"): 0.39322,
    ("left_shoulder", "right_shoulder"): 0.35484,
    ("left_hip", "right_hip"): 0.17150,
    ("neck", "left_shoulder"): 0.18136,
    ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_hip"): 0.51375,
    ("right_shoulder", "right_hip"): 0.51226,
}
```

### Fallback Factor
In cases where poses are incomplete or invalid and bone-based normalization cannot be computed, we apply a global average factor:

```python
M_PX_FACTOR_AVG = 3.07  # meters per 1000 pixels (approx.)
```
This ensures robust metric computation across all samples.

## Benchmark & Baselines

We provide a **DETR-based multi-view localization model** as baseline, evaluated with state-of-the-art backbones.

**Baseline performance (Test Set):**

| Model            | Type   | mAP  | mAPs (spatial) | mAPt (temporal) |
|------------------|--------|------|----------------|-----------------|
| MoViNet-A2       | Conv   | 33.5 | 43.8           | **60.9**        |
| SlowFast-R101    | Conv   | 40.2 | 50.4           | 53.2            |
| MViT-b           | Transf | **41.7** | **55.6**   | 58.2            |
| ViT-small        | Transf | 28.3 | 42.4           | 46.9            |
| ViT-base         | Transf | 31.1 | 45.7           | 47.0            |
| ViT-giant (frozen) | Transf | 38.5 | 50.3         | 58.0            |

---

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{mazzini2025retailaction,
  title={RetailAction: Dataset for Multi-View Spatio-Temporal Localization of Human-Object Interactions in Retail},
  author={Mazzini, Davide and Raimondi, Alberto and Abbate, Bruno and Fischetti, Daniel and Woollard, David M.},
  booktitle={ICCV Retail Vision Workshop},
  year={2025}
}
```

## License
The dataset is released by Standard AI.
See the full license terms in the [LICENSE](./LICENSE) file.


## Contact

For questions or collaborations, please contact:
{davide, bruno, david.woollard}@standard.ai