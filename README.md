# MC-3D-Detection

**MC3D-Net: 3D Detection of Meissner's Corpuscles in RCM Images**

University of Rochester — CSC249 Research Project  
Authors: Jingwen Hu, Ruby Gong

---

## Overview

This project develops an improved detection pipeline for Meissner's Corpuscles (MCs) in reflectance confocal microscopy (RCM) image stacks. MCs are small mechanoreceptors in the skin that appear as weak, low-contrast highlights across multiple depth slices.

The existing YOLOv4-based pipeline processes each 2D slice independently and reconstructs 3D structures via postprocessing. We address two key limitations:
- **False positives** from non-MC structures in superficial skin layers
- **False negatives** from MCs with weak highlights below the detection threshold

---

## Approach

All three pipelines share the same preprocessing steps:

1. **Starting layer filter** — discard the top 2 skin layers (retain layer 3 and below) to reduce false positives
2. **CLAHE enhancement** — apply Contrast Limited Adaptive Histogram Equalization to boost weak MC highlights before detection

Three parallel detection pipelines are then compared:

| Pipeline | Description |
|----------|-------------|
| Pipeline 1 | YOLOv4 → YOLOv8 swap, existing 3D postprocessing kept intact |
| Pipeline 2 | MedYOLO — native 3D detection on NIfTI volumes |
| Pipeline 3 | YOLOv8 extended to 3D — end-to-end volumetric detection |

All pipelines are evaluated using **mAP, precision, and recall**.

---

## Repository Structure

```
MC-3D-Detection/
├── main.py              # Main pipeline (YOLOv4 baseline with starting layer filter)
├── .gitignore
└── README.md
```

> **Note:** Image data is stored separately in Box and is not included in this repository.  
> Data path structure: `AI_MC_Analysis_Test_Images / [patient] / [visit] / [site] / VivaStack #N / *.bmp`

---

## Data

- **Source:** In-vivo RCM skin image stacks provided by medical collaborators
- **Format:** `.bmp` images, ~35 slices per stack, 7 µm spacing between slices
- **Target:** Meissner's corpuscles (~40 µm wide, 80–100 µm deep, visible in ≥4 consecutive slices)
- **Annotations:** Manual bounding boxes created in ImageJ

Data is stored in Box (access restricted). Contact the project team for access.

---

## Setup

### Requirements

```bash
pip install opencv-python numpy tqdm
# darknet (YOLOv4): follow official install instructions
# ultralytics (YOLOv8): pip install ultralytics
```

### Running Pipeline 1

```bash
python main.py \
  --path "/path/to/VivaStack #1" \
  --time 20260416 \
  --score_thresh 0.32 \
  --count_thresh 3
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--path` | required | Path to the data folder (stack / site / visit / patient / study) |
| `--time` | required | Timestamp string for output CSV filename |
| `--score_thresh` | 0.32 | Confidence threshold for detection |
| `--count_thresh` | 3 | Min number of slices an MC must appear in |
| `--scan_depth_gap` | 7.0 | Depth spacing between slices (µm) |
| `--store_predicted_images` | false | Save images with detection overlays |

---

## Current Status

- [x] Starting layer filter (skip top 2 layers)
- [x] CLAHE image enhancement
- [ ] Pipeline 1: YOLOv8 2D + existing 3D postprocessing
- [ ] Pipeline 2: MedYOLO native 3D
- [ ] Pipeline 3: YOLOv8 extended to 3D
- [ ] Evaluation (mAP / precision / recall)

---

## References

1. Bozkurt et al., "Skin strata delineation in RCM using attention-based recurrent CNNs," *IEEE TMI*, 2020.
2. Lboukili et al., "ML methods for RCM image analysis," *Medical Image Analysis*, 2021.
3. Baumgartner et al., "nnDetection," *Nature Machine Intelligence*, 2021.
4. Sobek et al., "MedYOLO," *arXiv:2204.03072*, 2022.
