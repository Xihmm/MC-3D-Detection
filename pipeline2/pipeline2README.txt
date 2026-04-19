==============================
Pipeline 2: Native 3D Detection with MedYOLO
==============================

Author: Jingwen Hu
Project: MC Detection (Medical Imaging)
Stage: Data Preparation (2D → 3D labels)

--------------------------------------------------
📌 Overview
--------------------------------------------------

Pipeline 2 replaces slice-wise 2D detection with a native 3D detection pipeline.

The key idea is:
- Convert raw image stacks → 3D volumes (.nii.gz)
- Convert per-slice 2D YOLO labels → volumetric 3D labels
- Train a 3D detector (MedYOLO)

This pipeline enables modeling of spatial continuity of Meissner's Corpuscles (MCs),
which typically appear across multiple consecutive slices.

--------------------------------------------------
📁 Directory Structure
--------------------------------------------------

D:\AI_MED\

├── pipeline2\
│   ├── README_pipeline2.txt
│   ├── build_volume.py
│   ├── build_volume_group.py
│   ├── convert_yolo2d_to_medyolo3d.py
│   ├── outputs\
│   │   └── medyolo_labels_set7\
│   └── medyolo_config\
│       └── mc_set7.yaml
│
├── generated_labels_diff_clean1\
│   └── <stack_name>\
│       ├── *.txt  (YOLO 2D labels per slice)
│
├── NII_Output_set7\
│   ├── *.nii.gz  (3D volumes)
│
├── Set 7\Set 7\
│   └── *_Raw\  (original image stacks)
│
├── MedYOLO\
│
└── MC Tool v8\

--------------------------------------------------
📥 Inputs
--------------------------------------------------

1. Raw image stacks:
   D:\AI_MED\Set 7\Set 7\*_Raw\

2. Per-slice YOLO 2D labels:
   D:\AI_MED\generated_labels_diff_clean1\<stack_name>\*.txt

   Format:
   class_id x_center y_center width height
   (normalized to [0,1])

3. Generated 3D volumes:
   D:\AI_MED\NII_Output_set7\*.nii.gz

--------------------------------------------------
📤 Outputs
--------------------------------------------------

MedYOLO 3D labels:
D:\AI_MED\pipeline2\outputs\medyolo_labels_set7\*.txt

Format per line:
class z_center x_center y_center z_length x_length y_length

All values are normalized to [0,1].

--------------------------------------------------
⚙️ Step 1: Build 3D Volumes
--------------------------------------------------

Script:
build_volume.py / build_volume_group.py

Function:
- Load slice images (bmp/png/tif)
- Sort by filename index
- Remove first N layers (start_layer)
- Apply CLAHE (optional)
- Stack into 3D array
- Save as .nii.gz

Important parameter:
start_layer = 2

This must be consistent across ALL steps.

--------------------------------------------------
⚙️ Step 2: Convert 2D Labels → 3D Labels
--------------------------------------------------

Script:
convert_yolo2d_to_medyolo3d.py

Core steps:
1. Read per-slice YOLO labels
2. Convert normalized boxes → pixel boxes
3. Match boxes across adjacent slices
4. Form 3D tracks
5. Convert tracks → 3D bounding boxes
6. Normalize and save

Matching criteria:
- IoU threshold
- Center distance threshold
- Slice continuity constraint

--------------------------------------------------
📐 Key Parameters
--------------------------------------------------

start_layer = 2
    Must match volume construction

max_slice_gap = 1
    Only connect adjacent slices

iou_thresh = 0.05
    Lower IoU helps connect slightly shifted detections

center_dist_thresh = 60.0
    Pixel distance threshold for matching across slices

min_layers_visible = 2
    Filters out noise (single-slice detections)

--------------------------------------------------
🧠 Design Rationale
--------------------------------------------------

- MCs are 3D structures and appear across multiple slices
- 2D detection alone loses spatial continuity
- 3D aggregation improves robustness and reduces false positives
- Using center distance + IoU handles detection jitter

--------------------------------------------------
🧪 Validation Strategy
--------------------------------------------------

Single-stack testing (completed):
- 7 2D boxes → 2–5 3D tracks
- Typical track spans: 3–10 slices

Checks:
- No excessive single-slice tracks
- No unrealistic long connections
- Reasonable spatial continuity

--------------------------------------------------
🚨 Important Notes
--------------------------------------------------

1. Coordinate consistency:
   - start_layer MUST match between volume and labels

2. Slicer visualization:
   - Orientation differences are display-only
   - Does NOT affect training

3. Boundary boxes:
   - Some boxes may be clamped at image edges (x=0)
   - Verify if due to detection or real structure

--------------------------------------------------
🚀 Next Steps
--------------------------------------------------

1. Batch convert all stacks in set7
2. Split dataset into train / val / test
3. Prepare MedYOLO dataset structure:
   images/ + labels/
4. Train MedYOLO 3D detector
5. Compare against Pipeline 1 (2D YOLO)

--------------------------------------------------
📊 Current Status
--------------------------------------------------

✔ NIfTI generation completed
✔ 2D YOLO labels available
✔ 2D → 3D conversion validated on multiple stacks
✔ Output format compatible with MedYOLO

Next milestone:
→ Full dataset generation + training

--------------------------------------------------