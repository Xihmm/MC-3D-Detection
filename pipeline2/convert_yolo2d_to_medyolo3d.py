import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import nibabel as nib


# =========================================================
# 基础工具
# =========================================================

def extract_number(filename: str) -> int:
    nums = re.findall(r'\d+', filename)
    if not nums:
        return -1
    return int(nums[-1])


def clamp(v, low, high):
    return max(low, min(v, high))


def yolo_norm_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int
) -> List[float]:
    """
    YOLO normalized:
        x_center, y_center, width, height in [0,1]
    转成像素坐标:
        [x1, y1, x2, y2]
    """
    xc = x_center * img_w
    yc = y_center * img_h
    bw = width * img_w
    bh = height * img_h

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0

    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, 0, img_w - 1)
    y2 = clamp(y2, 0, img_h - 1)

    return [x1, y1, x2, y2]


def box_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box1: List[float], box2: List[float]) -> float:
    c1x, c1y = box_center(box1)
    c2x, c2y = box_center(box2)
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def box_iou(box1: List[float], box2: List[float]) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# =========================================================
# Track3D：一个跨层的 3D object
# =========================================================

class Track3D:
    def __init__(self, class_id: int, slice_idx: int, box: List[float]):
        self.class_id = class_id
        self.slice_indices = [slice_idx]
        self.boxes = [box]

    def last_slice(self) -> int:
        return self.slice_indices[-1]

    def last_box(self) -> List[float]:
        return self.boxes[-1]

    def add(self, slice_idx: int, box: List[float]):
        self.slice_indices.append(slice_idx)
        self.boxes.append(box)

    def span(self) -> int:
        return len(self.slice_indices)

    def to_3d_box(self):
        z_min = min(self.slice_indices)
        z_max = max(self.slice_indices)

        x_min = min(b[0] for b in self.boxes)
        y_min = min(b[1] for b in self.boxes)
        x_max = max(b[2] for b in self.boxes)
        y_max = max(b[3] for b in self.boxes)

        return x_min, y_min, z_min, x_max, y_max, z_max


# =========================================================
# 读每层 YOLO 2D txt
# =========================================================

def read_yolo_slice_labels(
    label_dir: str,
    img_w: int,
    img_h: int,
    start_layer: int = 2,
    allowed_class_ids: List[int] = None
) -> Dict[int, List[Tuple[int, List[float]]]]:
    """
    读取一个 stack 文件夹下的每层 txt
    每行格式:
        class_id x_center y_center width height

    返回:
        slice_boxes = {
            z: [(class_id, [x1,y1,x2,y2]), ...],
            ...
        }

    注意:
    - z 是过滤掉前 start_layer 层之后重新编号的 z
    """
    label_dir = Path(label_dir)
    txt_files = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    txt_files = sorted(txt_files, key=lambda p: extract_number(p.name))

    if len(txt_files) == 0:
        raise ValueError(f"No txt files found in {label_dir}")

    slice_boxes: Dict[int, List[Tuple[int, List[float]]]] = {}

    for original_idx, txt_path in enumerate(txt_files):
        # 跳过前两层，对齐你生成 nii 时的 start_layer
        if original_idx < start_layer:
            continue

        z = original_idx - start_layer
        boxes_this_slice = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                if allowed_class_ids is not None and class_id not in allowed_class_ids:
                    continue

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                box = yolo_norm_to_xyxy(
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    img_w=img_w,
                    img_h=img_h
                )
                boxes_this_slice.append((class_id, box))

        slice_boxes[z] = boxes_this_slice

    return slice_boxes


# =========================================================
# 相邻层匹配
# =========================================================

def can_match(
    prev_box: List[float],
    curr_box: List[float],
    iou_thresh: float = 0.1,
    center_dist_thresh: float = 25.0
) -> bool:
    iou = box_iou(prev_box, curr_box)
    dist = center_distance(prev_box, curr_box)

    if iou >= iou_thresh:
        return True
    if dist <= center_dist_thresh:
        return True
    return False


def group_2d_boxes_to_3d_tracks(
    slice_boxes: Dict[int, List[Tuple[int, List[float]]]],
    max_slice_gap: int = 1,
    iou_thresh: float = 0.1,
    center_dist_thresh: float = 25.0
) -> List[Track3D]:
    """
    按 class_id 分开匹配，避免不同类串起来
    """
    tracks: List[Track3D] = []
    all_slices = sorted(slice_boxes.keys())

    for z in all_slices:
        boxes = slice_boxes[z]
        used = [False] * len(boxes)

        # 尝试接到已有 track 上
        for track in tracks:
            last_z = track.last_slice()

            if z - last_z < 1 or z - last_z > max_slice_gap:
                continue

            best_idx = -1
            best_score = -1e9

            for i, (class_id, box) in enumerate(boxes):
                if used[i]:
                    continue
                if class_id != track.class_id:
                    continue

                if can_match(
                    track.last_box(),
                    box,
                    iou_thresh=iou_thresh,
                    center_dist_thresh=center_dist_thresh
                ):
                    score = box_iou(track.last_box(), box) - 0.001 * center_distance(track.last_box(), box)
                    if score > best_score:
                        best_score = score
                        best_idx = i

            if best_idx != -1:
                _, best_box = boxes[best_idx]
                track.add(z, best_box)
                used[best_idx] = True

        # 没匹配上的开新 track
        for i, (class_id, box) in enumerate(boxes):
            if not used[i]:
                tracks.append(Track3D(class_id=class_id, slice_idx=z, box=box))

    return tracks


# =========================================================
# Track3D -> MedYOLO 3D label
# 格式:
# class z_center x_center y_center z_length x_length y_length
# volume_shape = (H, W, D)
# =========================================================

def track_to_medyolo_line(
    track: Track3D,
    volume_shape: Tuple[int, int, int]
) -> str:
    H, W, D = volume_shape

    x_min, y_min, z_min, x_max, y_max, z_max = track.to_3d_box()

    z_center = (z_min + z_max) / 2.0
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # z 是离散 slice，长度建议 +1
    z_length = (z_max - z_min + 1)
    x_length = (x_max - x_min)
    y_length = (y_max - y_min)

    # 归一化
    z_center /= D
    x_center /= W
    y_center /= H

    z_length /= D
    x_length /= W
    y_length /= H

    return (
        f"{track.class_id} "
        f"{z_center:.6f} {x_center:.6f} {y_center:.6f} "
        f"{z_length:.6f} {x_length:.6f} {y_length:.6f}"
    )


# =========================================================
# 从 nii 自动拿 volume shape
# =========================================================

def get_volume_shape_from_nii(nii_path: str) -> Tuple[int, int, int]:
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape}")
    return data.shape  # (H, W, D)


# =========================================================
# 主函数：一个 stack 的所有 per-slice txt -> 一个 MedYOLO 3D txt
# =========================================================

def convert_one_stack_yolo2d_to_medyolo3d(
    nii_path: str,
    slice_label_dir: str,
    save_txt_path: str,
    start_layer: int = 2,
    allowed_class_ids: List[int] = None,
    min_layers_visible: int = 1,
    max_slice_gap: int = 1,
    iou_thresh: float = 0.1,
    center_dist_thresh: float = 25.0,
    verbose: bool = True
):
    volume_shape = get_volume_shape_from_nii(nii_path)  # (H, W, D)
    H, W, D = volume_shape

    if verbose:
        print(f"NII: {nii_path}")
        print(f"Volume shape (H, W, D): {volume_shape}")

    slice_boxes = read_yolo_slice_labels(
        label_dir=slice_label_dir,
        img_w=W,
        img_h=H,
        start_layer=start_layer,
        allowed_class_ids=allowed_class_ids
    )

    if verbose:
        total_2d = sum(len(v) for v in slice_boxes.values())
        print(f"Read {len(slice_boxes)} slices after start_layer filter")
        print(f"Total 2D boxes: {total_2d}")

    tracks = group_2d_boxes_to_3d_tracks(
        slice_boxes=slice_boxes,
        max_slice_gap=max_slice_gap,
        iou_thresh=iou_thresh,
        center_dist_thresh=center_dist_thresh
    )

    # 过滤太短的 track
    tracks = [t for t in tracks if t.span() >= min_layers_visible]

    if verbose:
        print(f"3D tracks kept: {len(tracks)}")
        for i, t in enumerate(tracks[:10]):
            print(
                f"  Track {i}: class={t.class_id}, "
                f"slices={t.slice_indices}, "
                f"3D box={t.to_3d_box()}"
                )
    debug_txt_path = str(save_txt_path).replace(".txt", "_debug.txt")
    with open(debug_txt_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(tracks):
            f.write(
                f"Track {i}: class={t.class_id}, "
                f"slices={t.slice_indices}, "
                f"3D_box={t.to_3d_box()}\n"
            )
    print(f"Saved debug track info: {debug_txt_path}")
    lines = [track_to_medyolo_line(t, volume_shape) for t in tracks]

    save_txt_path = Path(save_txt_path)
    save_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved MedYOLO label: {save_txt_path}")


# =========================================================
# 批量版：多个 stack
# 假设:
#   nii_root/
#       1897_Stack4.nii.gz
#       1897_Stack5.nii.gz
#
#   yolo2d_label_root/
#       1897_Stack4/
#           slice_001.txt
#           slice_002.txt
#           ...
#       1897_Stack5/
#           ...
#
# 输出:
#   medyolo_label_root/
#       1897_Stack4.txt
#       1897_Stack5.txt
# =========================================================

def batch_convert_yolo2d_to_medyolo3d(
    nii_root: str,
    yolo2d_label_root: str,
    medyolo_label_root: str,
    start_layer: int = 2,
    allowed_class_ids: List[int] = None,
    min_layers_visible: int = 1,
    max_slice_gap: int = 1,
    iou_thresh: float = 0.1,
    center_dist_thresh: float = 25.0
):
    nii_root = Path(nii_root)
    yolo2d_label_root = Path(yolo2d_label_root)
    medyolo_label_root = Path(medyolo_label_root)
    medyolo_label_root.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(nii_root.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in {nii_root}")

    success = 0
    failed = 0

    for nii_path in nii_files:
        stack_name = nii_path.name.replace(".nii.gz", "")
        slice_label_dir = yolo2d_label_root / stack_name
        save_txt_path = medyolo_label_root / f"{stack_name}.txt"

        print("\n==============================")
        print(f"Processing stack: {stack_name}")

        if not slice_label_dir.exists():
            print(f"Label folder not found: {slice_label_dir}")
            failed += 1
            continue

        try:
            convert_one_stack_yolo2d_to_medyolo3d(
                nii_path=str(nii_path),
                slice_label_dir=str(slice_label_dir),
                save_txt_path=str(save_txt_path),
                start_layer=start_layer,
                allowed_class_ids=allowed_class_ids,
                min_layers_visible=min_layers_visible,
                max_slice_gap=max_slice_gap,
                iou_thresh=iou_thresh,
                center_dist_thresh=center_dist_thresh,
                verbose=True
            )
            success += 1
        except Exception as e:
            print(f"Failed: {stack_name}")
            print(f"Error: {e}")
            failed += 1

    print("\n==============================")
    print("Batch done")
    print(f"Success: {success}")
    print(f"Failed : {failed}")


# =========================================================
# 用法示例
# =========================================================

if __name__ == "__main__":
    convert_one_stack_yolo2d_to_medyolo3d(
        nii_path=r"D:\AI_MED\NII_Output_set7\0035_Stack14.nii.gz",
        slice_label_dir=r"D:\AI_MED\generated_labels_diff_clean1\0035_Stack14_Overlays_PNG",
        save_txt_path=r"D:\AI_MED\pipeline2\outputs\medyolo_labels_set7\0035_Stack14.txt",
        start_layer=2,
        allowed_class_ids=[0],
        min_layers_visible=1,
        max_slice_gap=1,
        iou_thresh=0.05,
        center_dist_thresh=80.0
    )
    # ---------- 批量 ----------
    # batch_convert_yolo2d_to_medyolo3d(
    #     nii_root=r"D:\AI_MED\NII_Output",
    #     yolo2d_label_root=r"D:\AI_MED\YOLO2D_Labels",
    #     medyolo_label_root=r"D:\AI_MED\MedYOLO_Labels",
    #     start_layer=2,
    #     allowed_class_ids=[0],
    #     min_layers_visible=1,
    #     max_slice_gap=1,
    #     iou_thresh=0.1,
    #     center_dist_thresh=25.0
    # )