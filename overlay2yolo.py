import cv2
import argparse
import numpy as np
from pathlib import Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def yolo_line_from_xyxy(x1, y1, x2, y2, img_w, img_h, cls_id=0):
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def clamp_box(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w - 1))
    y2 = max(0, min(int(y2), img_h - 1))
    return x1, y1, x2, y2


def merge_close_boxes(boxes, merge_gap=8):
    """
    Merge nearby overlapping boxes.
    Useful when one hand-drawn ROI becomes multiple small contours.
    """
    if not boxes:
        return []

    merged = True
    boxes = boxes[:]

    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            x1, y1, x2, y2 = boxes[i]
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue

                    a1, b1, a2, b2 = boxes[j]

                    # overlap or almost-touch condition
                    if not (
                        a1 > x2 + merge_gap or
                        a2 < x1 - merge_gap or
                        b1 > y2 + merge_gap or
                        b2 < y1 - merge_gap
                    ):
                        x1 = min(x1, a1)
                        y1 = min(y1, b1)
                        x2 = max(x2, a2)
                        y2 = max(y2, b2)
                        used[j] = True
                        changed = True
                        merged = True

            new_boxes.append((x1, y1, x2, y2))

        boxes = new_boxes

    return boxes


def find_overlay_pngs(root_dir):
    """
    Find all PNGs under folders whose names contain 'Overlays'.
    Example:
    D:/AI_MED/Set 7/Set 7/0035_Stack6_Overlays_PNG/VivaStack #60031.png
    """
    root = Path(root_dir)

    if not root.exists():
        raise ValueError(f"root_dir does not exist: {root}")

    all_files = list(root.rglob("*"))
    all_pngs = [p for p in all_files if p.is_file() and p.suffix.lower() == ".png"]

    png_paths = []
    for path in all_pngs:
        parent_parts = [part.lower() for part in path.parts]
        if any("overlays" in part for part in parent_parts):
            png_paths.append(path)

    print(f"DEBUG total files found = {len(all_files)}")
    print(f"DEBUG total png found = {len(all_pngs)}")
    print(f"DEBUG overlay png found = {len(png_paths)}")
    for p in png_paths[:10]:
        print("DEBUG sample overlay png:", p)

    return sorted(png_paths)


from pathlib import Path

def get_raw_path_from_overlay(root_dir, overlay_path):
    rel_path = overlay_path.relative_to(root_dir)

    # 1) overlay folder -> raw folder
    raw_folder = str(rel_path.parent).replace("_Overlays_PNG", "_Raw")

    # 2) filename: VivaStack #60031.png -> v0000031.bmp
    stem = overlay_path.stem  # e.g. "VivaStack #60031"
    number = stem.split("#")[-1].strip()  # "60031"

    # take last 4 digits: 0031
    frame_id = number[-4:]

    raw_name = f"v000{frame_id}.bmp"

    raw_path = Path(root_dir) / raw_folder / raw_name
    return raw_path


def process_one_image(
    overlay_path,
    raw_path,
    out_label_path,
    out_vis_path=None,
    diff_thresh=25,
    min_area=20,
    max_area_ratio=0.2,
    merge_gap=8,
    pad=3,
    kernel_size=3,
    cls_id=0,
):
    overlay = cv2.imread(str(overlay_path))
    raw = cv2.imread(str(raw_path))

    if overlay is None:
        raise ValueError(f"Cannot read overlay image: {overlay_path}")
    if raw is None:
        raise ValueError(f"Cannot read raw image: {raw_path}")

    if overlay.shape != raw.shape:
        raise ValueError(
            f"Shape mismatch:\nOverlay: {overlay_path} -> {overlay.shape}\nRaw: {raw_path} -> {raw.shape}"
        )

    img_h, img_w = overlay.shape[:2]

    # 1) overlay - raw
    diff = cv2.absdiff(overlay, raw)

    # 2) grayscale threshold
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, diff_thresh, 255, cv2.THRESH_BINARY)

    # 3) morphology cleanup
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4) contours -> boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = img_w * img_h * max_area_ratio
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x1 = x - pad
        y1 = y - pad
        x2 = x + w + pad
        y2 = y + h + pad
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    # 5) merge nearby boxes
    boxes = merge_close_boxes(boxes, merge_gap=merge_gap)

    # 6) write YOLO txt
    ensure_dir(out_label_path.parent)
    lines = [yolo_line_from_xyxy(*b, img_w, img_h, cls_id=cls_id) for b in boxes]
    with open(out_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 7) visualization
    if out_vis_path is not None:
        ensure_dir(out_vis_path.parent)
        vis = overlay.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out_vis_path), vis)

    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description="Convert overlay PNG annotations to YOLO txt using overlay-raw difference.")
    parser.add_argument("--root_dir", required=True, help='Example: "D:\\AI_MED\\Set 7\\Set 7"')
    parser.add_argument("--output_label_dir", required=True, help='Example: "D:\\AI_MED\\generated_labels"')
    parser.add_argument("--output_vis_dir", default=None, help='Example: "D:\\AI_MED\\generated_vis"')
    parser.add_argument("--diff_thresh", type=int, default=25, help="Threshold on overlay-raw grayscale diff")
    parser.add_argument("--min_area", type=int, default=20, help="Minimum contour area")
    parser.add_argument("--max_area_ratio", type=float, default=0.2, help="Maximum contour area as image fraction")
    parser.add_argument("--merge_gap", type=int, default=8, help="Merge boxes if they are close")
    parser.add_argument("--pad", type=int, default=3, help="Extra padding around each bbox")
    parser.add_argument("--kernel_size", type=int, default=3, help="Morphology kernel size")
    parser.add_argument("--cls_id", type=int, default=0, help="YOLO class id")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_label_dir = Path(args.output_label_dir)
    output_vis_dir = Path(args.output_vis_dir) if args.output_vis_dir else None

    print("RUNNING overlay2yolo.py DIFF VERSION")
    print("DEBUG root_dir =", root_dir)
    print("DEBUG absolute =", root_dir.resolve())
    print("DEBUG exists =", root_dir.exists())
    print("DEBUG is_dir =", root_dir.is_dir())

    overlay_paths = find_overlay_pngs(root_dir)
    if not overlay_paths:
        raise ValueError("No overlay PNG files found under folders containing 'Overlays'.")

    total_boxes = 0
    total_images = 0
    missing_raw = 0

    for overlay_path in overlay_paths:
        rel_path = overlay_path.relative_to(root_dir)
        raw_path = get_raw_path_from_overlay(root_dir, overlay_path)

        out_label_path = output_label_dir / rel_path.with_suffix(".txt")
        out_vis_path = output_vis_dir / rel_path if output_vis_dir else None

        if not raw_path.exists():
            print(f"WARNING: raw image not found for {overlay_path}")
            print(f"         expected raw path: {raw_path}")
            missing_raw += 1
            continue

        n = process_one_image(
            overlay_path=overlay_path,
            raw_path=raw_path,
            out_label_path=out_label_path,
            out_vis_path=out_vis_path,
            diff_thresh=args.diff_thresh,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            merge_gap=args.merge_gap,
            pad=args.pad,
            kernel_size=args.kernel_size,
            cls_id=args.cls_id,
        )

        total_boxes += n
        total_images += 1
        print(f"{rel_path} -> {n} boxes")

    print(f"Done. Images processed: {total_images}")
    print(f"Done. Missing raw pairs: {missing_raw}")
    print(f"Done. Total boxes: {total_boxes}")


if __name__ == "__main__":
    main()