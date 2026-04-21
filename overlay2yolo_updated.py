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


def contour_fill_ratio(cnt, w, h):
    rect_area = max(w * h, 1)
    cnt_area = cv2.contourArea(cnt)
    return cnt_area / rect_area


def process_one_image(
    overlay_path,
    out_label_path,
    out_vis_path=None,
    h_min=25,
    h_max=35,
    s_min=150,
    s_max=255,
    v_min=150,
    v_max=255,
    min_area=20,
    max_area_ratio=0.2,
    merge_gap=8,
    pad=3,
    kernel_size=3,
    dilate_iter=2,
    close_iter=1,
    open_iter=1,
    min_w=3,
    min_h=3,
    max_aspect_ratio=6.0,
    border_margin=0,
    min_fill_ratio=0.02,
    cls_id=0,
    save_mask_path=None,
):
    overlay = cv2.imread(str(overlay_path))
    if overlay is None:
        raise ValueError(f"Cannot read overlay image: {overlay_path}")

    img_h, img_w = overlay.shape[:2]

    # 1) yellow color threshold in HSV
    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_yellow = np.array([h_max, s_max, v_max], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2) morphology for thin yellow contour
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    if save_mask_path is not None:
        ensure_dir(save_mask_path.parent)
        cv2.imwrite(str(save_mask_path), mask)

    # 3) contours -> boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = img_w * img_h * max_area_ratio
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < min_w or h < min_h:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > max_aspect_ratio:
            continue

        fill_ratio = contour_fill_ratio(cnt, w, h)
        if fill_ratio < min_fill_ratio:
            continue

        # optional: skip objects touching extreme image borders
        # if (
         #    x <= border_margin or
        #     y <= border_margin or
        #     x + w >= img_w - border_margin or
            y + h >= img_h - border_margin
        # ):
         #    continue

        x1 = x - pad
        y1 = y - pad
        x2 = x + w + pad
        y2 = y + h + pad
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    # 4) merge nearby boxes
    boxes = merge_close_boxes(boxes, merge_gap=merge_gap)

    # 5) write YOLO txt
    ensure_dir(out_label_path.parent)
    lines = [yolo_line_from_xyxy(*b, img_w, img_h, cls_id=cls_id) for b in boxes]
    with open(out_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 6) visualization
    if out_vis_path is not None:
        ensure_dir(out_vis_path.parent)
        vis = overlay.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out_vis_path), vis)

    return len(boxes)


def main():
    parser = argparse.ArgumentParser(
        description="Convert overlay PNG annotations to YOLO txt using yellow-color extraction."
    )

    parser.add_argument("--root_dir", required=True, help='Example: "D:\\AI_MED\\Set 7\\Set 7"')
    parser.add_argument("--output_label_dir", required=True, help='Example: "D:\\AI_MED\\generated_labels"')
    parser.add_argument("--output_vis_dir", default=None, help='Example: "D:\\AI_MED\\generated_vis"')
    parser.add_argument("--output_mask_dir", default=None, help='Optional mask output for debugging')

    # HSV range around your sampled yellow: HSV = [30, 255, 255]
    parser.add_argument("--h_min", type=int, default=25, help="HSV hue lower bound for yellow")
    parser.add_argument("--h_max", type=int, default=35, help="HSV hue upper bound for yellow")
    parser.add_argument("--s_min", type=int, default=150, help="HSV saturation lower bound")
    parser.add_argument("--s_max", type=int, default=255, help="HSV saturation upper bound")
    parser.add_argument("--v_min", type=int, default=150, help="HSV value lower bound")
    parser.add_argument("--v_max", type=int, default=255, help="HSV value upper bound")

    parser.add_argument("--min_area", type=int, default=20, help="Minimum contour area")
    parser.add_argument("--max_area_ratio", type=float, default=0.2, help="Maximum contour area as image fraction")
    parser.add_argument("--merge_gap", type=int, default=8, help="Merge boxes if they are close")
    parser.add_argument("--pad", type=int, default=3, help="Extra padding around each bbox")
    parser.add_argument("--kernel_size", type=int, default=3, help="Morphology kernel size")
    parser.add_argument("--dilate_iter", type=int, default=2, help="Dilation iterations for thin yellow contours")
    parser.add_argument("--close_iter", type=int, default=1, help="Close iterations")
    parser.add_argument("--open_iter", type=int, default=1, help="Open iterations")

    parser.add_argument("--min_w", type=int, default=3, help="Minimum bbox width")
    parser.add_argument("--min_h", type=int, default=3, help="Minimum bbox height")
    parser.add_argument("--max_aspect_ratio", type=float, default=6.0, help="Reject overly thin / elongated boxes")
    parser.add_argument("--border_margin", type=int, default=1, help="Reject contours touching border within this margin")
    parser.add_argument("--min_fill_ratio", type=float, default=0.02, help="Reject very hollow / line-like contours")
    parser.add_argument("--cls_id", type=int, default=0, help="YOLO class id")

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_label_dir = Path(args.output_label_dir)
    output_vis_dir = Path(args.output_vis_dir) if args.output_vis_dir else None
    output_mask_dir = Path(args.output_mask_dir) if args.output_mask_dir else None

    print("RUNNING overlay2yolo.py YELLOW VERSION")
    print("DEBUG root_dir =", root_dir)
    print("DEBUG absolute =", root_dir.resolve())
    print("DEBUG exists =", root_dir.exists())
    print("DEBUG is_dir =", root_dir.is_dir())
    print("DEBUG yellow hsv =", (args.h_min, args.h_max, args.s_min, args.s_max, args.v_min, args.v_max))

    overlay_paths = find_overlay_pngs(root_dir)
    if not overlay_paths:
        raise ValueError("No overlay PNG files found under folders containing 'Overlays'.")

    total_boxes = 0
    total_images = 0

    for overlay_path in overlay_paths:
        rel_path = overlay_path.relative_to(root_dir)

        out_label_path = output_label_dir / rel_path.with_suffix(".txt")
        out_vis_path = output_vis_dir / rel_path if output_vis_dir else None
        #out_mask_path = output_mask_dir / rel_path if output_mask_dir else None

        n = process_one_image(
            overlay_path=overlay_path,
            out_label_path=out_label_path,
            out_vis_path=out_vis_path,
            h_min=args.h_min,
            h_max=args.h_max,
            s_min=args.s_min,
            s_max=args.s_max,
            v_min=args.v_min,
            v_max=args.v_max,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            merge_gap=args.merge_gap,
            pad=args.pad,
            kernel_size=args.kernel_size,
            dilate_iter=args.dilate_iter,
            close_iter=args.close_iter,
            open_iter=args.open_iter,
            min_w=args.min_w,
            min_h=args.min_h,
            max_aspect_ratio=args.max_aspect_ratio,
            border_margin=args.border_margin,
            min_fill_ratio=args.min_fill_ratio,
            cls_id=args.cls_id,
            #save_mask_path=out_mask_path,
        )

        total_boxes += n
        total_images += 1
        print(f"{rel_path} -> {n} boxes")

    print(f"Done. Images processed: {total_images}")
    print(f"Done. Total boxes: {total_boxes}")


if __name__ == "__main__":
    main()