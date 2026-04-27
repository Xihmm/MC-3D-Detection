import torch
import numpy as np

from models3D.yolov8_3d_model import build_yolov8_3d
from utils3D.dataset import MC3DDataset, collate_fn
from torch.utils.data import DataLoader
from utils3D.nms import nms_3d_safe


# =========================
# Config
# =========================
CKPT_PATH = r"runs/train_yolov8_3d_one_stack/best.pt"

IMAGE_DIR = r"D:/AI_MED/one_stack_dataset/images/train"
LABEL_DIR = r"D:/AI_MED/one_stack_dataset/labels/train"

IMG_SIZE = (33, 320, 320)  # D, H, W
NC = 1
TOPK = 20
CONF_THRES = 0.05


# =========================
# Helpers
# =========================
def zxydwh_to_zxyzxy(box):
    """
    box: [..., 6] = z, x, y, d, w, h
    return: [..., 6] = z1, x1, y1, z2, x2, y2
    """
    z, x, y, d, w, h = box.unbind(-1)

    z1 = z - d / 2
    x1 = x - w / 2
    y1 = y - h / 2
    z2 = z + d / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return torch.stack([z1, x1, y1, z2, x2, y2], dim=-1)


def box_iou_3d(box1, box2):
    """
    box1: [N, 6] z1 x1 y1 z2 x2 y2
    box2: [M, 6]
    return: [N, M]
    """
    b1 = box1[:, None, :]
    b2 = box2[None, :, :]

    inter_z1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_x1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_y1 = torch.maximum(b1[..., 2], b2[..., 2])

    inter_z2 = torch.minimum(b1[..., 3], b2[..., 3])
    inter_x2 = torch.minimum(b1[..., 4], b2[..., 4])
    inter_y2 = torch.minimum(b1[..., 5], b2[..., 5])

    inter_d = (inter_z2 - inter_z1).clamp(min=0)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)

    inter = inter_d * inter_w * inter_h

    vol1 = (
        (box1[:, 3] - box1[:, 0]).clamp(min=0)
        * (box1[:, 4] - box1[:, 1]).clamp(min=0)
        * (box1[:, 5] - box1[:, 2]).clamp(min=0)
    )

    vol2 = (
        (box2[:, 3] - box2[:, 0]).clamp(min=0)
        * (box2[:, 4] - box2[:, 1]).clamp(min=0)
        * (box2[:, 5] - box2[:, 2]).clamp(min=0)
    )

    union = vol1[:, None] + vol2[None, :] - inter + 1e-6

    return inter / union


def decode_outputs(outputs, img_size):
    """
    Decode native YOLOv8-3D outputs into pixel-space boxes.

    outputs:
      list of:
        box: [B, 6, Dz, Dy, Dx]
        obj: [B, 1, Dz, Dy, Dx]
        cls: [B, nc, Dz, Dy, Dx]

    return:
      pred: [N, 7] = z, x, y, d, w, h, conf
    """
    D, H, W = img_size
    all_preds = []

    for out in outputs:
        box = out["box"]
        obj = out["obj"]
        cls = out["cls"]

        B, _, Dz, Dy, Dx = box.shape
        assert B == 1, "debug script assumes batch size 1"

        stride_z = D / Dz
        stride_y = H / Dy
        stride_x = W / Dx

        box = torch.sigmoid(box[0])  # [6, Dz, Dy, Dx]
        obj = torch.sigmoid(obj[0, 0])  # [Dz, Dy, Dx]
        cls_score = torch.sigmoid(cls[0, 0])  # [Dz, Dy, Dx]

        conf = obj * cls_score

        zz, yy, xx = torch.meshgrid(
            torch.arange(Dz, device=box.device),
            torch.arange(Dy, device=box.device),
            torch.arange(Dx, device=box.device),
            indexing="ij"
        )

        z = (zz + box[0]) * stride_z
        x = (xx + box[1]) * stride_x
        y = (yy + box[2]) * stride_y

        # sizes are normalized to full volume, so convert to pixels
        d = box[3] * D
        w = box[4] * W
        h = box[5] * H

        pred = torch.stack([z, x, y, d, w, h, conf], dim=-1)
        pred = pred.reshape(-1, 7)

        all_preds.append(pred)

    all_preds = torch.cat(all_preds, dim=0)

    d = all_preds[:, 3]
    w = all_preds[:, 4]
    h = all_preds[:, 5]
    conf = all_preds[:, 6]

    keep = (
        (conf > CONF_THRES) &
        (d >= 3) & (d <= 20) &
        (w >= 10) & (w <= 70) &
        (h >= 10) & (h <= 70)
    )
    all_preds = all_preds[keep]

    return all_preds


def targets_to_pixel_boxes(targets, img_size):
    """
    targets: [N, 8] = batch class z x y d w h normalized
    return:
      [N, 6] zxydwh pixel
    """
    D, H, W = img_size

    if targets.numel() == 0:
        return torch.zeros((0, 6), device=targets.device)

    t = targets[:, 2:8].clone()
    t[:, 0] *= D
    t[:, 1] *= W
    t[:, 2] *= H
    t[:, 3] *= D
    t[:, 4] *= W
    t[:, 5] *= H

    return t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = MC3DDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = build_yolov8_3d(nc=NC).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    imgs, targets, paths = next(iter(loader))
    imgs = imgs.to(device)
    targets = targets.to(device)

    print("Image:", paths[0])
    print("GT targets:", targets.shape)

    with torch.no_grad():
        outputs = model(imgs)

    preds = decode_outputs(outputs, IMG_SIZE)
    print("Before NMS:", preds.shape)
    preds = nms_3d_safe(
        preds,
        iou_thres=0.3,
        conf_thres=CONF_THRES,
        max_det=50
    )
    print("After NMS:", preds.shape)

    print("Predictions after conf filter:", preds.shape)

    if preds.numel() == 0:
        print("No predictions above CONF_THRES.")
        return

    preds = preds[torch.argsort(preds[:, 6], descending=True)]

    gt_zxydwh = targets_to_pixel_boxes(targets, IMG_SIZE)
    gt_zxyzxy = zxydwh_to_zxyzxy(gt_zxydwh)

    pred_zxydwh = preds[:, :6]
    pred_zxyzxy = zxydwh_to_zxyzxy(pred_zxydwh)

    ious = box_iou_3d(pred_zxyzxy, gt_zxyzxy)
    best_iou_per_pred = ious.max(dim=1).values

    print("\nTop predictions:")
    for i in range(min(TOPK, preds.shape[0])):
        z, x, y, d, w, h, conf = preds[i].tolist()
        print(
            f"{i:02d}: conf={conf:.4f}, "
            f"bestIoU={best_iou_per_pred[i].item():.4f}, "
            f"zxy=({z:.1f}, {x:.1f}, {y:.1f}), "
            f"dwh=({d:.1f}, {w:.1f}, {h:.1f})"
        )

    best_iou_all, best_idx = best_iou_per_pred.max(dim=0)
    print("\nBest IoU prediction:")
    print("best IoU:", best_iou_all.item())
    print("prediction:", preds[best_idx].detach().cpu().numpy())

    print("\nGT boxes pixel zxydwh:")
    print(gt_zxydwh.detach().cpu().numpy())


if __name__ == "__main__":
    main()