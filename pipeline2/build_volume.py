import os
import cv2
import numpy as np
import nibabel as nib


def load_slices(stack_dir):
    """
    读取 stack 内所有 slice，并按顺序排序
    """
    files = [f for f in os.listdir(stack_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]

    # ⚠️ 按数字排序（非常重要！）
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    slices = []
    for f in files:
        path = os.path.join(stack_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: cannot read {f}")
            continue

        slices.append(img)

    return slices


def apply_starting_layer_filter(slices, start_layer=2):
    """
    去掉前几层（默认去掉 layer 0,1 → 从第2层开始）
    """
    return slices[start_layer:]


def apply_clahe(slices):
    """
    对每个 slice 做 CLAHE 增强（推荐）
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced = []
    for img in slices:
        img_clahe = clahe.apply(img)
        enhanced.append(img_clahe)

    return enhanced


def stack_to_volume(slices):
    """
    stack 成 3D volume
    输出 shape: (H, W, D)
    """
    volume = np.stack(slices, axis=-1)
    return volume


def normalize(volume):
    """
    简单归一化到 [0,1]
    """
    volume = volume.astype(np.float32)
    volume = volume / 255.0
    return volume


def save_nifti(volume, save_path):
    """
    保存为 .nii.gz
    """
    affine = np.eye(4)  # identity matrix
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, save_path)


def convert_stack_to_nifti(
    stack_dir,
    save_path,
    start_layer=2,
    use_clahe=True
):
    """
    主函数
    """
    print(f"Processing: {stack_dir}")

    # 1. 读 slices
    slices = load_slices(stack_dir)
    print(f"Total slices: {len(slices)}")

    # 2. starting layer filter
    slices = apply_starting_layer_filter(slices, start_layer)
    print(f"After layer filter: {len(slices)}")

    # 3. enhancement
    if use_clahe:
        slices = apply_clahe(slices)

    # 4. stack
    volume = stack_to_volume(slices)
    print(f"Volume shape: {volume.shape}")  # (H, W, D)

    # 5. normalize
    volume = normalize(volume)

    # 6. save
    save_nifti(volume, save_path)

    print(f"Saved to: {save_path}")


# ==========================
# 🧪 直接运行测试
# ==========================
if __name__ == "__main__":
    convert_stack_to_nifti(
        stack_dir=r"D:\AI_MED\Set 9-selected\1897_Stack4_Raw",
        save_path=r"D:\AI_MED\1897_Stack4.nii.gz",
        start_layer=2,     # 对应 proposal: 从 layer 3 开始
        use_clahe=True
    )