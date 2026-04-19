import os
import re
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib


def extract_number(filename):
    nums = re.findall(r'\d+', filename)
    if not nums:
        return -1
    return int(nums[-1])


def load_slices(stack_dir):
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = [f for f in os.listdir(stack_dir) if f.lower().endswith(valid_exts)]

    if len(files) == 0:
        raise ValueError(f"No image files found in {stack_dir}")

    files.sort(key=extract_number)

    slices = []
    used_files = []

    for f in files:
        path = os.path.join(stack_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: cannot read {path}")
            continue
        slices.append(img)
        used_files.append(f)

    if len(slices) == 0:
        raise ValueError(f"All image files failed to load in {stack_dir}")

    h0, w0 = slices[0].shape
    for i, img in enumerate(slices):
        if img.shape != (h0, w0):
            raise ValueError(
                f"Slice size mismatch in {stack_dir}: "
                f"{used_files[i]} has shape {img.shape}, expected {(h0, w0)}"
            )

    return slices, used_files


def apply_starting_layer_filter(slices, start_layer=2):
    if start_layer >= len(slices):
        raise ValueError(
            f"start_layer={start_layer} is too large for {len(slices)} slices"
        )
    return slices[start_layer:]


def apply_clahe(slices, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return [clahe.apply(img) for img in slices]


def stack_to_volume(slices):
    return np.stack(slices, axis=-1)   # (H, W, D)


def normalize(volume):
    return volume.astype(np.float32) / 255.0


def save_nifti(volume, save_path):
    if volume.size == 0:
        raise ValueError("volume is empty")
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, str(save_path))

    if not os.path.exists(save_path):
        raise IOError(f"File was not created: {save_path}")

    file_size = os.path.getsize(save_path)
    if file_size == 0:
        raise IOError(f"Saved file is 0 bytes: {save_path}")


def convert_stack_to_nifti(
    stack_dir,
    save_path,
    start_layer=2,
    use_clahe=True,
    clip_limit=2.0
):
    print(f"\nProcessing: {stack_dir}")

    slices, used_files = load_slices(stack_dir)
    print(f"  Loaded slices: {len(slices)}")
    print(f"  First file: {used_files[0]}")
    print(f"  Last file : {used_files[-1]}")

    slices = apply_starting_layer_filter(slices, start_layer=start_layer)
    print(f"  After layer filter: {len(slices)}")

    if len(slices) == 0:
        raise ValueError("No slices left after starting layer filter.")

    if use_clahe:
        slices = apply_clahe(slices, clip_limit=clip_limit)

    volume = stack_to_volume(slices)
    print(f"  Volume shape: {volume.shape}")

    volume = normalize(volume)
    save_nifti(volume, save_path)

    print(f"  Saved to: {save_path}")
    print(f"  File size: {os.path.getsize(save_path)} bytes")


def make_output_name(stack_folder_name):
    """
    例如:
    1897_Stack4_Raw -> 1897_Stack4.nii.gz
    """
    name = stack_folder_name
    if name.endswith("_Raw"):
        name = name[:-4]
    return f"{name}.nii.gz"


def batch_convert_stacks(
    input_root,
    output_root,
    folder_keyword="_Raw",
    start_layer=2,
    use_clahe=True,
    clip_limit=2.0,
    overwrite=False
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stack_dirs = [
        p for p in input_root.iterdir()
        if p.is_dir() and folder_keyword in p.name
    ]

    if len(stack_dirs) == 0:
        print(f"No stack folders found under: {input_root}")
        return

    print(f"Found {len(stack_dirs)} stack folders.")

    success = 0
    failed = 0
    failed_list = []

    for stack_dir in sorted(stack_dirs):
        output_name = make_output_name(stack_dir.name)
        save_path = output_root / output_name

        if save_path.exists() and not overwrite:
            print(f"\nSkipping (already exists): {save_path}")
            continue

        try:
            convert_stack_to_nifti(
                stack_dir=str(stack_dir),
                save_path=str(save_path),
                start_layer=start_layer,
                use_clahe=use_clahe,
                clip_limit=clip_limit
            )
            success += 1

        except Exception as e:
            failed += 1
            failed_list.append((stack_dir.name, str(e)))
            print(f"  Failed: {stack_dir.name}")
            print(f"  Error : {e}")

    print("\n====================")
    print("Batch conversion done")
    print("====================")
    print(f"Success: {success}")
    print(f"Failed : {failed}")

    if failed_list:
        print("\nFailed folders:")
        for name, err in failed_list:
            print(f"- {name}: {err}")


if __name__ == "__main__":
    batch_convert_stacks(
        input_root=r"D:\AI_MED\Set 7\Set 7",
        output_root=r"D:\AI_MED\NII_Output_set7",
        folder_keyword="_Raw",
        start_layer=2,
        use_clahe=True,
        clip_limit=2.0,
        overwrite=False
    )