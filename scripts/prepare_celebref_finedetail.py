import argparse
import csv
import os
import re
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def image_files(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])


def identity_from_name(name):
    stem = Path(name).stem
    stem = re.sub(r"\s*\([^)]*\)", "", stem)
    stem = re.sub(r"[-_ ]*\d+$", "", stem)
    stem = re.sub(r"\s+", " ", stem.replace("-", " ")).strip()
    return stem or Path(name).stem


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def add_subset(rows, source_dir, dest_root, split, variant, paired_names=None):
    paired_names = paired_names or set()
    for src in image_files(source_dir):
        identity = identity_from_name(src.name)
        rel = Path(variant) / identity / src.name
        copy_file(src, dest_root / rel)
        rows.append(
            {
                "split": split,
                "variant": variant,
                "identity": identity,
                "file_name": src.name,
                "path": str(rel).replace(os.sep, "/"),
                "has_pair": "yes" if src.name in paired_names else "no",
            }
        )


def write_readme(dest_root):
    text = """---
license: cc-by-nc-sa-4.0
task_categories:
- image-to-image
tags:
- face-restoration
- reference-based-restoration
- fine-details
- celebrity-faces
pretty_name: CelebRef-FineDetail
---

# CelebRef-FineDetail

CelebRef-FineDetail is a small reference-based face restoration evaluation set curated around identity-specific fine details. The images emphasize details that generic face restoration models often hallucinate or erase, such as moles, freckles, tattoos, distinctive eyebrows, facial hair, scars, piercings, and other local attributes that should be recovered from a same-identity reference image.

The dataset is organized into:

- `raw/`: original collected images.
- `aligned/`: 256x256 aligned images used by the RefineFIR inference pipeline when available.
- `smallYC_raw/` and `fine_detail_raw_extra/`: additional small raw-image subsets.
- `metadata.csv`: identity, file name, variant, and pair availability.

Images were collected from public internet sources for non-commercial research. The dataset is intended for evaluating reference-based restoration behavior rather than for identity recognition or biometric use. The images are not owned by the dataset authors; users are responsible for complying with applicable rights, privacy, and platform terms.
"""
    (dest_root / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--small-yc")
    parser.add_argument("--raw-extra")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dest_root = Path(args.output)
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    raw_names = {p.name for p in image_files(args.raw)}
    aligned_names = {p.name for p in image_files(args.aligned)}
    rows = []
    add_subset(rows, args.raw, dest_root, "main", "raw", paired_names=aligned_names)
    add_subset(rows, args.aligned, dest_root, "main", "aligned", paired_names=raw_names)
    if args.small_yc:
        add_subset(rows, args.small_yc, dest_root, "smallYC", "smallYC_raw")
    if args.raw_extra:
        add_subset(rows, args.raw_extra, dest_root, "extra", "fine_detail_raw_extra")

    with (dest_root / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "variant", "identity", "file_name", "path", "has_pair"])
        writer.writeheader()
        writer.writerows(rows)

    write_readme(dest_root)
    print(dest_root)
    print("rows", len(rows))


if __name__ == "__main__":
    main()

