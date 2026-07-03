# RefineFIR

This repository is for the code release of **Copy or Not? Reference-Based Face Image Restoration with Fine Details**.

The repository mentioned in the paper, [`RefineFIR/RefineFIR`](https://github.com/RefineFIR/RefineFIR), is currently not accessible. We plan to place the code here over the coming period.

Due to several constraints, including the original implementation no longer being accessible and productization-related changes, the code released here will be a reimplementation produced within limited time. It may therefore differ from the original research implementation.

- [Paper](https://jianwang-cmu.github.io/25Refine/main.pdf)
- [Poster](https://jianwang-cmu.github.io/25Refine/wacv25-2727-poster.pdf)
- [Video](https://www.dropbox.com/scl/fi/kzvj2aw95wiaxa3mk2fpg/Copy-or-not-WACV.mp4?rlkey=jq4bi8698zdm5u283lltb86yw&st=m66pom5j&dl=0)

## Status

The historical internal inference path used Snap/XCV face warping, which is not included here. This public version uses a MediaPipe Face Mesh based warp fallback. In our internal checks, the XCV warp is usually stronger, but the MediaPipe path is easier to release and can run without proprietary dependencies.

## Installation

```bash
conda create -n refinefir python=3.10 -y
conda activate refinefir
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your machine if the default `torch` wheel is not appropriate.

## Checkpoints

Put the generator checkpoint and ArcFace identity encoder in `weights/`:

```text
weights/latest.pt
weights/model_ir_se50.pth
weights/stylegan2-256-550000.pt
```

The release checkpoint is available on Hugging Face:

- [JianWang1/RefineFIR-MediaPipe](https://huggingface.co/JianWang1/RefineFIR-MediaPipe)

## Inference

The input and reference should be aligned face images. The script optionally synthesizes a low-quality input using the training degradation before restoration.

```bash
python infer_mediapipe.py \
  --input examples/input.png \
  --reference examples/reference.png \
  --checkpoint weights/latest.pt \
  --id-weight weights/model_ir_se50.pth \
  --output outputs/restored.png \
  --save-debug
```

Use `--skip-degrade` if your input image is already degraded and should be restored directly.

## Training

The training code is split into:

- `training_dataset.py`: loads clean same-identity image pairs and precomputed warped references.
- `train.py`: trains RefineFIR using GAN, LPIPS, identity, and degradation-cycle losses.

The historical training data uses precomputed Snap/XCV warped references. The public inference path uses MediaPipe, but this training artifact does not require releasing Snap/XCV code because the warped images are already generated.

Training artifacts are available on Hugging Face:

- [JianWang1/RefineFIR-TrainingData](https://huggingface.co/datasets/JianWang1/RefineFIR-TrainingData)

Example:

```bash
python train.py \
  --image-root data/CelebHQRefForRelease \
  --pair-list data/celebref_list_256_jw_0403.pkl \
  --warp-root data/celebaref_warped_snapCV_jw_0403/celebaref_warped_snapCV_jw_0403 \
  --stylegan-weight weights/stylegan2-256-550000.pt \
  --id-weight weights/model_ir_se50.pth \
  --output checkpoints/refinefir_256 \
  --iter 800000 \
  --batch 4
```

To continue from a released checkpoint, add `--ckpt weights/latest.pt` and set `--iter` to the target total iteration count.

## CelebRef-FineDetail

We also curate a small evaluation dataset, **CelebRef-FineDetail**, for reference-based restoration of identity-specific fine details. It contains public internet face images with attributes such as moles, freckles, tattoos, distinctive eyebrows, facial hair, scars, and piercings. The goal is to test whether the restoration can copy details from a same-identity reference image instead of hallucinating them.

The dataset layout follows:

```text
raw/
aligned/
smallYC_raw/
fine_detail_raw_extra/
metadata.csv
```

The dataset is available on Hugging Face:

- [JianWang1/CelebRef-FineDetail](https://huggingface.co/datasets/JianWang1/CelebRef-FineDetail)

## Citation

```bibtex
@inproceedings{chong2025copy,
  title={Copy or not? reference-based face image restoration with fine details},
  author={Chong, Min Jin and Xu, Dejia and Zhang, Yi and Wang, Zhangyang and Forsyth, David and Krishnan, Gurunandan and Wu, Yicheng and Wang, Jian},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={9660--9669},
  year={2025},
  organization={IEEE}
}
```
