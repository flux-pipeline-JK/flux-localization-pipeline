# FLUX Localization Pipeline
Automatic Object Transformation in Image via FLUX Inpainting with Multi-LoRA Condition Decomposition

<img width="1280" height="720" alt="pipeline" src="https://github.com/user-attachments/assets/c423934a-5fcf-4ca2-8542-b021fe7e9500" />

---

## Overview

This repository provides the official implementation of:

> **FLUX 인페인팅과 다중 LoRA 조건 분리를 이용한 영상 객체 자동 변환**

The proposed framework automatically transforms visual objects in images and video frames while preserving:

- background consistency
- object identity
- lighting coherence
- spatial alignment

Our method combines:

- Grounding DINO for object localization
- Segment Anything Model (SAM) for precise mask extraction
- FLUX Inpainting
- Multi-LoRA conditional decomposition

The pipeline separates:

- Fill condition (background)
- Subject condition (reference object)
- Denoising condition (generation stabilization)

to achieve robust object-level localization and replacement.

---

## Pipeline

The framework consists of four stages:

1. Object Localization (Grounding DINO)
2. Mask Extraction (SAM)
3. Condition Construction
4. FLUX Multi-LoRA Inpainting

### Multi-LoRA Composition

\[
\Delta \theta_{LoRA}
=
\Delta \theta_{fill}
+
\Delta \theta_{subj}
+
\Delta \theta_{denoise}
\]

- Fill LoRA:
  preserves scene-level texture and lighting

- Subject LoRA:
  preserves object identity and fine details

- Denoising LoRA:
  stabilizes iterative diffusion generation

---

## Features

- Automatic object localization
- Precise object segmentation
- Object-aware inpainting
- Multi-condition controllable generation
- Background-preserving object replacement
- Visual localization for media contents
- Robust under occlusion conditions

---

## Installation

Clone the repository:

```bash
git clone https://github.com/flux-pipeline-JK/flux-localization-pipeline.git

cd flux-localization-pipeline
```

Create environment:

```bash
conda create -n flux-localization python=3.10

conda activate flux-localization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```bash
torch>=2.0.0
torchvision
diffusers
transformers
accelerate
opencv-python
numpy
pillow
matplotlib
segment-anything
groundingdino-py
einops
safetensors
tqdm
```

---

## Project Structure

```bash
flux-localization-pipeline/
│
├── inference.py
├── prepare_inputs.py
├── force_composite.py
├── requirements.txt
│
├── src/
│   ├── UniCombinePipeline.py
│   ├── UniCombineTransformer2DModel.py
│   ├── UniCombineTransformerBlock.py
│   ├── condition.py
│   ├── dataloader.py
│   ├── hook.py
│   ├── text_encoder.py
│   └── lora_switching_module.py
│
├── examples/
│
└── demo_Condition_LoRA/
```

---

## Inference

Run object transformation:

```bash
python inference.py
```

Example workflow:

1. Detect target object using Grounding DINO
2. Generate segmentation mask using SAM
3. Construct:
   - background condition
   - subject condition
4. Generate transformed object using FLUX inpainting

---

## Experimental Settings

- Resolution: 512 × 512
- Diffusion steps: 6
- GPU: NVIDIA RTX 4090
- OS: Ubuntu 22.04

Recommended LoRA weights:

| LoRA Type | Recommended Weight |
|---|---|
| Fill | 0.6 ~ 0.9 |
| Subject | 1.0 ~ 1.4 |
| Denoising | 1.0 |

---

## Results

### Object Transformation

- Preserves object position
- Maintains lighting consistency
- Supports realistic replacement

### Occlusion Robustness

The proposed method remains stable even under partial object occlusion.

### Media Localization

Applicable to:

- advertisement localization
- virtual PPL
- media remastering
- global content adaptation

---

## Citation

```bibtex
@inproceedings{flux_localization_2026,
  title={Automatic Object Transformation in Image via FLUX Inpainting with Multi-LoRA Condition Decomposition},
  author={Kang, Minsoo and Jeong, Ayoung and Kim, Namho and Kim, Junhwa},
  year={2026}
}
```

---

## Acknowledgement

This work was supported by the IITP MSIT SW-centered University Program (2024-0-00047).

---

## Paper

Korean title:

> FLUX 인페인팅과 다중 LoRA 조건 분리를 이용한 영상 객체 자동 변환



