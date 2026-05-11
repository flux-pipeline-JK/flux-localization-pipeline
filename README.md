# CTG 후처리 모듈을 활용한 <br>Zero-Shot Open-Vocabulary Segmentation 개선 연구



<img width="988" height="488" alt="image" src="https://github.com/user-attachments/assets/3c095ff1-1a29-4f64-a4a4-317f9831ec01" />

## Installation

To get started, clone the repository and install dependencies:

```bash
!git clone --recurse-submodules https://github.com/AI-AYJ/AI-AYJ-CTG.git
%cd AI-AYJ-CTG/GEM
!pip install -e .
```

## Requirements

```bash
- torch>=1.9.0
- torchvision
- regex
- ftfy
- tqdm
- huggingface_hub
- sentencepiece
- protobuf
- timm
- einops
- open_clip_torch<=2.23.0
- opencv-python
- matplotlib
- numpy
- requests
- torchmetrics
```

## Datasets
We use three datasets (Pascal VOC, ADE20K and OpenImages V7) in our paper.

For `Pascal VOC`, you can download the dataset from [here](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset).

For `ADE20K`, you can download the dataset from [here](https://ade20k.csail.mit.edu/).

For `OpenImages V7`, you can download the dataset from [here](https://storage.googleapis.com/openimages/web/download_v7.html).



## Run on Pascal VOC

To reproduce the Pascal VOC experiments:

- **PascalVOC 재구현 코드**  
  Run: [`PascalVOC_재구현_코드.ipynb`](./PascalVOC_재구현_코드.ipynb)

- **PascalVOC + CTG 실험 코드**  
  Run: [`PascalVOC_+CTG_코드.ipynb`](./PascalVOC_+CTG_코드.ipynb)

- **PascalVOC Heatmap 시각화 코드**  
  Run: [`PascalVOC+heatmap_코드.ipynb`](./PascalVOC+heatmap_코드.ipynb)


---

## Run on ADE20K

To reproduce the ADE20K experiments:

- **ADE20K 재구현 코드**  
  Run: [`ADE20K_재구현_코드.ipynb`](./ADE20K_재구현_코드.ipynb)

- **ADE20K + CTG 실험 코드**  
  Run: [`ADE20K + CTG_코드.ipynb`](./ADE20K%20+%20CTG_%EC%BD%94%EB%93%9C.ipynb)

- **ADE20K Heatmap 시각화 코드**  
  Run: [`ADE20K+heatmap코드.ipynb`](./ADE20K+heatmap%EC%BD%94%EB%93%9C.ipynb)


  ---

## Run on OpenImages V7

To reproduce the OpenImages V7 experiments:

- **OpenImages V7 재구현 코드**  
  Run: [`Openimagesv7_재구현_코드.ipynb`](./Openimagesv7_%E1%84%8C%E1%85%A2%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB_%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3.ipynb)

- **OpenImages V7 + CTG 실험 코드**  
  Run: [`openimagesv7+CTG_코드.ipynb`](./openimagesv7%2BCTG_%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3.ipynb)



