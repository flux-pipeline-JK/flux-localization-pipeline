"""
facebookresearch/segment-anything 으로 cola.jpeg 의 콜라 부분만 마스킹.
중앙 점을 프롬프트로 사용, 배경은 검정으로 채워서 저장.

준비:
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install opencv-python torch torchvision numpy
  # 체크포인트 다운로드 (셋 중 하나):
  #   ViT-B (358MB):  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  #   ViT-L (1.2GB):  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  #   ViT-H (2.4GB):  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  
"""

import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# 설정
IMAGE_PATH  = "Fanta.png" #"cola.jpeg"
CHECKPOINT  = "sam_vit_b_01ec64.pth"   # 다운받은 체크포인트 경로
MODEL_TYPE  = "vit_b"                  # 체크포인트에 맞춰 vit_b / vit_l / vit_h
OUTPUT_PATH = "fanta_masked.png"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

#[2385, 68, 3472, 4083]

# 이미지 로드
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# SAM 로드
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(image)

# 이미지 중앙 점을 프롬프트로
input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])  # 1 = foreground
BBOX = np.array([19, 18, 159, 491])

# Bounding Box 를 프롬프트로
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=BBOX[None, :],         # (1, 4) xyxy
    multimask_output=False,    # box prompt 는 보통 단일 마스크
)
mask = masks[0]
best_mask = masks[np.argmax(scores)]

# 객체만 남기고 배경은 검정
result = np.zeros_like(image)
result[best_mask] = image[best_mask]

cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print(f"저장 완료: {OUTPUT_PATH}  (mask score={scores.max():.3f})")
