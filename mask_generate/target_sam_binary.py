"""
facebookresearch/segment-anything 으로 cola.jpeg 의 콜라 부분만 마스킹.
주어진 bounding box 를 프롬프트로 사용, 배경은 검정으로 채워서 저장.

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
IMAGE_PATH  = "model.png"
CHECKPOINT  = "sam_vit_b_01ec64.pth"
MODEL_TYPE  = "vit_b"
OUTPUT_PATH = "model_masked.png"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# GroundingDINO 등으로 얻은 Bounding Box (xyxy 픽셀 좌표)
BBOX = np.array([558,292,962,482])

# 이미지 로드
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# SAM 로드
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(image)

# Bounding Box 를 프롬프트로
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=BBOX[None, :],         # (1, 4) xyxy
    multimask_output=False,    # box prompt 는 보통 단일 마스크
)
mask = masks[0]

# 이진 마스크: 콜라=흰색(255), 배경=검정(0)
mask_img = (mask.astype(np.uint8)) * 255

cv2.imwrite(OUTPUT_PATH, mask_img)
print(f"저장 완료: {OUTPUT_PATH}  (mask score={scores[0]:.3f})")
