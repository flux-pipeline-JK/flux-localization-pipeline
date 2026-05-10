"""
GroundingDINO 로 'cola bottle' 의 bounding box 를 얻고,
그 box 를 SAM 에 prompt 로 넘겨 마스크 추출 → 배경 검정으로 저장.

준비:
  # GroundingDINO
  git clone https://github.com/IDEA-Research/GroundingDINO.git
  cd GroundingDINO && pip install -e . && cd ..
  # 또는: pip install groundingdino-py

  # SAM
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install opencv-python torch torchvision numpy

  # 체크포인트
  #   GroundingDINO: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  #   SAM (ViT-B):   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  # config 파일은 GroundingDINO 레포의 groundingdino/config/GroundingDINO_SwinT_OGC.py 사용

"""

import numpy as np
import cv2
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

# 설정

IMAGE_PATH       = 'Fanta.png'         # ex) "cola.jpeg"
TEXT_PROMPT      = "cola bottle ."     # GroundingDINO 는 끝에 . 붙이는 걸 권장
BOX_THRESHOLD    = 0.35
TEXT_THRESHOLD   = 0.25

GD_CONFIG        = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CHECKPOINT    = "weights/groundingdino_swint_ogc.pth"

SAM_CHECKPOINT   = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE   = "vit_b"

OUTPUT_PATH      = "cola_masked.png"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# 1) GroundingDINO 로 Bounding Box 획득

gd_model = load_model(GD_CONFIG, GD_CHECKPOINT)
image_source, image_tensor = load_image(IMAGE_PATH)   # image_source: HxWx3 numpy(RGB)

boxes, logits, phrases = predict(
    model=gd_model,
    image=image_tensor,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE,
)

# boxes 는 [N, 4], cxcywh 정규화(0~1) 형식 → xyxy 픽셀로 변환

H, W, _ = image_source.shape
boxes_xyxy = box_convert(
    boxes=boxes * torch.tensor([W, H, W, H]),
    in_fmt="cxcywh", out_fmt="xyxy",
).numpy()

print("Detections:")
for p, l, b in zip(phrases, logits, boxes_xyxy):
    print(f"  {p:20s} score={l:.3f}  xyxy=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

if len(boxes_xyxy) == 0:
    raise SystemExit("아무것도 감지하지 못함. THRESHOLD 또는 PROMPT 조정 필요.")

# 가장 confidence 높은 박스 1개 선택

best_idx = int(torch.argmax(logits).item())
best_box = boxes_xyxy[best_idx]
print(f"\n선택된 박스: {best_box.astype(int).tolist()}  (phrase='{phrases[best_idx]}')")


# 2) SAM 에 박스 prompt 로 마스크 추출

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(image_source)        # SAM 은 RGB numpy 입력

masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=best_box[None, :],               # (1,4) xyxy
    multimask_output=False,
)
mask = masks[0]


# 3) 콜라만 남기고 배경 검정

result = np.zeros_like(image_source)
result[mask] = image_source[mask]

cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print(f"\n저장 완료: {OUTPUT_PATH}  (sam score={scores[0]:.3f})")
