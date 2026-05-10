import numpy as np
import cv2
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


# 설정

IMAGE_PATH = "cola2.jpg"
OUTPUT_PATH = "masked_cola2_masked.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GD_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CHECKPOINT = "weights/groundingdino_swint_ogc.pth"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"


# 모델 로드

gd_model = load_model(GD_CONFIG, GD_CHECKPOINT)
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)
predictor = SamPredictor(sam)

image_source, image_tensor = load_image(IMAGE_PATH)
predictor.set_image(image_source)

H, W, _ = image_source.shape


# 1️. CAN detection

boxes_can, logits_can, _ = predict(
    model=gd_model,
    image=image_tensor,
    caption="soda can",
    box_threshold=0.3,
    text_threshold=0.25,
    device=DEVICE,
)

if len(boxes_can) == 0:
    raise RuntimeError("캔 감지 실패")

# 가장 confidence 높은 캔 선택
idx = int(torch.argmax(logits_can))
box_can = boxes_can[idx]

box_can_xyxy = box_convert(
    box_can * torch.tensor([W,H,W,H]),
    "cxcywh","xyxy"
).cpu().numpy()


# 2️. HAND detection

boxes_hand, logits_hand, _ = predict(
    model=gd_model,
    image=image_tensor,
    caption="hand",
    box_threshold=0.25,
    text_threshold=0.25,
    device=DEVICE,
)

# 3️. CAN mask (SAM)

mask_can, _, _ = predictor.predict(
    box=box_can_xyxy[None,:],
    multimask_output=False
)
mask_can = mask_can[0]


# 4️. HAND mask (SAM)

mask_hand_total = np.zeros_like(mask_can)

for box in boxes_hand:
    box_xyxy = box_convert(
        box * torch.tensor([W,H,W,H]),
        "cxcywh","xyxy"
    ).cpu().numpy()

    m, _, _ = predictor.predict(
        box=box_xyxy[None,:],
        multimask_output=False
    )
    mask_hand_total = np.logical_or(mask_hand_total, m[0])

mask_hand_total = mask_hand_total.astype(np.uint8)


# 5️. 최종 mask

final_mask = mask_can.copy()
final_mask[mask_hand_total > 0] = 0


# 6️. mask 정리

kernel = np.ones((5,5), np.uint8)
final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


# 7️. 결과 생성

result = np.zeros_like(image_source)
result[final_mask.astype(bool)] = image_source[final_mask.astype(bool)]

cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

print(" 완료: masked_cola2_masked.jpg")
