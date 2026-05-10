import cv2
import numpy as np


# 1. Image Load

img = cv2.imread("model.png")
fg = cv2.imread("model_masked_water.png")

# 2. mask 생성 (검정 기준)

gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY) # 검은 배경 = 0 → threshold

# 3. 반전 (캔 위치만 255)

mask_inv = cv2.bitwise_not(mask)


# 4. background 생성
background = img.copy()

background[mask == 255] = 0 # 캔 위치만 제거


# 저장
cv2.imwrite(
    "/home/vaill/work/tmp/UniCombine/examples/custom/background_model_masked.png",
    background
)

print("DONE")
