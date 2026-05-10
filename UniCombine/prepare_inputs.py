# prepare_inputs_v4.py

import os, json
import numpy as np
from PIL import Image

os.makedirs("examples/custom", exist_ok=True)
TARGET = 512

src_orig = Image.open("cola2.jpg").convert("RGB")
mask_orig = Image.open("cola2_masked.png").convert("L").resize(src_orig.size)
mask_arr = np.array(mask_orig) > 128

ys, xs = np.where(mask_arr)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

# 패딩 비대칭으로:
# - 가로(좌우): 30% (위치 정확도 위해 유지)
# - 위(top): 10% (병뚜껑 여유)
# - 아래(bottom): 0% (책상 영역 침범 안 함, 받침 생성 방지)

pad_x = int((x1 - x0) * 0.3)
pad_top = int((y1 - y0) * 0.1)
pad_bot = 0

x0p = max(0, x0 - pad_x)
y0p = max(0, y0 - pad_top)
x1p = min(src_orig.width, x1 + pad_x)
y1p = min(src_orig.height, y1 + pad_bot)   # 여기 핵심

bg_arr = np.array(src_orig)
bg_arr[y0p:y1p, x0p:x1p] = 0
bg = Image.fromarray(bg_arr)

W, H = bg.size
s = TARGET / max(W, H)
nw, nh = (int(W * s) // 16) * 16, (int(H * s) // 16) * 16

# 그냥 resize만 (비율 유지)
bg = bg.resize((nw, nh), Image.LANCZOS)
bg.save("examples/custom/background_cola2.png")

ref = Image.open("king.png").convert("RGB")
rw, rh = ref.size
rs = TARGET / max(rw, rh)
rnw, rnh = (int(rw * rs) // 16) * 16, (int(rh * rs) // 16) * 16

ref = ref.resize((rnw, rnh), Image.LANCZOS)
ref.save("examples/custom/subject_king.png")

with open("examples/custom/prompt.json", "w") as f:
    json.dump({
        "prompt": "The king can held in a hand, same shape, same position, realistic lighting, photorealistic, no distortion, no extra objects"
    }, f)
    
    # prepare_inputs_v4.py 마지막에 추가

with open("examples/custom/bbox.json", "w") as f:
    json.dump({
        "x0": int(x0p),
        "y0": int(y0p),
        "x1": int(x1p),
        "y1": int(y1p)
    }, f)

print(f"✓ bbox: ({x0p},{y0p}) - ({x1p},{y1p})")
print(f"  bottom edge matches bottle bottom (no padding below)")
