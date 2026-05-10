# prepare_inputs_v5.py
import os, json
import numpy as np
from PIL import Image

os.makedirs("examples/custom", exist_ok=True)
TARGET = 512

src_orig = Image.open("Monster.png").convert("RGB")
mask_orig = Image.open("Monster_masked.png").convert("L").resize(src_orig.size)
mask_arr = np.array(mask_orig) > 128

ys, xs = np.where(mask_arr)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

# 가로 30%, 위 10%, 아래 5%만 (받침대 생성 방지하면서 반사 공간 약간 제공)
pad_x = int((x1 - x0) * 0.30)
pad_top = int((y1 - y0) * 0.10)
pad_bot = int((y1 - y0) * 0.05)
x0p = max(0, x0 - pad_x)
y0p = max(0, y0 - pad_top)
x1p = min(src_orig.width, x1 + pad_x)
y1p = min(src_orig.height, y1 + pad_bot)

bg_arr = np.array(src_orig)
bg_arr[y0p:y1p, x0p:x1p] = 0
bg = Image.fromarray(bg_arr)

W, H = bg.size
s = TARGET / max(W, H)
nw, nh = (int(W * s) // 16) * 16, (int(H * s) // 16) * 16
bg = bg.resize((nw, nh), Image.LANCZOS)
bg.save("examples/custom/background.jpg")

# Subject 원본 비율 (stretch X)
ref = Image.open("Monster.png").convert("RGB")
rw, rh = ref.size
rs = TARGET / max(rw, rh)
rnw, rnh = (int(rw * rs) // 16) * 16, (int(rh * rs) // 16) * 16
ref.resize((rnw, rnh), Image.LANCZOS).save("examples/custom/subject.jpg")

with open("examples/custom/prompt.json", "w") as f:
    json.dump({
        "prompt": "A Monster soda can on a white office desk, no reflection, no duplicate"
    }, f)

print(f"✓ bbox: ({int(x0p*s)},{int(y0p*s)}) - ({int(x1p*s)},{int(y1p*s)})")
