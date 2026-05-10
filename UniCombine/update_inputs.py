# update_inputs.py

import os, json
import numpy as np
from PIL import Image

os.makedirs("examples/custom", exist_ok=True)

TARGET = 512

def fit(img, target):
    w, h = img.size
    s = target / max(w, h)
    nw, nh = (int(w * s) // 16) * 16, (int(h * s) // 16) * 16
    return img.resize((nw, nh), Image.LANCZOS)

# 핵심 변경: 콜라 영역을 흰색(255)이 아닌 회색(128)으로

bg = Image.open("cola.jpeg").convert("RGB")
mask = Image.open("cola_masked.png").convert("L").resize(bg.size)
bg_arr = np.array(bg)
mask_arr = np.array(mask) > 128
bg_arr[mask_arr] = 128                     # 흰색 → 회색
bg = Image.fromarray(bg_arr)
bg = fit(bg, TARGET)
bg.save("examples/custom/background.jpg")

# subject (환타)

ref = Image.open("Fanta.png").convert("RGB")
ref = fit(ref, TARGET)
ref.save("examples/custom/subject.jpg")

# json

with open("examples/custom/prompt.json", "w") as f:
    json.dump({"prompt": "A Fanta orange soda bottle on a white office desk"}, f)

print(f"✓ bg: {bg.size}, subj: {ref.size}")
