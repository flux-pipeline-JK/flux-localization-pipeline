from PIL import Image

img = Image.open("NewMountainDew.png").convert("RGB")

TARGET = 512

w, h = img.size
scale = TARGET / max(w, h)

nw = int(w * scale)
nh = int(h * scale)

img_resized = img.resize((nw, nh), Image.LANCZOS)

# 512x512 생성
canvas = Image.new("RGB", (TARGET, TARGET), (0, 0, 0))  # 검은 배경

# 가운데 정렬
x = (TARGET - nw) // 2
y = (TARGET - nh) // 2

canvas.paste(img_resized, (x, y))

canvas.save("NewMountainDew_512.png") # 마운틴 듀 512 리사이즈 

print("비율 유지 512 완성")
