"""生成应用图标 assets/shadowvox.ico：粉色方块上的环形频谱，四重对称（中心对称）。

中心实心圆向外辐射直角光线，长度按 cos(4θ) 起伏；小尺寸减少光线数保证可辨。
8 倍超采样后缩小抗锯齿。用法：python tools/make_icon.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ACCENT, INK = (0xB5, 0x2F, 0x68, 255), (255, 255, 255, 255)
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SCALE = 8


def draw(size):
    s = size * SCALE
    image = Image.new("RGBA", (s, s), ACCENT)
    pen = ImageDraw.Draw(image)
    c = s / 2
    small = size <= 24
    rays = 12 if small else 24
    inner, outer = s * (.22 if small else .2), s * .44
    width = s * (.07 if small else .034)
    for i in range(rays):
        angle = 2 * math.pi * i / rays
        level = .3 + .7 * (.5 + .5 * math.cos(4 * angle)) ** 1.5   # 4 重对称的频谱起伏
        end = inner + (outer - inner) * level
        start = inner * 1.15
        dx, dy = math.cos(angle), math.sin(angle)
        nx, ny = -dy * width / 2, dx * width / 2
        pen.polygon([(c + dx * start + nx, c + dy * start + ny), (c + dx * end + nx, c + dy * end + ny),
                     (c + dx * end - nx, c + dy * end - ny), (c + dx * start - nx, c + dy * start - ny)], fill=INK)
    dot = s * (.12 if small else .1)
    pen.ellipse((c - dot, c - dot, c + dot, c + dot), fill=INK)
    return image.resize((size, size), Image.LANCZOS)


def main():
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    images = [draw(size) for size in SIZES]
    images[-1].save(out / "shadowvox.ico", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    images[-1].save(out / "shadowvox.png")
    preview = Image.new("RGBA", (sum(SIZES) + 12 * len(SIZES), 256), (255, 250, 251, 255))
    x = 0
    for image in images:
        preview.paste(image, (x, 256 - image.height))
        x += image.width + 12
    preview.save(out / "icon-preview.png")
    print("icon ->", out / "shadowvox.ico")


if __name__ == "__main__":
    main()
