from __future__ import annotations

"""生成白光干涉测量图标：干涉条纹与重建表面轮廓。"""

import math
from pathlib import Path

from PIL import Image, ImageDraw


def build_icon(size: int) -> Image.Image:
    """在高分辨率画布绘制后缩小，保证任务栏小图标边缘平滑。"""
    canvas_size = 1024
    scale = canvas_size / 256
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def s(value: float) -> int:
        return round(value * scale)

    draw.rounded_rectangle((s(8), s(8), s(248), s(248)), radius=s(52), fill="#102C46")
    # 椭圆条纹表达干涉测量，白色轮廓表达从条纹恢复表面高度。
    for index in range(5):
        inset = 16 * index
        draw.ellipse(
            (s(37 + inset), s(35 + inset * 0.65), s(219 - inset), s(176 - inset * 0.65)),
            outline=("#2B8197", "#3FA4B3", "#58CCD0", "#76E2DB", "#B1F7E8")[index],
            width=s(5),
        )
    points = [
        (s(x), s(184 - 26 * math.exp(-((x - 91) / 27) ** 2) + 13 * math.exp(-((x - 155) / 25) ** 2)))
        for x in range(39, 219)
    ]
    draw.line(points, fill="#102C46", width=s(20), joint="curve")
    draw.line(points, fill="#F3FCFF", width=s(9), joint="curve")
    draw.ellipse((s(119), s(97), s(137), s(115)), fill="#F3FCFF")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    icon = build_icon(256)
    icon.save(assets_dir / "app_icon.ico", sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)])
    icon.save(assets_dir / "app_icon.png")


if __name__ == "__main__":
    main()
