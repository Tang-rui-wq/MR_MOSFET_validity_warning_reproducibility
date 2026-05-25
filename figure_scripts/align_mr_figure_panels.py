from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


BASE = Path(r"C:\Users\Tangrui\Desktop\MR_submission_latex")


def trim_whitespace(img: Image.Image, threshold: int = 246, margin: int = 22) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda p: 255 if p > (255 - threshold) else 0)
    bbox = mask.getbbox()
    if not bbox:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(rgb.width, right + margin)
    bottom = min(rgb.height, bottom + margin)
    return rgb.crop((left, top, right, bottom))


def fit_inside(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def label_font(size: int = 38) -> ImageFont.ImageFont:
    for candidate in [
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ]:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def make_two_panel(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    panel_w: int,
    panel_h: int,
    gap: int,
    label_h: int,
    margin_x: int,
    margin_y: int,
    threshold: int = 246,
) -> None:
    left = trim_whitespace(Image.open(left_path), threshold=threshold)
    right = trim_whitespace(Image.open(right_path), threshold=threshold)

    canvas_w = margin_x * 2 + panel_w * 2 + gap
    canvas_h = margin_y * 2 + panel_h + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    font = label_font()
    draw = ImageDraw.Draw(canvas)

    for idx, (img, label, x0) in enumerate(
        [
            (left, "(a)", margin_x),
            (right, "(b)", margin_x + panel_w + gap),
        ]
    ):
        fitted = fit_inside(img, panel_w, panel_h)
        x = x0 + (panel_w - fitted.width) // 2
        y = margin_y + (panel_h - fitted.height) // 2
        canvas.paste(fitted, (x, y))

        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_x = x0 + (panel_w - label_w) // 2
        label_y = margin_y + panel_h + 12
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    canvas.save(out_path, dpi=(300, 300))


def main() -> None:
    make_two_panel(
        BASE / "Figure_1a.jpeg",
        BASE / "Figure_1b.png",
        BASE / "Figure_1.png",
        panel_w=1000,
        panel_h=900,
        gap=80,
        label_h=64,
        margin_x=30,
        margin_y=30,
        threshold=246,
    )
    make_two_panel(
        BASE / "Figure_3a.png",
        BASE / "Figure_3b.png",
        BASE / "Figure_3.png",
        panel_w=1000,
        panel_h=900,
        gap=80,
        label_h=64,
        margin_x=30,
        margin_y=30,
        threshold=248,
    )
    print(BASE / "Figure_1.png")
    print(BASE / "Figure_3.png")


if __name__ == "__main__":
    main()
