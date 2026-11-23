"""Pixel transformations applied before the frozen visual encoder."""
from PIL import Image

ROLES = ("original", "rotate_left", "rotate_right", "zoom_in", "zoom_out_black")


def transformed(image, role, rotations, scale=0.8):
    image = image.convert("RGB")
    if role == "original":
        return image.copy()
    if role in ("rotate_left", "rotate_right"):
        angle = rotations[0] if role == "rotate_left" else rotations[1]
        return image.rotate(angle, Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    w, h = image.size
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    x, y = (w - cw) // 2, (h - ch) // 2
    if role == "zoom_in":
        return image.crop((x, y, x + cw, y + ch)).resize(image.size, Image.Resampling.BICUBIC)
    if role == "zoom_out_black":
        view = Image.new("RGB", image.size, (0, 0, 0))
        view.paste(image.resize((cw, ch), Image.Resampling.BICUBIC), (x, y))
        return view
    raise ValueError(f"Unknown view: {role}")

