# overlay_circles.py
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import math

def _color_from_id(pid: int) -> tuple[int, int, int]:
    """Deterministic pseudo-random color for a particle id."""
    # simple hash → hue; convert to RGB-ish
    h = (hash(int(pid)) & 0xFFFFFF)
    r = (h >> 16) & 255
    g = (h >> 8) & 255
    b = h & 255
    # avoid very dark colors
    return (max(r, 60), max(g, 60), max(b, 60))

def overlay_circles(
    csv_path: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "frame_{frame:05d}.bmp",  # filename template
    radius: int = 8,
    width: int = 2,                           # line thickness
    frame_offset: int = 0,                    # if CSV frame index starts at 1, set to -1
    antialias: int = 2,                       # >1 for smoother circles (draw big then downscale)
):
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"x", "y", "frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    has_particle = "particle" in df.columns
    # ensure numeric
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])
    df["frame"] = pd.to_numeric(df["frame"], downcast="integer")

    # group by frame to open each image once
    for frame, group in df.groupby("frame"):
        fnum = int(frame + frame_offset)
        img_path = images_dir / pattern.format(frame=fnum)
        if not img_path.exists():
            print(f"[warn] image not found for frame {frame} → {img_path}")
            continue

        # open and convert to RGB so we can draw colored overlays
        base = Image.open(img_path)
        if base.mode != "RGB":
            base = base.convert("RGB")

        # optional antialias trick: draw larger and downscale
        scale = max(1, int(antialias))
        work = base.resize((base.width * scale, base.height * scale), Image.NEAREST) if scale > 1 else base
        draw = ImageDraw.Draw(work)

        for _, row in group.iterrows():
            x = float(row["x"]) * scale
            y = float(row["y"]) * scale
            r = radius * scale
            bbox = [x - r, y - r, x + r, y + r]

            color = _color_from_id(int(row["particle"])) if has_particle else (255, 0, 0)
            # draw circle outline
            draw.ellipse(bbox, outline=color, width=width * scale)
            # optional center dot
            draw.ellipse([x-1*scale, y-1*scale, x+1*scale, y+1*scale], fill=color)

        out = work.resize((base.width, base.height), Image.LANCZOS) if scale > 1 else work
        out.save(out_dir / img_path.name)

    print(f"Done. Wrote overlays to: {out_dir}")

if __name__ == "__main__":
    # quick demo call; replace with Click if you want a CLI
    overlay_circles(
        csv_path="data/processed/test_passiveInteraction/tracks.csv",
        images_dir="data/processed/test_passiveInteraction/enhanced/",
        out_dir="data/processed/test_passiveInteraction/overlays",
        pattern="cropped_{frame}.bmp",
        radius=8,
        width=2,
        frame_offset=0,
    )