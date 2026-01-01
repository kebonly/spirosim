
"""
Play a folder of .bmp files as a movie.

Controls:
  q or ESC  : quit
  SPACE     : pause / resume
  ← / →     : step backward / forward one frame (while paused)
  r         : restart from first frame
"""

import argparse
import glob
import os
import re
import time
import cv2
import numpy as np

def natural_key(s: str):
    # Split into digit/non-digit chunks for natural sorting
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_bmps(folder: str):
    files = glob.glob(os.path.join(folder, "*.bmp"))
    if not files:
        raise FileNotFoundError(f"No .bmp files found in: {folder}")
    files.sort(key=natural_key)
    return files

def maybe_resize(frame, scale=None, max_width=None, max_height=None):
    h, w = frame.shape[:2]
    fx = fy = 1.0
    if scale is not None:
        fx = fy = float(scale)
    if max_width or max_height:
        sx = (max_width / w) if max_width else 1.0
        sy = (max_height / h) if max_height else 1.0
        m = min(sx, sy, 1.0)  # only shrink-to-fit unless scale explicitly given
        fx = fy = min(fx, m) if scale is None else scale
    if fx != 1.0 or fy != 1.0:
        frame = cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    return frame

def main():
    ap = argparse.ArgumentParser(description="Play a folder of .bmp files as a movie.")
    ap.add_argument("folder", help="Path to folder containing .bmp frames")
    ap.add_argument("--fps", type=float, default=24.0, help="Playback frames per second (default: 24)")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    ap.add_argument("--scale", type=float, default=None, help="Uniform scale factor (e.g., 0.5 to shrink by half)")
    ap.add_argument("--max-width", type=int, default=None, help="Shrink to fit this width (ignored if --scale given)")
    ap.add_argument("--max-height", type=int, default=None, help="Shrink to fit this height (ignored if --scale given)")
    ap.add_argument("--window", default="BMP Player", help="Window name")
    args = ap.parse_args()

    files = list_bmps(args.folder)
    delay_ms = int(1000.0 / args.fps) if args.fps > 0 else 1

    cv2.namedWindow(args.window, cv2.WINDOW_AUTOSIZE)
    idx = 0
    paused = False

    while True:
        # Load current frame (BGR or grayscale is fine; imshow handles both)
        img = cv2.imread(files[idx], cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to read {files[idx]}; skipping.")
        else:
            disp = maybe_resize(img, scale=args.scale, max_width=args.max_width, max_height=args.max_height)
            cv2.imshow(args.window, disp)

        start = time.time()
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFFFF

        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord(' '):      # Space toggles pause
            paused = not paused
        elif key == ord('r'):      # restart
            idx = 0
            continue
        elif key == 104:            # Left arrow
            paused = True
            idx = (idx - 1) % len(files)
            continue
        elif key == 108:            # Right arrow
            paused = True
            idx = (idx + 1) % len(files)
            continue

        if not paused:
            idx += 1
            if idx >= len(files):
                if args.loop:
                    idx = 0
                else:
                    break

        # Optional: timing correction if your system is fast and you want tighter FPS
        if not paused and args.fps > 0:
            elapsed_ms = (time.time() - start) * 1000.0
            leftover = delay_ms - int(elapsed_ms)
            if leftover > 1:
                cv2.waitKey(leftover)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
