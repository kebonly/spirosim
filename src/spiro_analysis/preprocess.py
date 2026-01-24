import pims
import numpy as np
import scipy
import json
from pathlib import Path

from skimage import exposure
import skimage
from typing import Any
from tqdm import tqdm


import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import matplotlib as mpl
import numpy as np
import trackpy as tp
import pims
import imageio.v2 as imageio

class Experiment:
    def __init__(self, name: str , root_dir: Path, input_data_dir: Path, path: Path = None):
        self.name = name
        self.root_dir = root_dir
        self.input_data_dir = input_data_dir
        self.path = path if path else Path(self.root_dir/"metadata.json")

        if not os.path.exists(self.root_dir / "metadata.json"):
            print(self.root_dir)
            print("data dir not exist")
            self._initialize_metadata()
        self.metadata = self.load_metadata()
        self.frames = None

    def _initialize_metadata(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)

        with open(self.root_dir / "metadata.json", "w") as f:
            data = {"experiment_name": self.name}
            json.dump(data, f)

        return
    
    def load_metadata(self) -> dict:
        with open(self.path, "r") as f:
            data = json.load(f)
        
        return data

    def read_metadata(self, key: str) -> Any:
        return self.metadata.get(key)

    def update_metadata(self, key, value):
        try:
            self.metadata[key] = value
            with open(self.path, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Unable to update experiment metadata: {e}")
        

    def update_cropping(self) -> None:
        # frame = os.listdir(f"data/raw/{self.name}")[0]
        # image_path = f"data/raw/{self.name}/{frame}"
        image_path = next((self.root_dir / "raw").iterdir())

        img = mpimg.imread(image_path)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_title("Click and drag to draw/resize/move circle. Press Enter to confirm.")

        selector = CircleSelector(ax, img)
        plt.show(block=True)
        # Need to close the plot

        self.update_metadata("center", selector.circle.center)
        self.update_metadata("radius", selector.circle.radius)

        return None

    def generate_cropped_images(self) -> None:
        # image_path = next((self.root_dir / "raw").iterdir())
        frames = self.open_frames() # TODO: Need to make opening_frames separate
        frames = circle_crop(frames, self.read_metadata("center"), self.read_metadata("radius"))
        # save_frames(frames, save_dir)
        self.frames = frames # so that we don't have to open it each time.

    def open_frames(self, dir=None, prefix="Pic", filetype="bmp") -> pims.ImageSequence:
        # Now we want to circle crop
        if dir is None:
            frames = pims.open(f"{self.root_dir}/{self.input_data_dir}/{prefix}*.{filetype}")
        else:
            frames = pims.open(f"{dir}/{prefix}*.{filetype}")
        
        self.frames = frames # TODO: This is a bit redundant to save attribute and then export too
        return frames

    
    def generate_background_subtracted_images(self, use_background_files=True):
        # TODO: Need to clean up this part of the pipeline
        frames = gray(self.frames)

        background = open_backgrounds(f"{self.root_dir}/raw/background*.bmp")

        if use_background_files:
            crop_background = circle_crop(background, self.read_metadata("center"), self.read_metadata("radius"))
            background = background_calculation(frames, crop_background)
        else:
            background = background_calculation(frames, None)

        # need to crop the background too
        frames = subtract_background(frames, background)
        self.frames = frames
        return
    
    def generate_threshold_images(self):
        frames = threshold(self.frames)
        self.frames = frames
        return
    
    def generate_remove_small_objects(self, min_size=70):
        frames = remove_small_objects(self.frames, min_size=min_size)
        self.frames = frames
        return


    def save_frames(self, dir_name, stretch_contrast=False):
        """ Saves experiment frames at whatever state they're currently in.
        """

        dir_path = f"{self.root_dir}/processed/{dir_name}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with tqdm(total = len(self.frames)) as pbar:
            for i, frame in enumerate(self.frames):

                if stretch_contrast:
                    frame_min = np.min(frame).astype(np.uint8)
                    frame_max = np.max(frame).astype(np.uint8)
                    frame = frame.astype(np.uint8)
                    fout = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                else:
                    fout = np.clip(frame, 0, 255).astype(np.uint8)

                imageio.imwrite(f"{dir_path}/img_{i}.bmp", fout)
                pbar.update()

        return
    
    def imshow(self, frame_number=0):
        plt.imshow(self.frames[frame_number], cmap=plt.cm.gray)
        plt.show()
    
    def get_frame(self, frame_number=0):
        return self.frames[frame_number]
    
@pims.pipeline
def gray(frame):
    arr = frame[:, :]
    return pims.Frame(arr, frame.frame_no, metadata=frame.metadata)  # Take just the green channel

@pims.pipeline
def threshold(frame):
    
    arr = np.asarray(frame, dtype=np.int16)

    thresh = skimage.filters.threshold_yen(arr)

    binary = arr > thresh
    return pims.Frame(binary, frame.frame_no, metadata=frame.metadata)


@pims.pipeline
def circle_crop(img, center_coordinates, radius):
    """
    Crop the image to select the region of interest
    """
    row_i, col_j = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    distance = (row_i - center_coordinates[1])**2 + (col_j - center_coordinates[0])**2

    mask = distance > radius**2
    img[mask] = 0
    
    xlim = [int(center_coordinates[0]-radius), int(center_coordinates[0]+radius)]
    ylim = [int(center_coordinates[1]-radius), int(center_coordinates[1]+radius)]
    cropped = img[ylim[0]:ylim[1], xlim[0]:xlim[1]]
    cropped = np.clip(cropped, 0, 255).astype(np.uint8)

    return cropped

@pims.pipeline
def remove_small_objects(frame, min_size=70):
    mask = frame == 255
    clean = skimage.morphology.remove_small_objects(mask, min_size=min_size)
    frame[~clean] = 0
    return pims.Frame(frame, frame.frame_no)


@pims.pipeline
def subtract_background(frame, background):
    subtracted = np.asarray(frame, dtype=np.int16) - np.asarray(background, dtype=np.int16)
    mask = subtracted < 0
    subtracted[mask] = 0 # NOTE: Shouldn't clamp to 0 so early on in pipeline
    return pims.Frame(subtracted, frame.frame_no)

def open_backgrounds(path):
    try:
        background = pims.open(path)
        print("done opening backgrounds")
    except:
        background = None
        print("there are no background files")
    return background

def background_calculation(img, background_data):
    """Calculate background based on the modal value
    Works more effectively if there are more than 100 frames, I think.
    """
    if background_data is None:
        print("No background file so using mean of image frames")
        # return scipy.stats.mode(img, axis=0) # Let's try mode
        background = np.average(img, axis=0)
    else:
        print("Doing average of background frames")
        background = np.average(gray(background_data), axis=0)

    return background

class CircleSelector:
    def __init__(self, ax, img):
        self.ax = ax
        self.img = img
        self.circle = None
        self.press = None
        self.mode = None  # 'draw', 'move', 'resize'
        self.drag_threshold = 10  # pixel distance for selecting edge

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        x0, y0 = event.xdata, event.ydata

        if self.circle is None:
            # Start drawing new circle
            self.mode = 'draw'
            self.press = (x0, y0)
            self.circle = Circle((x0, y0), 1, edgecolor='r', fill=False, linewidth=2)
            self.ax.add_patch(self.circle)
        else:
            cx, cy = self.circle.center
            r = self.circle.radius
            dist = np.hypot(x0 - cx, y0 - cy)

            if abs(dist - r) < self.drag_threshold:
                self.mode = 'resize'
                self.press = (cx, cy)
            elif dist < r:
                self.mode = 'move'
                self.press = (x0 - cx, y0 - cy)
            else:
                self.mode = None

    def on_motion(self, event):
        if event.inaxes != self.ax or self.circle is None or self.mode is None:
            return

        x, y = event.xdata, event.ydata

        if self.mode == 'draw':
            x0, y0 = self.press
            r = np.hypot(x - x0, y - y0)
            self.circle.set_radius(r)
        elif self.mode == 'move':
            dx, dy = self.press
            self.circle.center = (x - dx, y - dy)
        elif self.mode == 'resize':
            cx, cy = self.press
            r = np.hypot(x - cx, y - cy)
            self.circle.set_radius(r)

        self.ax.figure.canvas.draw()

    def on_release(self, event):
        self.mode = None
        self.press = None

    def on_key(self, event):
        if event.key == 'enter' and self.circle is not None:
            cx, cy = self.circle.center
            r = self.circle.radius
            # print(f"\n✅ Final circle:")
            # print(f"  Center: ({cx:.2f}, {cy:.2f})")
            # print(f"  Radius: {r:.2f}")
            plt.close()

def merge_strobe_triplet(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    *,
    glare_percentile: float = 95.0,
    use_texture_filter: bool = True,
    texture_thresh_percentile: float = 30.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Merge 3 images of the same aligned scene under different illumination angles,
    suppressing glare/specular highlights via per-image glare detection + robust fusion.

    Parameters
    ----------
    img1, img2, img3 : np.ndarray
        Images with identical shape (H, W) or (H, W, C). C can be 3 (RGB) etc.
        Assumed already registered/aligned.
    glare_percentile : float
        Pixels above this percentile (in intensity) are considered glare candidates.
    use_texture_filter : bool
        If True, also require low local texture to classify as glare (helps reject bright
        diffuse regions like white paper).
    texture_thresh_percentile : float
        Texture below this percentile is considered "low texture" (only used if enabled).
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    merged : np.ndarray
        Glare-suppressed fused image, same shape and dtype as the inputs.

    Notes
    -----
    - If all 3 images are marked as glare at a pixel, falls back to pixel-wise median.
    - Best results when highlights are additive (specular) and move with illumination.
    """

    imgs = [img1, img2, img3]
    if not (img1.shape == img2.shape == img3.shape):
        raise ValueError("All three images must have the same shape (must be aligned/registered).")

    # Work in float32 for computation; restore dtype at end.
    orig_dtype = img1.dtype
    imgs_f = [i.astype(np.float32) for i in imgs]

    # Convert to intensity for glare detection (per-pixel scalar).
    def to_intensity(im: np.ndarray) -> np.ndarray:
        if im.ndim == 2:
            return im
        # Luma-like weighting (works fine for RGB; for other channel counts, just average)
        if im.shape[2] >= 3:
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.mean(im, axis=2)

    intens = [to_intensity(im) for im in imgs_f]  # list of (H,W)

    # Simple local texture estimate: magnitude of finite differences.
    # (Avoids needing OpenCV/scipy; fast and decent.)
    def texture_map(I: np.ndarray) -> np.ndarray:
        dx = np.abs(I[:, 1:] - I[:, :-1])
        dy = np.abs(I[1:, :] - I[:-1, :])
        # pad back to (H,W)
        dx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
        dy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
        return dx + dy

    # Build glare masks per image.
    glare_masks = []
    for I in intens:
        bright_thr = np.percentile(I, glare_percentile)
        bright = I >= bright_thr

        if use_texture_filter:
            T = texture_map(I)
            tex_thr = np.percentile(T, texture_thresh_percentile)
            low_tex = T <= tex_thr
            glare = bright & low_tex
        else:
            glare = bright

        glare_masks.append(glare)

    # Stack for fusion: shape (3,H,W[,C])
    stack = np.stack(imgs_f, axis=0)
    masks = np.stack(glare_masks, axis=0)  # (3,H,W)

    # Expand masks to channels if needed
    if stack.ndim == 4:  # (3,H,W,C)
        masks_exp = masks[..., None]  # (3,H,W,1)
    else:  # (3,H,W)
        masks_exp = masks

    # Set glare pixels to NaN so we can ignore them in robust stats
    stack_masked = stack.copy()
    stack_masked[masks_exp] = np.nan

    # Robust fusion: median of non-glare samples per pixel (and per channel if RGB)
    merged = np.nanmedian(stack_masked, axis=0)

    # If all 3 were NaN at some pixel, nanmedian returns NaN. Fill those with plain median.
    fallback = np.median(stack, axis=0)
    nan_locs = np.isnan(merged)
    merged[nan_locs] = fallback[nan_locs]

    # Clip and cast back to original dtype
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        merged = np.clip(merged, info.min, info.max)
        return merged.astype(orig_dtype)
    else:
        # For float images, keep range as-is (or you can clip to [0,1] if that's your convention)
        return merged.astype(orig_dtype)

def merge_strobe_directory(
    input_dir,
    output_dir,
    *,
    suffix="_merged",
    ext=None,
    overwrite=False,
    **merge_kwargs,
):
    """
    Merge strobe triplets in a directory using merge_strobe_triplet
    and save the fused frames to a new directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw images ordered as A,B,C,A,B,C,...
    output_dir : str or Path
        Directory where merged images will be saved.
    suffix : str
        Suffix added to output filenames.
    ext : str or None
        Output extension (e.g. ".tif", ".png").
        If None, uses input file extension.
    overwrite : bool
        If False, skip files that already exist.
    merge_kwargs : dict
        Passed directly to merge_strobe_triplet.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    )

    if len(files) < 3:
        raise ValueError("Need at least 3 images to form one strobe triplet.")

    n_triplets = len(files) // 3
    if len(files) % 3 != 0:
        print(f"⚠️ Warning: {len(files)} files found, truncating to {3*n_triplets}.")

    print(f"Processing {n_triplets} strobe triplets...")

    for i in range(n_triplets):
        f1, f2, f3 = files[3*i : 3*i + 3]

        # Construct output filename
        out_ext = ext if ext is not None else f1.suffix
        out_name = f1.stem + suffix + out_ext
        out_path = output_dir / out_name

        if out_path.exists() and not overwrite:
            print(f"Skipping existing {out_path.name}")
            continue

        # Load images
        img1 = imageio.imread(f1)
        img2 = imageio.imread(f2)
        img3 = imageio.imread(f3)

        # Merge
        merged = merge_strobe_triplet(img1, img2, img3, **merge_kwargs)

        # Save
        imageio.imwrite(out_path, merged)

        if i % 50 == 0 or i == n_triplets - 1:
            print(f"  [{i+1}/{n_triplets}] saved {out_path.name}")

    print("✅ Done.")