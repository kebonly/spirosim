import pims
import numpy as np
import scipy
import json
from pathlib import Path
import matplotlib as mpl
import imageio.v2 as iio
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

# def save_frames(frames, dir):
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#         print(f"Created directory: {dir}")
#     for i, frame in enumerate(frames):
#         fout = np.clip(frame, 0, 255).astype(np.uint8)
#         imageio.imwrite(f"{dir}/cropped_{i}.bmp", fout)

#     return dir

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

# @pims.pipeline
# def enhance_contrast(frame,
#                      p_low=1, p_high=99,          # robust stretch percentiles
#                      method="clahe",              # "clahe" or "rescale"
#                      clip_limit=0.01,             # CLAHE contrast limit (0–1)
#                      tile_grid_size=(8, 8),       # CLAHE tile grid
#                      gamma=None,                  # e.g. 0.8 to brighten, 1.2 to darken
#                      out_dtype=np.float32,
#                      min_brightness=24):       # keep float for later steps
#     """Per-frame contrast enhancement for dim/bright particles.

#     - Robust percentile stretch to remove exposure variations.
#     - Optional CLAHE for local contrast in uneven illumination.
#     """
#     arr = np.asarray(frame)
#     print(np.max(arr))
#     # print(np.max(arr))
#     if np.max(arr) > min_brightness:
#         fin = arr.astype(np.float32) / np.max(arr)
#     else:
#         fin = arr.astype(np.float32) / min_brightness

#     # pick output dtype
#     if out_dtype == np.uint8:
#         fout = (np.clip(fin, 0, 1) * 255.0 + 0.5).astype(np.uint8)
#     else:
#         fout = fin.astype(out_dtype)
#     # print(getattr(frame, "frame_no", None))

#     return pims.Frame(fout, frame_no=getattr(frame, "frame_no", None),
#                       metadata=getattr(frame, "metadata", {}))


# def background_subtraction(experiment, save=False):
#     # TODO: make sure to flatten background?
#     frames = gray(open_frames(experiment))

#     background_frames = open_backgrounds(experiment)
#     background = background_calculation(frames, background_frames)
#     if save:
#         print("Saving calculated background image...")
#         background_path = Path(f"data/processed/{experiment}/background.bmp")
#         background_path.parent.mkdir(parents=True, exist_ok=True)
#         background_path.touch(exist_ok=True)
#         print(type(background))
#         iio.imwrite(f"data/processed/{experiment}/background.bmp", background.astype(np.uint8))

#     # background_arr = np.asarray(background, dtype=np.int16)
#     # subtracted = np.asarray(frames, dtype=np.int16) - background_arr
#     subtracted = subtract_background(frames, background)


#     return subtracted

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

