import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import trackpy as tp
import pims
import imageio.v2 as imageio
from pathlib import Path

from preprocess.background import *
from preprocess.crop_circles import *




if __name__ == "__main__":

    # Optionally, tweak styles.
    mpl.rc('image', cmap='gray')

    experiment = "backgroundSubtraction"
    json_file_path = Path(f"data/metadata/{experiment}/metadata.json")
    # Make sure the directory exists (create parents if needed)
    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Now create the file if it doesn't exist
    json_file_path.touch(exist_ok=True)
    
    
    # First, check to see if the relevant metadata is there
    experiment_exists = False
    with open(json_file_path, "r+") as f:
        try:
            data = json.load(f)

        except json.JSONDecodeError:
            json.dump({}, f)
            data = json.load(f)

        experiment_exists = experiment in data


    if not experiment_exists:
        frame = os.listdir(f"data/test/{experiment}")[0]
        center, radius = get_circle_crop_info(f"data/test/{experiment}/{frame}")

        update_experiment_json(json_file_path, experiment, center, radius)

    # background = imageio.imread(f"data/test/{experiment}/background_20250716162440161.bmp")
    background = pims.open(f"data/test/{experiment}/background*.bmp")
    
    # Now we want to circle crop
    frames = pims.open(f"data/test/{experiment}/Pic*.bmp")
    # Now want to do background subtraction

    with open(json_file_path, "r") as f:
        data = json.load(f)
        metadata = data[experiment]
        center = tuple(metadata.get("circle_center"))
        radius = metadata.get("circle_radius")
    
    
    frames = background_subtraction(gray(frames), background)
    frames = circle_crop(frames, center, radius)

    if True:
        cropped_path = f"data/processed/{experiment}/cropped"
        if not os.path.exists(cropped_path):
            os.makedirs(cropped_path)
            print(f"Created directory: {cropped_path}")
        for i, frame in enumerate(frames):
            xlim = [int(center[0]-radius), int(center[0]+radius)]
            ylim = [int(center[1]-radius), int(center[1]+radius)]
            cropped = frame[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            cropped = np.clip(cropped, 0, 255).astype(np.uint8)
            imageio.imwrite(f"{cropped_path}/cropped_{i}.bmp", cropped)

    # Need to do background subtraction

    # Now operate on the cropped image.
    # This will be moved to analysis at some point.
    DIAMETER_THRESHOLD = 39 # must be odd number
    BRIGHTNESS_THRESHOLD = 30

    files = os.listdir(f"data/processed/{experiment}/cropped")
    files = sorted(files)
    img = imageio.imread(f"{cropped_path}/{files[0]}")
    img_height, img_width = img.shape
    stack = np.zeros((len(files), img_height, img_width))
    for i, cropped_img in enumerate(files):
        stack[i,:,:] = imageio.imread(f"{cropped_path}/{cropped_img}")

    f = tp.batch(stack, DIAMETER_THRESHOLD, threshold=BRIGHTNESS_THRESHOLD, invert=False)
    t = tp.link(f, 100, memory=4)
    f.to_csv(f"data/processed/{experiment}/particles_in_frames.csv")
    t.to_csv(f"data/processed/{experiment}/tracks.csv")

    MICRONS_PER_PIXEL = 100
    FPS = 4
    msd = tp.motion.msd(t, MICRONS_PER_PIXEL, FPS, detail=True)
    msd.to_csv(f"data/processed/{experiment}/msd.csv")

    fig, ax = plt.subplots(figsize=(6,6))
    tp.plot_traj(t, ax=ax)
    # plt.xlim(0,img_width)
    # plt.ylim(0,img_height) # need to flip
    plt.gca().invert_yaxis()
    plt.savefig(f"data/processed/{experiment}/trajectory.png")

    # plt.show()