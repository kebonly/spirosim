
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import trackpy as tp
import pims
import imageio.v2 as imageio
import pandas as pd


from spiro_analysis.preprocess import *

import click

def save_frames(experiment, frames, center, radius, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created directory: {dir}")
    for i, frame in enumerate(frames):
        # xlim = [int(center[0]-radius), int(center[0]+radius)]
        # ylim = [int(center[1]-radius), int(center[1]+radius)]
        # cropped = frame[ylim[0]:ylim[1], xlim[0]:xlim[1]]
        # cropped = np.clip(cropped, 0, 255).astype(np.uint8)
        fout = np.clip(frame, 0, 255).astype(np.uint8)
        imageio.imwrite(f"{dir}/cropped_{i}.bmp", fout)

    return dir

# def load_stack(experiment, cropped_path):
#     files = os.listdir(f"data/processed/{experiment}/cropped")
#     files = sorted(files)
#     img = imageio.imread(f"{cropped_path}/{files[0]}")
#     img_height, img_width = img.shape
#     stack = np.zeros((len(files), img_height, img_width))
#     for i, cropped_img in enumerate(files):
#         stack[i,:,:] = imageio.imread(f"{cropped_path}/{cropped_img}")

#     return stack

@click.command()
@click.argument("experiment_name")   # required positional argument
def run(experiment_name: str):
    experiment = experiment_name

    # TODO: Maybe make an Experiment Class?
    radius, center = open_metadata(experiment)
    # background = open_backgrounds(experiment)
    
    if read_metadata(experiment, "background_subtraction_is_done"):
        frames = pims.open(f"data/processed/{experiment}/cropped/cropped_*")
        print("Background subtraction already done... skipping step.")
    else:
        frames_background_subtracted = background_subtraction(experiment)
        frames = circle_crop(frames_background_subtracted, center, radius)

        save_frames(experiment, frames, center, radius, f"data/processed/{experiment}/cropped")
        update_metadata(experiment, "background_subtraction_is_done", True)


    if read_metadata(experiment, "enhance_contrast_is_done"):
        print("Contrast enhancement done already... skipping.")
    else:
        print("Starting enhance_contrast...")
        frames = enhance_contrast(frames, method="clahe", out_dtype=np.uint8)
        save_frames(experiment, frames, center, radius, f"data/processed/{experiment}/enhanced")
        update_metadata(experiment, "enhance_contrast_is_done", True)
        print("Finished enhance_contrast")



    # Now operate on the cropped image.
    # This will be moved to analysis at some point.
    # TODO: put these settings in experiment metadata instead.
    DIAMETER_THRESHOLD = 31 # must be odd number
    BRIGHTNESS_THRESHOLD = 10
    MEMORY = 8

    stack = pims.open(f"data/processed/{experiment}/enhanced/cropped_*")#load_stack(experiment, f"data/processed/{experiment}/enhanced")
    print("done loading stack")

    if read_metadata(experiment, "tracking_is_done"):
        print("Tracking done alread. Loading files...")
        f = pd.read_csv(f"data/processed/{experiment}/particles_in_frames.csv")
        t = pd.read_csv(f"data/processed/{experiment}/tracks.csv")

    else:

        f = tp.batch(stack, DIAMETER_THRESHOLD,threshold=BRIGHTNESS_THRESHOLD, minmass=1000, invert=False)

        f.to_csv(f"data/processed/{experiment}/particles_in_frames.csv")

        t = tp.link(f, 20, memory=MEMORY)
        print("done tracking")
        t.to_csv(f"data/processed/{experiment}/tracks.csv")
        update_metadata(experiment, "tracking_is_done", True)

    # print(stack.shape)
    fig, ax = plt.subplots(figsize=(6,6))
    tp.plot_traj(t, ax=ax).axis

    
    plt.xlim(0,stack.shape[2])
    plt.ylim(0,stack.shape[1]) # need to flip
    plt.gca().invert_yaxis()
    perimeter = patches.Circle((stack.shape[2]/2, stack.shape[1]/2), read_metadata(experiment, "circle_radius"), fill=False, linewidth=2)
    ax.add_patch(perimeter)
    plt.savefig(f"data/processed/{experiment}/trajectory.png")

    MICRONS_PER_PIXEL = 100
    FPS = 4
    msd = tp.motion.msd(t, MICRONS_PER_PIXEL, FPS, detail=True)
    msd.to_csv(f"data/processed/{experiment}/msd.csv")



    # plt.show()

if __name__ == '__main__':
    run()