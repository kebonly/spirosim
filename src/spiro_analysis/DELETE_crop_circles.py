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

def get_circle_crop_info(image_path):
    print("getting circle crop info")
    img = mpimg.imread(image_path)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click and drag to draw/resize/move circle. Press Enter to confirm.")

    selector = CircleSelector(ax, img)
    plt.show()
    return selector.circle.center, selector.circle.radius

def update_experiment_json(json_path, experiment_name, circle_center, circle_radius):
    # Step 1: Read existing JSON (or start with empty dict if file doesn't exist)
    print("Updaing experiment json...")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Step 2: Check if experiment already exists
    if experiment_name in data:
        print(f"Experiment '{experiment_name}' already exists. No update made.")
    else:
        # Step 3: Add new experiment entry
        data[experiment_name] = {
            "circle_center": list(circle_center),
            "circle_radius": circle_radius
        }
        # Step 4: Write back to JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Added experiment '{experiment_name}' to {json_path}.")