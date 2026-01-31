import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import cKDTree

import pims
import napari
from pathlib import Path
import trackpy as tp

def calculate_velocity(trajectories, csv_out, fps=0.25, pixel_size=6.9e-6, unwrap=True):
    """Calculates velocity using finite differences
    """
    

    def _vel_for_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("frame").copy()

        # If you have irregular frames, compute dt from frame differences
        # assuming 'frame' is integer indices at FPS:
        t = g["frame"].to_numpy(dtype=float) / fps

        x = g["x"].to_numpy(dtype=float) * pixel_size
        y = g["y"].to_numpy(dtype=float) * pixel_size

        n = len(g)
        vx = np.full(n, np.nan)
        vy = np.full(n, np.nan)

        if n >= 2:
            # One-sided at ends
            vx[0] = (x[1] - x[0]) / (t[1] - t[0])
            vy[0] = (y[1] - y[0]) / (t[1] - t[0])
            vx[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
            vy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])

        if n >= 3:
            # Central difference for interior points (3-point stencil)
            dt = t[2:] - t[:-2]
            vx[1:-1] = (x[2:] - x[:-2]) / dt
            vy[1:-1] = (y[2:] - y[:-2]) / dt

        speed = np.hypot(vx, vy)  # magnitude
        # angles (radians) from +x axis
        theta = np.arctan2(vy, vx)  # [-pi, pi]
        # undefined when speed == 0 or NaN velocities
        theta[~np.isfinite(speed) | (speed == 0)] = np.nan

        # unwrap per particle if requested (keeps continuity over time)
        if unwrap:
            # apply unwrap on valid (finite) entries only
            valid_idx = np.where(np.isfinite(theta))[0]
            if len(valid_idx) > 0:
                theta_unwrapped = theta.copy()
                theta_unwrapped[valid_idx] = np.unwrap(theta[valid_idx])
                theta = theta_unwrapped

        theta_deg = np.degrees(theta)

        g["vx"] = vx
        g["vy"] = vy
        g["speed"] = speed
        g["angle_rad"] = theta
        return g

    out = trajectories.groupby("particle", group_keys=False).apply(_vel_for_group)

    out.to_csv(csv_out, index=False)

    return out

def pair_correlations(vel_df, r_max=1e10, dr=2.0):
    """
    Compute velocity correlation Cvv(r) from trajectory CSV
    with columns: frame, particle, x, y, vx, vy, theta_deg, ...
    """
    df = vel_df

    results = []

    for t, g in df.groupby("frame"):
        pts = g[["x","y"]].to_numpy(float)
        V   = g[["vx","vy"]].to_numpy(float)
        ids = g["particle"].to_numpy(int)

        if len(pts) < 2:
            continue

        tree = cKDTree(pts)
        pairs = tree.query_pairs(r_max)

        norms = np.linalg.norm(V, axis=1) + 1e-12
        Vhat = (V.T / norms).T  # normalize to unit vectors

        for i, j in pairs:
            r = np.linalg.norm(pts[i] - pts[j])
            corr = np.dot(Vhat[i], Vhat[j])  # cosine of angle difference
            results.append((t, ids[i], ids[j], r, corr))

    pairs_df = pd.DataFrame(results, columns=["frame","i","j","r","corr"])

    # bin by separation
    pairs_df["r_bin"] = (pairs_df["r"] // dr) * dr
    Cvv = pairs_df.groupby("r_bin")["corr"].mean().reset_index(name="Cvv")


    return pairs_df, Cvv

def clean_fiji_csv(experiment):
    fiji_df = pd.read_csv(f"data/processed/{experiment}/fiji_tracks.csv")
    df = fiji_df[["TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T", "FRAME", "TOTAL_INTENSITY_CH1", "ELLIPSE_THETA", "ELLIPSE_ASPECTRATIO", "AREA"]]

    df = df.rename(columns=({"TRACK_ID": "particle",
                        "TOTAL_INTENSITY_CH1": "mass",
                        "AREA": "size",
                        "POSITION_X": "x",
                        "POSITION_Y": "y",
                        "FRAME": "frame",
                        "ELLIPSE_THETA": "theta",
                        "ELLIPSE_ASPECTRATIO": "eccentricity"}))
    # df["frame"] = pd.to_numeric(df["frame"])
    df = df.drop(index=[0, 1, 2])
    
    return df

def labels_stack_from_centroids(
    features_df,
    image_stack_shape,     # (T, H, W)
    radius=5,
    frame_col="frame",
    x_col="x",
    y_col="y",
    brightness_col="mass",     # tp.batch output usually has 'mass' (integrated brightness)
    min_brightness=None,       # set e.g. to 200 or use np.percentile(features_df['mass'], 50)
):
    """
    Convert tp.batch detections into a napari Labels stack where ALL kept detections
    share the SAME label value = 1 (background = 0).

    This is ideal if you want to manually edit a "bright detection mask" in napari
    without assigning unique IDs yet.

    - If min_brightness is provided, only detections with features_df[brightness_col] >= min_brightness are used.
    - Each detection stamps a disk of radius `radius` into the labels.
    """
    T, H, W = image_stack_shape
    labels = np.zeros((T, H, W), dtype=np.int32)

    # Filter by brightness if requested
    f = features_df
    if min_brightness is not None:
        if brightness_col not in f.columns:
            raise ValueError(
                f"brightness_col='{brightness_col}' not in features_df columns: {list(f.columns)}"
            )
        f = f[f[brightness_col] >= min_brightness]

    # Precompute disk offsets
    rr = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(rr, rr, indexing="ij")
    disk = (xx**2 + yy**2) <= radius**2
    dy, dx = np.where(disk)
    dy = dy - radius
    dx = dx - radius

    # Group by frame; stamp label=1 for each detection
    for t, g in f.groupby(frame_col, sort=True):
        t = int(t)
        if t < 0 or t >= T:
            continue

        xs = np.rint(g[x_col].to_numpy()).astype(int)
        ys = np.rint(g[y_col].to_numpy()).astype(int)

        for cx, cy in zip(xs, ys):
            Y = cy + dy
            X = cx + dx
            valid = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
            labels[t, Y[valid], X[valid]] = 1  # <- single label for all detections

    return labels

def run_analysis(cfg: dict, config_path: Path):

    frames = pims.open(str(Path(cfg["analysis"]["input_dir"]) / "*.bmp"))
    print(f"Loaded {len(frames)} frames")

    TP_PARAMS = cfg["analysis"]["tp_batch"]
    features = tp.batch(
        frames,
        threshold=TP_PARAMS["brightness_threshold"],
        diameter=TP_PARAMS["diameter_threshold"],
        minmass=TP_PARAMS["minimum_mass_threshold"],
        invert=False
    )

    if TP_PARAMS["save_batch_frames"]:
        features.to_csv(f"{cfg["paths"]["output_dir"]}", index=False)

    LABELS_PARAMS = cfg["analysis"]["labels_stack"]
    labels_stack = labels_stack_from_centroids(
        features,
        frames.shape,
        radius=LABELS_PARAMS["radius"],
        brightness_col=LABELS_PARAMS["brightness_col"],
        min_brightness=LABELS_PARAMS["min_brightness"]
    )

    if cfg["analysis"]["do_napari"]: 
        viewer = napari.Viewer()

        viewer.add_image(
        np.asarray(frames),
        name="images",
        colormap="gray"
        )
        labels_layer = viewer.add_labels(labels_stack, name="seg_from_batch", opacity=0.5)
        napari.run()

        edited_labels = np.array(labels_layer.data) * 255

        features = tp.batch(
            edited_labels,
            threshold=10,
            diameter=13,
            invert=False
        )

        if LABELS_PARAMS["save_napari_edits"]:
            features.to_csv(Path(cfg["analysis"]["output_dir"]) / "edited_features_batch.csv")
    

    TRACK_PARAMS = cfg["analysis"]["track"]
    tracks_df = tp.link_df(
        features,
        search_range=TRACK_PARAMS["search_range"],
        memory=TRACK_PARAMS["memory"]
    )

    # print(f"{tracks_df['particle'].nunique()} tracks found")
    tracks_df.head()
    tracks_array = tracks_df[['particle', 'frame', 'y', 'x']].to_numpy()

    if TRACK_PARAMS["save_tracks_df"]:
        tracks_df.to_csv(Path(cfg["analysis"]["output_dir"]) / "tracks.csv")


# if __name__ == "__main__":
#     experiment = "20250714_passiveInteraction_2_numSpiros_2"
#     trajectories = clean_fiji_csv(experiment)
#     vel_df = calculate_velocity(trajectories, f"data/processed/{experiment}/velocities_fiji.csv")
#     pairs_df, Cvv = pair_correlations(vel_df)
#     Cvv.to_csv(f"data/processed/{experiment}/cvv.csv")
#     sns.relplot(data=Cvv, x="r_bin", y="Cvv")
#     plt.show()