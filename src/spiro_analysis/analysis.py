import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import cKDTree

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
    
if __name__ == "__main__":
    experiment = "20250714_passiveInteraction_2_numSpiros_2"
    trajectories = clean_fiji_csv(experiment)
    vel_df = calculate_velocity(trajectories, f"data/processed/{experiment}/velocities_fiji.csv")
    pairs_df, Cvv = pair_correlations(vel_df)
    Cvv.to_csv(f"data/processed/{experiment}/cvv.csv")
    sns.relplot(data=Cvv, x="r_bin", y="Cvv")
    plt.show()