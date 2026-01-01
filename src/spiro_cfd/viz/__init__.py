
import matplotlib.pyplot as plt
import numpy as np


def plot_quiver_field(
    X, Y, Ux, Uy,
    *,
    mask_center=None,
    mask_radius=0.05,
    step=4,
    scale=None,
    title=None,
    figsize=(6, 6)
):
    X, Y, Ux, Uy = map(np.asarray, (X, Y, Ux, Uy))

    if mask_center is not None:
        x0, y0 = mask_center
        mask = (X - x0)**2 + (Y - y0)**2 < mask_radius**2
        Ux = np.ma.array(Ux, mask=mask)
        Uy = np.ma.array(Uy, mask=mask)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")

    ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        Ux[::step, ::step], Uy[::step, ::step],
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.003
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_streamlines(
    X, Y, Ux, Uy,
    *,
    mask_center=None,
    mask_radius=0.05,
    density=1.3,
    linewidth=1.0,
    color_by_speed=True,
    title=None,
    figsize=(6, 6)
):
    X, Y, Ux, Uy = map(np.asarray, (X, Y, Ux, Uy))
    speed = np.sqrt(Ux**2 + Uy**2)

    if mask_center is not None:
        x0, y0 = mask_center
        mask = (X - x0)**2 + (Y - y0)**2 < mask_radius**2
        Ux = np.ma.array(Ux, mask=mask)
        Uy = np.ma.array(Uy, mask=mask)
        speed = np.ma.array(speed, mask=mask)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")

    ax.streamplot(
        X, Y, Ux, Uy,
        density=density,
        linewidth=linewidth,
        arrowsize=1.2,
        color=speed if color_by_speed else None
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
