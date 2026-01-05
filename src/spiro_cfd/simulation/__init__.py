from typing import Callable
import numpy as np

from spiro_cfd.faxen import faxen_translation_force_free

from numerics.operators import curl_2d

from numpy.typing import NDArray

Array = NDArray[np.floating]

def simulate_active_particle(
    *,
    u_inf: Callable[[Array], Array],
    x0: tuple[float, float],
    theta0: float,
    T: float,
    dt: float,
    a: float,
    v0: float,
    h: float,
    use_faxen: bool = True,
) -> tuple[Array, Array]:
    """
    Docstring for simulate_active_particle for spherical particle in Faxen flow.
    
    :param u_inf: Takes in position and outputs corresponding velocity.
    :type u_inf: Callable[[Array], Array]
    :param x0: Description
    :type x0: tuple[float, float]
    :param theta0: Description
    :type theta0: float
    :param T: Description
    :type T: float
    :param dt: Description
    :type dt: float
    :param a: Description
    :type a: float
    :param v0: Description
    :type v0: float
    :param h: Spatial discretization.
    :type h: float
    :param use_faxen: Description
    :type use_faxen: bool
    :return: Description
    :rtype: Array
    """
    
    n = int(T / dt)                 # number of time steps
    x = np.zeros((n, 2))            # trajectory list
    th = np.zeros(n)                # angle
    x[0] = np.asarray(x0, float)    # initial position
    th[0] = float(theta0)           # initial angle

    for k in range(n - 1):
        pos = x[k]
        
        theta = th[k]
        p = np.array([np.cos(theta), np.sin(theta)])

        # U_bg = u_inf(pos) # background flow

        U_bg = faxen_translation_force_free(u_inf, pos, a, h=h)

        U = U_bg + v0 * p

        omega = curl_2d(u_inf, pos, h=h)
        th_dot = 0.5 * omega

        x[k + 1] = pos + dt * U
        th[k + 1] = theta + dt * th_dot

    return x, th