from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
from numpy.typing import NDArray

import numpy as np
from numerics.operators import laplacian_vec

Array = NDArray[np.floating]
Flow = Callable[[Array], Array]

def faxen_translation_force_free(
        u_inf: Flow,
        x0: Array,
        a: float,
        h: float
) -> Array:
    """
    Faxen's 1st law for a force-free sphere:
        U = u_inf(x0) + (a^2/6) ∇^2 u_inf(x0)
    """

    u0 = np.asarray(u_inf(x0), dtype=float)
    lap_u0 = laplacian_vec(u_inf, x0, h)
    return u0 + ( a * a / 6 ) * lap_u0


