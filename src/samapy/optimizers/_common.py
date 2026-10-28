"""Shared helpers for the SAMA optimizers.

Settings are read from ``InData`` when an optimizer is constructed, not when
its module is imported, so changes made to ``InData`` before construction
(for example by the CLI runner or by tests) are always honoured.
"""
import numpy as np
from numba import njit


@njit
def _seed_numba(seed):
    # Numba keeps its own random state, separate from NumPy's, so it must be
    # seeded from inside a compiled function.
    np.random.seed(seed)


def seed_all(seed):
    """Seed NumPy and Numba random generators. Does nothing when seed is None."""
    if seed is None:
        return
    np.random.seed(seed)
    _seed_numba(seed)


def resolve_settings(InData, var_min, var_max, n_pop, max_it, run_time):
    """Return bounds and run settings, falling back to the current InData values.

    Default bounds are the SAMA design-variable bounds
    [PV modules, wind turbines, batteries, generator, inverter], scaled by
    the component switches, exactly as in earlier SAMA versions.
    """
    switches = [InData.PV, InData.WT, InData.Bat, InData.DG, 1]
    if var_min is None:
        var_min = InData.VarMin * switches
    else:
        var_min = np.asarray(var_min, dtype=float)
    if var_max is None:
        var_max = InData.VarMax * switches
    else:
        var_max = np.asarray(var_max, dtype=float)
    if np.shape(var_min) != np.shape(var_max):
        raise ValueError("var_min and var_max must have the same length")
    if np.any(np.asarray(var_max) < np.asarray(var_min)):
        raise ValueError("each var_max entry must be >= the matching var_min entry")

    n_pop = int(InData.nPop if n_pop is None else n_pop)
    max_it = int(InData.MaxIt if max_it is None else max_it)
    run_time = int(InData.Run_Time if run_time is None else run_time)
    if n_pop < 1 or max_it < 1 or run_time < 1:
        raise ValueError("n_pop, max_it and run_time must all be at least 1")
    return var_min, var_max, n_pop, max_it, run_time
