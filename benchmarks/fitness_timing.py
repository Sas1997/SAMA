"""Time one SAMA fitness evaluation with and without Numba compilation.

Usage:
    python benchmarks/fitness_timing.py               # compiled (Numba JIT)
    NUMBA_DISABLE_JIT=1 python benchmarks/fitness_timing.py   # interpreted

Uses the bundled default inputs (Input_Data.py) and the design
[PV 12 kW, no wind, 2 battery packs, no generator, 8 kW inverter]. Each
evaluation is a full fitness call: 8,760-hour dispatch, bills and lifecycle
cost. The first call (which includes Numba compilation) is excluded; the
median of the timed calls is reported.
"""
import os
import platform
import statistics
import sys
import time

import numpy as np

from samapy.core.Fitness import fitness

X = np.array([12.0, 0.0, 2.0, 0.0, 8.0])
interpreted = os.environ.get("NUMBA_DISABLE_JIT") == "1"
repeats = 20 if interpreted else 200

fitness(X)  # warm-up, includes compilation
times = []
for _ in range(repeats):
    t0 = time.perf_counter()
    fitness(X)
    times.append(time.perf_counter() - t0)

import numba  # noqa: E402

print(f"mode        : {'interpreted (NUMBA_DISABLE_JIT=1)' if interpreted else 'compiled (Numba)'}")
print(f"median      : {statistics.median(times) * 1000:.2f} ms per evaluation ({repeats} calls)")
print(f"python      : {sys.version.split()[0]}, numba {numba.__version__}, numpy {np.__version__}")
print(f"machine     : {platform.processor() or platform.machine()}, {platform.system()}")
