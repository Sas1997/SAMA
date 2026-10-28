"""Shared pytest setup for the SAMA test suite.

Importing samapy writes input files (samapy_inputs/) to the current working
directory, so the whole session runs inside a temporary directory to keep the
repository clean.
"""
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # no display is needed to run the tests

_WORKDIR = tempfile.mkdtemp(prefix="samapy_tests_")


def pytest_configure(config):
    os.chdir(_WORKDIR)
