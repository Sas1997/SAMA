"""Data files bundled with the package are present."""
import os

import pytest

from samapy import get_content_path

BUNDLED_FILES = [
    "house_load.xlsx",
    "METEO.csv",
]


@pytest.mark.parametrize("filename", BUNDLED_FILES)
def test_bundled_file_exists(filename):
    path = get_content_path(filename)
    assert os.path.isfile(path), f"missing bundled file: {path}"
    assert os.path.getsize(path) > 0, f"bundled file is empty: {path}"
