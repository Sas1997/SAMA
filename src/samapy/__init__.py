import os
from pathlib import Path

try:  # version of the installed package, from pyproject.toml
    from importlib.metadata import version as _version, PackageNotFoundError
    __version__ = _version("samapy")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"


def get_content_path(filename):
    """
    Get the full path to a file in the content folder.

    Parameters
    ----------
    filename : str
        Name of file in content folder (e.g., 'house_load.xlsx')
        Can include subfolders (e.g., 'weather/data.csv')

    Returns
    -------
    str
        Full path to the file
    """
    # Get the directory where this __init__.py file is located
    package_dir = Path(__file__).parent
    content_path = package_dir / 'content' / filename
    return str(content_path)