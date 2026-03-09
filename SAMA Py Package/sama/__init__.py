import os
from pathlib import Path


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