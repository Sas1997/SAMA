"""Console output helper for the command-line tools."""
import sys


def use_utf8_output():
    """
    Write console output as UTF-8.

    The command-line tools print symbols such as check marks and box lines.
    On Windows, when the output is redirected to a file or pipe, Python uses
    the system code page (for example cp1252), which cannot encode them and
    raises UnicodeEncodeError. Switching stdout and stderr to UTF-8 avoids
    this; characters that still cannot be written are replaced.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a reconfigurable text stream
            pass
