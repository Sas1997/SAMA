"""The command-line runner accepts the example configuration shipped with the repository."""
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "samapy_config_COMPLETE.yaml"


@pytest.mark.skipif(not CONFIG.is_file(), reason="example configuration not found")
def test_runner_dry_run_validates_example_config(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "from samapy.cli.runner import main; main()",
         "--dry-run", "--config", str(CONFIG)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dry run complete" in result.stdout


def test_runner_output_works_with_a_non_utf8_code_page(tmp_path):
    """Redirected output on Windows uses the system code page (e.g. cp1252)."""
    import os
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    config = Path(__file__).resolve().parents[1] / "samapy_config_COMPLETE.yaml"
    result = subprocess.run([sys.executable, "-c", "from samapy.cli.runner import main; main()",
                             "--config", str(config), "--dry-run"],
                            cwd=tmp_path, capture_output=True, env=env, timeout=600)
    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", "replace")
    assert "dry run complete" in result.stdout.decode("utf-8")
