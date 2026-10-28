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
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dry run complete" in result.stdout
