"""The package and its main modules import without errors."""
import importlib

import pytest

MODULES = [
    "samapy",
    "samapy.core.Input_Data",
    "samapy.core.Fitness",
    "samapy.ems.EMS",
    "samapy.results.Results",
    "samapy.optimizers.swarm",
    "samapy.optimizers.AdvancedDifferentialEvolution",
    "samapy.optimizers.ArtificialBeeColony",
    "samapy.optimizers.GreyWolfOptimizer",
    "samapy.cli.runner",
    "samapy.cli.config_loader",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None


def test_optimizer_classes_exist():
    from samapy.optimizers.swarm import Swarm
    from samapy.optimizers.AdvancedDifferentialEvolution import AdvancedDifferentialEvolution
    from samapy.optimizers.ArtificialBeeColony import ImprovedArtificialBeeColony
    from samapy.optimizers.GreyWolfOptimizer import GreyWolfOptimizer

    for cls in (Swarm, AdvancedDifferentialEvolution, ImprovedArtificialBeeColony, GreyWolfOptimizer):
        assert callable(getattr(cls, "optimize"))
