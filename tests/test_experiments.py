from pathlib import Path
from trecs import experiments
from trecs.util import load_module
import pytest


EXPERIMENTS = [
    path
    for path in Path(experiments.__file__).parent.glob("*/*.py")
    if not path.stem.startswith("_")
]


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda x: "/".join(x.parts[-2:]))
def test_setup_experiment(path: Path) -> None:
    module = load_module(path)
    assert hasattr(module, "setup")
    experiment = module.setup()
    assert isinstance(experiment, experiments.Experiment)
