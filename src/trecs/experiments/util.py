from flax import nnx
from grain import DataLoader
from grain.sources import RandomAccessDataSource
from jax import numpy as jnp
from pathlib import Path
import pydantic
from typing import Any


class Experiment(pydantic.BaseModel):
    """Base class for experiments.

    Each experiment comprises model, optimizer, and data pipelines defined in regular Python code,
    with configuration parameters as pydantic fields.
    """

    seed: int
    num_steps: int
    eval_every: int
    checkpoint_every: int
    batch_size: int
    learning_rate: float = pydantic.Field(gt=0)

    def create_model(self, rngs: nnx.Rngs) -> nnx.Module:
        """Create the model to be trained."""
        raise NotImplementedError

    def create_optimizer(self, model: nnx.Module) -> nnx.Optimizer:
        """Create an optimizer for the model."""
        raise NotImplementedError

    def create_data_source(self, split: str) -> RandomAccessDataSource:
        """Create a data source for a given split."""
        raise NotImplementedError

    def create_data_loader(
        self, split: str, data_source: RandomAccessDataSource, seed: int
    ) -> DataLoader:
        """Create data loader for a split with the given source.

        Args:
            split: Split name, e.g., to decide whether operations should include data
                imputation.
            data_source: Source from which samples are drawn for this split.
            seed: Random number generator seed for sampling (if shuffled).
        """
        raise NotImplementedError

    def evaluate_loss(
        self,
        model: nnx.Module,
        inputs: Any,
        labels: Any,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the loss function to optimize.

        Args:
            model: Model.
            inputs: Model inputs.
            labels: Desired model outputs.
            prng_key: Random number generator key for stochastic loss functions.
        """
        raise NotImplementedError

    def setup_output(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
