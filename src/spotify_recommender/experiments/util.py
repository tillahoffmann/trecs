from flax import nnx
from grain import DataLoader
from grain.sources import RandomAccessDataSource
from jax import numpy as jnp
from pathlib import Path
import pydantic
from typing import Any, Generic, TypeVar, Type


class TrainConfig(pydantic.BaseModel):
    num_steps: int = pydantic.Field(description="Number of training steps.")
    seed: int = pydantic.Field(description="Random number generator seed.")
    checkpoint_every: int = pydantic.Field(
        description="Checkpoint the training state every # steps.", default=1_000
    )
    validate_every: int = pydantic.Field(
        description="Evaluate the validation loss every # steps.", default=1_00
    )
    batch_size: int = pydantic.Field(description="Size of mini batches.")
    learning_rate: float = pydantic.Field(description="Optimizer learning rate.", gt=0)


T = TypeVar("T", bound=TrainConfig)
A = TypeVar("A", bound=pydantic.BaseModel)


class Config(pydantic.BaseModel, Generic[T, A]):
    train: T
    architecture: A


C = TypeVar("C", bound=Config)


class Experiment(Generic[C]):
    """Base class for experiments.

    Each experiment comprises two pieces:

    1. Model, optimizer, and data pipelines defined in regular Python code.
    2. Simple configuration parameters, such as data sources, as a Pydantic model.

    The distinction between "simple configuration parameters" and things that are "not
    simple" is of course blurry. Whenever we start implementing something that feels
    like a parser to construct 1. from 2., we've probably gone astray.
    """

    CONFIG_CLS: Type

    def __init__(self, config: C) -> None:
        self.config = config

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
