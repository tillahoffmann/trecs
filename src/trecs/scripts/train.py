from argparse import ArgumentParser
from flax import nnx
import grain
import functools
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm
from typing import Any, cast, Type
from ..experiments import DecoderOnlyExperiment, Experiment, Config, TrainConfig


def checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data_iterators: dict[str, grain.DataLoaderIterator],
    rngs: nnx.Rngs,
) -> None:
    flax_nodes = {"model": model, "optimizer": optimizer, "rngs": rngs}
    flax_args = {
        key: ocp.args.PyTreeSave(nnx.state(value))  # pyright: ignore[reportCallIssue]
        for key, value in flax_nodes.items()
    }
    checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            **{
                f"data_iterator_{split}": grain.checkpoint.CheckpointSave(
                    iterator  # pyright: ignore[reportCallIssue]
                )
                for split, iterator in data_iterators.items()
            },
            **flax_args,
            step=ocp.args.JsonSave(step),  # pyright: ignore[reportCallIssue]
        ),
    )


def restore(
    checkpoint_manager: ocp.CheckpointManager,
    step: int | None,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,
    data_iterators: dict[str, grain.DataLoaderIterator] | None = None,
    rngs: nnx.Rngs | None = None,
) -> int:
    flax_nodes = {"model": model, "optimizer": optimizer, "rngs": rngs}
    flax_states = {key: nnx.state(value) for key, value in flax_nodes.items() if value is not None}
    data_iterators = data_iterators or {}
    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            **{
                f"data_iterator_{split}": grain.checkpoint.CheckpointRestore(
                    iterator  # pyright: ignore[reportCallIssue]
                )
                for split, iterator in data_iterators.items()
            },
            **{
                key: ocp.args.PyTreeRestore(
                    abstract_state  # pyright: ignore[reportCallIssue]
                )
                for key, abstract_state in flax_states.items()
            },
            step=ocp.args.JsonRestore(),  # pyright: ignore[reportCallIssue]
        ),
    )
    for key, value in flax_nodes.items():
        if value is not None:
            nnx.update(value, restored[key])

    return restored["step"]


def as_train_step(loss_fn, safe_update: bool = False):
    """Transform a function to act as a train step."""

    @functools.wraps(loss_fn)
    def _train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, inputs, labels, prng_key
    ):
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, inputs, labels, prng_key)
        if safe_update:
            # We cannot capture optimizer because nnx.cond creates a new "trace
            # context." That's why we have an extra lambda, even though we should be
            # able to just use `optimizer.update` as the true function. For details, see
            # https://github.com/google/flax/discussions/3998#discussioncomment-9780795.
            nnx.cond(
                jnp.isfinite(loss),
                lambda optimizer, model, grad: optimizer.update(model, grad),
                lambda *_: None,
                optimizer,
                model,
                grads,
            )
        else:
            optimizer.update(model, grads)
        return loss

    return _train_step


class _Args:
    # Optional arguments for the experiments.
    seed: int | None
    num_steps: int | None
    eval_every: int | None
    resume: bool
    # Core to the results: the experiment and output directory.
    experiment: str
    config: Path
    output: Path


EXPERIMENTS: dict[str, Type[Experiment]] = {
    "DecoderOnly": DecoderOnlyExperiment,
}


def __main__(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Random generator seed, takes precedence over the experiment config.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of training steps, takes precedence over the experiment config.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        help="Evaluate validation loss every # steps, takes precedence over the experiment config.",
    )
    parser.add_argument("--resume", help="Resume training.", action="store_true")
    parser.add_argument("output", type=Path, help="Output directory.")
    parser.add_argument(
        "experiment",
        help="Experiment class.",
        choices=EXPERIMENTS,
    )
    parser.add_argument("config", type=Path, help="Path to configuration file.")
    args = cast(_Args, parser.parse_args(argv))

    # Complain if the output already exists and we don't intend to resume.
    output_exists = args.output.exists()
    if output_exists and not args.resume:
        raise FileExistsError(
            f"Output path '{args.output}' exists. Delete it to restart or use the "
            "'--resume' flag to resume training."
        )
    # Complain if we want to resume but the output does not exist.
    elif args.resume and not output_exists:
        raise FileNotFoundError(
            f"Cannot resume '{args.output}' because it does not exist."
        )

    # Load the experiment configuration, adjust it where desired, and create the
    # experiment.
    experiment_cls = EXPERIMENTS[args.experiment]
    config_cls = experiment_cls.CONFIG_CLS
    config: Config[TrainConfig, Any] = config_cls.model_validate_json(
        args.config.read_text()
    )
    if args.seed is not None:
        config.train.seed = args.seed
    if args.num_steps is not None:
        config.train.num_steps = args.num_steps
    if args.eval_every is not None:
        config.train.eval_every = args.eval_every
    experiment = experiment_cls(config)

    # Set up random number generation streams for:
    # 1. parameter initialization for model weights
    # 2. dropout regularization during training
    # 3. seed generation for the grain index samplers
    # 4. sampled softmax cross-entropy for the training batches
    # 5. ... and for the validation batches
    streams = ["params", "dropout", "index_sampler", "train_loss", "valid_loss"]
    keys = jax.random.split(jax.random.key(config.train.seed), len(streams))
    keys = dict(zip(streams, keys))
    rngs = nnx.Rngs(**keys)

    # Set up the output directory and take care of any setup business.
    experiment.setup_output(args.output)

    # Create data sources, loaders, and instantiate iterators.
    data_sources = {
        split: experiment.create_data_source(split) for split in ["train", "valid"]
    }
    data_loaders = {
        split: experiment.create_data_loader(
            split,
            data_source,
            seed=int(jax.random.randint(rngs.index_sampler(), (), 0, 2**31 - 1)),
        )
        for split, data_source in data_sources.items()
    }
    data_iterators = {key: iter(value) for key, value in data_loaders.items()}

    checkpoint_path = (args.output / "checkpoints").resolve()
    checkpoint_manager: ocp.CheckpointManager
    with ocp.CheckpointManager(
        checkpoint_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    ) as checkpoint_manager:
        if args.resume:
            model = nnx.eval_shape(lambda: experiment.create_model(nnx.Rngs(0)))
            optimizer = nnx.eval_shape(lambda: experiment.create_optimizer(model))
            step = restore(
                checkpoint_manager,
                model=model,
                optimizer=optimizer,
                data_iterators=data_iterators,
                rngs=rngs,
                step=None,
            )
        else:
            model = experiment.create_model(rngs)
            optimizer = experiment.create_optimizer(model)
            step = 0

        # We use a "safe" update that is only applied when the loss is finite. This
        # mitigates issues with noise from minibatch sampling occasionally messing with
        # the optimization.
        train_step = as_train_step(experiment.evaluate_loss, safe_update=True)
        train_step = nnx.jit(train_step)
        evaluate_loss = nnx.jit(experiment.evaluate_loss)
        valid_loss = jnp.nan

        # Start the training loop.
        with (
            tqdm(total=config.train.num_steps, initial=step) as progress,
            # One writer each for train and validation
            # (https://stackoverflow.com/a/37156491/1150961). Maybe "valid" would be a
            # better name than "eval", but tensorboard orders the plots alphabetically.
            SummaryWriter(str(args.output / "logdir/train")) as train_writer,
            SummaryWriter(str(args.output / "logdir/eval")) as valid_writer,
        ):
            while step < config.train.num_steps:
                # Run one training step.
                inputs, labels = next(data_iterators["train"])
                train_loss = train_step(
                    model, optimizer, inputs, labels, prng_key=rngs.train_loss()
                )
                train_writer.add_scalar("loss", train_loss, global_step=step)
                if not jnp.isfinite(train_loss):
                    print(
                        "WARNING: Training loss was not finite for batch with shape "
                        f"{labels.shape} at step {step:,}: {train_loss}"
                    )

                # Evaluate the validation loss.
                if step % config.train.eval_every == 0:
                    inputs, labels = next(data_iterators["valid"])
                    valid_loss = evaluate_loss(
                        model, inputs, labels, prng_key=rngs.valid_loss()
                    )
                    valid_writer.add_scalar("loss", valid_loss, global_step=step)

                # Checkpoint the model.
                if step % config.train.checkpoint_every == 0:
                    checkpoint(
                        checkpoint_manager,
                        model=model,
                        optimizer=optimizer,
                        data_iterators=data_iterators,
                        rngs=rngs,
                        step=step,
                    )

                # Advance the state.
                step += 1
                progress.update()
                progress.set_description(
                    f"train loss: {train_loss:.2f}; valid loss: {valid_loss:.2f}"
                )

            checkpoint(
                checkpoint_manager,
                model=model,
                optimizer=optimizer,
                data_iterators=data_iterators,
                rngs=rngs,
                step=step,
            )
            checkpoint_manager.wait_until_finished()

        print("done")


if __name__ == "__main__":
    __main__()
