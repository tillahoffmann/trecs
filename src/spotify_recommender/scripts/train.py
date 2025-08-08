from argparse import ArgumentParser
import collectiontools
from flax import nnx
import grain
import jax
from jax import numpy as jnp
import optax
from orbax import checkpoint as ocp
import numpy
from pathlib import Path
import pickle
import sqlite3
from tensorboardX import SummaryWriter
from tqdm import tqdm
from typing import Any, cast
from ..models import PlaylistDecoder
from ..data import (
    Sqlite3Dataset,
    SELECT_PLAYLISTS_BY_SPLIT,
    SELECT_DISTINCT_TRACK_IDS_BY_SPLIT,
    pad_batch,
    truncate_batch,
    BatchTransform,
    Encoder,
    LambdaMap,
    LambdaRandomMap,
)
from ..util import sampled_dot_cross_entropy_with_integer_labels


Batch = list[dict[str, Any]]


class _Args:
    num_layers: int
    num_features: int
    num_hidden: int
    num_heads: int
    dropout: float
    seed: int
    context_length: int
    resume: bool
    input: Path
    output: Path
    batch_size: int
    learning_rate: float
    num_epochs: int | None
    num_steps: int | None
    encoder: Path | None
    checkpoint_every: int
    valid_every: int
    unk_proba: float


class Counter(nnx.Metric):
    def __init__(self):
        """Counting metric."""
        self.total = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> None:
        self.total.value = jnp.array(0, dtype=jnp.float32)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *, count: int = 1
    ) -> None:
        self.total.value += count

    def compute(self) -> int:
        return self.total.value


def _create_decoder(args: _Args, rngs: nnx.Rngs, num_tracks: int) -> PlaylistDecoder:
    return PlaylistDecoder(
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=args.num_features,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
        rngs=rngs,
        num_tracks=num_tracks,
    )


def _create_optimizer(args: _Args, model: PlaylistDecoder) -> nnx.Optimizer:
    return nnx.Optimizer(
        model, optax.adamw(args.learning_rate, weight_decay=0.01), wrt=nnx.Param
    )


def encode_track_ids(batch: dict, encoder: Encoder):
    batch["track_id"] = jax.tree.map(encoder, batch["track_id"])
    return batch


def inputs_and_labels(batch: dict) -> tuple[dict, jnp.ndarray]:
    return (
        {key: value[:, :-1] for key, value in batch.items()},
        batch["track_id"][:, 1:],
    )


def loss_fn(
    model: PlaylistDecoder,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    key: jnp.ndarray,
    eop_token: int,
) -> jnp.ndarray:
    """Evaluate the masked cross-entropy loss function for track labels. Masking is
    applied such that only the first <EOP> token contributes to the loss.

    Args:
        model: Decoder model.
        inputs: Input to the model.
        labels: Outputs with shape (batch_size, num_tokens).
        key: Random number generator key for sampling the loss function.
        eop_token: End of playlist token for masking.
    """
    embeddings = model(inputs)
    batch_size, num_tokens, num_features = embeddings.shape
    # Flatten all the data so we have a standard (batch_size, event_size) setup.
    flat_embeddings = embeddings.reshape((batch_size * num_tokens, num_features))
    flat_labels = labels.reshape((batch_size * num_tokens,))

    # Evaluate the loss.
    sampled_loss = sampled_dot_cross_entropy_with_integer_labels(
        key, flat_embeddings, model.track_embedding.embedding.value, flat_labels
    )

    # Mask out anything after the first end of playlist token.
    first = jnp.argmax(labels == eop_token, axis=1)
    length = labels.shape[-1]
    mask = (jnp.arange(length) <= first[:, None]).ravel()
    return sampled_loss @ mask.ravel() / mask.sum()


@nnx.jit
def train_step(
    model: PlaylistDecoder,
    optimizer: nnx.Optimizer,
    inputs: jnp.ndarray,
    labels: jnp.ndarray,
    key: jnp.ndarray,
    eop_token: int,
) -> float:
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, inputs, labels, key, eop_token)
    optimizer.update(model, grads)
    return loss


def checkpoint_training_state(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics,
    train_data_iter,
    valid_data_iter,
    loss_rngs,
):
    nnx_items = {
        "model": model,
        "loss_rngs": loss_rngs,
        "optimizer": optimizer,
        "my-metrics": metrics,
    }
    nnx_args = {
        key: ocp.args.PyTreeSave(nnx.state(value))  # pyright: ignore[reportCallIssue]
        for key, value in nnx_items.items()
    }
    checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            train_data_iter=grain.checkpoint.CheckpointSave(  # pyright: ignore[reportCallIssue]
                train_data_iter  # pyright: ignore[reportCallIssue]
            ),
            valid_data_iter=grain.checkpoint.CheckpointSave(  # pyright: ignore[reportCallIssue]
                valid_data_iter  # pyright: ignore[reportCallIssue]
            ),
            **nnx_args,
        ),
    )
    checkpoint_manager.wait_until_finished()


def restore_training_state(
    checkpoint_manager: ocp.CheckpointManager,
    step: int | None = None,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics,
    train_data_iter,
    valid_data_iter,
    loss_rngs,
) -> dict:
    nnx_items = {
        "model": model,
        "loss_rngs": loss_rngs,
        "optimizer": optimizer,
        "my-metrics": metrics,  # BUG: You cannot save a field called metrics.
    }
    nnx_splits = {key: nnx.split(value) for key, value in nnx_items.items()}
    nnx_args = {
        key: ocp.args.PyTreeRestore(state)  # pyright: ignore[reportCallIssue]
        for key, (_, state) in nnx_splits.items()
    }

    # Restore state.
    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            train_data_iter=grain.checkpoint.CheckpointRestore(  # pyright: ignore[reportCallIssue]
                train_data_iter  # pyright: ignore[reportCallIssue]
            ),
            valid_data_iter=grain.checkpoint.CheckpointRestore(  # pyright: ignore[reportCallIssue]
                valid_data_iter  # pyright: ignore[reportCallIssue]
            ),
            **nnx_args,
        ),
    )

    # Reassemble graph def and state and return.
    nnx_items = {
        key: nnx.merge(graphdef, getattr(restored, key))
        for key, (graphdef, _) in nnx_splits.items()
    }
    return {
        "train_data_iter": train_data_iter,
        "valid_data_iter": valid_data_iter,
        **nnx_items,
    }


def __main__(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--num-layers", help="Number of transformer layers.", type=int, default=6
    )
    parser.add_argument(
        "--num-features", help="Number of embedding features.", type=int, default=128
    )
    parser.add_argument(
        "--num-hidden",
        help="Number of hidden units in feed-forward network of transformer block.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-heads", help="Number of attention heads.", type=int, default=8
    )
    parser.add_argument(
        "--dropout", help="Dropout probability.", type=float, default=0.1
    )
    parser.add_argument(
        "--seed", help="Random number generator seed.", type=int, default=42
    )
    parser.add_argument(
        "--context-length", help="Context window size.", type=int, default=50
    )
    parser.add_argument(
        "--num-epochs",
        help="Number of training epochs, incompatible with `--num-steps`. May be a float for partial epochs.",
        type=float,
    )
    parser.add_argument(
        "--num-steps",
        help="Number of training steps, incompatible with `--num-epochs`.",
        type=int,
    )
    parser.add_argument("--batch-size", type=int, help="Batch size.", default=8)
    parser.add_argument(
        "--checkpoint-every",
        help="Checkpoint every x number of steps.",
        default=1_000,
        type=int,
    )
    parser.add_argument(
        "--valid-every",
        help="Evaluate validation loss every x number of steps.",
        default=100,
        type=int,
    )
    parser.add_argument("--resume", help="Resume training.", action="store_true")
    parser.add_argument(
        "--learning-rate", help="Optimizer learning rate", type=float, default=5e-4
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        help="Path to a pickled encoder. Defaults to 'encoder.pkl' in the output directory.",
    )
    parser.add_argument(
        "--unk-proba",
        type=float,
        help="Probability of inserting '<UNK>' tokens into training data.",
        default=0,
    )
    parser.add_argument("input", type=Path, help="Sqlite3 database of playlists.")
    parser.add_argument("output", type=Path, help="Output directory.")
    args = cast(_Args, parser.parse_args(argv))

    # Complain if the output already exists and we don't intend to resume.
    output_exists = args.output.exists()
    if output_exists and not args.resume:
        raise FileExistsError(
            f"Output path '{args.output}' exists. Delete it to restart or use the "
            "'--resume' flag to resume training."
        )

    # Create the datasets which are stateless. We pass in the path to the database
    # instead of a connection so the dataset can handle thread-local connections. Sqlite
    # does not support multi-threaded connections.
    datasets = {
        split: Sqlite3Dataset(
            args.input,
            SELECT_PLAYLISTS_BY_SPLIT,
            """
            SELECT
                tracks.id as track_id,
                ptm.pos
            FROM playlist_track_memberships AS ptm
            INNER JOIN tracks
            ON ptm.track_id = tracks.id
            WHERE ptm.playlist_id = :id
            """,
            {"split": split},
        )
        for split in ["train", "valid"]
    }

    if args.num_steps and args.num_epochs:
        raise ValueError(
            "`--num-steps` and `--num-epochs` are incompatible, use at most one."
        )
    if args.num_steps:
        num_steps = args.num_steps
    elif args.num_epochs:
        num_steps = int(args.num_epochs * len(datasets["train"]) / args.batch_size)
    else:
        # Run one epoch by default.
        num_steps = int(len(datasets["train"]) / args.batch_size)

    # Encoder for turning ids into consecutive integers for indexing embeddings.
    # Building the encoder can be expensive because we need to find all the tracks in
    # all the playlists that belong to the training set. So we allow the user to specify
    # a path to an encoder outside the output directory. For full reproducibility,
    # sticking with the default tokenizer path is recommended.
    encoder_path = args.encoder or (args.output / "encoder.pkl")
    if encoder_path.is_file():
        with encoder_path.open("rb") as fp:
            track_encoder = pickle.load(fp)
        print(
            f"Loaded encoder with {len(track_encoder):,} unique tokens from '{encoder_path}'."
        )
    else:
        print("Creating encoder. This may take a while ...")
        track_encoder = Encoder(
            ["<EOP>", "<UNK>"], on_unknown="default", default="<UNK>"
        )
        conn = sqlite3.connect(args.input, check_same_thread=False)
        cursor = conn.execute(SELECT_DISTINCT_TRACK_IDS_BY_SPLIT, {"split": "train"})
        track_encoder.update(track_id for (track_id,) in cursor)
        encoder_path.parent.mkdir(exist_ok=True)
        with encoder_path.open("wb") as fp:
            pickle.dump(track_encoder, fp)
        print(
            f"Created encoder with {len(track_encoder):,} unique tokens and saved it to '{encoder_path}'."
        )
    num_tokens = len(track_encoder)

    # Randomly sample unknown tokens.
    def f(element: dict, generator: numpy.random.Generator, proba: float | None):
        if proba is None:
            return element
        return element | {
            "track_id": [
                "<UNK>" if generator.random() < proba else x
                for x in element["track_id"]
            ]
        }

    # Datasets have some state, but they should be reproducible over different runs.
    # Unlike the samplers, we do not need to checkpoint them.
    operations = [
        BatchTransform(args.batch_size, on_short="drop"),
        LambdaMap(truncate_batch, max_length=args.context_length),
        LambdaMap(pad_batch, fill_value={"track_id": "<EOP>", "pos": 0}),
        LambdaMap(collectiontools.transpose_to_dict),
        LambdaMap(encode_track_ids, track_encoder),
        LambdaMap[dict, dict](lambda x: collectiontools.map_values(jnp.asarray, x)),
        LambdaMap(inputs_and_labels),
    ]
    data_loaders = {
        split: grain.DataLoader(
            data_source=cast(grain.sources.RandomAccessDataSource, dataset),
            sampler=grain.samplers.IndexSampler(
                len(dataset),
                shuffle=True,
                # Create a seed based on the split name and the specified seed.
                seed=int.from_bytes(split.encode()) & (2**32 - 1) + args.seed,
            ),
            operations=[
                LambdaRandomMap(f, proba=args.unk_proba if split == "train" else None),
                *operations,
            ],
            worker_count=0,
        )
        for split, dataset in datasets.items()
    }
    data_iters = {key: iter(value) for key, value in data_loaders.items()}

    # Create metrics and random number generator state. Their state will be restored
    # if we're loading from a checkpoint. We handle model and optimizer differently
    # below because just instantiating the model to infer the abstract pytree
    # representation can be very expensive.
    metrics = {
        "step": Counter(),
        "epoch": Counter(),
    }
    loss_rngs = nnx.Rngs(args.seed + 1)

    # Create a checkpoint manager which we'll use to persist model weights, data loader
    # state, and the rng used for sampled softmax.
    with (
        ocp.CheckpointManager(
            (args.output / "checkpoints").resolve(),
            item_names=(
                "loss_rngs",
                "my-metrics",
                "model",
                "optimizer",
                "train_data_iter",
                "valid_data_iter",
            ),
            options=ocp.CheckpointManagerOptions(
                preservation_policy=ocp.checkpoint_managers.LatestN(5)
            ),
        ) as checkpoint_manager,
        tqdm(total=num_steps) as progress,
        SummaryWriter(str(args.output / "logdir")) as writer,
    ):

        # Construct the model or load it from the checkpoint.
        if args.resume:
            model = nnx.eval_shape(
                lambda: _create_decoder(args, nnx.Rngs(args.seed), num_tokens)
            )
            optimizer = nnx.eval_shape(lambda: _create_optimizer(args, model))
            training_state = restore_training_state(
                checkpoint_manager,
                metrics=metrics,
                loss_rngs=nnx.Rngs(0),
                model=model,
                optimizer=optimizer,
                valid_data_iter=data_iters["valid"],
                train_data_iter=data_iters["train"],
            )
            model = training_state["model"]
            optimizer = training_state["optimizer"]
            loss_rngs = training_state["loss_rngs"]
            metrics = training_state["my-metrics"]
            progress.n = int(metrics["step"].compute())
        else:
            # Set up all state.
            model = _create_decoder(args, nnx.Rngs(args.seed), num_tokens)
            optimizer = _create_optimizer(args, model)

        valid_loss = jnp.nan
        while (step := int(metrics["step"].compute())) < num_steps:
            # Apply a training step.
            inputs, labels = next(data_iters["train"])
            train_loss = train_step(
                model,
                optimizer,
                inputs,
                labels,
                loss_rngs(),
                track_encoder["<EOP>"],
            )
            writer.add_scalar("train_loss", train_loss, global_step=step)

            # Checkpoint the model if it's time.
            if step % args.checkpoint_every == 0:
                checkpoint_training_state(
                    checkpoint_manager,
                    step,
                    model=model,
                    train_data_iter=data_iters["train"],
                    valid_data_iter=data_iters["valid"],
                    loss_rngs=loss_rngs,
                    metrics=metrics,
                    optimizer=optimizer,
                )

            # Evaluate the training loss if it's time.
            if step % args.valid_every == 0:
                inputs, labels = next(data_iters["valid"])
                valid_loss = loss_fn(
                    model, inputs, labels, loss_rngs(), track_encoder["<EOP>"]
                )
                writer.add_scalar("valid_loss", valid_loss, global_step=step)

            metrics["step"].update()
            progress.update()
            progress.set_description(
                f"train-loss={train_loss:.2f}, valid-loss={valid_loss:.2f}"
            )

        # Checkpoint after training finishes.
        checkpoint_training_state(
            checkpoint_manager,
            step,
            model=model,
            train_data_iter=data_iters["train"],
            valid_data_iter=data_iters["valid"],
            loss_rngs=loss_rngs,
            metrics=metrics,
            optimizer=optimizer,
        )


if __name__ == "__main__":
    __main__()
