from argparse import ArgumentParser
import collectiontools
from flax import nnx
import grain
import jax
from jax import numpy as jnp
import optax
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
        dropout=args.num_hidden,
        rngs=rngs,
        num_tracks=num_tracks,
    )


def _create_optimizer(args: _Args, model: PlaylistDecoder) -> nnx.Optimizer:
    return nnx.Optimizer(model, optax.adamw(args.learning_rate), wrt=nnx.Param)


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
        "--checkpoint-every", help="Checkpoint every x number of steps.", default=1_000
    )
    parser.add_argument("--resume", help="Resume training.", action="store_true")
    parser.add_argument(
        "--learning-rate", help="Optimizer learning rate", type=float, default=1e-3
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        help="Path to a pickled encoder. Defaults to 'encoder.pkl' in the output directory.",
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
        num_steps = int(args.num_epochs * len(datasets["train"]))
    else:
        # Run one epoch by default.
        num_steps = len(datasets["train"])

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

    # Construct the model.
    if args.resume:
        raise NotImplementedError
    else:
        # Set up all state.
        model = _create_decoder(args, nnx.Rngs(args.seed), num_tokens)
        optimizer = _create_optimizer(args, model)
        # We don't use a multi-metric because it requires we specify a value every time.
        # We may not want to do that, e.g., when we advance the step but not the epoch.
        metrics = {
            "step": Counter(),
            "epoch": Counter(),
        }
        # Random number generator for sampled softmax.
        softmax_rngs = nnx.Rngs(args.seed + 1)

    # Data loaders are stateful and must be checkpointed for reproducibility.
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
                # Only run one epoch at a time.
                num_epochs=1,
            ),
            operations=operations,
            worker_count=0,
        )
        for split, dataset in datasets.items()
    }

    with (
        tqdm(total=num_steps, initial=int(metrics["step"].compute())) as progress,
        SummaryWriter(str(args.output / "logdir")) as writer,
    ):
        valid_loss = jnp.nan
        train_batches = iter(data_loaders["train"])
        while (step := int(metrics["step"].compute())) < num_steps:
            # Try to get a new batch. If the iterator is exhausted, create a new one and
            # restart the loop.
            try:
                inputs, labels = next(train_batches)
                metrics["epoch"].update()
            except StopIteration:
                train_batches = iter(data_loaders["train"])
                continue

            # Apply a training step.
            train_loss = train_step(
                model, optimizer, inputs, labels, softmax_rngs(), track_encoder["<EOP>"]
            )
            writer.add_scalar("train_loss", train_loss, global_step=step)

            metrics["step"].update()
            progress.update()
            progress.set_description(
                f"train-loss={train_loss:.2f}, valid-loss={valid_loss:.2f}"
            )


if __name__ == "__main__":
    __main__()
