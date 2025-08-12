from pathlib import Path
import collectiontools
import contextlib
from flax import nnx
from grain.sources import RandomAccessDataSource
from grain.samplers import IndexSampler
from grain import DataLoader
import optax
import os
from jax import numpy as jnp
import pydantic
import sqlite3
from typing import cast, Literal
from ..util import Experiment
from ...models import PlaylistDecoder
from ...data import (
    Sqlite3Dataset,
    SELECT_DISTINCT_TRACK_IDS_BY_SPLIT,
    SELECT_PLAYLISTS_BY_SPLIT,
    LambdaMap,
    LambdaRandomMap,
    create_input_and_label_batches,
    pad_batch,
    BatchTransform,
    Encoder,
)
from ...util import (
    sampled_dot_cross_entropy_with_integer_labels_and_label_in_denominator,
    sampled_dot_cross_entropy_with_integer_labels_uniform,
    evaluate_eop_loss_mask,
)


class DecoderOnlyExperiment(Experiment):
    context_length: int
    num_layers: int
    num_heads: int
    num_features: int
    num_hidden: int
    loss_function: Literal[
        "label_in_denominator",
        "uniform",
    ]
    dropout: float = pydantic.Field(ge=0, le=1)
    num_tracks: int | None
    unk_proba: float = pydantic.Field(ge=0, le=1)
    weight_decay: float = pydantic.Field(ge=0)

    start_token: int | None = None
    eop_token: int | None = None
    unk_token: int | None = None
    track_encoder: Encoder | None = None

    # Because the `Encoder` is not a standard class.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def create_model(self, rngs: nnx.Rngs) -> PlaylistDecoder:
        assert self.num_tracks, "Number of tracks must be specified or inferred."
        return PlaylistDecoder(
            context_length=self.context_length,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_tracks=self.num_tracks,
            num_features=self.num_features,
            num_hidden=self.num_hidden,
            dropout=self.dropout,
            rngs=rngs,
        )

    def create_optimizer(self, model: nnx.Module) -> nnx.Optimizer:
        return nnx.Optimizer(
            model,
            optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
            wrt=nnx.Param,
        )

    def create_data_source(self, split: str) -> RandomAccessDataSource:
        dataset = Sqlite3Dataset(
            self.db_path,
            SELECT_PLAYLISTS_BY_SPLIT,
            """
            SELECT
                tracks.id as track_id,
                ptm.pos
            FROM playlist_track_memberships AS ptm
            INNER JOIN tracks
            ON ptm.track_id = tracks.id
            WHERE ptm.playlist_id = :id
            ORDER BY ptm.pos
            LIMIT :context_length
            """,
            {"split": split},
            {"context_length": self.context_length},
        )
        return cast(RandomAccessDataSource, dataset)

    def create_data_loader(
        self, split: str, data_source: RandomAccessDataSource, seed: int
    ) -> DataLoader:
        assert self.track_encoder, "Create track encoder first."
        operations = [
            # {INPUT}: {"pos": [0, 1, ...], "track_id": [43, 7, ...]}
            # Inject a start token.
            LambdaMap[dict, dict](
                lambda x: {
                    "track_id": ["<START>", *x["track_id"]],
                    "pos": list(range(len(x["pos"]) + 1)),
                }
            ),
            # Encode tracks and truncate to the maximum context length:
            # {"pos": [0, 1, ...], "track_id": [0, 1, ...]}
            LambdaMap[dict, dict](
                lambda x: {
                    "track_id": [
                        self.track_encoder(y)  # pyright: ignore[reportOptionalCall]
                        for y in x["track_id"]
                    ],
                    "pos": x["pos"],
                },
                validate_output=lambda y: all(
                    len(seq) <= self.context_length + 1 for seq in y.values()
                ),
            ),
            # Batch records: [{"track_id": [0, 1], ...}, {"track_id": [4, 5], ...}, ...]
            BatchTransform(self.batch_size, on_short="drop"),
            # Pad values to the same length.
            LambdaMap(
                pad_batch,
                fill_value={"track_id": self.eop_token, "pos": 0},
                length=self.context_length + 1,
            ),
            # Transpose to get a dictionary keyed by `track_id`, `pos`, etc. Then
            # convert to jax arrays.
            LambdaMap[dict, dict](
                lambda x: collectiontools.map_values(
                    jnp.asarray, collectiontools.transpose_to_dict(x)
                )
            ),
            # Shift the labels backwards by one and truncate to the same shape.
            LambdaMap(create_input_and_label_batches, label_key="track_id"),
        ]

        # Insert an operation that randomly replaces values with an <UNK> token at the
        # *beginning* of the pipeline.
        if split == "train":
            operations = [
                LambdaRandomMap[dict, dict](
                    lambda x, generator: x
                    | {
                        "track_id": [
                            ("<UNK>" if generator.binomial(1, self.unk_proba) else y)
                            for y in x["track_id"]
                        ]
                    }
                )
            ] + operations

        sampler = IndexSampler(len(data_source), shuffle=True, seed=seed)

        return DataLoader(
            data_source=data_source, sampler=sampler, operations=operations
        )

    def evaluate_loss(
        self,
        model: nnx.Module,
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        prng_key: jnp.ndarray,
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
        model = cast(PlaylistDecoder, model)
        embeddings = model(inputs)
        batch_size, num_tokens, num_features = embeddings.shape
        # Flatten all the data so we have a standard (batch_size, event_size) setup.
        flat_embeddings = embeddings.reshape((batch_size * num_tokens, num_features))
        flat_labels = labels.reshape((batch_size * num_tokens,))

        # Evaluate the loss.
        if self.loss_function == "label_in_denominator":
            func = (
                sampled_dot_cross_entropy_with_integer_labels_and_label_in_denominator
            )
        elif self.loss_function == "uniform":
            func = sampled_dot_cross_entropy_with_integer_labels_uniform
        else:
            raise ValueError(self.loss_function)

        sampled_loss = func(
            prng_key,
            flat_embeddings,
            model.track_embedding.embedding.value,
            flat_labels,
        )

        # Mask out anything after the first end of playlist token.
        assert self.eop_token is not None, "EOP token has not been initialized."
        mask = evaluate_eop_loss_mask(labels, self.eop_token)
        return sampled_loss @ mask.ravel() / mask.sum()

    @property
    def db_path(self) -> Path:
        db_path = os.environ.get("MPD")
        if db_path is None:
            raise ValueError(
                "Million Playlist Dataset sqlite3 path must be set as environment "
                "variable 'MPD'."
            )
        db_path = Path(db_path)
        assert (
            db_path.is_file()
        ), f"Million Playlist Database does not exist at '{db_path}'."
        return db_path

    def setup_output(self, output: Path) -> None:
        super().setup_output(output)
        encoder_path = os.environ.get("ENCODER")
        if encoder_path:
            encoder_path = Path(encoder_path)
        else:
            encoder_path = output / "track_encoder.pkl"

        if encoder_path.is_file():
            self.track_encoder = Encoder.from_pickle(encoder_path)
            print(
                f"Loaded track encoder with {len(self.track_encoder):,} tokens from '{encoder_path}'."
            )
        else:
            print(
                f"Building a new tokenizer from '{self.db_path}'. This may take a few minutes ..."
            )
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.execute(
                    SELECT_DISTINCT_TRACK_IDS_BY_SPLIT, {"split": "train"}
                )
                self.track_encoder = Encoder(
                    [
                        "<START>",
                        "<EOP>",
                        "<UNK>",
                        *(track_id for (track_id,) in cursor),
                    ],
                    on_unknown="default",
                    default="<UNK>",
                )
            self.track_encoder.to_pickle(encoder_path)
            print(f"Built new track encoder with {len(self.track_encoder):,} tokens.")

        # Get named special tokens.
        self.start_token = self.track_encoder("<START>")
        self.eop_token = self.track_encoder("<EOP>")
        self.unk_token = self.track_encoder("<UNK>")

        num_tracks = len(self.track_encoder)
        if self.num_tracks is None:
            self.num_tracks = num_tracks
        elif num_tracks > self.num_tracks:
            raise ValueError(
                f"Dataset contains {num_tracks:,} tracks, but model only supports "
                f"{self.num_tracks:,}."
            )
