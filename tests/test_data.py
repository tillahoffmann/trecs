from jax import numpy as jnp
from pathlib import Path
import pytest
from spotify_recommender.data import (
    create_input_and_label_batches,
    Encoder,
    pad_batch,
    SELECT_PLAYLISTS_BY_SPLIT,
    Sqlite3Dataset,
    truncate_batch,
)
import sqlite3
from typing import Any


Batch = list[dict[str, list[Any]]]


@pytest.fixture
def example_batch() -> Batch:
    lengths = [3, 5, 7]
    return [
        {"a": list(range(length)), "b": list(range(10, 10 + length))}
        for length in lengths
    ]


def test_pad_batch(example_batch: Batch) -> None:
    padded = pad_batch(example_batch, 99)
    assert all(len(element["a"]) == 7 for element in padded)
    assert padded[0]["a"] == [0, 1, 2, 99, 99, 99, 99]

    padded = pad_batch(example_batch, 99, 30)
    assert all(len(element["a"]) == 30 for element in padded)

    padded = pad_batch(example_batch, length=5)
    lengths = [len(element["a"]) for element in padded]
    assert lengths == [5, 5, 7]

    padded = pad_batch(example_batch, {"a": 99, "b": 77})
    element = padded[0]
    assert element["a"] == [0, 1, 2, 99, 99, 99, 99]
    assert element["b"] == [10, 11, 12, 77, 77, 77, 77]


def test_truncate_batch(example_batch: Batch) -> None:
    truncated = truncate_batch(example_batch, 4)
    lengths = [len(element["a"]) for element in truncated]
    assert lengths == [3, 4, 4]


def test_sqlite3_dataset(example_conn: sqlite3.Connection) -> None:
    for split, expected_size in {"train": 80, "test": 10, "valid": 10}.items():
        dataset = Sqlite3Dataset(
            example_conn,
            SELECT_PLAYLISTS_BY_SPLIT,
            """
            SELECT playlist_id, track_id, pos
            FROM playlist_track_memberships
            WHERE playlist_id = :id
            """,
            {"split": split},
        )
        # These numbers depend on the random number generator in `conftest.py`.
        assert expected_size == len(dataset)
        assert dataset._idx
        for i, playlist_id in enumerate(dataset._idx):
            element = dataset[i]
            assert all(x == playlist_id for x in element["playlist_id"])


def test_encoder(tmp_path: Path) -> None:
    encoder = Encoder("abc")
    assert encoder["c"] == 2
    with pytest.raises(KeyError):
        encoder["d"]

    encoder = Encoder("abc", on_unknown="default", default="b")
    assert encoder["d"] == encoder["b"]

    path = tmp_path / "encoder.pkl"
    encoder.to_pickle(path)
    encoder = Encoder.from_pickle(path)
    assert encoder["d"] == encoder["b"]


def test_create_input_and_label_batches() -> None:
    batch = {
        "a": jnp.ones((5, 1)) * jnp.arange(7),
        "x": jnp.arange(105).reshape((5, 7, 3)),
    }
    inputs, labels = create_input_and_label_batches(batch, label_key="a")
    assert labels.shape == (5, 6)
    assert inputs["a"].shape == (5, 6)
    assert inputs["x"].shape == (5, 6, 3)
    assert inputs["a"].max() == 5
    assert labels.min() == 1
