from pathlib import Path
import pytest
from spotify_recommender.data import (
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


@pytest.mark.parametrize("split", ["train", "test", "valid"])
def test_sqlite3_dataset(example_conn: sqlite3.Connection, split: str) -> None:
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
    assert {"train": 36, "test": 4, "valid": 11}[split] == len(dataset)
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
