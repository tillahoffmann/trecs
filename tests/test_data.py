import contextlib
from pathlib import Path
import pytest
import spotify_recommender
from spotify_recommender.data import (
    pad_batch,
    truncate_batch,
    Sqlite3Dataset,
    SELECT_PLAYLISTS_BY_SPLIT,
)
import sqlite3
from typing import Any, Generator


Batch = list[dict[str, list[Any]]]


@pytest.fixture
def example_batch() -> Batch:
    lengths = [3, 5, 7]
    return [
        {"a": list(range(length)), "b": list(range(10, 10 + length))}
        for length in lengths
    ]


@pytest.fixture
def example_conn() -> Generator[sqlite3.Connection, None, None]:
    schema_path = Path(spotify_recommender.__file__).parent / "schema.sql"
    schema = schema_path.read_text()

    example_data = {
        "playlists": (
            ("id", "name", "collaborative", "modified_at", "num_followers"),
            [
                (1, "playlist1", False, 1234, 0),
                (2, "playlist2", True, 5678, 19),
                (3, "playlist3", False, 1234, 0),
                (4, "playlist4", True, 5678, 19),
            ],
        ),
        "artists": (
            ("id", "name", "uri"),
            [
                (1, "artist1", "spotify:artist:1"),
                (2, "artist2", "spotify:artist:2"),
                (3, "artist3", "spotify:artist:3"),
                # The "artist1" here is deliberate because track names are not unique.
                (4, "artist1", "spotify:artist:4"),
            ],
        ),
        "albums": (
            ("id", "name", "uri"),
            [
                (1, "album1", "spotify:album:1"),
                (2, "album2", "spotify:album:2"),
                # The "album1" here is deliberate because track names are not unique.
                (3, "album1", "spotify:album:3"),
            ],
        ),
        "tracks": (
            ("id", "name", "artist_id", "album_id", "uri", "duration_ms"),
            [
                (1, "track1", 1, 1, "spotify:track:1", 12345),
                (2, "track2", 1, 1, "spotify:track:2", 12345),
                # The "track1" here is deliberate because track names are not unique.
                (3, "track1", 1, 1, "spotify:track:3", 12345),
                (4, "track4", 1, 1, "spotify:track:4", 12345),
                (5, "track5", 1, 1, "spotify:track:5", 12345),
            ],
        ),
        "playlist_track_memberships": (
            ("playlist_id", "track_id", "pos"),
            [
                (1, 1, 0),
                (1, 2, 1),
                (2, 3, 0),
                (2, 4, 1),
                (3, 5, 0),
                (3, 2, 1),
                (3, 3, 2),
                (4, 5, 0),
                (4, 2, 1),
                (4, 3, 2),
            ],
        ),
        "splits": (
            ("id", "name"),
            [(1, "train"), (2, "test")],
        ),
        "split_playlist_memberships": (
            ("split_id", "playlist_id"),
            [(1, 1), (1, 2), (1, 3), (2, 4)],
        ),
    }

    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(schema)
        for table, (columns, rows) in example_data.items():
            query = (
                f"INSERT INTO {table} ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})"
            )
            conn.executemany(query, rows)

        conn.commit()
        yield conn


def test_pad_batch(example_batch: Batch) -> None:
    padded = pad_batch(example_batch, 99)
    assert all(len(element["a"]) == 7 for element in padded)
    assert padded[0]["a"] == [0, 1, 2, 99, 99, 99, 99]

    padded = pad_batch(example_batch, 99, 30)
    assert all(len(element["a"]) == 30 for element in padded)

    padded = pad_batch(example_batch, length=5)
    lengths = [len(element["a"]) for element in padded]
    assert lengths == [5, 5, 7]


def test_truncate_batch(example_batch: Batch) -> None:
    truncated = truncate_batch(example_batch, 4)
    lengths = [len(element["a"]) for element in truncated]
    assert lengths == [3, 4, 4]


@pytest.mark.parametrize("split", ["train", "test"])
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
    assert {"train": 3, "test": 1}[split] == len(dataset)
    assert dataset._idx
    for i, playlist_id in enumerate(dataset._idx):
        element = dataset[i]
        assert all(x == playlist_id for x in element["playlist_id"])
