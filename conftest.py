import contextlib
from pathlib import Path
import pytest
import spotify_recommender
import sqlite3
from typing import Generator


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
