import collectiontools
import contextlib
from pathlib import Path
import pytest
import random
import spotify_recommender
import sqlite3
from typing import Generator


@pytest.fixture
def example_conn() -> Generator[sqlite3.Connection, None, None]:
    rng = random.Random(42)

    num_albums = 13
    num_artists = 9
    num_tracks = 102
    num_playlists = 51

    playlist_track_memberships = []
    for playlist_id in range(num_playlists):
        tracks_in_playlist = 5 + (playlist_id % 13)
        for pos in range(tracks_in_playlist):
            playlist_track_memberships.append(
                {
                    "playlist_id": playlist_id,
                    "track_id": rng.randint(0, num_tracks - 1),
                    "pos": pos,
                }
            )

    example_data = {
        "playlists": {
            "id": range(num_playlists),
            # Names can repeat.
            "name": [f"playlist name: {i % 7}" for i in range(num_playlists)],
            "collaborative": [i % 2 == 0 for i in range(num_playlists)],
            "modified_at": [1500 * i for i in range(num_playlists)],
            "num_followers": [i % 12 for i in range(num_playlists)],
        },
        "artists": {
            "id": range(num_artists),
            "name": [f"artist name: {i % 5}" for i in range(num_artists)],
            "uri": [f"spotify:artist:{i}" for i in range(num_artists)],
        },
        "albums": {
            "id": range(num_albums),
            "name": [f"album name: {i % 11}" for i in range(num_albums)],
            "uri": [f"spotify:album:{i}" for i in range(num_albums)],
        },
        "tracks": {
            "id": range(num_tracks),
            "name": [f"track name: {i % 75}" for i in range(num_tracks)],
            "uri": [f"spotify:track:{i}" for i in range(num_tracks)],
            "album_id": [rng.randint(0, num_albums - 1) for _ in range(num_tracks)],
            "artist_id": [rng.randint(0, num_artists - 1) for _ in range(num_tracks)],
            "duration_ms": [2000 * i for i in range(num_tracks)],
        },
        "playlist_track_memberships": playlist_track_memberships,
        "splits": {"id": range(3), "name": ["train", "valid", "test"]},
        "split_playlist_memberships": {
            "split_id": [
                rng.choices([0, 1, 2], [0.7, 0.2, 0.1])[0] for _ in range(num_playlists)
            ],
            "playlist_id": [i for i in range(num_playlists)],
        },
    }

    schema_path = Path(spotify_recommender.__file__).parent / "schema.sql"
    schema = schema_path.read_text()
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(schema)
        for table, data in example_data.items():
            if isinstance(data, dict):
                data = collectiontools.transpose_to_list(data)
            cols = list(data[0])
            values = ", ".join(f":{name}" for name in cols)
            query = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({values})"

            try:
                conn.executemany(query, data)
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to insert data into '{table}': {ex}"
                ) from ex

        conn.commit()
        yield conn
