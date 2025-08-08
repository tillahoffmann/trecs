from argparse import ArgumentParser
from contextlib import closing
import numpy
from pathlib import Path
import sqlite3
from tqdm import tqdm
from typing import cast, Hashable
from ..data.models import PlaylistSlice


def maybe_insert_entity(
    conn: sqlite3.Connection,
    table: str,
    data: dict,
    *,
    key: Hashable | None = None,
    lookup: dict | None = None,
) -> int:
    """Insert an entity if it does not already exist, as determined by presence of a
    `key` in a `lookup` table.

    Args:
        conn: Connection to insert into.
        table: Table to insert into.
        data: Data to insert.
        key: Key to check in the lookup table.
        lookup: Lookup table mapping keys to row ids.

    Returns:
        Cached or newly inserted row id.
    """
    if key is not None:
        assert lookup is not None, "A lookup table must be given if a key is given."
        rowid = lookup.get(key)
        if rowid is not None:
            return rowid

    columns = ", ".join(data)
    values = ", ".join(f":{column}" for column in data)
    cursor = conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({values})", data)
    rowid = cursor.lastrowid
    assert (
        rowid is not None
    ), f"Inserting '{data}' into '{table}' did not yield a row id."
    if key:
        assert lookup is not None, "We should never get here."
        lookup[key] = rowid
    return rowid


def insert_slices(conn: sqlite3.Connection, slice_paths: list[Path]) -> list[int]:
    """Insert all playlist slices into the database.

    Args:
        conn: Connection to database to insert into.
        slice_paths: Paths to Million Playlist Dataset slices.

    Returns:
        List of playlist ids that can be used to construct train-validation-test splits.
    """
    playlist_id_lookup = {}
    track_id_lookup = {}
    album_id_lookup = {}
    artist_id_lookup = {}

    for slice_path in tqdm(slice_paths):
        playlist_slice = PlaylistSlice.model_validate_json(slice_path.read_text())
        for playlist in playlist_slice.playlists:
            # Set up the playlist.
            playlist_id = maybe_insert_entity(
                conn,
                "playlists",
                {
                    "id": playlist.pid,
                    "name": playlist.name,
                    "collaborative": playlist.collaborative,
                    "num_followers": playlist.num_followers,
                    "modified_at": playlist.modified_at,
                    "num_edits": playlist.num_edits,
                },
                key=playlist.pid,
                lookup=playlist_id_lookup,
            )

            # Go through all tracks with some validation.
            unique_artists = set()
            unique_albums = set()
            total_duration_ms = 0
            playlist_track_memberships = []
            for pos, track in enumerate(playlist.tracks):
                # Parent album.
                album_id = maybe_insert_entity(
                    conn,
                    "albums",
                    {"name": track.album_name, "uri": track.album_uri},
                    key=track.album_uri,
                    lookup=album_id_lookup,
                )
                unique_albums.add(track.album_uri)

                # Parent artist.
                artist_id = maybe_insert_entity(
                    conn,
                    "artists",
                    {"name": track.artist_name, "uri": track.artist_uri},
                    key=track.artist_uri,
                    lookup=artist_id_lookup,
                )
                unique_artists.add(track.artist_uri)

                # Track.
                assert track.pos == pos, "Position mismatch."
                track_id = maybe_insert_entity(
                    conn,
                    "tracks",
                    {
                        "uri": track.track_uri,
                        "name": track.track_name,
                        "duration_ms": track.duration_ms,
                        "artist_id": artist_id,
                        "album_id": album_id,
                    },
                    key=track.track_uri,
                    lookup=track_id_lookup,
                )
                total_duration_ms += track.duration_ms
                playlist_track_memberships.append(
                    {
                        "pos": pos,
                        "playlist_id": playlist_id,
                        "track_id": track_id,
                    }
                )

            assert len(playlist.tracks) == playlist.num_tracks
            assert total_duration_ms == playlist.duration_ms
            assert len(unique_artists) == playlist.num_artists
            assert len(unique_albums) == playlist.num_albums

    return list(playlist_id_lookup.values())


def insert_splits(
    conn: sqlite3.Connection,
    split_fracs: dict[str, float],
    playlist_ids: list[int] | numpy.ndarray,
    seed: int,
) -> None:
    # Permute the indices randomly and then split them to the desired sizes.
    rng = numpy.random.RandomState(seed)
    playlist_ids = rng.permutation(playlist_ids)
    indices = numpy.cumsum(
        [playlist_ids.size * frac for frac in split_fracs.values()]
    ).astype(int)[:-1]
    split_ids = numpy.split(playlist_ids, indices)
    for split, ids in zip(split_fracs, split_ids, strict=True):
        split_id = maybe_insert_entity(conn, "splits", {"name": split})
        conn.executemany(
            "INSERT INTO split_playlist_memberships (split_id, playlist_id) "
            "VALUES (?, ?)",
            [(split_id, int(id)) for id in ids],  # Need to cast to int from numpy int.
        )


class _Args:
    tvt: str
    slices: list[Path]
    output: Path
    seed: int


def __main__(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--tvt",
        help="Comma-separated train, validation, test split sizes as fractions.",
        default="0.8,0.1,0.1",
    )
    parser.add_argument(
        "--seed", help="Random number generator seed.", type=int, default=42
    )
    parser.add_argument("output", help="Database output path.", type=Path)
    parser.add_argument(
        "slices", help="Million playlist dataset slices.", nargs="+", type=Path
    )
    args = cast(_Args, parser.parse_args(argv))

    # Get the split sizes as fractions which we will turn into sizes once we've consumed
    # the entire dataset.
    fracs = [float(x) for x in args.tvt.split(",")]
    assert sum(fracs) == 1, "--tvt (train-validation-test) arg must some to one."
    split_fracs = dict(zip(["train", "valid", "test"], fracs, strict=True))

    # Open the connection and create the schema.
    assert not args.output.exists(), f"Database '{args.output}' already exists."
    args.output.parent.mkdir(exist_ok=True)
    with closing(sqlite3.connect(args.output)) as conn:
        schema_path = Path(__file__).parent / "schema.sql"
        schema = schema_path.read_text()
        conn.executescript(schema)

        playlist_ids = insert_slices(conn, args.slices)
        insert_splits(conn, split_fracs, playlist_ids, args.seed)
        conn.commit()
