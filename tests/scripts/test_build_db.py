from contextlib import closing
from pathlib import Path
from spotify_recommender.scripts import build_db
import sqlite3


def test_build_db(tmp_path: Path) -> None:
    mpd_slice_path = Path(__file__).parent.parent / "assets/test_mpd_slice.json"
    assert mpd_slice_path.is_file()
    db_path = tmp_path / "mpd.db"
    build_db.__main__([str(db_path), str(mpd_slice_path)])

    with closing(sqlite3.connect(db_path)) as conn:
        data = conn.execute(
            """
            SELECT splits.name, COUNT(split_playlist_memberships.playlist_id)
            FROM splits
            INNER JOIN split_playlist_memberships
            ON splits.id = split_playlist_memberships.split_id
            GROUP BY splits.name
            ORDER BY splits.name
        """
        ).fetchall()
        assert data == [("test", 10), ("train", 80), ("valid", 10)]
        (total,) = conn.execute("SELECT COUNT(*) FROM playlists").fetchone()
        assert total == 100

        # All tables must have something in them.
        tables = [
            "albums",
            "artists",
            "playlist_track_memberships",
            "split_playlist_memberships",
            "splits",
            "tracks",
        ]
        for table in tables:
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert count, f"'{table}' is empty."
