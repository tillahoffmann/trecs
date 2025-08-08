from contextlib import closing
from pathlib import Path
import pytest
from spotify_recommender.scripts import build_db
import sqlite3
from typing import Generator


@pytest.fixture
def example_conn(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    # Create a temporary database.
    mpd_slice_path = Path(__file__).parent / "tests/assets/test_mpd_slice.json"
    assert mpd_slice_path.is_file()
    db_path = tmp_path / "example-database.db"
    build_db.__main__([str(db_path), str(mpd_slice_path)])

    # Connect to it and yield the connection.
    with closing(sqlite3.connect(db_path)) as conn:
        yield conn
