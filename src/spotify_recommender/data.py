import collectiontools
import sqlite3
from torch.utils.data import Dataset
from typing import Any


SELECT_PLAYLISTS_BY_SPLIT = """
SELECT split_playlist_memberships.playlist_id
FROM splits
INNER JOIN split_playlist_memberships
ON splits.id = split_playlist_memberships.split_id
WHERE splits.name = :split
"""


class Sqlite3Dataset(Dataset):
    """Dataset from a sqlite database. Each element is a dictionary keyed by the column
    names of the :attr:`data_query`.

    Args:
        conn: Database connection.
        idx_query: Query to fetch identifiers which must yield a single element. This
            query is only run once.
        data_query: Query to fetch data given an :code:`:id` parameter. This query is
            run every time when an element of the dataset is fetched.
        idx_parameters: Parameters passed to the index query.
        data_parameters: Parameters passed to the data query. These may not include an
            `id` parameter because it is automatically inserted from the index query.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        idx_query: str,
        data_query: str,
        idx_parameters: dict[str, Any] | None = None,
        data_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.conn = conn
        self.idx_query = idx_query
        self.data_query = data_query
        self.idx_parameters = idx_parameters or {}
        self.data_parameters = data_parameters or {}
        assert "id" not in self.data_parameters

        self._idx: list[Any] | None = None
        self._columns: list[str] | None = None

    @property
    def idx(self) -> list[Any]:
        if self._idx is None:
            self._idx = [
                id for (id,) in self.conn.execute(self.idx_query, self.idx_parameters)
            ]
        return self._idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, key) -> dict[str, list[Any]]:
        id = self.idx[key]
        parameters = {"id": id} | self.data_parameters
        cursor = self.conn.execute(self.data_query, parameters)
        if self._columns is None:
            self._columns = [column for (column, *_) in cursor.description]
        return collectiontools.transpose_to_dict(
            [dict(zip(self._columns, row)) for row in cursor]
        )


def pad_batch(
    batch: list[dict[str, Any]], fill_value: Any = None, length: int | None = None
) -> list[dict[str, Any]]:
    """Pad elements of a batch to a desired length.

    Args:
        batch: List of batch elements, each being a dictionary with list values.
        fill_value: Value to pad with.
        length: Length to pad to or :code:`None` to pad to the longest sequence in the
            batch.

    Returns:
        Batch with all elements padded to the desired length.
    """
    if length is None:
        length = max(max(map(len, element.values())) for element in batch)
    return [
        {
            key: value + [fill_value] * (length - len(value))
            for key, value in element.items()
        }
        for element in batch
    ]


def truncate_batch(
    batch: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    """Truncate a batch to the desired length. Shorter sequences are left unchanged.

    Args:
        List of batch elements, each being a dictionary with list values.
        max_length: Maximum length of sequences.

    Returns:
        Batch with all elements truncated to the desired length.
    """
    return [
        {key: value[:max_length] for key, value in element.items()} for element in batch
    ]
