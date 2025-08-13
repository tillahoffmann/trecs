import collectiontools
import grain
import itertools
from jax import numpy as jnp
from numpy.random import Generator
from pathlib import Path
import pickle
import sqlite3
import threading
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Self,
    TypeVar,
)
from ..util import safe_write

T = TypeVar("T")
U = TypeVar("U")


SELECT_PLAYLISTS_BY_SPLIT = """
SELECT split_playlist_memberships.playlist_id
FROM splits
INNER JOIN split_playlist_memberships
ON splits.id = split_playlist_memberships.split_id
WHERE splits.name = :split
ORDER BY split_playlist_memberships.playlist_id
"""

SELECT_DISTINCT_TRACK_IDS_BY_SPLIT = """
SELECT DISTINCT track_id
FROM playlist_track_memberships AS ptm
INNER JOIN split_playlist_memberships AS spm
ON ptm.playlist_id = spm.playlist_id
INNER JOIN splits
ON splits.id = spm.split_id
WHERE splits.name = :split
ORDER BY track_id
"""


class Sqlite3Dataset:
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
        conn: sqlite3.Connection | str | Path,
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
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        # Just return the connection if one was explicitly passed.
        if isinstance(self.conn, sqlite3.Connection):
            return self.conn
        try:
            return self._local.conn
        except AttributeError:
            self._local.conn = sqlite3.connect(self.conn)
            return self._local.conn

    def _close_conn(self) -> None:
        # Close the connection if there is one.
        if isinstance(self.conn, sqlite3.Connection):
            self.conn.close()
        elif isinstance(self._local.conn, sqlite3.Connection):
            self._local.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self._close_conn()

    @property
    def idx(self) -> list[Any]:
        if self._idx is None:
            self._idx = [
                id
                for (id,) in self._get_conn().execute(
                    self.idx_query, self.idx_parameters
                )
            ]
        return self._idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, key) -> dict[str, list[Any]]:
        id = self.idx[key]
        parameters = {"id": id} | self.data_parameters
        cursor = self._get_conn().execute(self.data_query, parameters)
        if self._columns is None:
            self._columns = [column for (column, *_) in cursor.description]
        return collectiontools.transpose_to_dict(
            [dict(zip(self._columns, row)) for row in cursor]
        )

    def __repr__(self) -> str:
        cls = self.__class__
        return f"<{cls.__module__}.{cls.__name__} connected to '{self.conn}' with {len(self)} records>"


def pad_batch(
    batch: list[dict[str, Any]],
    fill_value: Any | dict[str, Any] = None,
    length: int | None = None,
) -> list[dict[str, Any]]:
    """Pad elements of a batch to a desired length.

    Args:
        batch: List of batch elements, each being a dictionary with list values.
        fill_value: Value to pad with or a mapping from keys in each element of the
            batch to a fill value.
        length: Length to pad to or :code:`None` to pad to the longest sequence in the
            batch.

    Returns:
        Batch with all elements padded to the desired length.
    """
    if length is None:
        length = max(max(map(len, element.values())) for element in batch)
    return [
        {
            key: value
            + [fill_value[key] if isinstance(fill_value, Mapping) else fill_value]
            * (length - len(value))
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


class Encoder(Mapping):
    """Encoder that maps from arbitrary identifiers to consecutive integers.

    Args:
        lookup: Lookup table or sequence of tokens to encode.
        on_unknown: Whether to use a default value or raise an exception.
        default: Default token if not found in the lookup table. E.g., if the default is
            '<UNK>', then an element not in the lookup table will be encoded *as the
            integer* representing the '<UNK>' token.
    """

    def __init__(
        self,
        lookup: dict | Iterable,
        on_unknown: Literal["raise", "default"] = "raise",
        default: Any = None,
    ) -> None:
        if isinstance(lookup, Iterable) and not isinstance(lookup, dict):
            lookup = {key: i for i, key in enumerate(lookup)}
        if on_unknown == "default" and default not in lookup:
            raise ValueError(f"Default '{default}' is not in the lookup.")

        self.lookup = lookup
        self.on_unknown = on_unknown
        self.default = default

    def add(self, token) -> int:
        return self.lookup.setdefault(token, len(self.lookup))

    def update(self, tokens) -> None:
        for token in tokens:
            self.add(token)

    def __repr__(self):
        default = "raises KeyError" if self.on_unknown == "raise" else self.default
        return f"<Encoder with {len(self.lookup):,} tokens, default: {default}>"

    def __getitem__(self, key: Any) -> Any:
        try:
            return self.lookup[key]
        except KeyError:
            if self.on_unknown == "raise":
                raise
            return self.lookup[self.default]

    def __call__(self, key: Any) -> Any:
        return self[key]

    def __len__(self) -> int:
        return len(self.lookup)

    def __iter__(self) -> Iterator:
        raise NotImplementedError("Iteration is not implemented for token encoders.")

    def to_pickle(self, path: str | Path) -> None:
        """Save an encoder to a pickle file."""
        with safe_write(path, mode="wb") as fp:
            pickle.dump(
                {
                    "lookup": self.lookup,
                    "on_unknown": self.on_unknown,
                    "default": self.default,
                },
                fp,
            )

    @classmethod
    def from_pickle(cls, path: str | Path) -> Self:
        """Load an encoder from a pickle file."""
        with open(path, mode="rb") as fp:
            state = pickle.load(fp)
        return cls(**state)


class BatchTransform:
    """Batch elements like :class:`grain._src.python.operations.BatchOperation`,
    although each batch is a list of arbitrary elements.

    Args:
        size: Batch size.
        on_short: What to do if the last batch is short.
    """

    def __init__(
        self, size: int, on_short: Literal["keep", "drop", "raise"] = "keep"
    ) -> None:
        self.size = size
        self.on_short = on_short

    def __call__(self, iterable):
        for batch in itertools.batched(
            iterable, self.size, strict=self.on_short == "raise"
        ):
            if len(batch) == self.size or self.on_short == "keep":
                yield grain.Record(
                    batch[-1].metadata.remove_record_key(), data=[x.data for x in batch]
                )


class BatchMapTransform(grain.transforms.Map):
    """Apply a function to elements of a batch."""

    def __init__(self, func: Callable, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def map(self, element):  # pyright: ignore[reportIncompatibleMethodOverride]
        return [self.func(x, *self.args, **self.kwargs) for x in element]


class LambdaReprMixin:
    func: Callable

    def __repr__(self) -> str:
        _id = id(self)
        name = self.__class__.__name__
        return f"<{name} at 0x{_id:x}: {self.func.__name__}>"


class LambdaRandomMap(grain.transforms.RandomMap, LambdaReprMixin, Generic[T, U]):
    """Apply a callable to all elements of a map, receiving a random number generator."""

    def __init__(
        self, func: Callable[Concatenate[T, Generator, ...], U], *args, **kwargs
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def random_map(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, element: T, rng: Generator
    ) -> U:
        return self.func(element, rng, *self.args, **self.kwargs)


class LambdaMap(grain.transforms.Map, LambdaReprMixin, Generic[T, U]):
    """Apply a callable to all elements of a map."""

    def __init__(
        self,
        func: Callable[Concatenate[T, ...], U],
        *args,
        validate_input: Callable[[T], bool] | None = None,
        validate_output: Callable[[U], bool] | None = None,
        **kwargs,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.validate_input = validate_input
        self.validate_output = validate_output

    def map(self, element: T) -> U:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.validate_input and not self.validate_input(element):
            raise ValueError(f"Invalid input for `{self.func}`: {element}")
        result = self.func(element, *self.args, **self.kwargs)
        if self.validate_output and not self.validate_output(result):
            raise ValueError(f"Invalid output from `{self.func}`: {result}")
        return result


def create_input_and_label_batches(
    batch: dict[str, jnp.ndarray], *, label_key: str = "label"
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Create inputs and labels for training an autoregressive model.

    Args:
        batch: Mapping of tensors with shape `(batch_size, num_tokens, ...)`.
        label_key: Key of the labels to predict.

    Returns:
        Pair of inputs and labels with token length reduced by one.
    """
    labels = batch[label_key][:, 1:]
    inputs = {key: value[:, :-1] for key, value in batch.items()}
    return inputs, labels
