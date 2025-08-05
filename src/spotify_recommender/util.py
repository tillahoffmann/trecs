import contextlib
from jax import numpy as jnp
from jax import random
from pathlib import Path
from typing import Generator, IO


@contextlib.contextmanager
def safe_write(
    path: str | Path,
    *,
    mode: str = "w",
    suffix: str = ".tmp",
    parents: bool = True,
    **kwargs,
) -> Generator[IO]:
    """Safely open a file for writing.

    Args:
        path: Path to write to.
        mode: Mode for writing to file.
        suffix: Suffix for temporary file.
        parents: Create parent directories if they do not exist.
        **kwargs: Keyword arguments passed to :meth:`Path.open`.

    Yields:
        Handle for temporary file.
    """
    if isinstance(path, str):
        path = Path(path)

    if parents and not path.parent.exists():
        path.parent.mkdir(parents=True)

    tmp_path = path.with_suffix(suffix)

    with tmp_path.open(mode, **kwargs) as fp:
        yield fp

    if not tmp_path.is_file():
        raise FileNotFoundError(tmp_path)

    tmp_path.rename(path)


def sampled_dot_cross_entropy_with_integer_labels(
    key: jnp.ndarray,
    query: jnp.ndarray,
    embedding: jnp.ndarray,
    labels: jnp.ndarray,
    num_samples: int = 20,
) -> jnp.ndarray:
    """Evaluate the sampled cross entropy based on logits obtained through a dot
    product `query @ embedding.T`. This function never evaluates the full dot product
    but only considers a sampled subset of the embedding matrix.

    Args:
        key: Random number generator key.
        query: Context to contract with the output embedding with shape
            `(batch_size, num_features)`.
        embedding: Output embedding with shape `(num_classes, num_features)`.
        labels: Target labels with shape `(batch_size,)`.

    Returns:
        Sampled cross-entropy with shape `(batch_size,)`.
    """
    batch_size, num_features = query.shape
    num_classes, embedding_num_features = embedding.shape
    assert (
        num_features == embedding_num_features
    ), "Feature dimensions of query and embeddings do not match."
    (batch_size_labels,) = labels.shape
    assert (
        batch_size == batch_size_labels
    ), "Batch size of query and labels do not match."

    # We sample indices with replacement. This is not as good as without replacement,
    # but much, much faster. It shouldn't matter provided that the number of samples is
    # much smaller than the number of classes. That's the setting where we'd use this
    # anyway.
    idx = random.randint(key, (batch_size, num_samples), 0, num_classes)
    sampled_embedding = embedding[idx]
    assert sampled_embedding.shape == (batch_size, num_samples, num_features)
    sampled_logits = jnp.vecdot(query[:, None, :], sampled_embedding)
    label_logits = jnp.vecdot(query, embedding[labels])

    # We take off the maximum value for numerical stability just like log-sum-exp.
    max_logits = jnp.maximum(label_logits, sampled_logits.max(axis=-1))
    sampled_logits = sampled_logits - max_logits[:, None]
    label_logits = label_logits - max_logits

    # We have to make sure we don't double-count the labeled logit, so we count the
    # number of times we accidentally sampled the label.
    num_hits = (labels[:, None] == idx).sum(axis=-1)
    sampled_negative_exp = jnp.sum(
        jnp.exp(sampled_logits), axis=-1
    ) - num_hits * jnp.exp(label_logits)

    # Edge case of only sampling the positive class.
    effective_num_neg_samples = num_samples - num_hits
    scale = jnp.where(
        effective_num_neg_samples > 0,
        (num_classes - 1) / effective_num_neg_samples,
        0.0,
    )

    return -(
        label_logits - jnp.log(jnp.exp(label_logits) + scale * sampled_negative_exp)
    )
