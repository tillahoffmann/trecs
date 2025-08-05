from flax import nnx
from jax import numpy as jnp
from jax import random
from jax.scipy.special import softmax
import numpy
from optax import softmax_cross_entropy_with_integer_labels
from spotify_recommender.util import sampled_dot_cross_entropy_with_integer_labels


def test_sampled_dot_cross_entropy_with_integer_labels() -> None:
    # Generate synthetic data.
    rngs = nnx.Rngs(17)
    batch_size = 1024
    num_classes = 2048
    num_features = 16
    scale = 1 / jnp.sqrt(num_features)
    query = random.normal(rngs(), (batch_size, num_features)) * scale
    embedding = random.normal(rngs(), (num_classes, num_features)) * scale
    logits = query @ embedding.T
    assert logits.shape == (batch_size, num_classes)
    probas = softmax(logits, axis=-1)
    labels = jnp.argmax(random.multinomial(rngs(), 1, probas), axis=1)

    # Evaluate the true cross-entropy based on all logits.
    expected = softmax_cross_entropy_with_integer_labels(logits, labels)
    assert expected.shape == (batch_size,)

    # Evaluate the sampled cross-entropy and fit a linear regression model to verify
    # slope and intercept.
    actual = sampled_dot_cross_entropy_with_integer_labels(
        rngs(), query, embedding, labels, num_samples=60
    )
    fit = numpy.polynomial.Polynomial.fit(expected, actual, 1).convert()
    intercept, coef = fit.coef
    assert abs(coef - 1) < 0.01
    assert abs(intercept) < 0.05
