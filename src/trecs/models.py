from flax import nnx
from jax import numpy as jnp


class TransformerBlock(nnx.Module):
    """Transformer black with causal self attention and pre-layer norm.

    Args:
        num_features: Number of embedding features.
        num_hidden: Number of hidden units in feed-forward network.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        *,
        num_features: int,
        num_hidden: int,
        num_heads: int,
        dropout: float,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.norm1 = nnx.LayerNorm(num_features=num_features, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=num_features,
            use_bias=False,
            rngs=rngs,
            dropout_rate=dropout,
            decode=False,
            deterministic=False,
        )
        self.norm2 = nnx.LayerNorm(num_features=num_features, rngs=rngs)
        self.feed_forward = nnx.Sequential(
            nnx.Linear(in_features=num_features, out_features=num_hidden, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(in_features=num_hidden, out_features=num_features, rngs=rngs),
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Construct a causal attention mask.
        batch_size, num_tokens, num_features = x.shape
        mask = jnp.tril(jnp.ones((num_tokens, num_tokens)))

        # Self-attention layer.
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        # No dropout here, because the attention layer already has dropout.
        x = x + shortcut

        # Dense feed-forward layer.
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class PlaylistDecoder(nnx.Module):
    """GPT-style decoder model for playlists with weight-tying.

    The output of the model are the final embeddings which can be contracted with the
    embeddings to obtain logits. We do not calculate logits because the vocabulary size
    is potentially large.

    Args:
        context_length: Size of the context window.
        num_layers: Number of transformer layers.
        num_tracks: Vocabulary size.
        num_features: Number of embedding features.
        num_hidden: Number of hidden units in feed-forward network.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        *,
        context_length: int,
        num_layers: int,
        num_heads: int,
        num_tracks: int,
        num_features: int,
        num_hidden: int,
        dropout: float,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.pos_embedding = nnx.Embed(
            num_embeddings=context_length, features=num_features, rngs=rngs
        )
        self.track_embedding = nnx.Embed(
            num_embeddings=num_tracks, features=num_features, rngs=rngs
        )
        self.initial_droput = nnx.Dropout(dropout, rngs=rngs)
        self.layers = nnx.Sequential(
            *[
                TransformerBlock(
                    num_features=num_features,
                    num_hidden=num_hidden,
                    dropout=dropout,
                    num_heads=num_heads,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nnx.LayerNorm(num_features=num_features, rngs=rngs)
        # We do not have an output projection layer because we use weight-tying.

    def __call__(self, inputs) -> jnp.ndarray:
        pos_embedding = self.pos_embedding(inputs["pos"])
        track_embedding = self.track_embedding(inputs["track_id"])
        embedding = pos_embedding + track_embedding
        embedding = self.initial_droput(embedding)
        embedding = self.layers(embedding)
        embedding = self.final_norm(embedding)
        return embedding
