import chex
from flax import linen as nn
import jax.numpy as jnp
import jax.tree_util
from math import prod
from typing import Callable, Sequence


def num_params(params):
    n_outputs = jax.tree_map(lambda x: prod(jnp.shape(x)), params)
    n_outputs, *_ = jax.tree_util.tree_flatten(n_outputs)
    return sum(n_outputs)


def causal_mask(shape):
    return jnp.triu(jnp.ones(shape, dtype=jnp.bool_), k=1)


def sin_pos_enc(sequence_length, embed_dim):
    """create sin/cos positional encodings

    Paramters
    =========
    sequence_length : int
        The max length of the input sequences for this model
    embed_dim : int
        the embedding dimension

    Returns
    =======
    a matrix of shape: (sequence_length, embed_dim)
    """
    chex.assert_is_divisible(embed_dim, 2)
    X = jnp.expand_dims(jnp.arange(sequence_length), 1) / \
        jnp.power(10000, jnp.arange(embed_dim, step=2) / embed_dim)
    out = jnp.empty((sequence_length, embed_dim))
    out = out.at[:, 0::2].set(jnp.sin(X))
    out = out.at[:, 1::2].set(jnp.cos(X))
    return out


def masked_softmax(args, mask):
    if mask is not None:
        args = args + (mask.astype(args.dtype) * -10_000.0)
    return nn.softmax(args)


def dot_prod_attn(q, k, v, dropout=lambda x: x, mask=None):
    # NxD @ DxM => NxM
    # (B[, H], N, M)
    attn_scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(q.shape[-1])
    attn_weights = masked_softmax(attn_scores, mask)
    # (B[, H], N, D)
    out = dropout(attn_weights) @ v
    return out, attn_weights


class MultiHeadAttention(nn.Module):
    n_heads: int
    size_per_head: int
    attn_dropout: float
    fc_dropout: float
    attn_fn: Callable = dot_prod_attn

    @nn.compact
    def __call__(self, q, k, v, mask=None, training=False):
        "expected shape: Batch, [N|M], Dim"
        B, N, D = q.shape
        _, M, _ = k.shape

        def qkv_layer(x, name):
            x = nn.Dense(self.n_heads * self.size_per_head, name=name)(x)
            x = x.reshape((B, -1, self.n_heads, self.size_per_head)).swapaxes(1, 2)
            return x
        # BxNxD => BxHxNxP
        q = qkv_layer(q, 'query_linear')
        # BxMxD => BxHxMxP
        k = qkv_layer(k, 'key_linear')
        # BxMxD => BxHxMxP
        v = qkv_layer(v, 'value_linear')
        if mask is not None:
            # accounting for reshape in qkv_layer
            # B[xN]xN   => Bx1[xN]xN
            mask = jnp.expand_dims(mask, 1)
            if mask.ndim < q.ndim:
                # softmax is applied to dim -1
                # Bx1xN => Bx1x1xN
                mask = jnp.expand_dims(mask, -2)
        attn_do = nn.Dropout(self.attn_dropout, deterministic=not training,
                             name='attn_dropout')
        out, attn_weights = self.attn_fn(q, k, v, attn_do, mask=mask)
        # uncomment to keep attention weights in state
        # self.sow('intermediates', 'weights', attn_weights)
        out = out.swapaxes(1, 2).reshape((B, N, -1))
        out = nn.Dense(D, name='output_linear')(out)
        out = nn.Dropout(self.fc_dropout, deterministic=not training,
                         name='fc_dropout')(out)
        return out


class AddAndNorm(nn.Module):
    """The add and norm."""

    @nn.compact
    def __call__(self, X, X_out):
        return nn.LayerNorm()(X + X_out)


class FeedForward(nn.Module):
    """a 2-layer feed-forward network."""
    hidden_dim: int

    @nn.compact
    def __call__(self, X):
        D = X.shape[-1]
        X = nn.Dense(self.hidden_dim)(X)
        X = nn.relu(X)
        X = nn.Dense(D)(X)
        return X


class EncoderLayer(nn.Module):
    hidden_dim: int
    n_heads: int
    size_per_head: int
    attn_dropout: float
    fc_dropout: float

    def setup(self):
        self.attn = MultiHeadAttention(n_heads=self.n_heads,
                                       size_per_head=self.size_per_head,
                                       attn_dropout=self.attn_dropout,
                                       fc_dropout=self.fc_dropout)
        self.aan_0 = AddAndNorm()
        self.ff = FeedForward(hidden_dim=self.hidden_dim)
        self.aan_1 = AddAndNorm()

    def __call__(self, X, mask=None, training=False):
        X1 = self.attn(X, X, X, mask=mask, training=training)
        X = self.aan_0(X, X1)
        X1 = self.ff(X)
        X = self.aan_1(X, X1)
        return X


class Encoder(nn.Module):
    pos_encoding: Callable[[int, int], jnp.array]
    vocab_size: int
    embed_dim: int
    layers: Sequence[EncoderLayer]

    @nn.compact
    def __call__(self, X, mask=None, training=False):
        B, N = X.shape
        if mask is not None:
            chex.assert_shape(mask, (B, N))
        X = nn.Embed(self.vocab_size, self.embed_dim, name='embed')(X)
        X = X * jnp.sqrt(self.embed_dim)
        # X.shape[-2] is the sequence length
        X = X + self.pos_encoding(X.shape[-2], self.embed_dim)
        for layer in self.layers:
            X = layer(X, mask=mask, training=training)
        return X


class DecoderLayer(nn.Module):
    hidden_dim: int
    n_heads: int
    size_per_head: int
    attn_dropout: float
    fc_dropout: float

    @nn.compact
    def __call__(self, X_enc, X_dec, enc_mask, dec_mask, training=False):

        def attn(q, kv, mask, training, name):
            mdl = MultiHeadAttention(n_heads=self.n_heads,
                                     size_per_head=self.size_per_head,
                                     attn_dropout=self.attn_dropout,
                                     fc_dropout=self.fc_dropout,
                                     name=f'{name}_attn')
            out = mdl(q, kv, kv, mask=mask, training=training)
            aan = AddAndNorm(name=f'{name}_addnorm')
            return aan(q, out)
        X_dec = attn(X_dec, X_dec, dec_mask, training, 'self')
        X_dec = attn(X_dec, X_enc, enc_mask, training, 'src')
        X1 = FeedForward(hidden_dim=self.hidden_dim)(X_dec)
        X_dec = AddAndNorm()(X_dec, X1)
        return X_dec


class Decoder(nn.Module):
    pos_encoding: Callable[[int, int], jnp.array]
    vocab_size: int
    embed_dim: int
    layers: Sequence[DecoderLayer]

    @nn.compact
    def __call__(self, X_enc, X_dec, enc_mask, training=False):
        B, N = X_dec.shape[:2]
        dec_mask = causal_mask((1, N, N))
        X_dec = nn.Embed(self.vocab_size, self.embed_dim, name='embed')(X_dec)
        X_dec = X_dec * jnp.sqrt(self.embed_dim)
        # X.shape[-2] is the sequence length
        X_dec = X_dec + self.pos_encoding(X_dec.shape[-2], self.embed_dim)
        for layer in self.layers:
            X_dec = layer(X_enc, X_dec, enc_mask, dec_mask, training=training)
        X_dec = nn.Dense(self.vocab_size, name='final')(X_dec)
        return X_dec


class EncoderDecoderTransformer(nn.Module):
    pos_encoding: Callable[[int, int], jnp.array]
    in_vocab_size: int
    out_vocab_size: int
    embed_dim: int
    n_layers: int
    hidden_dim: int
    attn_dropout: float
    fc_dropout: float
    n_heads: int
    size_per_head: int

    def setup(self):
        self.encoder = Encoder(
            pos_encoding=self.pos_encoding,
            vocab_size=self.in_vocab_size,
            embed_dim=self.embed_dim,
            layers=[EncoderLayer(hidden_dim=self.hidden_dim,
                                 attn_dropout=self.attn_dropout,
                                 fc_dropout=self.fc_dropout,
                                 n_heads=self.n_heads,
                                 size_per_head=self.size_per_head,
                                 name=f'encoder_{i}')
                    for i in range(self.n_layers)])
        self.decoder = Decoder(
            pos_encoding=self.pos_encoding,
            vocab_size=self.out_vocab_size,
            embed_dim=self.embed_dim,
            layers=[DecoderLayer(hidden_dim=self.hidden_dim,
                                 attn_dropout=self.attn_dropout,
                                 fc_dropout=self.fc_dropout,
                                 n_heads=self.n_heads,
                                 size_per_head=self.size_per_head,
                                 name=f'decoder_{i}')
                    for i in range(self.n_layers)])

    def __call__(self, X, Y, source_mask, training=False):
        # required for dot product attention
        chex.assert_equal(self.encoder.embed_dim, self.decoder.embed_dim)
        encodings = self.encoder(X, source_mask, training=training)
        self.sow('intermediates', 'encodings', encodings)
        return self.decoder(encodings, Y, source_mask, training=training)


class EncoderOnlyTransformer(nn.Module):
    pos_encoding: Callable[[int, int], jnp.array]
    vocab_size: int
    embed_dim: int
    n_layers: int
    hidden_dim: int
    attn_dropout: float
    fc_dropout: float
    n_heads: int
    size_per_head: int

    def setup(self):
        self.encoder = Encoder(
            pos_encoding=self.pos_encoding,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            layers=[EncoderLayer(hidden_dim=self.hidden_dim,
                                 attn_dropout=self.attn_dropout,
                                 fc_dropout=self.fc_dropout,
                                 n_heads=self.n_heads,
                                 size_per_head=self.size_per_head,
                                 name=f'encoder_{i}')
                    for i in range(self.n_layers)])

    def __call__(self, X, mask, training=False):
        return self.encoder(X, mask, training=training)


class DecoderOnlyTransformer(nn.Module):
    pos_encoding: Callable[[int, int], jnp.array]
    vocab_size: int
    embed_dim: int
    n_layers: int
    hidden_dim: int
    attn_dropout: float
    fc_dropout: float
    n_heads: int
    size_per_head: int

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.embed_dim)
        self.decoder = Decoder(
            pos_encoding=self.pos_encoding,
            vocab_size=self.out_vocab_size,
            embed_dim=self.embed_dim,
            layers=[DecoderLayer(hidden_dim=self.hidden_dim,
                                 attn_dropout=self.attn_dropout,
                                 fc_dropout=self.fc_dropout,
                                 n_heads=self.n_heads,
                                 size_per_head=self.size_per_head,
                                 name=f'decoder_{i}')
                    for i in range(self.n_layers)])

    def __call__(self, static, X, source_mask, training=False):
        encodings = self.embed(static)
        return self.decoder(encodings, X, source_mask, training=training)
