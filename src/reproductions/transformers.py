import chex
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence


def masked_softmax(args, mask):
    "mask is [0, 1] -- 1 for keeps, 0 for nots"
    if mask is not None:
        chex.assert_shape(mask, (args.shape[0],))
        mask = jnp.expand_dims(mask, 1)
        mask = jnp.broadcast_to(jnp.arange(args.shape[-2]),
                                (args.shape[0], args.shape[-2])) < mask
        mask = (1 - mask) * -10_000
        if args.ndim == 4:
            mask = jnp.expand_dims(mask, (1, -1))
        elif args.ndim == 3:
            mask = jnp.expand_dims(mask, (-1,))
        else:
            raise NotImplementedError('only dims 3 or 4 are implemented')
        args += mask
    return nn.softmax(args)


def dot_prod_attn(q, k, v, dropout=lambda x: x, mask=None):
    """dot product attention
    q (B[, H], N, D)
    k (B[, H], M, D)
    v (B[, H], M, D)
    m (B[, H], M) or None
    """
    # NxD @ DxM => NxM
    # (B[, H], N, M)
    attn_scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(q.shape[-1])
    attn_weights = masked_softmax(attn_scores, mask)
    # (B[, H], N, D)
    out = dropout(attn_weights) @ v
    return out, attn_weights


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    dropout: float

    @nn.compact
    def __call__(self, q, k, v, mask=None, training=False):
        dropout = nn.Dropout(self.dropout, deterministic=not training)
        out, attn_weights = dot_prod_attn(q, k, v, dropout, mask=mask)
        return out, attn_weights


class MultiHeadAttention(nn.Module):
    """Scaled dot product attention

    Example
    -------
    >>> import jax.random as jran
    >>> rng = jran.PRNGKey(0)
    >>> mdl = MultiHeadAttention(attn_dropout=0.1, fc_dropout=0.1,
    ...                          n_heads=7, size_per_head=13)
    >>> q = jnp.ones((2, 5, 7 * 11))
    >>> kv = jnp.ones((2, 3, 7 * 11))
    >>> mask = jran.randint(rng, (2,), 0, 5)
    >>> rng, now_rng = jran.split(rng)
    >>> params = mdl.init(now_rng, q, kv, kv)
    >>> rng, now_rng = jran.split(rng)
    >>> resp, state = mdl.apply(params, q, kv, kv, mask=mask, training=True,
    ...                         mutable=['intermediates'],
    ...                         rngs={'dropout': now_rng})
    >>> resp.shape
    (2, 5, 77)
    """
    n_heads: int
    size_per_head: int
    attn_dropout: float
    fc_dropout: float
    attn_fn: Callable = dot_prod_attn

    @nn.compact
    def __call__(self, q, k, v, mask=None, training=False):
        # B : batch_size
        # N : input sequence length
        # M : output sequence length
        # D : size of embeddings

        # helpful nomenclature
        # H : number of heads

        # q (B, N, D)
        # k (B, M, D)
        # v (B, M, D)
        # m (B, M) or None
        B, N, D = q.shape
        M = k.shape[1]
        chex.assert_shape(q, (B, N, D))
        chex.assert_shape(k, (B, M, D))
        chex.assert_shape(v, (B, M, D))
        _shape = (B, -1, self.n_heads, self.size_per_head)
        if mask is not None:
            chex.assert_shape(mask, (B,))

        def qkv_layer(x, name):
            x = nn.Dense(self.n_heads * self.size_per_head, name=name)(x)
            x = x.reshape(_shape).swapaxes(1, 2)
            return x
        # BxNxD => BxHxAxD
        q = qkv_layer(q, 'query')
        # BxMxD => BxHxBxD
        k = qkv_layer(k, 'key')
        # BxMxD => BxHxBxD
        v = qkv_layer(v, 'value')
        attn_do = nn.Dropout(self.attn_dropout, deterministic=not training)
        out, attn_weights = self.attn_fn(q, k, v, attn_do, mask=mask)
        # uncomment to keep attention weights in state
        # self.sow('intermediates', 'weights', attn_weights)
        out = out.swapaxes(1, 2).reshape((B, N, -1))
        out = nn.Dense(D, name='output')(out)
        out = nn.Dropout(self.fc_dropout, deterministic=not training)(out)
        return out


class AddAndNorm(nn.Module):
    """The add and norm."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, X, X_out):
        return nn.LayerNorm(self.eps)(X + X_out)


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
    add_norm_eps: float = 1e-6

    def setup(self):
        self.attn = MultiHeadAttention(n_heads=self.n_heads,
                                       size_per_head=self.size_per_head,
                                       attn_dropout=self.attn_dropout,
                                       fc_dropout=self.fc_dropout)
        self.norm_0 = AddAndNorm(self.add_norm_eps)
        self.ff = FeedForward(hidden_dim=self.hidden_dim)
        self.norm_1 = AddAndNorm(self.add_norm_eps)

    def __call__(self, X, mask=None, training=False):
        X1 = self.attn(X, X, X, mask=mask, training=training)
        X = self.norm_0(X, X1)
        X1 = self.ff(X)
        X = self.norm_1(X, X1)
        return X


def sin_pos_enc(sequence_length, embed_dim):
    chex.assert_is_divisible(embed_dim, 2)
    X = jnp.expand_dims(jnp.arange(sequence_length), 1) / \
        jnp.power(10000, jnp.arange(embed_dim, step=2) / embed_dim)
    out = jnp.empty((sequence_length, embed_dim))
    out = out.at[:, 0::2].set(jnp.sin(X))
    out = out.at[:, 1::2].set(jnp.cos(X))
    return out


class Encoder(nn.Module):
    """The transformer encoder network

    Example
    =======
    >>> import jax.random as jran
    >>> def layer():
    ...     return EncoderLayer(hidden_dim=13,
    ...                         attn_dropout=0.1,
    ...                         fc_dropout=0.1,
    ...                         n_heads=7,
    ...                         size_per_head=17,
    ...                         add_norm_eps=1e-6)
    >>> mdl = Encoder(pos_encoding=sin_pos_enc, vocab_size=23,
    ...               embed_dim=2 * 3 * 5,
    ...               layers=[layer() for _ in range(3)])
    >>> rng = jran.PRNGKey(0)
    >>> rng, now_rng = jran.split(rng)
    >>> a = jran.randint(rng, (2, 5), 0, mdl.vocab_size)
    >>> rng, now_rng = jran.split(rng)
    >>> mask = jran.randint(now_rng, (2,), 0, 5)
    >>> rng, now_rng = jran.split(rng)
    >>> params = mdl.init(now_rng, a)
    >>> rng, now_rng = jran.split(rng)
    >>> resp, state = mdl.apply(params, a, mask=mask, training=True,
    ...                         mutable=['intermediates'],
    ...                         rngs={'dropout': now_rng})
    >>> resp.shape
    (2, 5, 30)
    """
    pos_encoding: Callable[[int, int], jnp.array]
    vocab_size: int
    embed_dim: int
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, X, mask=None, training=False):
        X = nn.Embed(self.vocab_size, self.embed_dim)(X)
        X = X * jnp.sqrt(self.embed_dim)
        # X.shape[-2] is the sequence length
        X = X + self.pos_encoding(X.shape[-2], self.embed_dim)
        for layer in self.layers:
            X = layer(X, mask=mask, training=training)
        return X


class DecoderLayer(nn.Module):
    """transformer decoder layer

    Example
    =======
    >>> import jax.random as jran
    >>> mdl = DecoderLayer(hidden_dim=13, attn_dropout=0.1, fc_dropout=0.1,
    ...                    n_heads=2, size_per_head=5, add_norm_eps=1e-6)
    >>> rng = jran.PRNGKey(0)
    >>> rng, now_rng = jran.split(rng)
    >>> X = jran.uniform(rng, (2, 5, 2 * 7))
    >>> rng, now_rng = jran.split(rng)
    >>> enc_mask = jran.randint(now_rng, (2,), 0, 5)
    >>> rng, now_rng = jran.split(rng)
    >>> Y = jran.uniform(rng, (2, 3, 2 * 7))
    >>> rng, now_rng = jran.split(rng)
    >>> dec_mask = jran.randint(now_rng, (2,), 0, 3)
    >>> rng, now_rng = jran.split(rng)
    >>> params = mdl.init(now_rng, X, Y, enc_mask, dec_mask)
    >>> rng, now_rng = jran.split(rng)
    >>> resp, state = mdl.apply(params, X, Y, enc_mask, dec_mask,
    ...                         training=True,
    ...                         mutable=['intermediates'],
    ...                         rngs={'dropout': now_rng})
    >>> resp.shape
    (2, 3, 14)
    >>> # print(mdl.tabulate(rng, X, Y, enc_mask, dec_mask))
    """
    hidden_dim: int
    n_heads: int
    size_per_head: int
    attn_dropout: float
    fc_dropout: float
    add_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, X, Y, enc_mask, dec_mask, training=False):
        def attn(q, kv, mask, training, name):
            mdl = MultiHeadAttention(n_heads=self.n_heads,
                                     size_per_head=self.size_per_head,
                                     attn_dropout=self.attn_dropout,
                                     fc_dropout=self.fc_dropout,
                                     name=f'{name}_attn')
            out = mdl(q, kv, kv, mask=mask, training=training)
            aan = AddAndNorm(self.add_norm_eps, name=f'{name}_addnorm')
            return aan(q, out)
        Y = attn(Y, Y, dec_mask, training, 'self')
        Y = attn(Y, X, enc_mask, training, 'src')
        Y1 = FeedForward(hidden_dim=self.hidden_dim)(Y)
        Y = AddAndNorm(self.add_norm_eps)(Y, Y1)
        return Y


class Decoder(nn.Module):
    """The transformer decoder network

    Example
    =======
    >>> import jax.random as jran
    >>> def layer():
    ...     return DecoderLayer(hidden_dim=13,
    ...                         attn_dropout=0.1,
    ...                         fc_dropout=0.1,
    ...                         n_heads=7,
    ...                         size_per_head=17,
    ...                         add_norm_eps=1e-6)
    >>> mdl = Decoder(pos_encoding=sin_pos_enc, vocab_size=37,
    ...               embed_dim=2 * 3 * 5,
    ...               layers=[layer() for i in range(3)])
    >>> rng = jran.PRNGKey(0)
    >>> rng, now_rng = jran.split(rng)
    >>> X = jran.uniform(now_rng, (2, 5, mdl.embed_dim))
    >>> rng, now_rng = jran.split(rng)
    >>> enc_mask = jran.randint(now_rng, (2,), 0, 5)
    >>> rng, now_rng = jran.split(rng)
    >>> Y = jran.randint(rng, (2, 3), 0, mdl.vocab_size)
    >>> rng, now_rng = jran.split(rng)
    >>> dec_mask = jran.randint(now_rng, (2,), 0, 3)
    >>> rng, now_rng = jran.split(rng)
    >>> params = mdl.init(now_rng, X, Y, enc_mask, dec_mask)
    >>> rng, now_rng = jran.split(rng)
    >>> resp, state = mdl.apply(params, X, Y, enc_mask, dec_mask,
    ...                         training=True,
    ...                         mutable=['intermediates'],
    ...                         rngs={'dropout': now_rng})
    >>> resp.shape  # (B, X_sequence_length, mdl.embed_dim)
    (2, 3, 37)
    """
    pos_encoding: Callable[[int, int], jnp.array]
    vocab_size: int
    embed_dim: int
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, X, Y, enc_mask, dec_mask, training=False):
        Y = nn.Embed(self.vocab_size, self.embed_dim)(Y)
        Y = Y * jnp.sqrt(self.embed_dim)
        # X.shape[-2] is the sequence length
        Y = Y + self.pos_encoding(Y.shape[-2], self.embed_dim)
        for i, layer in enumerate(self.layers):
            Y = layer(X, Y, enc_mask, dec_mask, training=training)
        Y = nn.Dense(self.vocab_size, name='final')(Y)
        Y = nn.softmax(Y)
        return Y


class Transformer(nn.Module):
    """encoder/decoder transformer

    Example
    =======
    >>> import jax.random as jran
    >>> mdl = Transformer(
    ...     pos_encoding=sin_pos_enc,
    ...     in_vocab_size=37,
    ...     out_vocab_size=23,
    ...     embed_dim=2 * 3 * 5,
    ...     n_layers=3,
    ...     hidden_dim=13,
    ...     attn_dropout=0.1,
    ...     fc_dropout=0.1,
    ...     n_heads=7,
    ...     size_per_head=17)
    >>> rng = jran.PRNGKey(0)
    >>> rng, now_rng = jran.split(rng)
    >>> X = jran.randint(rng, (2, 5), 0, mdl.in_vocab_size)
    >>> rng, now_rng = jran.split(rng)
    >>> enc_mask = jran.randint(now_rng, (2,), 0, 5)
    >>> rng, now_rng = jran.split(rng)
    >>> Y = jran.randint(rng, (2, 3), 0, mdl.out_vocab_size)
    >>> rng, now_rng = jran.split(rng)
    >>> dec_mask = jran.randint(now_rng, (2,), 0, 3)
    >>> rng, now_rng = jran.split(rng)
    >>> params = mdl.init(now_rng, X, Y, enc_mask, dec_mask)
    >>> rng, now_rng = jran.split(rng)
    >>> resp, state = mdl.apply(params, X, Y, enc_mask, dec_mask,
    ...                         training=True,
    ...                         mutable=['intermediates'],
    ...                         rngs={'dropout': now_rng})
    >>> resp.shape  # (B, Y_sequence_length, decoder.vocab_size)
    (2, 3, 23)
    """
    pos_encoding: int
    in_vocab_size: int
    out_vocab_size: int
    embed_dim: int
    n_layers: int
    hidden_dim: int
    attn_dropout: int
    fc_dropout: int
    n_heads: int
    size_per_head: int
    add_norm_eps: int = 1e-6

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
                                 add_norm_eps=self.add_norm_eps,
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
                                 add_norm_eps=self.add_norm_eps,
                                 name=f'decoder_{i}')
                    for i in range(self.n_layers)])

    def __call__(self, X, Y, source_mask, target_mask, training=False):
        # required for dot product attention
        chex.assert_equal(self.encoder.embed_dim, self.decoder.embed_dim)
        encodings = self.encoder(X, source_mask, training=training)
        self.sow('intermediates', 'encodings', encodings)
        return self.decoder(encodings, Y, source_mask, target_mask,
                            training=training)
