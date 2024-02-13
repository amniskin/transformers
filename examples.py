from clu import metrics
from flax import struct, core
from flax.training import train_state
import jax
import jax.numpy as jnp
import jax.nn
import jax.random as jran
import jax.tree_util
import logging
import mlflow
import numpy as np
import optax
from reproductions.transformers import (
    EncoderDecoderTransformer, sin_pos_enc, num_params
)
from tqdm.auto import tqdm


logging.getLogger('mlflow').setLevel(logging.ERROR)


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    """immutable state for parameters and such"""
    metrics: Metrics = Metrics.empty()


class DataGen(struct.PyTreeNode):
    vocab: core.FrozenDict[str, int] = struct.field(pytree_node=True)
    tokens: tuple
    batch_size: int
    max_i: int
    in_len: int
    out_len: int
    reverse: bool
    s: int
    e: int

    @classmethod
    def init(cls, batch_size, tokens, reverse=False, max_i=6,
             s='s', e=None):
        if e is None:
            e = '0' if reverse else 'e'
        if isinstance(tokens, str):
            tokens = [x for x in tokens if len(x.strip()) > 0]
        tokens += [s, e]
        tokens = sorted(set(tokens))
        vocab = core.FrozenDict({k: i for i, k in enumerate(tokens)})
        return cls(vocab=vocab,
                   tokens=tuple(tokens),
                   batch_size=batch_size,
                   max_i=max_i,
                   reverse=reverse,
                   # +1 for next 2 bc 10^i requires i + 1 chars
                   in_len=int(np.log10(max_i) + 1) * 2 + 1,
                   out_len=int(np.log10((max_i - 1) * 2)) + 1,
                   s=vocab[s],
                   e=vocab[e])

    def pad(self, arr, max_len):
        assert len(arr) <= max_len, f'ruh roh {arr} -- {max_len}'
        if len(arr) == max_len:
            return jnp.array(arr)
        return jnp.pad(jnp.array(arr),
                       (0, max_len - len(arr)),
                       'constant',
                       constant_values=(0, self.e))

    def str2ids(self, _str):
        X = [self.vocab[y] for y in str(_str)]
        if self.reverse:
            X = list(reversed(X))
        return X

    def strs2ids(self, str_arr, max_len):
        X = [self.str2ids(_str) for _str in str_arr]
        if max_len is None:
            max_len = max(len(X))
        X = jnp.stack([self.pad(x, max_len) for x in X])
        return X

    def ids2str(self, int_arr, remove=[]):
        out = [self.tokens[int(i)] for i in int_arr]
        out = [x for x in out if x not in remove]
        if self.reverse:
            out = reversed(out)
        return ''.join(out)

    def ids2strs(self, _2d_arr, remove=[]):
        return [self.ids2str(x, remove=remove) for x in _2d_arr]

    def next(self, key):
        X = []
        Yh = []
        for i in range(self.batch_size):
            ints = jran.randint(jran.fold_in(key, 2*i),
                                (2,), 0, self.max_i)
            Yh.append(str(ints.sum()))
            ints = [str(x) for x in ints]
            pads = jran.uniform(jran.fold_in(key, 2*i+1),
                                (self.in_len - sum(map(len, ints)) - 1,))
            for p in pads:
                ints[int(p < 0.5)] = '0' + ints[int(p < 0.5)]
            X.append('+'.join(ints))
        X = self.strs2ids(X, self.in_len)
        Yh = self.strs2ids(Yh, self.out_len)
        Y = Yh.at[:, 1:].set(Yh[:, :-1]).at[:, 0].set(self.s)
        return {'args': (X, Y, (X == self.e).astype(jnp.float32)),
                'label': Yh}


def get_state(mdl, batch, key, opt_conf, show=False):
    params = mdl.init(key, *batch['args'])
    mlflow.log_param('num_params', num_params)
    if show:
        print(mdl.tabulate(key, *batch['args']))
    else:
        print('num_params: ', num_params(params))
    lr = opt_conf['lr']
    if not isinstance(lr, (float, int)):
        lr = optax.warmup_exponential_decay_schedule(**lr)
    tx = optax.chain(
        optax.clip_by_global_norm(opt_conf['grad_clip']),
        optax.sgd(
            learning_rate=lr,
        )
    )
    state = TrainState.create(apply_fn=mdl.apply,
                              params=params['params'],
                              tx=tx)
    return state


def compute_metrics(state, loss, logits, batch):
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def train_step(state, batch, key):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, *batch['args'],
                                training=True, rngs={'dropout': key})
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = compute_metrics(state, loss, preds, batch)
    return state, preds


def fit(conf):
    datagen = DataGen.init(**conf['data'])
    mdl = EncoderDecoderTransformer(in_vocab_size=len(datagen.vocab),
                                    out_vocab_size=len(datagen.vocab),
                                    **conf['model'])
    rng, key = jran.split(jran.PRNGKey(conf['seed']), 2)
    with mlflow.start_run():
        mlflow.log_params(conf)
        state = get_state(mdl, datagen.next(key), key, conf['opt'])
        for epoch in range(conf['n_epochs']):
            for step in tqdm(range(conf['n_steps_per_epoch']), leave=False,
                             desc=f'epoch: {epoch}'):
                data_key, mdl_key = jran.split(jran.fold_in(rng, state.step), 2)
                batch = datagen.next(data_key)
                state, preds = train_step(state, batch, mdl_key)
            metrics = state.metrics.compute()
            state = state.replace(metrics=state.metrics.empty())
            mlflow.log_metrics({k: np.float64(v) for k, v in metrics.items()},
                               step=epoch)
            mlflow.llm.log_predictions(
                datagen.ids2strs(batch['args'][0][:2]),
                datagen.ids2strs(jnp.argmax(preds[:2], axis=-1)),
                datagen.ids2strs(batch['label'][:2]),
            )
            print(f"train epoch: {epoch}, "
                  f"loss: {metrics['loss']:.3E}, "
                  f"accuracy: {(metrics['accuracy']):.04}")
    return state


if __name__ == '__main__':
    conf = dict(
        model=dict(
            pos_encoding=sin_pos_enc,
            embed_dim=2,
            n_layers=4,
            hidden_dim=30,
            attn_dropout=0.0,
            fc_dropout=0.1,
            n_heads=10,
            size_per_head=4,
        ),
        opt=dict(
            lr=dict(
                init_value=0.5, peak_value=0.8, warmup_steps=1000,
                transition_steps=2000, decay_rate=0.5,
                transition_begin=1000, staircase=False, end_value=0.05
            ),
            grad_clip=1
        ),
        data=dict(
            s='s',
            batch_size=200,
            max_i=10000,
            tokens='+0123456789',
            reverse=True,
        ),
        seed=42,
        n_epochs=5000,
        n_steps_per_epoch=100,
    )
    state = fit(conf)
    if False:
        print('========================')
        print('======== state =========')
        print('========================')
        print(jax.tree_util.tree_map(jnp.shape, state))
        print('========================')
        print('======== model =========')
        print('========================')
        # print(mdl.tabulate(rng, X, Y, enc_mask, dec_mask))
