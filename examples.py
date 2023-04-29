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
from reproductions.transformers import Transformer, sin_pos_enc
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
            return arr
        return jnp.pad(jnp.array(arr),
                       (0, max_len - len(arr)),
                       'constant',
                       constant_values=(0, self.e))

    def str2ids(self, _str, max_len):
        X = [self.vocab[y] for y in str(_str)]
        if self.reverse:
            X = list(reversed(X))
        _len = len(X)
        X = self.pad(X, max_len)
        return X, _len

    def strs2ids(self, str_arr, max_len):
        X, lens = map(jnp.array,
                      zip(*[self.str2ids(_str, max_len)
                            for _str in str_arr]))
        return X, lens

    def ids2str(self, int_arr, remove=[]):
        out = [self.tokens[int(i)] for i in int_arr]
        out = [x for x in out if x not in remove]
        if self.reverse:
            out = reversed(out)
        return ''.join(out)

    def ids2strs(self, _2d_arr, remove=[]):
        return [self.ids2str(x, remove=remove) for x in _2d_arr]

    def next(self, key):
        args = jran.randint(key, (self.batch_size, 2), 0, self.max_i)
        X = ['+'.join(map(str, x)) for x in args]
        X, enc_lens = self.strs2ids(X, self.in_len)
        Yh, _ = self.strs2ids(jnp.sum(args, axis=1), self.out_len)
        Y = Yh.at[:, 1:].set(Yh[:, :-1]).at[:, 0].set(self.s)
        return {'args': (X, Y, jnp.array(enc_lens), None), 'label': Yh}


def get_state(mdl, batch, key, opt_conf, show=False):
    params = mdl.init(key, *batch['args'])
    if show:
        print(mdl.tabulate(key, *batch['args']))
    else:
        print(
            'num_params:',
            sum(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(
                        lambda x: np.product(jnp.shape(x)),
                        params))))
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
    mdl = Transformer(in_vocab_size=len(datagen.vocab),
                      out_vocab_size=len(datagen.vocab),
                      **conf['model'])
    rng, key = jran.split(jran.PRNGKey(conf['seed']), 2)
    state = get_state(mdl, datagen.next(key), key, conf['opt'])
    with mlflow.start_run():
        mlflow.log_params(conf)
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
                datagen.ids2strs(batch['args'][1][:2]),
            )
            print(f"train epoch: {epoch}, "
                  f"loss: {metrics['loss']:.3E}, "
                  f"accuracy: {(metrics['accuracy']):.04}")
    return state


if __name__ == '__main__':
    from pprint import pprint
    conf = dict(
        model=dict(
            pos_encoding=sin_pos_enc,
            embed_dim=16,
            n_layers=5,
            hidden_dim=8,
            attn_dropout=0.0,
            fc_dropout=0.1,
            n_heads=2,
            size_per_head=4,
        ),
        opt=dict(
            lr=dict(
                init_value=0.1, peak_value=0.5, warmup_steps=200,
                transition_steps=500, decay_rate=0.5,
                transition_begin=1000, staircase=False, end_value=0.001
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
        n_epochs=100,
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
