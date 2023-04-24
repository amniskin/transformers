from clu import metrics
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
import jax.random as jran
import jax.tree_util
import optax
from reproductions.transformers import Transformer, sin_pos_enc


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    """immutable state for parameters and such"""
    metrics: Metrics = Metrics.empty()


@struct.dataclass
class DataGen:
    vocab_size: int
    slen: int
    batch_size: int

    def next(self, key):
        X = jran.randint(key, (self.batch_size, self.slen), 0, self.vocab_size)
        Y = X[:, :-1]
        Yh = X[:, 1:]
        enc_mask = jnp.zeros((self.batch_size,)) + self.slen
        dec_mask = enc_mask - 1
        return {'args': (X, Y, enc_mask, dec_mask), 'label': Yh}


def get_state(mdl, batch, key, opt_conf):
    params = mdl.init(key, *batch['args'])
    tx = optax.chain(
        optax.clip_by_global_norm(opt_conf['grad_clip']),
        optax.sgd(
            learning_rate=opt_conf['lr'],
            momentum=0.9,
            accumulator_dtype='bfloat16',
        )
    )
    state = TrainState.create(apply_fn=mdl.apply,
                              params=params['params'],
                              tx=tx)
    return state


@jax.jit
def train_step(state, batch, key):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, *batch['args'],
                                training=True, rngs={'dropout': key})
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(state, batch):
    logits = state.apply_fn({'params': state.params}, *batch['args'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def fit(conf):
    mdl = Transformer(**conf['model'])
    datagen = DataGen(vocab_size=mdl.in_vocab_size,
                      slen=conf['data']['sequence_length'],
                      batch_size=conf['data']['batch_size'])
    metrics_history = {'train_loss': [],
                       'train_accuracy': []}
    rng, key = jran.split(jran.PRNGKey(conf['seed']), 2)
    state = get_state(mdl, datagen.next(key), key, conf['opt'])
    for step in range(conf['n_epochs'] * conf['n_steps_per_epoch']):
        data_key, mdl_key = jran.split(jran.fold_in(rng, step), 2)
        batch = datagen.next(data_key)
        state = train_step(state, batch, mdl_key)
        state = compute_metrics(state=state, batch=batch)
        if (step + 1) % conf['n_steps_per_epoch'] == 0:
            for metric, value in state.metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value)
            state = state.replace(metrics=state.metrics.empty())
            print(f"train epoch: {(step+1) // conf['n_steps_per_epoch']}, "
                  f"loss: {metrics_history['train_loss'][-1]}, "
                  f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
    return state, metrics_history


if __name__ == '__main__':
    from pprint import pprint
    conf = dict(
        model=dict(
            pos_encoding=sin_pos_enc,
            in_vocab_size=10,
            out_vocab_size=10,
            embed_dim=10,
            n_layers=3,
            hidden_dim=13,
            attn_dropout=0.1,
            fc_dropout=0.1,
            n_heads=7,
            size_per_head=17
        ),
        opt=dict(lr=0.1, grad_clip=1.0),
        data=dict(sequence_length=42, batch_size=20),
        seed=42,
        n_epochs=500,
        n_steps_per_epoch=10,
    )
    state, metrics_history = fit(conf)
    if False:
        print('========================')
        print('======== state =========')
        print('========================')
        print(jax.tree_util.tree_map(jnp.shape, state))
        print('========================')
        print('======== model =========')
        print('========================')
        # print(mdl.tabulate(rng, X, Y, enc_mask, dec_mask))
        pprint(metrics_history)
