from functools import partial
import os
from typing import Any, Callable, Sequence, Tuple
from typing import NamedTuple
from typing import List

from absl import app
from absl import flags

import jax
import jax.numpy as jnp
import flax
import optax

import matplotlib
import matplotlib.pyplot as plt

from ml_collections import config_dict
from ml_collections import config_flags

import numpy as onp
import ot
import pandas as pd
import tqdm  # progress bar
import wandb

import tensorflow_datasets as tfds
import tensorflow as tf

from jax.config import config
import jax.tree_util as tree_util
from flax.training import train_state
from flax import linen as nn
from flax import struct
from jaxopt.tree_util import tree_l2_norm

import plotly as py
import plotly.subplots
import plotly.graph_objects as go

import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.datasets import make_moons, make_circles

import cnqr
from cnqr.layers import StiefelDense, Normalized2ToInftyDense
from cnqr.layers import fullsort, groupsort2
from cnqr.layers import PositiveDense


# Good default values.
cfg = config_dict.ConfigDict()
cfg.dataset = 'two-moons'
cfg.noise = 0.05
cfg.dataset_size = 1024
cfg.batch_size = 1024
cfg.center_PQ = False
cfg.disc_lr = 1e-2
cfg.gen_lr = 5e-3
cfg.heuristic_lr = True
cfg.warmup_disc = 1600
cfg.num_epochs = 500
cfg.num_steps_gen = 1
cfg.num_steps_disc = 128
cfg.plot_freq = 0
cfg.log_wandb = "run"

_CONFIG = config_flags.DEFINE_config_dict('cfg', cfg)

project_name = "flaxGanW2"
sweep_config = {
  'method': 'bayes',
  'name': 'default',
  'metric': {'goal': 'minimize', 'name': 'w2_rel_gap'},
  'early_terminate': {'type': 'hyperband', 'min_iter': 15, 'eta': 2},
  'parameters': {
      'disc_lr': {
        'max': 1e-1,
        'min': 1e-5,
        'distribution': 'log_uniform_values'},
      'gen_lr': {
        'max': 5e-2,
        'min': 1e-5,
        'distribution': 'log_uniform_values'},
      'num_steps_disc': {
        'values': [16, 32, 64, 128, 256],
        'distribution': 'categorical'},
      'batch_size': {
        'values': [32, 64, 128, 256, 512, 1024],
        'distribution': 'categorical'},
      'num_steps_gen': {
        'values': [1, 2, 4, 8, 16],
        'distribution': 'categorical'},
  }
}


@jax.jit
def update_lipschitz(state, grads, lip_vars):
  """Updates model parameters."""
  return state.apply_gradients(grads=grads, lip_state=lip_vars)


@jax.jit
def update_convex(state, grads, convex_vars):
  """Updates model parameters."""
  return state.apply_gradients(grads=grads, convex_state=convex_vars)


@jax.jit
def predict_lipschitz(train_state, points):
  """Predicts on a batch of points."""
  model_params = {'params': train_state.params, 'lip': train_state.lip_state}
  preds = train_state.apply_fn(model_params, points, train=False)
  return preds


@jax.jit
def predict_convex(train_state, points):
  """Predicts on a batch of points."""
  model_params = {'params': train_state.params, 'convex': train_state.convex_state}
  preds = train_state.push(model_params, points, train=False)
  return preds


def grad_icnn(model):

  def apply_single_input(params, single_input, train, mutable):
    preds_and_aux = model.apply(params, single_input[jnp.newaxis, ...], train=train, mutable=mutable)
    if mutable is False:
      return preds_and_aux, None  # no aux output.
    return preds_and_aux
  
  def grad_single_input(params, single_input, train, mutable):
    grad_fn = jax.grad(apply_single_input, argnums=1, has_aux=True)
    potential, convex_variables = grad_fn(params, single_input, train, mutable)
    return potential, convex_variables
  
  def grad_inputs(params, single_input, train, mutable=False):
    grad_fns = jax.vmap(grad_single_input, in_axes=(None, 0, None, None))
    potentials, convex_variables = grad_fns(params, single_input, train, mutable)
    # extract convex variables that are shared accross the batch
    if convex_variables is None:
      return potentials
    convex_variables = tree_util.tree_map(lambda x: x[0], convex_variables)
    return potentials, convex_variables

  return grad_inputs


def predict_ds(train_state, ds, predict_model):
  """Predicts on a dataset."""
  jnp_preds = [predict_model(train_state, tf_to_jax(batch)) for batch in ds]
  jnp_preds = jnp.concatenate(jnp_preds)
  return jnp_preds


class LipschitzTrainState(train_state.TrainState):
  """Train state with Lipschitz constraint."""
  lip_state: Any


@jax.jit
def apply_lipschitz(params, state, inputs):
  model_params = {'params': params, 'lip': state.lip_state}
  out, variables = state.apply_fn(model_params, inputs, train=True, mutable='lip')
  return out, variables


def unbalanced_KR_W1(fP, fQ):
  """Unbalanced Kantorovich-Rubinstein (KR) loss for Wasserstein-1 distance.
  The loss is unbalanced in the sense that fP and fQ can have arbitrary sizes.
  
  Args:
    fP: array of shape (batch_size,).
    fQ: array of shape (batch_size,).
  Returns:
    loss: float.
  """
  loss = -(fP.mean() - fQ.mean())  # maximize E_P[f(x)] - E_Q[f(x)].
  return loss


def discriminator_loss(params, disc_state, gP, Q, has_aux=False):
  """Computes discriminator loss.
  
  Args:
    params: dict of parameters.
    disc_state: discriminator state.
    gP: array of shape (batch_size, dim).
    Q: array of shape (batch_size, dim).
    has_aux: bool, whether to return auxiliary variables.
    
  Returns:
    loss: float.
    aux: tuple of auxiliary variables containing Lipschitz constant, fP and fQ.
  """
  fgP, _ = apply_lipschitz(params, disc_state, gP)
  fQ, variables = apply_lipschitz(params, disc_state, Q)
  loss = unbalanced_KR_W1(fgP, fQ)
  if has_aux:
    return loss, (variables['lip'], fgP, fQ)
  return loss


@jax.jit
def apply_discriminator(gan_state, P, Q):
  """Computes gradients, loss and accuracy for a single batch."""
  gP, _ = apply_convex(gan_state.gen_state.params, gan_state.gen_state, P)
  grad_fn = jax.value_and_grad(discriminator_loss, has_aux=True)
  (loss, aux), grads = grad_fn(gan_state.disc_state.params, gan_state.disc_state, gP, Q, has_aux=True)
  return grads, aux, loss


class GanStates(NamedTuple):
  disc_state: LipschitzTrainState
  gen_state: LipschitzTrainState


def pushforward_loss(gP, Q, disc_state):
  """Computes Wasserstein-2 regularized loss between T#P and Q.

  Args:
    gP: array of shape (batch_size, dim).
    Q: array of shape (batch_size, dim).
    disc_state: LipschitzTrainState.
  Returns:
    loss: float.
  """
  # minimize the Wasserstein distance hence minus (-) sign.
  loss = -discriminator_loss(disc_state.params, disc_state, gP, Q)
  return loss


@jax.jit
def apply_generator(gan_state, P, Q):
  """Computes gradients, loss and update Lipschitz variables for a single batch.
  
  Args:
    gan_state: GanStates.
    P: array of shape (batch_size, dim).
    Q: array of shape (batch_size, dim).
  
  Returns:
    grads: dict of gradients.
    aux: tuple of auxiliary variables.
    loss: float, generator loss.
  """

  def generator_loss(params):
    gP, variables = apply_convex(params, gan_state.gen_state, P)
    loss = pushforward_loss(gP, Q, gan_state.disc_state)
    return loss, get_convex_state(variables)

  grad_fn = jax.value_and_grad(generator_loss, has_aux=True)
  (loss, aux), grads = grad_fn(gan_state.gen_state.params)
  return grads, aux, loss


def log_metrics(prefix, metrics, grads=None, epoch=None, pbar=None):
  """Logs metrics to wandb and prints them with tqdm progress bar."""
  metrics = {(prefix+key): float(onp.array(metric.mean())) for key, metric in metrics.items()}
  if grads is not None:
    metrics[f'{prefix}gradnorm'] = float(tree_l2_norm(grads))
  wandb.log(metrics)
  if pbar is not None:
    pbar.set_description(desc=f"[Epoch={epoch}] ", refresh=True)
    pbar.set_postfix({k:f"{m:.5f}" for k, m in metrics.items()})
    pbar.update()
  else:
    print(metrics)


def tf_to_jax(arr):
  return jnp.array(arr)
  # return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(tf_arr))


def get_test_ds(onp_array):
  """Returns a tf.data.Dataset from a numpy array solely for iteration purposes."""
  ds = tf.data.Dataset.from_tensor_slices(onp_array)
  ds = ds.batch(cfg.batch_size)
  return ds


def compute_Monge_ground_truth(onp_P, onp_Q):
  """Compute the Monge map between two distributions.

  Args:
    onp_P: array of shape (n, d).
    onp_Q: array of shape (n, d).

  Returns:
    TP: array of shape (n, d). Monge map T(P) = Q.
    cost: float.
  """
  assert len(onp_P) == len(onp_Q)
  n = len(onp_P)
  M = ot.dist(onp_P, onp_Q, metric='sqeuclidean', p=2)
  a, b = onp.ones((n,)) / n, onp.ones((n,)) / n  # uniform distribution on samples.
  G0, log = ot.emd(a, b, M, log=True)
  # Monge map <=> Kantorovich relaxation equivalent when the samples have the same size.
  # Hence for each p_i in P we have a unique associated q_i.
  TP = onp_Q[onp.argmax(G0, axis=-1)]
  return TP, log['cost']


def compute_w1_global(disc_state, jnp_GP, onp_Q):
  """Compute the Wasserstein-1 distance between two distributions.
  
  Args:
    disc_state: discriminator state.
    jnp_GP: jax.numpy.ndarray of shape (n, d) where n is the number of samples and d the dimension.
    onp_Q: numpy.ndarray of shape (n, d) where n is the number of samples and d the dimension.
    
  Returns:
    dict of metrics comparing the ground truth of POT and the Wasserstein-1 distance computed by the discriminator.
    keys:
      'w1_disc': Wasserstein-1 distance computed by the discriminator.
      'w1_gt': Wasserstein-1 distance computed by POT.
      'w1_abs_gap': absolute gap between 'w1_disc' and 'w1_gt'.
      'w1_rel_gap': relative gap between 'w1_disc' and 'w1_gt'.
  """
  ds_GP = get_test_ds(jnp_GP)
  ds_Q = get_test_ds(onp_Q)
  yGP = predict_ds(disc_state, ds_GP, predict_lipschitz).flatten()
  yQ = predict_ds(disc_state, ds_Q, predict_lipschitz).flatten()

  w1_disc = jnp.mean(yGP) - jnp.mean(yQ)

  n = len(jnp_GP)
  M = ot.dist(jnp_GP, onp_Q, metric='euclidean', p=2)
  a, b = jnp.ones((n,)) / n, jnp.ones((n,)) / n  # uniform distribution on samples.
  _, log = ot.emd(a, b, M, log=True)
  w1_gt = jnp.array(log['cost'])  # positive.

  # by construction observe that w1_gt > w1_disc.
  # proof: duality arguments.
  w1_abs_gap = w1_gt - w1_disc
  w1_rel_gap = w1_abs_gap / jnp.maximum(w1_gt, 1e-6) * 100

  return {'w1_abs_gap':w1_abs_gap, 'w1_rel_gap':w1_rel_gap, 'w1_gt': w1_gt, 'w1_disc':w1_disc}


def evaluate(gan_state, onp_P, onp_Q, monge_gt):
  """Evaluate the quality of the generator.
  
  Args:
    gan_state: GAN state.
    onp_P: numpy.ndarray of shape (n, d) where n is the number of samples and d the dimension.
    onp_Q: numpy.ndarray of shape (n, d) where n is the number of samples and d the dimension.
    monge_gt: tuple (TP, cost_gt) where TP is the Monge map T(P) = Q and cost_gt is the cost of the OT problem.
  
  Returns:
    dict of metrics.
    keys:
      'w2_abs_gap': absolute gap between G#P and T#P.
      'w2_rel_gap': relative gap between G#P and T#P.
      and the metrics returned by compute_w1_global.
  """
  ds_P = get_test_ds(onp_P)
  jnp_GP = predict_ds(gan_state.gen_state, ds_P, predict_convex)

  # Compute gap between G#P and T#P. Target: zero loss.
  TP, cost_gt = monge_gt
  w2_abs_gap = jnp.mean(jnp.sum((jnp_GP - TP)**2, axis=-1))  # mean L2 norm (squared)
  w2_rel_gap = w2_abs_gap / jnp.maximum(cost_gt, 1e-6) * 100.  # positive.

  # Check that W1 works well.
  w1_metrics = compute_w1_global(gan_state.disc_state, jnp_GP, onp_Q)
  return {'w2_abs_gap': w2_abs_gap, 'w2_rel_gap':w2_rel_gap, **w1_metrics}


class Discriminator(nn.Module):
  hidden_widths: Sequence[int] = (64, 64, 64, 64)

  @nn.compact
  def __call__(self, inputs, train):

    x = inputs
    for width in self.hidden_widths:
      x = StiefelDense(features=width)(x, train=train)
      x = fullsort(x)
    x = Normalized2ToInftyDense(features=1)(x, train=train)

    return x
  

def get_lip_state(params):
  """Extracts the lipschitz variables from the discriminator parameters."""
  return params.get('lip', None)


def create_discriminator_state(rng, batch_size, features, learning_rate):
  """Creates initial `TrainState`."""
  model = Discriminator()
  keys = dict(zip(['params', 'lip'], jax.random.split(rng, 2)))
  dummy_batch = jnp.zeros([batch_size, features])
  model_params = model.init(keys, dummy_batch, train=True)
  # compatible_tabulate(model, keys, dummy_batch)
  lip_state = get_lip_state(model_params)
  params = model_params['params']
  tx = optax.adam(learning_rate)
  return LipschitzTrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    lip_state=lip_state)


def cumulative_max(x, group_size):
  """Cumulative max of a vector.

  This function is convex and differentiable, because the max is
  differentiable and convex.
  
  Args:
    x: vector of shape (..., n).
    group_size: number of entries to consider in the max.
    
  Returns:
    vector of shape (..., n) where the i-th entry is the max of the first i entries."""
  n = x.shape[-1]
  assert n % group_size == 0, 'n must be divisible by group_size'
  x = jnp.reshape(x, (-1, group_size))
  y = jax.lax.cummax(x, axis=-1)
  return jnp.reshape(y, (-1, n))


class ICNN(nn.Module):
  """Input Convex Neural Network (ICNN) architecture with initialization.
  
  Implementation of input convex neural networks as introduced in
  Amos+(2017).
  
  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The
    output dimension of the last layer is 1 by default.
    dim_y: data dimensionality (default: 2).
    init_fn: choice of initialization method for weight matrices (default:
    `jax.nn.initializers.normal`).
    epsilon_init: value of standard deviation of weight initialization method.
    *sigma_act_fn: choice of activation function used in convex network architecture
  """
  dim_hidden: Sequence[int] = (32, 32, 16)
  init_fn: Callable = jax.nn.initializers.normal
  epsilon_init: float = 1e-2
  sigma_act_fn: Callable = nn.softplus  # alternative -> ReLU, ELU

  @nn.compact
  def __call__(self, y, train=None):
    """Forward pass.
    Args:
      y: observation from source distribution.
        array of shape (B, n_features_y) with B the batch_size.
      train: boolean, whether to use the train or eval mode. Ignored here.
    Returns:
      outputs: array of shape (B,)
    """
    dim_hidden = list(self.dim_hidden) + [1]

    # ===== Inference =====
    z = y
    for i, width in enumerate(dim_hidden):
      V = nn.Dense(features=width, use_bias=True)  # shortcut from the input to the intermediate layers.

      if i == 0:
        h = V(y)
      else:
        U = PositiveDense(features=width, kernel_init='inv_act_fun', use_bias=False)
        h = U(z, train=train) + V(y)

      if i+1 < len(dim_hidden):
        z = self.sigma_act_fn(h)
      else:
        z = h
      
    return z.squeeze()


class ConvexTrainState(train_state.TrainState):
  """Train state for convex networks."""
  convex_state: Any
  push: Callable = struct.field(pytree_node=False)


def get_convex_state(params):
  """Extracts the convex variables from the discriminator parameters."""
  return params.get('convex', None)

@jax.jit
def apply_convex(params, state, inputs):
  model_params = {'params': params, 'convex': state.convex_state}
  out, variables = state.push(model_params, inputs, train=True, mutable='convex')
  return out, variables


def create_generator_state(rng, batch_size, features, learning_rate):
  """Creates initial `TrainState`."""
  model = ICNN()
  keys = dict(zip(['params', 'convex'], jax.random.split(rng, 2)))
  dummy_batch = jnp.zeros([batch_size, features])
  model_params = model.init(keys, dummy_batch, train=True)
  # compatible_tabulate(model, keys, dummy_batch)
  params = model_params['params']
  convex_state = get_convex_state(model_params)
  tx = optax.adam(learning_rate)
  return ConvexTrainState.create(
    apply_fn=model.apply,
    push=grad_icnn(model),
    params=params,
    tx=tx,
    convex_state=convex_state)


def create_gan_state(rng, features):
  """Creates initial `GanStates`."""
  rng_disc, rng_gen = jax.random.split(rng, 2)
  disc_state = create_discriminator_state(rng_disc, cfg.batch_size, features, cfg.disc_lr)
  gen_lr = cfg.gen_lr
  gen_state = create_generator_state(rng_gen, cfg.batch_size, features, gen_lr)
  return GanStates(disc_state=disc_state, gen_state=gen_state)


@jax.jit
def update_discriminator(gan_state, grads, lip_vars):
  disc_state, gen_state = gan_state
  disc_state = disc_state.apply_gradients(grads=grads, lip_state=lip_vars)
  gan_state = GanStates(disc_state=disc_state, gen_state=gen_state)
  return gan_state


def train_discriminator_epoch(gan_state, ds_P, ds_Q, num_steps_disc, epoch):
  """Train discriminator for a single epoch."""
  pbar = tqdm.tqdm(total=num_steps_disc)

  for step, P, Q in zip(range(num_steps_disc), ds_P, ds_Q):
    P, Q = tf_to_jax(P), tf_to_jax(Q)
    grads, aux, loss = apply_discriminator(gan_state, P, Q)
    (lip_vars, fP, fQ) = aux
    gan_state = update_discriminator(gan_state, grads, lip_vars)
    log_metrics('disc_', {'fP':fP, 'fQ':fQ, 'loss': loss},
                         grads=grads, epoch=epoch, pbar=pbar)

  return gan_state


@jax.jit
def update_generator(gan_state, grads, convex_vars):
  disc_state, gen_state = gan_state
  gen_state = gen_state.apply_gradients(grads=grads, convex_state=convex_vars)
  gan_state = GanStates(disc_state=disc_state, gen_state=gen_state)
  return gan_state


def train_generator_epoch(gan_state, ds_P, ds_Q, num_steps_gen, epoch):
  """Train discriminator for a single epoch."""
  pbar = tqdm.tqdm(total=num_steps_gen)
  for step, P, Q in zip(range(num_steps_gen), ds_P, ds_Q):
    P, Q = tf_to_jax(P), tf_to_jax(Q)
    grads, convex_vars, loss = apply_generator(gan_state, P, Q)
    gan_state = update_generator(gan_state, grads, convex_vars)
    log_metrics('gen_', {'loss': loss},
                          grads=grads, epoch=epoch, pbar=pbar)
  return gan_state


def build_dataset(x, batch_size):  # create a tf.Dataset from a numpy array.
  """Build a tf.Dataset from a numpy array."""
  x = onp.random.permutation(x).astype('float32')
  ds = tf.data.Dataset.from_tensor_slices(x)
  to_shuffle = 2
  ds = ds.repeat()
  ds = ds.shuffle(to_shuffle*batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(2)
  return iter(ds)


def infinite_uniform(dim, batch_size):
  while True:
    yield tf.random.uniform(shape=(batch_size,dim), minval=-1., maxval=1.)


def show(gen_state, source, target, epoch, fig=None, row=1, col=1, plot_lst='pqt', upload=False):
  """Show the Generator plot in 2D.

  Args:
    gen_state: Lipschitz state of the generator.
    source: source dataset of shape (num_samples, num_features).
    target: source dataset of shape (num_samples, num_features).
    epoch: current epoch.
    fig: figure to update.
    row: row of the subplot.
    col: column of the subplot.
    plot_lst: list of plots to show. Can be 'p', 'q' or 't' or any combination.
    upload: whether to upload the plot to wandb.
  """
  gen = lambda inputs: predict_convex(gen_state, inputs)

  pp = source
  qq = target
  tp = gen(pp)
    
  size = 3.

  showlegend = row == col == 1
  opacity = 0.3
  data = []

  if 'q' in plot_lst:
    q_plot = go.Scatter(x=qq[:,0], y=qq[:,1], marker={'color':'green',
                                                      'size':size,
                                                      'opacity':opacity},
                        mode='markers', name='Q', showlegend=showlegend)
    data.append(q_plot)
  if 'p' in plot_lst:
      offset_x = 0.
      offset_y = 0.
      p_plot = go.Scatter(x=pp[:,0] + offset_x, y=pp[:,1] + offset_y, marker={'color':'blue',
                                                                              'size':size,
                                                                              'opacity':opacity},
                          mode='markers', name='P', showlegend=showlegend)
      data.append(p_plot)
  if 't' in plot_lst:
    tp_plot = go.Scatter(x=tp[:,0], y=tp[:,1], marker={'color':'red',
                                                      'size':size,
                                                      'opacity':opacity},
                        mode='markers', name='G#P', showlegend=showlegend)
    data.append(tp_plot)
  if fig is None:
    fig = go.Figure(data=data)
    fig.update_yaxes(
      scaleratio = 1,
      title=dict(text='y')
    )
    fig.update_xaxes(
      scaleanchor = "y",
      title=dict(text='x'),
      scaleratio = 1,
    )
    fig.update_layout(font=dict(size=16),
                      legend= {'itemsizing': 'constant'},
                      autosize=True, width=500, height=500)
    fig.write_image("figures/G#P.png")
    if upload:
      wandb.log({"G#P": wandb.Plotly(fig)})


def center_ds(points):
  """Center the dataset.

  Remark: Wasserstein-2 distance is invariant to translation.
  
  Args:
    points: dataset of shape (num_samples, num_features).
    
  Returns:
    centered dataset of shape (num_samples, num_features)."""
  mean = jnp.mean(points, axis=0, keepdims=True)
  return points - mean


def get_datasets():
  """Get the datasets."""
  if cfg.dataset == 'two-moons':
    make_ds = make_moons
  elif cfg.dataset == 'two-circles':
    make_ds = make_circles
  n_samples = cfg.dataset_size // 2
  X, y = make_ds(n_samples=(n_samples, n_samples), noise=cfg.noise)
  X = X.astype(jnp.float32)
  X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)  # centered
  Q = X[y == 0, :]
  P = X[y == 1, :]
  if cfg.center_PQ:
    P = center_ds(P)
    Q = center_ds(Q)
  return P, Q


def init_wandb():
  if cfg.log_wandb == 'run':
    # Log all hyper-parameters because config=cfg.

    wandb.init(project=project_name, mode="online", config=cfg)
  elif cfg.log_wandb == 'disabled':
    wandb.init(project=project_name, mode="disabled", config=cfg)
  else:  # this is a sweep.
    wandb.init()  # warning: do not log all hyper-parameters!
    # instead wandb.config contains the assignments of hyper_parameters.
    # made by the sweep agent. We retrieve them, put them in the config dict,
    # and log them manually.
    for param in sweep_config['parameters']:
      cfg[param] = wandb.config[param]
  # log all hyper-parameters in every case!
  df = pd.DataFrame.from_dict(data={k: [v] for k, v in cfg.items()}, orient='columns')
  print(df)
  hyper_params_table = wandb.Table(data=df)
  wandb.log({"hyper_params": hyper_params_table})


def train():
  init_wandb()
  
  # get the data.
  onp_P, onp_Q = get_datasets()
  ds_P = build_dataset(onp_P, batch_size=cfg.batch_size)
  ds_Q = build_dataset(onp_Q, batch_size=cfg.batch_size)

  # compute ground truth.
  monge_gt = compute_Monge_ground_truth(onp.array(onp_P), onp.array(onp_Q))
  print(f"Ground truth={monge_gt[1]:.5f}")

  # create the models.
  rng = jax.random.PRNGKey(seed=63168)
  features = onp_P.shape[-1]
  gan_state = create_gan_state(rng, features)

  # Pushforward between gP ~= P and Q.
  gan_state = train_discriminator_epoch(gan_state, ds_P, ds_Q, cfg.warmup_disc, epoch=0)
  # evaluate the initial error.
  metrics = evaluate(gan_state, onp_P, onp_Q, monge_gt)

  log_metrics('', metrics, epoch=0, pbar=None)
  if cfg.plot_freq > 0:
    show(gan_state.gen_state, onp_P, onp_Q, epoch=0, plot_lst='tpq', row=1, col=1)

  # train the models.
  for epoch in range(1, cfg.num_epochs+1):
    # alternate training of generator and discriminator.
    gan_state = train_generator_epoch(gan_state, ds_P, ds_Q, cfg.num_steps_gen, epoch)
    gan_state = train_discriminator_epoch(gan_state, ds_P, ds_Q, cfg.num_steps_disc, epoch)

    # evaluate errors.
    metrics = evaluate(gan_state, onp_P, onp_Q, monge_gt)
    log_metrics('', metrics, epoch=epoch, pbar=None)
    
    if cfg.plot_freq > 0 and epoch%cfg.plot_freq == 0:
      show(gan_state.gen_state, onp_P, onp_Q, epoch=epoch, plot_lst='tpq', row=1, col=1)

  show(gan_state.gen_state, onp_P, onp_Q, epoch=epoch, plot_lst='tpq', row=1, col=1, upload=True)


def main(_):
  tf.config.experimental.set_visible_devices([], 'GPU')
  matplotlib.use('Agg')

  wandb.login()
  
  if cfg.log_wandb in ['run', 'disabled']:
    train()
    return

  if cfg.log_wandb.startswith('sweep_'):
    sweep_name = cfg.log_wandb[len('sweep_'):]
    sweep_config['name'] = sweep_name
    for key, value in cfg.items():
      if key not in sweep_config['parameters']:
        sweep_config['parameters'][key] = dict(value=value, distribution='constant')
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)      

  if cfg.log_wandb.startswith('resume_'):
    sweep_id = cfg.log_wandb[len('resume_'):]
   
  wandb.agent(sweep_id, function=train, count=None)


if __name__ == '__main__':
  app.run(main)
