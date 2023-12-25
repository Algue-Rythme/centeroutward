from functools import partial
from typing import Callable, Sequence

from absl import app
from absl import flags

import jax
import jax.numpy as jnp
import optax

import matplotlib

from ml_collections import config_dict
from ml_collections import config_flags

import numpy as onp
import pandas as pd
import wandb

import tensorflow as tf

import jax.nn.initializers as initializers
from flax import linen as nn

import pandas as pd

from cnqr.layers import StiefelDense, Normalized2ToInftyDense
from cnqr.layers import fullsort, groupsort2
from cnqr._src.parametrizations import CachedParametrization

from convex_nn import ConvexDense, PositiveOrthant, NonNegativeOrthant, StochasticMatrix, OrthostochasticMatrix
from convex_nn import cumulative_max, group_logsumexp, stable_logsumexp, non_negative_orthant_squared_norm
from datasets import get_datasets, build_dataset
from evaluate import evaluate, compute_Monge_ground_truth
from train import train_discriminator_epoch, train_generator_epoch, log_metrics, grad_icnn
from train import GanStates, LipschitzTrainState, ConvexTrainState, get_lip_state, get_convex_state
from plot import plot_generator


# Good default values.
cfg = config_dict.ConfigDict()

cfg.dataset = 'center-outward-2'
cfg.noise = 0.1
cfg.dataset_size = 2048

cfg.batch_size = 256
cfg.center_PQ = True  # possible due W2 properties.

cfg.disc_lr = 2.5e-4
cfg.gen_lr = 2.5e-4

cfg.warmup_disc = 1200
cfg.num_epochs = 1500
cfg.num_steps_gen = 2
cfg.num_steps_disc = 128

cfg.convex = 'orthostochastic'
cfg.sigma_act_fn = 'cummax'
cfg.out_act_fn = 'logsumexp'
cfg.kernel_init_skip = 'zeros'
cfg.quadratic = True
cfg.dim_hidden = [32, 32]

cfg.plot_freq = 50
cfg.log_wandb = "disabled"

_CONFIG = config_flags.DEFINE_config_dict('cfg', cfg)

project_name = "flaxGanW2"
sweep_config = {
  'method': 'bayes',
  'name': 'default',
  'metric': {'goal': 'minimize', 'name': 'w2_abs_gap'},
  'early_terminate': {'type': 'hyperband', 'min_iter': 15, 'eta': 2},
  'parameters': {
      'disc_lr': {
        'max': 5e-3,
        'min': 5e-5,
        'distribution': 'log_uniform_values'},
      'gen_lr': {
        'max': 5e-3,
        'min': 5e-5,
        'distribution': 'log_uniform_values'},
      'convex': {
        'values': ['orthostochastic', 'stochastic', 'stochastic-bis'],
        'distribution': 'categorical'},
      'sigma_act_fn': {
        'values': ['logsumexp', 'cummax'],
        'distribution': 'categorical'},
      'out_act_fn': {
        'values': ['logsumexp', 'identity'],
        'distribution': 'categorical'},
      'kernel_init_skip': {
        'values': ['orthogonal', 'zeros', 'glorot'],
        'distribution': 'categorical'},
  }
}


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


class ICNN(nn.Module):
  """Input Convex Neural Network (ICNN) architecture with initialization.
  
  Implementation of input convex neural networks as introduced in
  Amos+(2017).
  
  Args:
    dim_hidden: sequence specifying size of hidden dimensions. The
    output dimension of the last layer is 1 by default.
    epsilon_init: value of standard deviation of weight initialization method.
    sigma_act_fn: choice of activation function used in intermediate layers.
    out_act_fn: choice of activation function used in output layer.
    convex_dense: inner convex layer.
    kernel_init_skip: kernel initialization method for skip connections from input.
    quadratic: whether to include quadratic term in input.
  """
  dim_hidden: Sequence[int]
  sigma_act_fn: Callable = nn.relu  # alternative -> ReLU, ELU
  out_act_fn: Callable = stable_logsumexp
  convex_dense: Callable = ConvexDense
  kernel_init_skip: Callable = initializers.orthogonal()
  quadratic: bool = True

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
    dim_hidden = list(self.dim_hidden)

    # ===== Inference =====
    if self.quadratic:
      low_rank = nn.Dense( 
        features=dim_hidden[0],
        kernel_init=initializers.glorot_uniform(),
        use_bias=False)
      convex_wrt_y = low_rank(y) / jnp.sqrt(dim_hidden[0])  # Py
      # Quadratic y^TP^TPy = y^TQy with Q = P^TP
      # Remark: y |-> y^TQy is convex since Q is positive definite.
      convex_wrt_y = jnp.sum(convex_wrt_y**2, axis=-1, keepdims=True)  
    else:
      convex_wrt_y = y
    
    for i, width in enumerate(dim_hidden):
      affine = nn.Dense(features=width,
                        kernel_init=self.kernel_init_skip,
                        use_bias=True)  # shortcut from the input to the intermediate layers.

      affine_y = affine(y)
      if i > 0 or self.quadratic:  # compose convex functions.
        convexity_preserving_op = self.convex_dense(features=width, use_bias=False)  # preserves convexity.
        convex_wrt_y = convexity_preserving_op(convex_wrt_y, train=train)
        convex_wrt_y = affine_y + convex_wrt_y
      else:
        convex_wrt_y = affine_y

      if i+1 < len(dim_hidden):
        convex_wrt_y = self.sigma_act_fn(convex_wrt_y)
      else:
        convex_wrt_y = self.out_act_fn(convex_wrt_y)
      
    return convex_wrt_y.squeeze(axis=-1)


def create_generator_state(rng, batch_size, features, learning_rate):
  """Creates initial `TrainState`."""

  if cfg.sigma_act_fn == 'relu':
    sigma_act_fn = nn.relu
  elif cfg.sigma_act_fn == 'softplus':
    sigma_act_fn = nn.softplus
  elif cfg.sigma_act_fn == 'cummax':
    sigma_act_fn = partial(cumulative_max, group_size=4)
  elif cfg.sigma_act_fn == 'logsumexp':
    sigma_act_fn = partial(group_logsumexp, group_size=4)
  else:
    raise ValueError(f'Unknown activation function {cfg.sigma_act_fn}')

  if cfg.out_act_fn == 'identity':
    out_act_fn = lambda x: jnp.sum(x, axis=-1)  # identity on vectors of length 1.
  elif cfg.out_act_fn == 'squared_norm':
    out_act_fn = non_negative_orthant_squared_norm
  elif cfg.out_act_fn == 'logsumexp':
    out_act_fn = stable_logsumexp
  else:
    raise ValueError(f'Unknown activation function {cfg.out_act_fn}')

  dim_hidden = cfg.dim_hidden
  if cfg.out_act_fn != 'logsumexp':
    dim_hidden += [1]
  else:
    dim_hidden += [dim_hidden[-1]]

  if cfg.convex == 'positive':
    convex_dense = partial(ConvexDense,
      kernel_init = initializers.glorot_uniform(),
      positive_parametrization = PositiveOrthant)
  elif cfg.convex == 'non-negative':
    convex_dense = partial(ConvexDense,
      kernel_init = initializers.glorot_uniform(),
      positive_parametrization = NonNegativeOrthant)
  elif cfg.convex == 'stochastic':
    convex_dense = partial(ConvexDense,
      kernel_init = initializers.glorot_uniform(),
      positive_parametrization = partial(StochasticMatrix, axis=0))
  elif cfg.convex == 'stochastic-bis':
    convex_dense = partial(ConvexDense,
      kernel_init = initializers.glorot_uniform(),
      positive_parametrization = partial(StochasticMatrix, axis=1))
  elif cfg.convex == 'orthostochastic':
    convex_dense = partial(ConvexDense,
      kernel_init = initializers.orthogonal(),
      positive_parametrization = OrthostochasticMatrix)
  else:
    raise ValueError(f'Unknown convex={cfg.convex}.')

  if cfg.kernel_init_skip == 'glorot':
    kernel_init_skip = initializers.glorot_uniform()
  elif cfg.kernel_init_skip == 'orthogonal':
    kernel_init_skip = initializers.orthogonal()
  elif cfg.kernel_init_skip == 'zeros':
    kernel_init_skip = initializers.zeros
  else:
    raise ValueError(f'Unknown kernel_init_skip={cfg.kernel_init_skip}.')

  model = ICNN(dim_hidden=dim_hidden,
               sigma_act_fn=sigma_act_fn,
               out_act_fn=out_act_fn,
               convex_dense=convex_dense,
               kernel_init_skip=kernel_init_skip,
               quadratic=cfg.quadratic)

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
  onp_P, onp_Q = get_datasets(cfg)
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
    plot_generator(gan_state.gen_state, onp_P, onp_Q, epoch=0, plot_lst='tpq', row=1, col=1)

  # train the models.
  for epoch in range(1, cfg.num_epochs+1):
    # alternate training of generator and discriminator.
    gan_state = train_generator_epoch(gan_state, ds_P, ds_Q, cfg.num_steps_gen, epoch)
    gan_state = train_discriminator_epoch(gan_state, ds_P, ds_Q, cfg.num_steps_disc, epoch)

    # evaluate errors.
    metrics = evaluate(gan_state, onp_P, onp_Q, monge_gt)
    log_metrics('', metrics, epoch=epoch, pbar=None)
    
    if cfg.plot_freq > 0 and epoch%cfg.plot_freq == 0:
      plot_generator(gan_state.gen_state, onp_P, onp_Q, epoch=epoch, plot_lst='tpq', row=1, col=1)

  plot_generator(gan_state.gen_state, onp_P, onp_Q, epoch=epoch, plot_lst='tpq', row=1, col=1, upload=True)


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
