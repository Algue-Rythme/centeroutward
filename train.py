from typing import Any, Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

import numpy as onp
import tqdm  # progress bar
import wandb

import jax.tree_util as tree_util
from flax.training import train_state
from flax import struct
from jaxopt.tree_util import tree_l2_norm


@jax.jit
def update_lipschitz(state, grads, lip_vars):
  """Updates model parameters."""
  return state.apply_gradients(grads=grads, lip_state=lip_vars)


@jax.jit
def update_convex(state, grads, convex_vars):
  """Updates model parameters."""
  return state.apply_gradients(grads=grads, convex_state=convex_vars)


def grad_icnn_vmap(model):
  """Returns a function that computes the gradient of the model wrt its inputs.

  Based on vmap of individual gradients.
  
  Args:
    model: a flax model.
    
  Returns:
    A function that computes the gradient of the model wrt its inputs.
  """

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


def grad_icnn(model):
  """Returns a function that computes the gradient of the model wrt its inputs.

  Based on global gradient.
  
  Args:
    model: a flax model.
    
  Returns:
    A function that computes the gradient of the model wrt its inputs.
  """

  def apply_inputs(params, single_input, train, mutable):
    out = model.apply(params, single_input, train=train, mutable=mutable)
    if mutable is False:
      preds, aux = out, None
    else:
      preds, aux = out
    preds = jnp.sum(preds)  # combine all scores.
    return preds, aux
  
  def grad_inputs(params, single_input, train, mutable=False):
    grad_fn = jax.grad(apply_inputs, argnums=1, has_aux=True)
    potentials, convex_variables = grad_fn(params, single_input, train, mutable)
    if convex_variables is None:
      return potentials
    return potentials, convex_variables

  return grad_inputs


class LipschitzTrainState(train_state.TrainState):
  """Train state with Lipschitz constraint."""
  lip_state: Any


def get_lip_state(params):
  """Extracts the lipschitz variables from the discriminator parameters."""
  return params.get('lip', None)


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