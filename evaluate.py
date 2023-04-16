import jax
import jax.numpy as jnp

import numpy as onp
import ot

import tensorflow as tf

from train import tf_to_jax


def get_test_ds(onp_array, batch_size):
  """Returns a tf.data.Dataset from a numpy array solely for iteration purposes."""
  ds = tf.data.Dataset.from_tensor_slices(onp_array)
  ds = ds.batch(batch_size)
  return ds


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


def predict_ds(train_state, ds, predict_model):
  """Predicts on a dataset."""
  jnp_preds = [predict_model(train_state, tf_to_jax(batch)) for batch in ds]
  jnp_preds = jnp.concatenate(jnp_preds)
  return jnp_preds


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
  ds_GP = get_test_ds(jnp_GP, batch_size=512)
  ds_Q = get_test_ds(onp_Q, batch_size=512)
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
  ds_P = get_test_ds(onp_P, batch_size=512)
  jnp_GP = predict_ds(gan_state.gen_state, ds_P, predict_convex)

  # Compute gap between G#P and T#P. Target: zero loss.
  TP, cost_gt = monge_gt
  w2_abs_gap = jnp.mean(jnp.sum((jnp_GP - TP)**2, axis=-1))  # mean L2 norm (squared)
  w2_rel_gap = w2_abs_gap / jnp.maximum(cost_gt, 1e-6) * 100.  # positive.

  # Check that W1 works well.
  w1_metrics = compute_w1_global(gan_state.disc_state, jnp_GP, onp_Q)
  return {'w2_abs_gap': w2_abs_gap, 'w2_rel_gap':w2_rel_gap, **w1_metrics}
