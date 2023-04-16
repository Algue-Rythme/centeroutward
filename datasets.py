import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from sklearn.datasets import make_moons, make_circles


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


def center_ds(points):
  """Center the dataset.

  Remark: Wasserstein-2 distance is invariant to translation:

    if T#P = Q , if t is a translation, and if T'#(P+t) = Q
    then for all x~(P + t), T'(x) = T(x-t)
         for all x~P, T(x) = T'(x+t)
  
  Args:
    points: dataset of shape (num_samples, num_features).
    
  Returns:
    centered dataset of shape (num_samples, num_features)."""
  mean = jnp.mean(points, axis=0, keepdims=True)
  centered = points - mean
  min_points = jnp.min(centered, axis=None, keepdims=True)
  max_points = jnp.max(centered, axis=None, keepdims=True)
  scale = 0.5 * (max_points - min_points)
  return centered / scale


def generate_spherical_uniform(n, dim, seed=0):
    onp.random.seed(seed)
    unit_norm = onp.random.normal(loc=0., scale=1., size=(n, dim))
    euclidean_norm = onp.sum(onp.square(unit_norm), axis=1, keepdims=True) ** 0.5
    unit_norm = unit_norm / euclidean_norm
    radius = onp.random.uniform(low=0., high=1., size=(n, 1))
    X = radius * unit_norm  # broadcasting.
    return X.astype('float32')


def make_center_outward_2(n_samples, noise=0.1, random_state=None):
    mean_u = onp.array([0.,1.]) * (noise**0.5)
    mean_h = onp.array([1.,0.]) * (noise**0.5)
    cov_1 = onp.array([[5., -4.], [-4., 5.]]) * noise
    gaussian_1 = onp.random.multivariate_normal(mean=-3. * mean_h, cov=cov_1, size=int(n_samples*3/8))
    cov_2 = onp.array([[5., 4.], [4., 5.]]) * noise
    gaussian_2 = onp.random.multivariate_normal(mean= 3. * mean_h, cov=cov_2, size=int(n_samples*3/8))
    cov_3 = onp.array([[4., 0.], [0., 1.]]) * noise
    gaussian_3 = onp.random.multivariate_normal(mean=-5./2. * mean_u, cov=cov_3, size=n_samples - 2*int(n_samples*3/8))
    mixture = onp.concatenate([gaussian_1, gaussian_2, gaussian_3], axis=0)
    onp.random.shuffle(mixture)
    return mixture.astype('float32')


def get_datasets(cfg):
  """Get the datasets."""
  if cfg.dataset in ['two-moons', 'two-circles']:  # classical OT.

    if cfg.dataset == 'two-moons':
      make_ds = make_moons
    elif cfg.dataset == 'two-circles':
      make_ds = make_circles
    
    n_samples = cfg.dataset_size // 2
    X, y = make_ds(n_samples=(n_samples, n_samples), noise=cfg.noise)
    X = X.astype(jnp.float32)
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)  # centered moons.
    Q = X[y == 0, :]
    P = X[y == 1, :]

  else:  # Center-Outward.

    P = generate_spherical_uniform(n=cfg.dataset_size, dim=2, seed=0)
    if cfg.dataset == 'center-outward-2':
      Q = make_center_outward_2(n_samples=cfg.dataset_size, noise=cfg.noise)

  if cfg.center_PQ:
    P = center_ds(P)
    Q = center_ds(Q)
  
  return P, Q
