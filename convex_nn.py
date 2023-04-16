from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import initializers
from jax import random

from jaxopt.tree_util import tree_add, tree_sub

import flax.linen as nn

from cnqr._src.parametrizations import CachedParametrization, BjorckParametrization


Shape = Tuple[int]
Dtype = Any
Array = Any
KeyArray = random.KeyArray
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex

EPSILON_NORMALIZATION = 1e-10


def straight_through(fun):
  """Straight-through estimator for identity parametrization."""

  def _straight_through(tree):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    zero = tree_sub(tree, jax.lax.stop_gradient(tree))
    return tree_add(zero, jax.lax.stop_gradient(fun(tree)))
  
  return _straight_through


def l2_normalize(vec, epsilon=EPSILON_NORMALIZATION):
  norm = jnp.linalg.norm(vec, ord=2) + epsilon
  return vec / norm


class PositiveOrthant(CachedParametrization):
  """Tensor with positive weights (> 0).

  This is a canonical reparametrization of the positive orthant.
  Softplus is a diffeomorphism from R to the positive orthant.
  Identity derivative is a trick that mimics Riemannian gradient on
  the Bregman divergence associated to the positive orthant (open manifold).  
  
  Attributes:
    beta: positive scalar to scale the weights.
    train: whether to use perform orthogonalization or re-use the cached kernel.
    auto_diff: support 'auto' (defaults to 'identity'), 'unroll' and 'identity' modes.
      See documentation of CachedParametrization for more details.
  """
  beta = 1.
  train: Optional[bool] = None
  groupname: str = 'convex'
  auto_diff: str = 'auto'

  def inv_act_fun(self, tensor):
    return (1. / self.beta) * jnp.log(jnp.exp(tensor * self.beta) - 1.)

  def act_fun(self, tensor):
    return (1. / self.beta) * nn.softplus(tensor * self.beta)

  @nn.compact
  def __call__(self, tensor, train=None):
    """Forward pass.

    Args:
      tensor: array of arbitrary shape.
      train: whether to use perform orthogonalization or re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    # init params
    tensor_shape = tensor.shape

    # init mutable variables
    positive_tensor = self.variable(self.groupname, 'positive_tensor', jnp.zeros, tensor_shape)

    train = nn.merge_param('train', self.train, train)
    if train:

      auto_diff = self.auto_diff
      if self.auto_diff == 'auto':
        auto_diff = 'identity'

      act_fun = self.act_fun
      if auto_diff == 'identity':
        act_fun = straight_through(act_fun)

      positive_ker = act_fun(tensor)
      positive_tensor.value = positive_ker
      
    else:
      positive_ker = positive_tensor.value

    return positive_ker


class NonNegativeOrthant(CachedParametrization):
  """Tensor with non negative weights (>= 0).

  This is a manifold with boundary. Hence, no biejctive reparametrization from R^n to the manifold is possible.
  Indeed, no diffeomorphism exists from R^n to a manifold with boundary.
  
  Attributes:
    train: whether to use perform orthogonalization or re-use the cached kernel.
    auto_diff: support 'auto' (default), 'unroll' and 'identity' modes.
      See documentation of CachedParametrization for more details.
  """
  train: Optional[bool] = None
  groupname: str = 'convex'
  auto_diff: str = 'auto'
  pos_fn: Callable = nn.relu

  def inv_act_fun(self, tensor):
    raise ValueError('Non negative orthant is not a diffeomorphism.')

  def act_fun(self, tensor):
    return nn.pos_fn(tensor)

  @nn.compact
  def __call__(self, tensor, train=None):
    """Forward pass.

    Args:
      tensor: array of arbitrary shape.
      train: whether to use perform orthogonalization or re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    # init params
    tensor_shape = tensor.shape

    # init mutable variables
    non_negative_tensor = self.variable(self.groupname, 'non_negative_tensor', jnp.zeros, tensor_shape)

    train = nn.merge_param('train', self.train, train)
    if train:
      auto_diff = self.auto_diff
      if self.auto_diff == 'auto':
        auto_diff = 'unroll'
      assert auto_diff == 'unroll', 'Only unrolling is available for non negative orthant.'

      non_negative_ker = self.act_fun(tensor)
      non_negative_tensor.value = non_negative_ker
      
    else:
      non_negative_ker = non_negative_tensor.value

    return non_negative_ker


class StochasticMatrix(CachedParametrization):
  """Matrix with positive weight summing to 1 along axis.

  This is a manifold with boundary. Hence, no biejctive reparametrization from R^n to the manifold is possible.
  
  Attributes:
    train: whether to use perform orthogonalization or re-use the cached kernel.
    groupname: name of the group of variables.
    auto_diff: support 'auto' (default), 'unroll' and 'identity' modes.
      See documentation of CachedParametrization for more details.
    axis: axis along which the weights sum to 1.
    temperature: temperature for the softmax.
  """
  train: Optional[bool] = None
  groupname: str = 'convex'
  auto_diff: str = 'auto'
  axis: int = 0
  temperature: float = 1.

  @nn.compact
  def __call__(self, matrix, train=None):
    """Forward pass.

    Args:
      matrix: matrix of shape (f1, f2).
      train: whether to use perform orthogonalization or re-use the cached kernel.

    Returns:
      outputs: matrix of shape (f1, f2)
    """
    # init params
    matrix_shape = matrix.shape

    # init mutable variables
    stochastic_matrix = self.variable(self.groupname, 'stochastic_matrix', jnp.zeros, matrix_shape)

    train = nn.merge_param('train', self.train, train)
    if train:
      auto_diff = self.auto_diff
      if self.auto_diff == 'auto':
        auto_diff = 'unroll'
      assert auto_diff == 'unroll', 'Only unrolling is available for stochastic matrix.'

      inv_temperature = matrix_shape[self.axis] / self.temperature
      stochastic_mat = jax.nn.softmax(matrix * inv_temperature, axis=self.axis)
      stochastic_matrix.value = stochastic_mat
      
    else:
      stochastic_mat = stochastic_matrix.value

    return stochastic_mat


class OrthostochasticMatrix(CachedParametrization):
  """Orthostochastic matrix.

  Orthostochastic matrices are the element-wise square of an orthogonal matrix.
  They are a subset of the Birkhoff polytope, i.e the set of doubly stochastic matrices.

  Birkhoff polytope is of dimension (n-1)^2, whereas the orthostochastic matrices are obtained
  with a surjective and differentiable mapping from a space of dimension n(n-1)/2.

  Hence the orthostochastic matrices are a proper subset of the Birkhoff polytope.
  
  Attributes:
    train: whether to use perform orthogonalization or re-use the cached kernel.
    groupname: name of the group of variables.
    auto_diff: support 'auto' (default), 'unroll' and 'identity' modes.
      See documentation of CachedParametrization for more details.
    axis: axis along which the weights sum to 1.
    temperature: temperature for the softmax.
  """
  train: Optional[bool] = None
  groupname: str = 'convex'
  auto_diff: str = 'auto'
  stiefel_parametrization: Callable = BjorckParametrization

  @nn.compact
  def __call__(self, matrix, train=None):
    """Forward pass.

    Args:
      matrix: array of arbitrary shape.
      train: whether to use perform orthogonalization or re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    # init params
    matrix_shape = matrix.shape

    # init mutable variables
    stochastic_matrix = self.variable(self.groupname, 'stochastic_matrix', jnp.zeros, matrix_shape)

    train = nn.merge_param('train', self.train, train)
    if train:
      auto_diff = self.auto_diff
      if self.auto_diff == 'auto':
        auto_diff = 'unroll'
      assert auto_diff == 'unroll', 'Only unrolling is available for non negative orthant.'

      stiefel_parametrization = self.stiefel_parametrization(groupname=self.groupname,
                                                             auto_diff=auto_diff)
      ortho = stiefel_parametrization(matrix, train=train)
      stochastic_mat = ortho ** 2
      stochastic_matrix.value = stochastic_mat
      
    else:
      stochastic_mat = stochastic_matrix.value

    return stochastic_mat


def inv_act_fun_initializer(input_dims, inv_act_fun):
  """Initializer for the inverse softplus function.

  Return a function with signature `init(key, shape, dtype=jnp.float_) -> Array`

  Args:
    input_dims: (integer) size of input dimension.
    inv_act_fun: inverse of the activation function.
  """
  constant = inv_act_fun(1.0 / input_dims)
  constant_init = initializers.constant(constant)
  return constant_init


@dataclass
class ConvexDense(nn.Module):
  """Dense layer with normalized matrix weights.
  
  Attributes:
    normalize_fun: function to normalize the matrix weights.
    features: number of output features.
    use_bias: whether to add a bias to the output (default: True).
    kernel_init: initializer for the kernel.
    bias_init: initializer for the bias.
  """
  features: int
  use_bias: bool = True
  kernel_init: Union[Callable, str] = initializers.glorot_uniform()
  bias_init: Callable = initializers.zeros
  positive_parametrization: CachedParametrization = StochasticMatrix

  @nn.compact
  def __call__(self, inputs: Array, train: bool = None) -> Array:
    """Forward pass.

    Args:
      inputs: array of shape (B, f_1) with B the batch_size.
      train: whether to use perform orthogonalization of re-use the cached kernel.

    Returns:
      outputs: array of shape (B, features)
    """
    positive_param = self.positive_parametrization()

    input_dims = inputs.shape[-1]
    kernel_init = self.kernel_init
    if isinstance(kernel_init, str) and kernel_init == 'inv_act_fun':
      kernel_init = inv_act_fun_initializer(input_dims, positive_param.inv_act_fun)

    # init params
    kernel_shape = (input_dims, self.features)
    kernel = self.param('kernel', kernel_init, kernel_shape)

    positive_kernel = positive_param(kernel, train=train)

    y = jnp.matmul(inputs, positive_kernel)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.reshape(bias, (1, self.features))
      y = y + bias

    return y


def cumulative_max(x, group_size):
  """Cumulative max of a vector.

  This function is convex and differentiable, because the max preserves
  convexity and is differentiable.
  
  Args:
    x: vector of shape (..., n).
    group_size: number of entries to consider in the max.
    
  Returns:
    vector of shape (..., n) where the i-th entry is the max of the first i entries."""
  n = x.shape[-1]
  assert n % group_size == 0, 'n must be divisible by group_size'
  x = jnp.reshape(x, (-1, group_size))
  axis = x.ndim - 1
  y = jax.lax.cummax(x, axis=axis)
  return jnp.reshape(y, (-1, n))


def stable_logsumexp(x):
  """Numerically stable logsumexp."""
  d = x.shape[-1]
  return jax.scipy.special.logsumexp(x * d, axis=-1, keepdims=True)


def group_logsumexp(x, group_size):
  """Logsumexp of a vector.

  This function is convex and differentiable, because the logsumexp is
  differentiable and convex.
  
  Args:
    x: vector of shape (..., n).
    group_size: number of entries to consider in the logsumexp.
    
  Returns:
    vector of shape (..., n) where the i-th entry is the logsumexp of the first i entries."""
  n = x.shape[-1]
  assert n % group_size == 0, 'n must be divisible by group_size'
  num_groups = n // group_size
  x = jnp.reshape(x, (-1, num_groups, group_size))
  y = stable_logsumexp(x)
  return jnp.reshape(y, (-1, num_groups))


def non_negative_orthant_squared_norm(x):
  """Squared norm of the positive orthant."""
  pos_x = nn.relu(x)
  return 0.5 * jnp.sum(pos_x**2, axis=-1)
