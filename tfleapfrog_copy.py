
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow import Print as tfPrint
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import tf_logging as logging

__all__ = [
    'leapfrog_integrator',
    'leapfrog_step'
]

def leapfrog_integrator(step_size, time, initial_position, initial_momentum,
                        potential_and_grad, initial_grad, name=None):
  """
  Example: Simple quadratic potential.
  ```python
  def potential_and_grad(position):
    return tf.reduce_sum(0.5 * tf.square(position)), position
  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)
  potential, grad = potential_and_grad(position)
  new_position, new_momentum, new_potential, new_grad = hmc.leapfrog_integrator(
    0.1, 3, position, momentum, potential_and_grad, grad)
  sess = tf.Session()
  position_val = np.random.randn(10)
  momentum_val = np.random.randn(10)
  potential_val, grad_val = sess.run([potential, grad],
                                     {position: position_val})
  positions = np.zeros([100, 10])
  for i in xrange(100):
    position_val, momentum_val, potential_val, grad_val = sess.run(
      [new_position, new_momentum, new_potential, new_grad],
      {position: position_val, momentum: momentum_val})
    positions[i] = position_val
  # Should trace out sinusoidal dynamics.
  plt.plot(positions[:, 0])
  ```
  """
  def leapfrog_wrapper(step_size, time, x, m, grad, l):
    #input is call from while statement, must be same as counter_fn
    x, m, _, grad = leapfrog_step(step_size, x, m, potential_and_grad, grad)
    return step_size, time, x, m, grad, l + 1

  def time_fn(step_size, time, c, d, e, counter):  # pylint: disable=unused-argument
    counter = tfPrint(counter, [counter, step_size*counter, time])
    return (counter + 1.)*step_size  < time

  with ops.name_scope(name, 'leapfrog_integrator',
                      [step_size, time, initial_position, initial_momentum,
                       initial_grad]):
    _, _, new_x, new_m, new_grad, count = control_flow_ops.while_loop(
        time_fn, leapfrog_wrapper, [step_size, time, initial_position,
                                       initial_momentum, initial_grad,
                                       array_ops.constant(np.float64(0.0))], back_prop=False)
    # We're counting on the runtime to eliminate this redundant computation.
    new_potential, new_grad = potential_and_grad(new_x)

    dt_tiny = time - count*step_size
    x, m, _, g = leapfrog_step(dt_tiny, new_x, new_m, potential_and_grad, new_grad)
  return new_x, new_m, new_potential, new_grad, x, m, count*step_size


def leapfrog_step(step_size, position, momentum, potential_and_grad, grad,
                  name=None):
  """  Example: Simple quadratic potential.
  ```python
  def potential_and_grad(position):
    # Simple quadratic potential
    return tf.reduce_sum(0.5 * tf.square(position)), position
  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)
  potential, grad = potential_and_grad(position)
  new_position, new_momentum, new_potential, new_grad = hmc.leapfrog_step(
    0.1, position, momentum, potential_and_grad, grad)
  sess = tf.Session()
  position_val = np.random.randn(10)
  momentum_val = np.random.randn(10)
  potential_val, grad_val = sess.run([potential, grad],
                                     {position: position_val})
  positions = np.zeros([100, 10])
  for i in xrange(100):
    position_val, momentum_val, potential_val, grad_val = sess.run(
      [new_position, new_momentum, new_potential, new_grad],
      {position: position_val, momentum: momentum_val})
    positions[i] = position_val
  # Should trace out sinusoidal dynamics.
  plt.plot(positions[:, 0])
  ```
  """
  with ops.name_scope(name, 'leapfrog_step', [step_size, position, momentum,
                                              grad]):
    momentum -= np.float64(0.5)* step_size * grad
    position += step_size * momentum
    potential, grad = potential_and_grad(position)
    momentum -= np.float64(0.5) * step_size * grad

  return position, momentum, potential, grad
