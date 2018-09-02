"""leap frog integrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
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

def leapfrog_integrator(step_size, dt, initial_position, initial_momentum,
                        potential_and_grad, initial_grad, name=None):
  def leapfrog_wrapper(step_size, dt, x, m, grad, l):
    x, m, _, grad = leapfrog_step(step_size, x, m, grad, potential_and_grad)
    return step_size, x, m, grad, l + 1

  def time_fn(step_size, dt, x, m, grad, counter):
      counter = tf.Print(counter, [counter, step_size*counter, dt])
      return step_size*counter < dt

  #take normal step sizes until you get close to the observed time
  with ops.name_scope(name, 'leapfrog_integrator',
                      [step_size, dt, initial_position, initial_momentum, initial_grad]):
    _, new_x, new_m, new_grad, count = control_flow_ops.while_loop(
        time_fn, leapfrog_wrapper, [step_size, dt, initial_position,
                                       initial_momentum, initial_grad,
                                       array_ops.constant(0.)], back_prop=False)
    #take a tiny step to get to t_obs
    dt_tiny = dt - step_size*count
    x, m, pot, grad = leapfrog_step(dt_tiny, new_x, new_m, new_grad, potential_and_grad)
    # We're counting on the runtime to eliminate this redundant computation.
    new_potential, new_grad = potential_and_grad(new_x)
    #return new_x, new_m, new_potential, new_grad
  return x, m, pot, grad, new_x, new_m, new_potential, new_grad

def leapfrog_step(step_size, position, momentum, grad, potential_and_grad,
                  name=None):
  with ops.name_scope(name, 'leapfrog_step', [step_size, position, momentum,
                                              grad]):
    momentum -= 0.5 * step_size * grad
    position += step_size * momentum
    potential, grad = potential_and_grad(position)
    momentum -= 0.5 * step_size * grad

  return position, momentum, potential, grad
