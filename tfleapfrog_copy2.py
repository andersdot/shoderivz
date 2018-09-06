
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
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
                        potential_and_grad, initial_grad, previous_time,
                        k=np.float64(0.0), name=None):

  def leapfrog_wrapper(step_size, time, x, m, grad, k, l):
    #input is call from while statement, must be same as counter_fn
    x, m, _, grad = leapfrog_step(step_size, x, m, potential_and_grad, grad, k=k)
    return step_size, time, x, m, grad, k, l + 1

  def time_fn(step_size, time, c, d, e, f, counter):  # pylint: disable=unused-argument
    counter = tfPrint(counter, [counter, step_size*counter, time])
    return (counter + 1.)*step_size  < time

  with ops.name_scope(name, 'leapfrog_integrator',
                      [step_size, time, initial_position, initial_momentum,
                       initial_grad]):
    _, _, new_x, new_m, new_grad, k, count = control_flow_ops.while_loop(
        time_fn, leapfrog_wrapper, [step_size, time, initial_position,
                                       initial_momentum, initial_grad, k,
                                       array_ops.constant(np.float64(0.0))], back_prop=False)
    # We're counting on the runtime to eliminate this redundant computation.
    new_potential, new_grad = potential_and_grad(new_x, k=k)

    #ip = tf.assign(initial_position, new_x)
    #im = tf.assign(initial_momentum, new_m)
    #ig = tf.assign(initial_grad, new_grad)
    #it = tf.assign(previous_time, count*step_size)
    #with tf.control_dependencies([ip, im, ig, it]):
    dt_tiny = time - count*step_size
    x, m, _, g = leapfrog_step(dt_tiny, new_x, new_m, potential_and_grad, new_grad, k=k)
  #print(initial_grad)
  #print(initial_position)

  return new_x, new_m, new_grad, count*step_size, x, m


def leapfrog_step(step_size, position, momentum, potential_and_grad, grad,
                  k=np.float64(0.0), name=None):

  with ops.name_scope(name, 'leapfrog_step', [step_size, position, momentum,
                                              grad]):
    momentum -= np.float64(0.5)* step_size * grad
    position += step_size * momentum
    potential, grad = potential_and_grad(position, k=k)
    momentum -= np.float64(0.5) * step_size * grad

  return position, momentum, potential, grad
