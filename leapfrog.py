# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Leapfrog Integrator for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops

def _assert_increasing(t):
  assert_increasing = control_flow_ops.Assert(
      math_ops.reduce_all(t[1:] > t[:-1]), ['`t` must be monotonic increasing'])
  return ops.control_dependencies([assert_increasing])


def _check_input_types(t, y0):
  if not (y0.dtype.is_floating or y0.dtype.is_complex):
    raise TypeError('`y0` must have a floating point or complex floating '
                    'point dtype')
  if not t.dtype.is_floating:
    raise TypeError('`t` must have a floating point dtype')


def _leap_frog_step(step_size, k, leapfrog_state, name=None):
  """Take an arbitrary Runge-Kutta step and estimate error.
  Args:
    step_size: step size of each leap frog call
    k: spring constant for potential
    leapfrog_state: collection.namedtuple for current state of leapfrog
    name: optional name for the operation.
  Returns:
    Tuple `(position1, momentum1, grad1, total_time1)` giving the estimated position, momentum,
    gradient of the potential, and time after the leap.
  """
  #position = leapfrog_state.position1
  #momentum = leapfrog_state.momentum1
  #grad     = leapfrog_state.grad1
  #total_time = leapfrog_state.total_time1

  position, momentum, grad, total_time = leapfrog_state
  with ops.name_scope(name, 'leap_frog_step',
  [step_size, position, momentum, grad, k, total_time]) as scope:

    step_size = ops.convert_to_tensor(step_size, name='step_size')
    momentum  = ops.convert_to_tensor(leapfrog_state.momentum1, name = 'momentum')
    position  = ops.convert_to_tensor(leapfrog_state.position1, name = 'position')
    grad      = ops.convert_to_tensor(leapfrog_state.grad1,     name = 'grad')
    k         = ops.convert_to_tensor(k,          name = 'k')
    total_time = ops.convert_to_tensor(total_time, name='total_time')

    momentumi = momentum - np.float64(0.5)* step_size * grad
    positioni = position + step_size * momentumi
    potentiali, gradi = potential_and_grad(positioni, k)
    momentumi = momentumi - np.float64(0.5) * step_size * gradi
    total_timei = total_time + step_size

    momentum1  = array_ops.identity(momentumi, name='{0}/momentum1'.format(scope))
    position1  = array_ops.identity(positioni, name='{0}/position1'.format(scope))
    potential1 = array_ops.identity(potentiali, name='{0}/potential1'.format(scope))
    grad1      = array_ops.identity(gradi,      name='{0}/grad1'.format(scope))
    total_time1 = array_ops.identity(total_timei, name='{0}/total_time1'.format(scope))
  return _Leap_Frog_State(position1, momentum1, grad1, total_time1)

class _Leap_Frog_State(
    collections.namedtuple('_Leap_Frog_State',
                           'position1, momentum1, grad1, total_time1')):
  """Saved state of the Leap Frog solver.
  Attributes:
    position1: the position at the end of the last time step.
    momentum1: the momentum at the end of the last time step.
    grad1: the gradient of the potential at the end of the last time step.
    total_time1: total time the system has been integrated at the end of the last time step.
  """

def potential_and_grad(position, k):
    #function that returns the potential and it's gradient at a given position
    return 0.5 * k * tf.square(position), k * position


def _leapfrog(x0,
              v0,
              k,
              t_obs,
              step_size, name=None):
  """Model the positions and velocities at t_obs in a potential described in function potential_and_grad."""

  with ops.name_scope(name, 'integrate',
                    [x0, v0, k, t_obs, step_size]) as scope:

    def leapfrog_wrapper(step_size, time, k, leapfrog_state, l):
      #input is call from while statement, must be same as counter_fn
      leapfrog_state = _leap_frog_step(step_size, k, leapfrog_state)
      return step_size, time, k, leapfrog_state, l + 1

    def time_fn(step_size, time, k, leapfrog_state, l):  # pylint: disable=unused-argument
      return leapfrog_state.total_time1 + step_size  <  time

    def leap_to_tobs(x, v, leapfrog_state, num_times, step_size, k, i):
      """Integrate to next t_obs point."""

      with ops.name_scope('leapfrog_to_obs'):
        #loop through stepsize to t_obs
        step_size, time, k, leapfrog_state, count = control_flow_ops.while_loop(
            time_fn, leapfrog_wrapper, (step_size, tf.gather(t_obs, i), k, leapfrog_state, 0.0),
            name='leapfrog_loop')
        dt_tiny = tf.gather(t_obs, i) - leapfrog_state.total_time1
        xt, vt, _, _ = _leap_frog_step(dt_tiny, k, leapfrog_state)
        x = x.write(i, xt)
        v = v.write(i, vt)
        #solution = solution.write(i, y)
        return x, v, leapfrog_state, num_times, step_size, k, i + 1

    def tobs_fn(x, v, leapfrog_state, num_times, step_size, k, i):
        return i < num_times

    num_times = array_ops.size(t_obs)
    x = tensor_array_ops.TensorArray(
        x0.dtype, size=num_times, name='x')
    v = tensor_array_ops.TensorArray(
        v0.dtype, size=num_times, name='v')

    potential0, grad0 = potential_and_grad(x0, k)
    total_time0 = np.float64(0.0)
    leapfrog_state = _Leap_Frog_State(
        x0, v0, grad0, total_time0)

    #loop through t_obs
    #print(x, v, leapfrog_state, num_times)
    x, v, leapfrog_state, _, _, _, _ = control_flow_ops.while_loop(
    tobs_fn, leap_to_tobs, (x, v, leapfrog_state, num_times, step_size, k, 0),
    name='tobs_loop')

    xreturn = x.stack(name='{0}/xvalues'.format(scope))
    vreturn = v.stack(name='{0}/vvalues'.format(scope))
    return (xreturn, vreturn)


def leapfrog(x0,
             v0,
             k,
             t_obs,
             step_size, name=None):
  """Integrate a system in a potential and observe at t_obs times.
  Implements leapfrog integration
  Args:
    x0: The initial position of the system
    v0: The initial velocity of the system
    grad0: The initial gradient of the system
    k: spring constant of the system
    t_obs: array of times to observe the system
    step_size: the step size for each leap
  Returns:
    x: the observed positions
    v: the observed velocities
  """

  with ops.name_scope(name, 'leapfrog', [x0, v0, k, t_obs, step_size]) as scope:

    #x0 = ops.convert_to_tensor(x0, dtype=x0.dtype, name='x0')
    #v0 = ops.convert_to_tensor(v0, dtype=v0.dtype, name='v0')
    #k = ops.convert_to_tensor(k, dtype=k.dtype, name = 'k')
    #t_obs = ops.convert_to_tensor(t_obs, dtype=t_obs.dtype, name='t_obs')

    return _leapfrog(
            x0,
            v0,
            k,
            t_obs,
            step_size, name=scope)
