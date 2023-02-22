# Copyright 2016 Google Inc.
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
# limitations under the License
# ==============================================================================
"""Learning 2 Learn problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mock
import os
import numpy as np
import tarfile
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import pdb
#from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset

import tensorflow_probability as tfp

tfd = tfp.distributions

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}

def simple():
  """Simple problem: f(x) = x^2."""

  def build():
    """Builds loss graph."""
    x = tf.get_variable(
        "x",
        shape=[],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    return tf.square(x, name="x_squared")

  return build


def simple_multi_optimizer(num_dims=2):
  """Multidimensional simple problem."""

  def get_coordinate(i):
    return tf.get_variable("x_{}".format(i),
                           shape=[],
                           dtype=tf.float32,
                           initializer=tf.ones_initializer())

  def build():
    coordinates = [get_coordinate(i) for i in xrange(num_dims)]
    x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
    return tf.reduce_sum(tf.square(x, name="x_squared"))

  return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('quadratic') as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims, 1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.matmul(w, tf.expand_dims(x, -1))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    return tf.reduce_mean(left_term)

  return build



def train_quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32, mnist_classes=None):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('train_quadratic'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=0.01))

      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims, 1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                        trainable=False)

    product = tf.matmul(w, tf.expand_dims(x, -1))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    return tf.reduce_mean(left_term)

  return build

def train_l2_reg(batch_size=128, num_dims=10, stddev=0.01, l=0.005, dtype=tf.float32, mnist_classes=None):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('train_l2_reg'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=0.01))

      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    
    left_term = 0.5*tf.reduce_sum((product - y) ** 2, 1)
    other_term = l * tf.norm(x, ord=2, axis=1)
    
    result =  tf.reduce_mean(left_term + other_term)
    return result
  return build




def test_quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('test_quadratic') as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=0.01))

      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    return tf.reduce_mean(0.5*tf.reduce_sum((product - y) ** 2, 1))

  return build


def lasso(batch_size=128, num_dims=10, stddev=0.01, l=0.005, dtype=tf.float32):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('lasso') as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims, 1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.matmul(w, tf.expand_dims(x, -1))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    other_term = l * tf.norm(x, ord=1, axis=1, keepdims=True)
    result = tf.reduce_mean(left_term + other_term)
    return result

  return build


def lasso_train(batch_size=128, num_dims=10, stddev=0.01, l=0.005, dtype=tf.float32, mnist_classes=None):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('lasso_train'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=0.01))

      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims, 1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                        trainable=False)

    product = tf.matmul(w, tf.expand_dims(x, -1))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    other_term = l * tf.norm(x, ord=1, axis=1, keepdims=True)
    result = tf.reduce_mean(left_term + other_term)
    return result

  return build

def lasso_train_mix(batch_size=128, num_dims=10, stddevs=[1,0.5,0.1], l=0.005, dtype=tf.float32, mnist_classes=None):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""

  def build():
    """Builds loss graph."""
    with tf.variable_scope('lasso_train'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=0.01))
      w1 = np.random.rand(batch_size // 3, num_dims, num_dims) * stddevs[0]
      w2 = np.random.rand(batch_size // 3, num_dims, num_dims) * stddevs[1]
      w3 = np.random.rand(batch_size -  batch_size // 3 * 2, num_dims, num_dims) * stddevs[2]
      w_array = np.concatenate([w1, w2, w3], 0)
      # Non-trainable variables.
      w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.constant_initializer(w_array),
                        trainable=False)
      y = tf.get_variable("y",
                        shape=[batch_size, num_dims, 1],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                        trainable=False)

    product = tf.matmul(w, tf.expand_dims(x, -1))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    other_term = l * tf.norm(x, ord=1, axis=1, keepdims=True)
    result = tf.reduce_mean(left_term + other_term)
    return result

  return build

def rastrigin_train_mix(batch_size=128, num_dims=10, alphas=[1,0.5,2], l=0.005, dtype=tf.float32, mnist_classes=None):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""
  
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rastrigin_vary_alpha'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims, 1],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-3, 3)
      )
      alpha = np.zeros((batch_size, num_dims, 1))
      alpha[:batch_size // 3] = alphas[0]
      alpha[batch_size // 3:(batch_size // 3) * 2] = alphas[1]
      alpha[(batch_size // 3) * 2:] = alphas[2]

      alpha = tf.get_variable(
        'alpha',
        shape=[batch_size, num_dims, 1],
        dtype=dtype,
        trainable=False,
        initializer=tf.constant_initializer(alpha)
      )
      x_cos = tf.math.cos(2*np.pi*x)

      res = tf.reduce_sum(x*x - alpha * x_cos, 1)
      res = tf.reduce_mean(res)
      return res
  return build


def lasso_fixed(data_A, data_b, stddev=0.01, l=0.005, dtype=tf.float32):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""
  a = data_A
  b = data_b

  print("=" * 100)
  print("LASSO: A_size={} b_size={}".format(a.shape, b.shape))
  print("=" * 100)
  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[a.shape[0], a.shape[2]],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=a.shape,
                        dtype=dtype,
                        initializer=tf.constant_initializer(a),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=b.shape,
                        dtype=dtype,
                        initializer=tf.constant_initializer(b),
                        trainable=False)

    # product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    
    product = tf.matmul(w, tf.expand_dims(x, -1))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    other_term = l * tf.norm(x, ord=1, axis=1, keepdims=True)
    result = tf.reduce_mean(left_term + other_term)
    return result

  return build

def rastrigin_vary_alpha(batch_size=128, num_dims=10, alpha=10, stddev=1, dtype=tf.float32, mnist_classes=None):
  # not rastrigin family
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rastrigin_vary_alpha'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims, 1],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-3, 3)
      )

      res = tf.reduce_sum(x*x - alpha*tf.math.cos(2*np.pi*x), 1)+ alpha*num_dims
      res = tf.reduce_mean(res)
      return res
  return build

def rastrigin_ori_vary_std(batch_size=128, num_dims=10, alpha=10, stddev=1, dtype=tf.float32, mnist_classes=None):
  # not rastrigin family
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rastrigin_ori_vary_std'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims, 1],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev)
      )

      res = tf.reduce_sum(x*x - alpha*tf.math.cos(2*np.pi*x), 1)+ alpha*num_dims
      res = tf.reduce_mean(res)
      return res
  return build

def rastrigin(batch_size=128, num_dims=10, alpha=10, stddev=1, dtype=tf.float32, mnist_classes=None):
  alpha = 5
  num_dims = 2
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rastrigin'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims, 1],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=1)
      )

      # Non-trainable variables.
      A = tf.get_variable("A",
                        dtype=dtype,
                        shape=[batch_size, num_dims, num_dims],
                        initializer=tf.random_normal_initializer(stddev=1),
                        trainable=False)
      B = tf.get_variable("B",
                        dtype=dtype,
                        shape=[batch_size, num_dims, 1],
                        initializer=tf.random_normal_initializer(stddev=1),
                        trainable=False)
      C = tf.get_variable("C",
                        dtype=dtype,
                        shape=[batch_size, num_dims, 1],
                        initializer=tf.random_normal_initializer(stddev=1),
                        trainable=False)

    product = tf.matmul(A, x)
    ras_norm=tf.norm(product-B,ord=2,axis=[-2,-1])
    
    cqTcos=tf.squeeze(tf.matmul(tf.transpose(C,perm=[0,2,1]),tf.cos(2*np.pi*x)))

    return tf.reduce_mean(0.5*(ras_norm**2)-alpha*cqTcos+alpha*num_dims)

  return build

def rastrigin(batch_size=128, num_dims=10, alpha=10, stddev=0.001, dtype=tf.float32, mnist_classes=None):
  alpha = 10
  num_dims = 2
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rastrigin'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=1)
      )

    # product = tf.matmul(A, x)
    ras_norm = tf.norm(x,ord=2,axis=-1) ** 2
    
    cqTcos = tf.reduce_sum(tf.cos(2*np.pi*x), -1)

    return tf.reduce_mean(ras_norm - alpha * cqTcos + alpha * num_dims)

  return build


def rastrigin_train_mix(batch_size=128, num_dims=10, alphas=[1,0.5,2], l=0.005, dtype=tf.float32, mnist_classes=None):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""
  alpha = 10
  num_dims = 2
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rastrigin'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-3, 3)
      )

    # product = tf.matmul(A, x)
    ras_norm = tf.norm(x,ord=2,axis=-1) ** 2
    
    cqTcos = tf.reduce_sum(tf.cos(2*np.pi*x), -1)

    return tf.reduce_mean(ras_norm - alpha * cqTcos + alpha * num_dims)
  return build


def rosenbrock(batch_size=128, num_dims=10, alpha=10, stddev=0.001, dtype=tf.float32, mnist_classes=None):
  alpha = 10
  num_dims = 2
  def build():
    """Builds loss graph."""
    with tf.variable_scope('rosenbrock'+str(mnist_classes)) as scope:
      # Trainable variable.
      x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=1)
      )

    # product = tf.matmul(A, x)
    # (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    item1 = (1 - x[:, 0]) ** 2
    item2 = 100 * (x[:, 1] - x[:, 0] ** 2) ** 2

    return tf.reduce_mean(item1 + item2, 0)

  return build


def ensemble(problems, weights=None):
  """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
  if weights and len(weights) != len(problems):
    raise ValueError("len(weights) != len(problems)")

  build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
               for p in problems]

  def build():
    loss = 0
    for i, build_fn in enumerate(build_fns):
      with tf.variable_scope("problem_{}".format(i)):
        loss_p = build_fn()
        if weights:
          loss_p *= weights[i]
        loss += loss_p
    return loss

  return build


