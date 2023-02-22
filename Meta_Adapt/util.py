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
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems
import random
import time

def run_epoch(sess, cost_op, ops, reset, num_unrolls,
              scale=None, rd_scale=False, rd_scale_bound=3.0, assign_func=None, var_x=None,
              step=None, unroll_len=None,
              task_i=-1, data=None, label_pl=None, input_pl=None,if_eval=True,net_params=None,opt_var_arr=None,assign_func_all_vars=None):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  
  '''
  *****************************************************
  '''
  
  if(net_params != None):
    assign_func_all_vars(net_params)
  
  '''
  *****************************************************
  '''

  cost = None
  if task_i == -1:
      if rd_scale:
        assert scale is not None
        r_scale = []
        for k in scale:
          r_scale.append(np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                            size=k.shape)))
        assert var_x is not None
        k_value_list = []
        for k_id in range(len(var_x)):
          k_value = sess.run(var_x[k_id])
          k_value = k_value / r_scale[k_id]
          k_value_list.append(k_value)
        assert assign_func is not None
        assign_func(k_value_list)
        feed_rs = {p: v for p, v in zip(scale, r_scale)}
      else:
        feed_rs = {}
      feed_dict = feed_rs
      
      if opt_var_arr != None:
        opt_var, assign_func_opt, assign_func_all_vars = opt_var_arr


      for i in xrange(num_unrolls):
        if step is not None:
            feed_dict[step] = i*unroll_len+1
        '''
        *****************************************************
        '''
        if opt_var_arr != None:
          #save the optimizer params
          opt_value_list = []
          for opt_id in range(len(opt_var)):
            opt_value = sess.run(opt_var[opt_id])
            #print(opt_value.shape)
            opt_value_list.append(opt_value)
        '''
        *****************************************************
        '''
        xxxx = sess.run([cost_op] + ops, feed_dict=feed_dict)
        cost = xxxx[0]
        
  else:
      assert data is not None
      assert input_pl is not None
      assert label_pl is not None
      feed_dict = {}
      for ri in xrange(num_unrolls):
          for pl, dat in zip(label_pl, data["labels"][ri]):
              feed_dict[pl] = dat
          for pl, dat in zip(input_pl, data["inputs"][ri]):
              feed_dict[pl] = dat
          if step is not None:
              feed_dict[step] = ri * unroll_len + 1
          cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
  return timer() - start, cost




def run_eval_epoch(sess, cost_op, ops, num_unrolls, step=None, unroll_len=None):
  """Runs one optimization epoch."""

  objective_values_cur = []
  training_acc_cur = []
  
  start = timer()
  # sess.run(reset)
  total_cost = []
  feed_dict = {}

  for i in xrange(num_unrolls):
    if step is not None:
        feed_dict[step] = i * unroll_len + 1
    cost, acc = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
    objective_values_cur.append(cost)
    training_acc_cur.append(acc)
    print('iteration = ', i)
    print('acc = ',sum(training_acc_cur)/len(training_acc_cur))
    print('loss = ',sum(objective_values_cur)/len(objective_values_cur))
    print()
    total_cost.append(cost)
  return timer() - start, total_cost, training_acc_cur, objective_values_cur



def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_default_net_config(path):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": path
  }


def get_config(problem_name, path=None, mode=None, num_hidden_layer=None, net_name=None, mnist_classes=None, stddev=None, l1_weight=None):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
        "cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": path
        },
        "adam": {
            "net": "Adam",
            "net_options": {"learning_rate": 0.01}
        }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    problem = problems.quadratic(batch_size=128, num_dims=10)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "rastrigin_vary_alpha":
    print('###########rastrigin_vary_alpha##################')
    print('#############################num_dims = ', 10)
    print('##################################alpha = ', l1_weight)
    problem = problems.rastrigin_vary_alpha(batch_size=128, num_dims=10, mnist_classes=mnist_classes, alpha=l1_weight)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "test_quadratic":
    print('##################################Standard Dev = ', stddev)
    problem = problems.test_quadratic(batch_size=128, num_dims=10, stddev=stddev)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "lasso_train":
    print('##################################Standard Dev = ', stddev)
    print('##################################L1 weight = ', l1_weight)
    problem = problems.lasso_train(batch_size=128, num_dims=10, stddev=stddev, l=l1_weight, mnist_classes=mnist_classes)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "lasso_train_mix":
    print('##################################Standard Dev = ', stddev)
    print('##################################L1 weight = ', l1_weight)
    problem = problems.lasso_train_mix(batch_size=128, num_dims=10, stddevs=[1,0.5,0.1], l=l1_weight, mnist_classes=mnist_classes)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "rastrigin_train_mix":
    print('##################################Standard Dev = ', stddev)
    print('##################################L1 weight = ', l1_weight)
    problem = problems.rastrigin_train_mix(batch_size=128, num_dims=10, l=l1_weight, mnist_classes=mnist_classes)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "rastrigin_ori_vary_std":
    print('###########rastrigin_ori_vary_std##################stddev = ', stddev)
    print('#############################num_dims = ', 10)
    problem = problems.rastrigin_ori_vary_std(batch_size=128, num_dims=10, mnist_classes=mnist_classes, stddev=stddev)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "train_l2_reg":
    print('##################################Standard Dev = ', stddev)
    print('##################################L2 weight = ', l1_weight)
    problem = problems.train_l2_reg(batch_size=128, num_dims=10, stddev=stddev, l=l1_weight, mnist_classes=mnist_classes)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  elif problem_name == "rastrigin":
    problem = problems.rastrigin(batch_size=128, num_dims=2)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  
  elif problem_name == "rosenbrock":
    problem = problems.rosenbrock(batch_size=128, num_dims=2)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None

  elif problem_name == "lasso":
    problem = problems.lasso(batch_size=128, num_dims=2)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": path
    }}
    net_assignments = None
  
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  if net_name == "RNNprop":
      default_config = {
              "net": "RNNprop",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "fc",
                  "preprocess_options": {"dim": 20},
                  "scale": 0.01,
                  "tanh_output": True
              },
              "net_path": path
          }
      net_config = {"rp": default_config}

  return problem, net_config, net_assignments
