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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pdb
import pickle

from six.moves import xrange
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import tensorflow as tf
import numpy as np
import meta_rnnprop_eval as meta
import util

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("problem", "simple", "Type of problem.")

flags.DEFINE_string("optimizee_path", None, "initial optimizee")

flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_string("output_path", None, "Path to output results.")

flags.DEFINE_integer("num_epochs", 1, "Number of evaluation epochs.")
flags.DEFINE_integer("num_steps", 1000, "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")

flags.DEFINE_float("beta1", 0.95, "")
flags.DEFINE_float("beta2", 0.95, "")


def main(_):
    # Configuration.
    num_unrolls = FLAGS.num_steps
    if FLAGS.seed:
        tf.set_random_seed(FLAGS.seed)

    # Problem.

    tested_problem_name = 'lasso_train'
    problem, net_config, net_assignments = util.get_config(tested_problem_name, FLAGS.path, net_name="RNNprop", stddev=100, l1_weight=0.005,mnist_classes=1)

    # Optimizer setup.
    if FLAGS.optimizer == "Adam":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        
        var_x = problem_vars
        step=None

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]
    elif FLAGS.optimizer == "sgd":
        cost_op = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)
        
        var_x = problem_vars

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tested_problem_name)
        all_vars_temp = []
        for iii in all_vars:
          if(('_mt' in iii.name) or ('_vt' in iii.name)):
            continue
          else:
            all_vars_temp.append(iii)
        all_vars = all_vars_temp
        for iii in all_vars:
          print(iii.shape)
        print(all_vars)

        step=None

        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]
    elif FLAGS.optimizer == "L2L":
        if FLAGS.path is None:
            logging.warning("Evaluating untrained L2L optimizer")
        optimizer = meta.MetaOptimizer(FLAGS.beta1, FLAGS.beta2, **net_config)
        meta_loss, _, var_x, step = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
        _, update, reset, cost_op, _ = meta_loss
        
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tested_problem_name)
        all_vars_temp = []
        for iii in all_vars:
          if(('_mt' in iii.name) or ('_vt' in iii.name)):
            continue
          else:
            all_vars_temp.append(iii)

        all_vars = all_vars_temp
        for iii in all_vars:
          print(iii.shape)
        print(all_vars)
    else:
        raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    '''
    ##########################
    '''
    p_val_x = []
    for k in var_x:
        p_val_x.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops = [tf.assign(var_x[k_id], p_val_x[k_id]) for k_id in range(len(p_val_x))]

    p_val_all = []
    for k in all_vars:
        p_val_all.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops_all_vars = [tf.assign(all_vars[k_id], p_val_all[k_id]) for k_id in range(len(p_val_all))]
    '''
    ##########################
    '''

    with ms.MonitoredSession() as sess:
        '''
        ##########################
        '''
        def assign_func(val_x):
            sess.run(assign_ops, feed_dict={p: v for p, v in zip(p_val_x, val_x)})

        def assign_func_all_vars(val_x):
            sess.run(assign_ops_all_vars, feed_dict={p: v for p, v in zip(p_val_all, val_x)})
        '''
        ##########################
        '''

        sess.run(reset)
        # Prevent accidental changes to the graph.
        tf.get_default_graph().finalize()
        '''
        ##########################
        '''
        if(FLAGS.optimizee_path!=None):
          with open(FLAGS.optimizee_path, 'rb') as qwe:
            optimizee_params = pickle.load(qwe)
            print('#########################')
            for k_id in optimizee_params:
              print(k_id.shape)
            assign_func_all_vars(optimizee_params)
        else:
          print('################################')
          print('SAVE INITIAL')
          print('################################')
          optimizee_params = []
          for k_id in range(len(all_vars)):
            k_value = sess.run(all_vars[k_id])
            optimizee_params.append(k_value)
          save_dir = FLAGS.output_path
          if not os.path.exists(save_dir):
            os.makedirs(save_dir)
          with open(os.path.join(save_dir, 'initial_optimizee_params.pickle'), 'wb') as l3_record:
                pickle.dump(optimizee_params, l3_record)


        total_time = 0
        total_cost = 0
        loss_record = []
        for e in xrange(FLAGS.num_epochs):
            # Training.
            time, cost, training_acc_cur, objective_values_cur = util.run_eval_epoch(sess, cost_op, [update],
                                             num_unrolls, step=step, unroll_len=1, var_x=var_x)
            total_time += time
            total_cost += sum(cost) / num_unrolls
            loss_record += cost

        #save trained optimizee params
        trained_optimizee_params = []

        for k_id in range(len(all_vars)):
          k_value = sess.run(all_vars[k_id])
          trained_optimizee_params.append(k_value)
        #save training acc and loss
        '''
        **********************************************************
        '''
        save_dir = FLAGS.output_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'seed{}_training_loss_record.pickle'.format(FLAGS.seed)), 'wb') as l1_record:
                pickle.dump(objective_values_cur, l1_record)

        with open(os.path.join(save_dir, 'seed{}_trained_optimizee_params.pickle'.format(FLAGS.seed)), 'wb') as l3_record:
                pickle.dump(trained_optimizee_params, l3_record)
        '''
        **********************************************************
        '''

        # Results.
        util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                         total_time, FLAGS.num_epochs)

    if FLAGS.output_path is not None:
        if not os.path.exists(FLAGS.output_path):
            os.mkdir(FLAGS.output_path)
    output_file = '{}/{}_eval_loss_record.pickle-{}'.format(FLAGS.output_path, FLAGS.optimizer, FLAGS.problem)
    with open(output_file, 'wb') as l_record:
        pickle.dump(loss_record, l_record)
    print("Saving evaluate loss record {}".format(output_file))


if __name__ == "__main__":
    tf.app.run()
