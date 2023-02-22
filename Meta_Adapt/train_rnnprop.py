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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
from timeit import default_timer as timer

from six.moves import xrange
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import tensorflow as tf

from data_generator import data_loader
import meta_rnnprop_train as meta
import util

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer_path", None, "pretrained optimizer")
flags.DEFINE_string("optimizee_path", None, "pretrained optimizee")


flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")

flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("evaluation_period", 10, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")
flags.DEFINE_integer("num_steps", 100, "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")

flags.DEFINE_float("beta1", 0.95, "")
flags.DEFINE_float("beta2", 0.95, "")

flags.DEFINE_string("problem", "mnist", "Type of problem.")

flags.DEFINE_boolean("if_scale", False, "")
flags.DEFINE_float("rd_scale_bound", 3.0, "Bound for random scaling on the main optimizee.")

flags.DEFINE_boolean("if_cl", False, "")
flags.DEFINE_integer("min_num_eval", 3, "")

flags.DEFINE_boolean("if_mt", False, "")
flags.DEFINE_integer("num_mt", 1, "")
flags.DEFINE_string("optimizers", "adam", ".")
flags.DEFINE_float("mt_ratio", 0.3, "")
flags.DEFINE_string("mt_ratios", "0.3 0.3 0.3", "")
flags.DEFINE_integer("k", 1, "")


def main(_):
    # Configuration.
    if FLAGS.if_cl:
        num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]

        num_unrolls = [int(ns / FLAGS.unroll_length) for ns in num_steps]
        num_unrolls_eval = num_unrolls[1:]
        curriculum_idx = 0
    else:
        num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

    # Output path.
    if FLAGS.save_path is not None:
        if not os.path.exists(FLAGS.save_path):
            os.mkdir(FLAGS.save_path)

    # Problem.
    tested_problem_name = FLAGS.problem
    problem, net_config, net_assignments = util.get_config(tested_problem_name, FLAGS.optimizer_path, net_name="RNNprop", stddev=100,l1_weight=0.005, mnist_classes=1)

    # Optimizer setup.
    optimizer = meta.MetaOptimizer(FLAGS.num_mt, FLAGS.beta1, FLAGS.beta2, **net_config)
    minimize, scale, var_x, constants, subsets, seq_step, \
        loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs, opt_var = optimizer.meta_minimize(
            problem, FLAGS.unroll_length,
            learning_rate=FLAGS.learning_rate,
            net_assignments=net_assignments,
            second_derivatives=FLAGS.second_derivatives)
    step, update, reset, cost_op, _ = minimize
        
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

    # Data generator for multi-task learning.
    if FLAGS.if_mt:
        data_mt = data_loader(problem, var_x, constants, subsets, scale,
                              FLAGS.optimizers, FLAGS.unroll_length)
        if FLAGS.if_cl:
            mt_ratios = [float(r) for r in FLAGS.mt_ratios.split()]

    # Assign func.
    p_val_x = []
    for k in var_x:
        p_val_x.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops = [tf.assign(var_x[k_id], p_val_x[k_id]) for k_id in range(len(p_val_x))]

    p_val_all = []
    for k in all_vars:
        p_val_all.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops_all_vars = [tf.assign(all_vars[k_id], p_val_all[k_id]) for k_id in range(len(p_val_all))]
    '''
    ********************************
    '''
    # Assign opt func.
    p_val_opt = []
    for k in opt_var:
        p_val_opt.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops_opt = [tf.assign(opt_var[k_id], p_val_opt[k_id]) for k_id in range(len(p_val_opt))]

    opt_var_arr=[opt_var,assign_ops_opt]
    '''
    ******************************
    '''
    #for adapt on trained optimizee
    
    if(FLAGS.optimizee_path != None):
      with open(FLAGS.optimizee_path, 'rb') as qwe:
        net_params = pickle.load(qwe)
    else:
      net_params = None
    
    '''
    ******************************
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with ms.MonitoredSession() as sess:
        def assign_func(val_x):
            sess.run(assign_ops, feed_dict={p: v for p, v in zip(p_val_x, val_x)})
        def assign_func_opt(val_opt):
          sess.run(assign_ops_opt, feed_dict={p: v for p, v in zip(p_val_opt, val_opt)})

        def assign_func_all_vars(val_x):
            sess.run(assign_ops_all_vars, feed_dict={p: v for p, v in zip(p_val_all, val_x)})

        opt_var_arr=[opt_var,assign_func_opt, assign_func_all_vars]
        # tf1.14
        for rst in [reset] + reset_mt:
            sess.run(rst)

        # Start.
        start_time = timer()
        tf.get_default_graph().finalize()
        best_evaluation = float("inf")
        num_eval = 0
        improved = False
        mti = -1
        for e in xrange(FLAGS.num_epochs):
            
            # Pick a task if it's multi-task learning.
            if FLAGS.if_mt:
                if FLAGS.if_cl:
                    if curriculum_idx >= len(mt_ratios):
                        mt_ratio = mt_ratios[-1]
                    else:
                        mt_ratio = mt_ratios[curriculum_idx]
                else:
                    mt_ratio = FLAGS.mt_ratio
                if random.random() < mt_ratio:
                    mti = (mti + 1) % FLAGS.num_mt
                    task_i = mti
                else:
                    task_i = -1
            else:
                task_i = -1

            # Training.
            if FLAGS.if_cl:
                num_unrolls_cur = num_unrolls[curriculum_idx]
            else:
                num_unrolls_cur = num_unrolls

            if task_i == -1:
                time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                            num_unrolls_cur,
                                            scale=scale,
                                            rd_scale=FLAGS.if_scale,
                                            rd_scale_bound=FLAGS.rd_scale_bound,
                                            assign_func=assign_func,
                                            var_x=var_x,
                                            step=seq_step,
                                            unroll_len=FLAGS.unroll_length,
                                            net_params=net_params,
                                            opt_var_arr=opt_var_arr,
                                            assign_func_all_vars=assign_func_all_vars)
            else:
                data_e = data_mt.get_data(task_i, sess, num_unrolls_cur, assign_func, FLAGS.rd_scale_bound,
                                        if_scale=FLAGS.if_scale, mt_k=FLAGS.k)
                time, cost = util.run_epoch(sess, loss_mt[task_i], [update_mt[task_i], steps_mt[task_i]], reset_mt[task_i],
                                            num_unrolls_cur,
                                            scale=scale,
                                            rd_scale=FLAGS.if_scale,
                                            rd_scale_bound=FLAGS.rd_scale_bound,
                                            assign_func=assign_func,
                                            var_x=var_x,
                                            step=seq_step,
                                            unroll_len=FLAGS.unroll_length,
                                            task_i=task_i,
                                            data=data_e,
                                            label_pl=mt_labels[task_i],
                                            input_pl=mt_inputs[task_i])
            print ("training_loss={}".format(cost))

            # Evaluation.
            if (e + 1) % FLAGS.evaluation_period == 0:
                if FLAGS.if_cl:
                    num_unrolls_eval_cur = num_unrolls_eval[curriculum_idx]
                else:
                    num_unrolls_eval_cur = num_unrolls
                num_eval += 1

                eval_cost = 0
                for _ in xrange(FLAGS.evaluation_epochs):
                    time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                num_unrolls_eval_cur,
                                                step=seq_step,
                                                assign_func=assign_func,
                                                net_params=net_params,
                                                unroll_len=FLAGS.unroll_length,
                                                assign_func_all_vars=assign_func_all_vars)
                    eval_cost += cost

                if FLAGS.if_cl:
                    num_steps_cur = num_steps[curriculum_idx]
                else:
                    num_steps_cur = FLAGS.num_steps
                print ("epoch={}, num_steps={}, eval_loss={}".format(
                    e, num_steps_cur, eval_cost / FLAGS.evaluation_epochs), flush=True)

                if not FLAGS.if_cl:
                    if eval_cost < best_evaluation:
                        best_evaluation = eval_cost
                        optimizer.save(sess, FLAGS.save_path, e + 1)
                        optimizer.save(sess, FLAGS.save_path, 0)
                        print ("Saving optimizer of epoch {}...".format(e + 1))
                    continue

                # Curriculum learning.
                # update curriculum
                if eval_cost < best_evaluation:
                    print('saving time, cur epoch = ', e+1)
                    best_evaluation = eval_cost
                    improved = True
                    # save model
                    optimizer.save(sess, FLAGS.save_path, curriculum_idx)
                    optimizer.save(sess, FLAGS.save_path, 0)
                elif num_eval >= FLAGS.min_num_eval and improved:
                    # restore model
                    optimizer.restore(sess, FLAGS.save_path, curriculum_idx)
                    num_eval = 0
                    improved = False
                    curriculum_idx += 1
                    if curriculum_idx >= len(num_unrolls):
                        curriculum_idx = -1

                    # initial evaluation for next curriculum
                    eval_cost = 0
                    for _ in xrange(FLAGS.evaluation_epochs):
                        time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                    num_unrolls_eval[curriculum_idx],
                                                    step=seq_step,
                                                    assign_func=assign_func,
                                                    net_params=net_params,
                                                    unroll_len=FLAGS.unroll_length,
                                                    assign_func_all_vars=assign_func_all_vars)
                        eval_cost += cost
                    best_evaluation = eval_cost
                    print("epoch={}, num_steps={}, eval loss={}".format(
                        e, num_steps[curriculum_idx], eval_cost / FLAGS.evaluation_epochs), flush=True)
                elif num_eval >= FLAGS.min_num_eval and not improved:
                    print ("no improve during curriculum {} --> stop".format(curriculum_idx))
                    break
        optimizer.save(sess, FLAGS.save_path, 'final')
        print ("total time = {}s...".format(timer() - start_time))


if __name__ == "__main__":
    tf.app.run()
