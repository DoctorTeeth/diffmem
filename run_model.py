#!/usr/bin/env python

import argparse
import autograd.numpy as np
from util.sequences import SequenceGen
from ntm.ntm import NTM
from util.optimizers import RMSProp
from util.util import gradCheck, serialize, deserialize, visualize
import os
import warnings
import time
import pdb

# Comment next line to remove determinism
np.random.seed(0)

# Uncomment below for debugging numerical issues
# warnings.simplefilter("error")

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="location of the serialized model",
                    default=None)
parser.add_argument("--task", help="the task to train or test on",
                    default='copy')
parser.add_argument("--N", help="the number of memory locations",
                    default=15, type=int)
parser.add_argument("--M", help="the number of cells in a memory location",
                    default=7, type=int)
parser.add_argument("--vec_size", help="width of input vector (the paper uses 8)",
                    default=3, type=int)
parser.add_argument("--hi", help="upper bound on seq length",
                    default=3, type=int)
parser.add_argument("--lo", help="lower bound on seq length",
                    default=1, type=int)
parser.add_argument("--units", help="number of hidden units",
                    default=100, type=int)
parser.add_argument("--lr_rate", help="maybe halve learning rate at this interval",
                    default=None, type=int)
parser.add_argument("--serialize_freq", help="serialize every <this many sequences>",
                    default=100, type=int)
parser.add_argument("--log_freq", help="how often to log diagnostic information",
                    default=100, type=int)
parser.add_argument("--serialize_to", help="where to save models",
                    default=None)
parser.add_argument('--test_mode', dest='test_mode', action='store_true')
parser.add_argument('--grad_check', dest='grad_check', action='store_true')
parser.add_argument('--manual_grad', dest='manual_grad', action='store_true')
parser.set_defaults(test_mode=False)
parser.set_defaults(grad_check=False)
parser.set_defaults(manual_grad=False)
args = parser.parse_args()

# create directory for serializations if necessary
if args.serialize_to is not None:
  print "Serializing to:", args.serialize_to
  try:
    os.mkdir(args.serialize_to)
  except OSError:
    pass

# if we use lr schedule, we need to keep track of errors over time
if args.lr_rate is not None:
  print "Using lr schedule rate of:", args.lr_rate
  errors = {}
  error_sum = 0

# deserialize saved model if path given
if args.model is None:
  # If not using a saved model, initialize from params
  vec_size = args.vec_size
  seq = SequenceGen(args.task, vec_size, args.hi, args.lo)
  hidden_size = args.units # Size of hidden layer of neurons
  N = args.N # number of memory locations
  M = args.M # size of a memory location
  model = NTM(seq.in_size, seq.out_size, hidden_size, N, M, vec_size)
else:
  # otherwise, load the model from specified file
  print "Using saved model:", args.model
  model = deserialize(args.model)
  vec_size = model.vec_size # vec size comes from model
  seq = SequenceGen(args.task, vec_size, args.hi, args.lo)

# An object that keeps the optimizer state during training
optimizer = RMSProp(model.W)

n = 0 # counts the number of sequences trained on
bpc = None # keeps track of trailing bpc (cost)

while n < 10000:

  i, t, seq_len = seq.make()
  inputs = np.matrix(i)
  targets = np.matrix(t)

  loss, deltas, outputs, r, w, a, e = model.lossFun(inputs, targets, args.manual_grad)

  newbpc = np.sum(loss) / ((seq_len*2 + 2) * vec_size)
  if bpc is not None:
    bpc = 0.99 * bpc + 0.01 * newbpc
  else:
    bpc = newbpc

  # sometimes print out diagnostic info
  if ((n % args.log_freq) == 0) or args.test_mode:
    print 'iter %d' % (n)
    visualize(inputs, outputs, r, w, a, e)

    # log ratio of delta l2 norm to weight l2 norm
    print "update/weight ratios:"
    for k in model.W.keys():
      print k + ": " + str(optimizer.qs[k])

    print "trailing bpc estimate: ", bpc
    print "this bpc: ", newbpc

  # maybe serialize
  if (args.serialize_to is not None) and (n % args.serialize_freq) == 0:
    ts = time.strftime("%Y-%m-%d-%h-%m-%s")
    filename = args.serialize_to + '/params_n-' + str(n) + '_' + ts + '.pkl'
    serialize(filename,model)

  if args.grad_check:
    # Check weights using finite differences
    check = gradCheck(model, deltas, inputs, targets, 1e-5, 1e-7)
    print "PASS DIFF CHECK?: ", check

  if not args.test_mode:
    # optionally halve the learning rate if error has not gone down
    if args.lr_rate is not None:
      error_sum += newbpc
      if n % args.lr_rate == 0:
        errors[n] = error_sum
        error_sum = 0
        if n > args.lr_rate: # don't do this for the first "epoch"
          if errors[n] > errors[n - args.lr_rate]:
            print "halving learning rate!"
            optimizer.lr /= 2
            print "learning rate is now:", optimizer.lr 

    optimizer.update_weights(model.W, deltas)

  n += 1
