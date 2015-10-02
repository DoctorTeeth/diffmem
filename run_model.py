#!/usr/bin/env python

import argparse
import autograd.numpy as np
from util.sequences import SequenceGen
from ntm.ntm import NTM
from util.optimizers import RMSProp
from util.util import gradCheck, serialize, deserialize, visualize
import sys
import os
import warnings
import time

# Comment next line to remove determinism
np.random.seed(0)

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
parser.add_argument("--serialize_to", help="where to save models",
                    default=None)
parser.add_argument('--test_mode', dest='test_mode', action='store_true')
parser.add_argument('--grad_check', dest='grad_check', action='store_true')
parser.set_defaults(test_mode=False)
parser.set_defaults(grad_check=False)
args = parser.parse_args()

if args.serialize_to is not None:
  try:
    os.mkdir(args.serialize_to)
  except OSError:
    pass

warnings.simplefilter("error")

vec_size = args.vec_size

seq = SequenceGen(args.task, vec_size, args.hi, args.lo)

hidden_size = 100 # Size of hidden layer of neurons

N = args.N
M = args.M

# An object that keeps the network state during training.
model = NTM(seq.in_size, seq.out_size, hidden_size, N, M)

if args.model is not None:
  model.weights = deserialize(args.model)

# An object that keeps the optimizer state during training
optimizer = RMSProp(model.weights)

n = 0 # counts the number of sequences trained on

npc = None # keeps track of trailing bpc

while True:

  if (n % 100) == 0:
    verbose = True
  else:
    verbose = False

  i, t, seq_len = seq.make()
  inputs = np.matrix(i)
  targets = np.matrix(t)

  loss, deltas, outputs, r, w, a, e = model.lossFun(inputs, targets, verbose)

  newnpc = np.sum(loss) / ((seq_len*2 + 2) * vec_size)
  if npc is not None:
    npc = 0.99 * npc + 0.01 * newnpc
  else:
    npc = newnpc

  # sometimes print out diagnostic info
  if verbose or args.test_mode:
    print 'iter %d' % (n)
    visualize(inputs, outputs, r, w, a, e)

    # log ratio of delta l2 norm to weight l2 norm
    print "FANCY:"
    for name, q in zip(model.names, optimizer.qs):
      print name + ": " + str(q)

    print "totalnpc: ", npc
    print "thisnpc: ", newnpc

    # maybe serialize
    if args.serialize_to is not None:
        timestring = time.strftime("%Y-%m-%d-%h-%m-%s")
        filename = args.serialize_to + '/params_n-' + str(n) + '_' + timestring  + '.pkl'
        serialize(filename,model.weights)

    if args.grad_check:
      # Check weights using finite differences
      check = gradCheck(model, deltas, inputs, targets, 1e-5, 1e-7)
      print "PASS DIFF CHECK?: ", check
      if not check:
        sys.exit(1)

  if not args.test_mode:
    optimizer.update_weights(model.weights, deltas)

  n += 1
