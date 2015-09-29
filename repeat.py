#!/usr/bin/env python

import autograd.numpy as np
import sequences
from ntm import NTM
from optimizers import RMSProp
from util import gradCheck, serialize, deserialize, visualize
import sys
import warnings
import time

warnings.simplefilter("error")

# Comment next line to remove determinism
np.random.seed(0)

# Set to True to perform gradient checking
GRAD_CHECK = False
TEST_MODE  = True

inFile = 'best_repeat.pkl' 
#inFile = None
vec_size = 3

# TODO: when deserializing, set these automatically
# TODO: each type of sequence should somehow generate the out and in sizes
out_size = vec_size + 1# Size of output bit vector at each time step
in_size = vec_size + 2 # Input vector size, bigger because of start+stop bits
hidden_size = 100 # Size of hidden layer of neurons

max_length = 6
N = 5
M = 7

# RMSProp params - graves says momentum 0.9
rms_lr    = 10e-5
rms_decay = 0.95
rms_blend = 0.95
# An object that keeps the network state during training.
model = NTM(in_size, out_size, hidden_size, N, M)

if inFile is not None:
  model.weights = deserialize(inFile)

# An object that keeps the optimizer state during training
optimizer = RMSProp(model.weights, rms_lr, rms_decay, rms_blend)

max_repeats = 5

n = 0 # counts the number of sequences trained on

npc = 1 # keeps track of trailing bpc
# TODO: we can't just initialize this to none
# TODO: we ought to be using bits instead of nats

while True:

  if (n % 100) == 0:
    verbose = True
    # if npc < 0.01:
    #     max_length = min(max_length + 1, 21)
  else:
    verbose = False

  # train on sequences of length from 1 to (max_length - 1)
  seq_length = np.random.randint(1,3)
  repeats    = np.random.randint(1,3)
  i, t = sequences.repeat_copy(seq_length, vec_size, repeats, max_repeats)
  inputs = np.matrix(i)
  targets = np.matrix(t)

  # forward seq_length characters through the net and fetch gradient
  # deltas is a list of deltas oriented same as list of weights
  loss, deltas, outputs, r, w, a, e = model.lossFun(inputs, targets, verbose)

  newnpc = np.sum(loss) / ((seq_length*2 + 2) * vec_size)
  npc = 0.99 * npc + 0.01 * newnpc

  # sometimes print out diagnostic info
  if verbose or TEST_MODE:
    print 'iter %d' % (n)
    hi = inputs.shape[1] - 1
    wi = inputs.shape[0]
    visualize(inputs, outputs, r, w, a, e, hi, wi)

    # check on the fancy quotients
    print "max_length: ", max_length
    print "FANCY:"
    for name, q in zip(model.names, optimizer.qs):
      print name + ": " + str(q)

    # calculate the NPC
    print "totalnpc: ", npc
    print "thisnpc: ", newnpc

    # always serialize
    timestring = time.strftime("%Y-%m-%d-%h-%m-%s")
    filename = 'serializations/params_n-' + str(n) + '_' + timestring  + '.pkl'
    serialize(filename,model.weights)

    if GRAD_CHECK:
      # Check weights using finite differences
      check = gradCheck(model, deltas, inputs, targets, 1e-5, 1e-7)
      print "PASS DIFF CHECK?: ", check
      if not check:
        sys.exit(1)

  if not TEST_MODE:
    optimizer.update_weights(model.weights, deltas)

  n += 1
