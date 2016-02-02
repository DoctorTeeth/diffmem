"""
This module implements a neural turing machine.
"""
import math
import autograd.numpy as np
from autograd import grad
from util.util import rando, sigmoid, softmax, softplus, unwrap, sigmoid_prime, tanh_prime, compare_deltas
import memory
import addressing
import sys

class NTM(object):
  """
  NTM with a single-layer feed-forward controller, using autodiff
  """

  def __init__(self, in_size, out_size, hidden_size, N, M, vec_size):

    self.N = N  # the number of memory locations
    self.M = M # the number of columns in a memory location
    self.out_size = out_size
    self.vec_size = vec_size
    shift_width = min(3,self.N) # seems necessary for generalization

    self.stats = None

    self.W = {} # maps parameter names to tensors

    # non-head parameters
    self.W['xh'] = rando(hidden_size, in_size)
    self.W['ho'] = rando(hidden_size, hidden_size)
    self.W['oy'] = rando(out_size, hidden_size)
    self.W['bh']  = rando(hidden_size, 1)
    self.W['by']  = rando(out_size, 1)
    self.W['bo']  = rando(hidden_size, 1)

    # weights from last read head output to hidden layer
    self.W['rh'] = rando(hidden_size, self.M)

    # weights
    self.W['ok_r'] = rando(self.M,hidden_size)
    self.W['ok_w'] = rando(self.M,hidden_size)

    self.W['obeta_r'] = rando(1,hidden_size)
    self.W['obeta_w'] = rando(1,hidden_size)

    # the interpolation gate is a scalar
    self.W['og_r'] = rando(1,hidden_size)
    self.W['og_w'] = rando(1,hidden_size)

    self.W['os_r'] = rando(shift_width,hidden_size)
    self.W['os_w'] = rando(shift_width,hidden_size)

    # gamma is also a scalar
    self.W['ogamma_r'] = rando(1,hidden_size)
    self.W['ogamma_w'] = rando(1,hidden_size)

    self.W['oadds']   = rando(self.M,hidden_size)
    self.W['oerases'] = rando(self.M,hidden_size)

    # biases
    self.W['bk_r'] = rando(self.M,1)
    self.W['bk_w'] = rando(self.M,1)

    self.W['bbeta_r'] = rando(1,1)
    self.W['bbeta_w'] = rando(1,1)

    self.W['bg_r'] = rando(1,1)
    self.W['bg_w'] = rando(1,1)

    self.W['bs_r'] = rando(shift_width,1)
    self.W['bs_w'] = rando(shift_width,1)

    self.W['bgamma_r'] = rando(1,1)
    self.W['bgamma_w'] = rando(1,1)

    self.W['badds']  = rando(self.M,1)
    self.W['erases'] = rando(self.M,1)

    # parameters specifying initial conditions
    self.W['rsInit'] = np.random.uniform(-1,1,(self.M,1))
    self.W['w_wsInit'] = np.random.randn(self.N,1)*0.01
    self.W['w_rsInit'] = np.random.randn(self.N,1)*0.01

    # initial condition of the memory
    self.W['memsInit'] = np.random.randn(self.N,self.M)*0.01

  def lossFun(self, inputs, targets, manual_grad=False):
    """
    Returns the loss given an inputs,targets tuple
    """

    def fprop(params):
      """
      Forward pass of the NTM.
      """

      W = params # aliasing for brevity

      xs, zhs, hs, ys, ps, ts, zos, os = {}, {}, {}, {}, {}, {}, {}, {}

      def l():
        """
        Silly utility function that should be called in init.
        """
        return {}

      rs = l()
      zk_rs = l()
      k_rs, beta_rs, g_rs, s_rs, gamma_rs = l(),l(),l(),l(),l()
      k_ws, beta_ws, g_ws, s_ws, gamma_ws = l(),l(),l(),l(),l()
      adds, erases = l(),l()
      w_ws, w_rs = l(),l() # read weights and write weights
      rs[-1] = self.W['rsInit'] # stores values read from memory
      w_ws[-1] = softmax(self.W['w_wsInit'])
      w_rs[-1] = softmax(self.W['w_rsInit'])

      mems = {} # the state of the memory at every timestep
      mems[-1] = self.W['memsInit']
      loss = 0

      for t in xrange(len(inputs)):

        xs[t] = np.reshape(np.array(inputs[t]),inputs[t].shape[::-1])

        rsum = np.dot(W['rh'], np.reshape(rs[t-1],(self.M,1)))
        zhs[t] = np.dot(W['xh'], xs[t]) + rsum + W['bh']
        hs[t] = np.tanh(zhs[t])

        zos[t] = np.dot(W['ho'], hs[t]) + W['bo']
        os[t] = np.tanh(zos[t])

        # parameters to the read head
        zk_rs[t] =np.dot(W['ok_r'],os[t]) + W['bk_r']
        k_rs[t] = np.tanh(zk_rs[t])
        beta_rs[t] = softplus(np.dot(W['obeta_r'],os[t])
                                    + W['bbeta_r'])
        g_rs[t] = sigmoid(np.dot(W['og_r'],os[t]) + W['bg_r'])
        s_rs[t] = softmax(np.dot(W['os_r'],os[t]) + W['bs_r'])
        gamma_rs[t] = 1 + sigmoid(np.dot(W['ogamma_r'], os[t])
                                        + W['bgamma_r'])

        # parameters to the write head
        k_ws[t] = np.tanh(np.dot(W['ok_w'],os[t]) + W['bk_w'])
        beta_ws[t] = softplus(np.dot(W['obeta_w'], os[t])
                                    + W['bbeta_w'])
        g_ws[t] = sigmoid(np.dot(W['og_w'],os[t]) + W['bg_w'])
        s_ws[t] = softmax(np.dot(W['os_w'],os[t]) + W['bs_w'])
        gamma_ws[t] = 1 + sigmoid(np.dot(W['ogamma_w'], os[t])
                                        + W['bgamma_w'])

        # the erase and add vectors
        # these are also parameters to the write head
        # but they describe "what" is to be written rather than "where"
        adds[t] = np.tanh(np.dot(W['oadds'], os[t]) + W['badds'])
        erases[t] = sigmoid(np.dot(W['oerases'], os[t]) + W['erases'])

        w_ws[t] = addressing.create_weights(   k_ws[t]
                                                , beta_ws[t]
                                                , g_ws[t]
                                                , s_ws[t]
                                                , gamma_ws[t]
                                                , w_ws[t-1]
                                                , mems[t-1])

        w_rs[t] = addressing.create_weights(   k_rs[t]
                                                , beta_rs[t]
                                                , g_rs[t]
                                                , s_rs[t]
                                                , gamma_rs[t]
                                                , w_rs[t-1]
                                                , mems[t-1])

        ys[t] = np.dot(W['oy'], os[t]) + W['by']
        ps[t] = sigmoid(ys[t])

        one = np.ones(ps[t].shape)
        ts[t] = np.reshape(np.array(targets[t]),(self.out_size,1))

        epsilon = 2**-23 # to prevent log(0)
        a = np.multiply(ts[t] , np.log2(ps[t] + epsilon))
        b = np.multiply(one - ts[t], np.log2(one-ps[t] + epsilon))
        loss = loss - (a + b)

        # read from the memory
        rs[t] = memory.read(mems[t-1],w_rs[t])

        # write into the memory
        mems[t] = memory.write(mems[t-1],w_ws[t],erases[t],adds[t])

      self.stats = [loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs, w_ws, adds, erases]
      return np.sum(loss)

    def manual_grads(params):
      """
      Compute the gradient of the loss WRT the parameters
      Ordering of the operations is reverse of that in fprop()
      """
      deltas = {}
      for key, val in params.iteritems():
        deltas[key] = np.zeros_like(val)

      loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs, w_ws, adds, erases = self.stats
      dd = {}
      for t in reversed(xrange(len(targets))):
        if t < len(inputs) - 1:
         # grab gradient from the future
         dnext = dd[t+1]
         # propagate the gradients to the first input of read().
         dread1 = np.dot(mems[t-1], dnext)
         # propagate the gradients to the second input of read().
         dread2 = np.dot(w_rs[t], dnext.T)
         # TODO: propagate the gradients through write()

        ts = np.reshape(np.array(targets[t]),(self.out_size,1))
        # gradient of cross entropy loss function.
        dt = (ps[t] - ts) / (math.log(2) * ps[t] * (1 - ps[t]))

        # propagate the gradient backwards through the flow graph,
        # updating parameters as we go
        dt *= sigmoid_prime(ys[t])
        deltas['oy'] = np.dot(dt, os[t].T)
        deltas['by'] = dt

        if t < len(inputs) - 1:
            # TODO: Update parameters oadds, oerases, ok_r, bbeta_r, og_r, os_r, ok_w ...
            # use dread1 and dread2 computed above as the starting gradients
            pass

        dt = np.dot(params['oy'].T, dt)
        dt *= tanh_prime(zos[t])
        deltas['ho'] = np.dot(dt, hs[t].T)
        deltas['bo'] = dt

        dt = np.dot(params['ho'].T, dt)
        dt *= tanh_prime(zhs[t])
        deltas['xh'] = np.dot(dt, xs[t].T)
        deltas['bh'] = dt

        deltas['rh'] += np.dot(dt, rs[t-1].reshape((self.M, 1)).T)
        # save the gradient for propagating backwards through time
        dd[t] = np.dot(params['rh'].T, dt)
      return deltas

    def bprop(params, manual_grad):
      """
      Compute the gradient of the loss WRT the parameters (W)
      using backward-mode differentiation.
      """
      if manual_grad:
        # compute gradients manually
        fprop(params)
        deltas = manual_grads(params)
        f = grad(fprop)
        auto_deltas = f(params)
        failed_keys = []
        passed_keys = []
        for k in auto_deltas.keys():
          rval = compare_deltas(baseline=auto_deltas[k], candidate=deltas[k])
          if not rval:
            print "compare deltas FAILED for key:", k
            print "baseline"
            print auto_deltas[k]
            print "candidate"
            print deltas[k]
            failed_keys.append(k)
          else:
            passed_keys.append(k)
        if len(failed_keys) > 0:
          print "FAILED KEYS:"
          for k in failed_keys:
            print k
          print "PASSED KEYS:"
          for k in passed_keys:
            print k
          sys.exit(1)
      else:
        # compute gradients automatically
        f = grad(fprop)
        deltas = f(params)
      return deltas

    deltas = bprop(self.W, manual_grad)
    loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs, w_ws, adds, erases = map(unwrap, self.stats)

    return loss, deltas, ps, w_rs, w_ws, adds, erases
