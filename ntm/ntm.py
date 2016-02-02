"""
This module implements a neural turing machine.
"""
import math
import autograd.numpy as np
from autograd import grad
from util.util import rando, sigmoid, softmax, softplus, unwrap, sigmoid_prime, tanh_prime
import memory
import addressing

class NTM(object):
  """
  NTM with a single-layer feed-forward controller, using autodiff
  """

  def __init__(self, in_size, out_size, hidden_size, N, M, vec_size, heads):

    self.N = N  # the number of memory locations
    self.M = M # the number of columns in a memory location
    self.out_size = out_size
    self.vec_size = vec_size
    self.heads = heads
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

    # head parameters
    for idx in range(self.heads):

      # weights from last read head output to hidden layer
      self.W['rh' + str(idx)] = rando(hidden_size, self.M)

      # weights
      self.W['ok_r' + str(idx)] = rando(self.M,hidden_size)
      self.W['ok_w' + str(idx)] = rando(self.M,hidden_size)

      self.W['obeta_r' + str(idx)] = rando(1,hidden_size)
      self.W['obeta_w' + str(idx)] = rando(1,hidden_size)

      # the interpolation gate is a scalar
      self.W['og_r' + str(idx)] = rando(1,hidden_size)
      self.W['og_w' + str(idx)] = rando(1,hidden_size)

      self.W['os_r' + str(idx)] = rando(shift_width,hidden_size)
      self.W['os_w' + str(idx)] = rando(shift_width,hidden_size)

      # gamma is also a scalar
      self.W['ogamma_r' + str(idx)] = rando(1,hidden_size)
      self.W['ogamma_w' + str(idx)] = rando(1,hidden_size)

      self.W['oadds' + str(idx)]   = rando(self.M,hidden_size)
      self.W['oerases' + str(idx)] = rando(self.M,hidden_size)

      # biases
      self.W['bk_r' + str(idx)] = rando(self.M,1)
      self.W['bk_w' + str(idx)] = rando(self.M,1)

      self.W['bbeta_r' + str(idx)] = rando(1,1)
      self.W['bbeta_w' + str(idx)] = rando(1,1)

      self.W['bg_r' + str(idx)] = rando(1,1)
      self.W['bg_w' + str(idx)] = rando(1,1)

      self.W['bs_r' + str(idx)] = rando(shift_width,1)
      self.W['bs_w' + str(idx)] = rando(shift_width,1)

      self.W['bgamma_r' + str(idx)] = rando(1,1)
      self.W['bgamma_w' + str(idx)] = rando(1,1)

      self.W['badds' + str(idx)]  = rando(self.M,1)
      self.W['erases' + str(idx)] = rando(self.M,1)

      # parameters specifying initial conditions
      self.W['rsInit' + str(idx)] = np.random.uniform(-1,1,(self.M,1))
      self.W['w_wsInit' + str(idx)] = np.random.randn(self.N,1)*0.01
      self.W['w_rsInit' + str(idx)] = np.random.randn(self.N,1)*0.01

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
        return [{} for _ in xrange(self.heads)]

      rs = l()
      zk_rs = l()
      k_rs, beta_rs, g_rs, s_rs, gamma_rs = l(),l(),l(),l(),l()
      k_ws, beta_ws, g_ws, s_ws, gamma_ws = l(),l(),l(),l(),l()
      adds, erases = l(),l()
      w_ws, w_rs = l(),l() # read weights and write weights
      for idx in range(self.heads):
        rs[idx][-1] = self.W['rsInit' + str(idx)] # stores values read from memory
        w_ws[idx][-1] = softmax(self.W['w_wsInit' + str(idx)])
        w_rs[idx][-1] = softmax(self.W['w_rsInit' + str(idx)])

      mems = {} # the state of the memory at every timestep
      mems[-1] = self.W['memsInit']
      loss = 0

      for t in xrange(len(inputs)):

        xs[t] = np.reshape(np.array(inputs[t]),inputs[t].shape[::-1])

        rsum = 0
        for idx in range(self.heads):
          rsum = rsum + np.dot(W['rh' + str(idx)], np.reshape(rs[idx][t-1],(self.M,1)))
        zhs[t] = np.dot(W['xh'], xs[t]) + rsum + W['bh']
        hs[t] = np.tanh(zhs[t])

        zos[t] = np.dot(W['ho'], hs[t]) + W['bo']
        os[t] = np.tanh(zos[t])

        for idx in range(self.heads):
          # parameters to the read head
          zk_rs[idx][t] =np.dot(W['ok_r' + str(idx)],os[t]) + W['bk_r' + str(idx)]
          k_rs[idx][t] = np.tanh(zk_rs[idx][t])
          beta_rs[idx][t] = softplus(np.dot(W['obeta_r' + str(idx)],os[t])
                                     + W['bbeta_r' + str(idx)])
          g_rs[idx][t] = sigmoid(np.dot(W['og_r' + str(idx)],os[t]) + W['bg_r' + str(idx)])
          s_rs[idx][t] = softmax(np.dot(W['os_r' + str(idx)],os[t]) + W['bs_r' + str(idx)])
          gamma_rs[idx][t] = 1 + sigmoid(np.dot(W['ogamma_r' + str(idx)], os[t])
                                         + W['bgamma_r' + str(idx)])

          # parameters to the write head
          k_ws[idx][t] = np.tanh(np.dot(W['ok_w' + str(idx)],os[t]) + W['bk_w' + str(idx)])
          beta_ws[idx][t] = softplus(np.dot(W['obeta_w' + str(idx)], os[t])
                                     + W['bbeta_w' + str(idx)])
          g_ws[idx][t] = sigmoid(np.dot(W['og_w' + str(idx)],os[t]) + W['bg_w' + str(idx)])
          s_ws[idx][t] = softmax(np.dot(W['os_w' + str(idx)],os[t]) + W['bs_w' + str(idx)])
          gamma_ws[idx][t] = 1 + sigmoid(np.dot(W['ogamma_w' + str(idx)], os[t])
                                         + W['bgamma_w' + str(idx)])

          # the erase and add vectors
          # these are also parameters to the write head
          # but they describe "what" is to be written rather than "where"
          adds[idx][t] = np.tanh(np.dot(W['oadds' + str(idx)], os[t]) + W['badds' + str(idx)])
          erases[idx][t] = sigmoid(np.dot(W['oerases' + str(idx)], os[t]) + W['erases' + str(idx)])

          w_ws[idx][t] = addressing.create_weights(   k_ws[idx][t]
                                                    , beta_ws[idx][t]
                                                    , g_ws[idx][t]
                                                    , s_ws[idx][t]
                                                    , gamma_ws[idx][t]
                                                    , w_ws[idx][t-1]
                                                    , mems[t-1])

          w_rs[idx][t] = addressing.create_weights(   k_rs[idx][t]
                                                    , beta_rs[idx][t]
                                                    , g_rs[idx][t]
                                                    , s_rs[idx][t]
                                                    , gamma_rs[idx][t]
                                                    , w_rs[idx][t-1]
                                                    , mems[t-1])

        ys[t] = np.dot(W['oy'], os[t]) + W['by']
        ps[t] = sigmoid(ys[t])

        one = np.ones(ps[t].shape)
        ts[t] = np.reshape(np.array(targets[t]),(self.out_size,1))

        epsilon = 2**-23 # to prevent log(0)
        a = np.multiply(ts[t] , np.log2(ps[t] + epsilon))
        b = np.multiply(one - ts[t], np.log2(one-ps[t] + epsilon))
        loss = loss - (a + b)

        for idx in range(self.heads):
          # read from the memory
          rs[idx][t] = memory.read(mems[t-1],w_rs[idx][t])

          # write into the memory
          mems[t] = memory.write(mems[t-1],w_ws[idx][t],erases[idx][t],adds[idx][t])

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
      dd = [{} for _ in range(self.heads)]
      for t in reversed(xrange(len(targets))):
        if t < len(inputs) - 1:
          for idx in range(self.heads):
            # grab gradient from the future
            dnext = dd[idx][t+1]
            # propagate the gradients to the first input of read().
            dread1 = np.dot(mems[t-1], dnext)
            # propagate the gradients to the second input of read().
            dread2 = np.dot(w_rs[idx][t], dnext.T)
            # TODO: propagate the gradients through write()
            pass

        ts = np.reshape(np.array(targets[t]),(self.out_size,1))
        # gradient of cross entropy loss function.
        dt = (ps[t] - ts) / (math.log(2) * ps[t] * (1 - ps[t]))

        # propagate the gradient backwards through the flow graph,
        # updating parameters as we go
        dt *= sigmoid_prime(ys[t])
        deltas['oy'] = np.dot(dt, os[t].T)
        deltas['by'] = dt

        if t < len(inputs) - 1:
          for idx in range(self.heads):
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

        for idx in range(self.heads):
            deltas['rh' + str(idx)] += np.dot(dt, rs[idx][t-1].reshape((self.M, 1)).T)
            # save the gradient for propagating backwards through time
            dd[idx][t] = np.dot(params['rh' + str(idx)].T, dt)
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
      else:
        # compute gradients automatically
        f = grad(fprop)
        deltas = f(params)
      return deltas

    deltas = bprop(self.W, manual_grad)
    loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs, w_ws, adds, erases = map(unwrap, self.stats)

    return loss, deltas, ps, w_rs, w_ws, adds, erases
