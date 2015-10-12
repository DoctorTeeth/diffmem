"""
NTM with feed-forward controller, using autodiff
"""
import autograd.numpy as np
from autograd import grad
from util.util import rando, sigmoid, softmax, softplus, unwrap
import memory
import addressing
import sys
import pdb

class NTM(object):

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

  def lossFun(self, inputs, targets):
    """
    inputs,targets are both list of integers.
    where in this case, H is hidden_size from above
    returns the loss, gradients on model parameters, and last hidden state
    n is the counter we're on, just for debugging
    """

    def fprop(params):

      W = params # aliasing for brevity

      xs, hs, ys, ps, ts, os = {}, {}, {}, {}, {}, {}

      def l():
        return [{} for _ in xrange(self.heads)]

      rs = l()
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
        hs[t] = np.tanh(np.dot(W['xh'], xs[t]) + rsum + W['bh'])

        os[t] = np.tanh(np.dot(W['ho'], hs[t]) + W['bo'])


        for idx in range(self.heads):
          # parameters to the read head
          k_rs[idx][t] = np.tanh(np.dot(W['ok_r' + str(idx)],os[t]) + W['bk_r' + str(idx)])
          beta_rs[idx][t] = softplus(np.dot(W['obeta_r' + str(idx)],os[t]) + W['bbeta_r' + str(idx)])
          g_rs[idx][t] = sigmoid(np.dot(W['og_r' + str(idx)],os[t]) + W['bg_r' + str(idx)])
          s_rs[idx][t] = softmax(np.dot(W['os_r' + str(idx)],os[t]) + W['bs_r' + str(idx)])
          gamma_rs[idx][t] = 1 + sigmoid(np.dot(W['ogamma_r' + str(idx)], os[t]) + W['bgamma_r' + str(idx)])

          # parameters to the write head
          k_ws[idx][t] = np.tanh(np.dot(W['ok_w' + str(idx)],os[t]) + W['bk_w' + str(idx)])
          beta_ws[idx][t] = softplus(np.dot(W['obeta_w' + str(idx)],os[t]) + W['bbeta_w' + str(idx)])
          g_ws[idx][t] = sigmoid(np.dot(W['og_w' + str(idx)],os[t]) + W['bg_w' + str(idx)])
          s_ws[idx][t] = softmax(np.dot(W['os_w' + str(idx)],os[t]) + W['bs_w' + str(idx)])
          gamma_ws[idx][t] = 1 + sigmoid(np.dot(W['ogamma_w' + str(idx)], os[t]) + W['bgamma_w' + str(idx)])

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

      self.stats = [loss, ps, w_rs, w_ws, adds, erases]
      return np.sum(loss)

    def bprop(params):
      f = grad(fprop)
      return f(params)

    deltas = bprop(self.W)

    loss, ps, reads, writes, adds, erases = map(unwrap,self.stats)

    return loss, deltas, ps, reads, writes, adds, erases
