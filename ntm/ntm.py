"""
NTM with feed-forward controller, using autodiff
"""
import autograd.numpy as np
from autograd import grad
from util.util import rando, sigmoid, softmax, softplus
import memory
import addressing
import sys
import pdb

class NTM(object):

  def __init__(self, in_size, out_size, hidden_size, N, M):

    self.N = N  # the number of memory locations
    self.M = M # the number of columns in a memory location
    self.out_size = out_size

    self.W = {} # maps parameter names to tensors

    self.W['xh'] = rando(hidden_size, in_size) 
    self.W['rh'] = rando(hidden_size, self.M)
    self.W['ho'] = rando(hidden_size, hidden_size)
    self.W['oy'] = rando(out_size, hidden_size)

    self.W['ok_r'] = rando(self.M,hidden_size)
    self.W['ok_w'] = rando(self.M,hidden_size)

    self.W['obeta_r'] = rando(1,hidden_size)
    self.W['obeta_w'] = rando(1,hidden_size)

    # the interpolation gate is a scalar
    self.W['og_r'] = rando(1,hidden_size)
    self.W['og_w'] = rando(1,hidden_size)

    shift_width = min(3,self.N)
    self.W['os_r'] = rando(shift_width,hidden_size)
    self.W['os_w'] = rando(shift_width,hidden_size)

    # gamma is also a scalar
    self.W['ogamma_r'] = rando(1,hidden_size)
    self.W['ogamma_w'] = rando(1,hidden_size)

    self.W['oadds']   = rando(self.M,hidden_size)
    self.W['oerases'] = rando(self.M,hidden_size)

    self.W['bh']  = rando(hidden_size, 1) 
    self.W['by']  = rando(out_size, 1) 
    self.W['bo']  = rando(hidden_size, 1) 

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

    # initialize some recurrent things to bias values
    self.rsInit = np.random.uniform(-1,1,(self.M,1))
    self.memsInit = np.random.randn(self.N,self.M)*0.01
    self.w_ws_initInit = np.random.randn(self.N,1)*0.01
    self.w_rs_initInit = np.random.randn(self.N,1)*0.01


  def lossFun(self, inputs, targets, verbose):
    """
    inputs,targets are both list of integers.
    where in this case, H is hidden_size from above
    returns the loss, gradients on model parameters, and last hidden state
    n is the counter we're on, just for debugging
    """

    def fprop(params):

      W = params # aliasing for brevity

      xs, hs, ys, ps, ts, rs, os = {}, {}, {}, {}, {}, {}, {}
      k_rs, beta_rs, g_rs, s_rs, gamma_rs = {},{},{},{},{}
      k_ws, beta_ws, g_ws, s_ws, gamma_ws = {},{},{},{},{}
      adds, erases = {},{}
      w_ws, w_rs = {},{} # read weights and write weights
      mems = {} # the state of the memory at every timestep
      # rs stores the value read out of memory
      rs[-1] = self.rsInit 
      mems[-1] = self.memsInit 
      w_ws_init = self.w_ws_initInit
      w_rs_init = self.w_rs_initInit
      w_ws[-1] = softmax(w_ws_init)
      w_rs[-1] = softmax(w_rs_init)
      loss = 0

      for t in xrange(len(inputs)):

        xs[t] = np.reshape(np.array(inputs[t]),inputs[t].shape[::-1])

        hs[t] = np.tanh(np.dot(W['xh'], xs[t]) + np.dot(W['rh'], np.reshape(rs[t-1],(self.M,1))) + W['bh'])

        os[t] = np.tanh(np.dot(W['ho'], hs[t]) + W['bo'])

        # parameters to the read head
        k_rs[t] = np.tanh(np.dot(W['ok_r'],os[t]) + W['bk_r'])
        beta_rs[t] = softplus(np.dot(W['obeta_r'],os[t]) + W['bbeta_r'])
        g_rs[t] = sigmoid(np.dot(W['og_r'],os[t]) + W['bg_r'])
        s_rs[t] = softmax(np.dot(W['os_r'],os[t]) + W['bs_r'])
        gamma_rs[t] = 1 + sigmoid(np.dot(W['ogamma_r'], os[t]) + W['bgamma_r'])

        # parameters to the write head
        k_ws[t] = np.tanh(np.dot(W['ok_w'],os[t]) + W['bk_w'])
        beta_ws[t] = softplus(np.dot(W['obeta_w'],os[t]) + W['bbeta_w'])
        g_ws[t] = sigmoid(np.dot(W['og_w'],os[t]) + W['bg_w'])
        s_ws[t] = softmax(np.dot(W['os_w'],os[t]) + W['bs_w'])
        gamma_ws[t] = 1 + sigmoid(np.dot(W['ogamma_w'], os[t]) + W['bgamma_w'])

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

      return loss, ps, w_rs, w_ws, adds, erases

    def training_loss(params):

      # wrap so that autograd works
      loss, ps, reads, writes, adds, erases = fprop(params)
      return np.sum(loss) 

    def bprop(params):
      f = grad(training_loss)
      return f(params)

    # TODO: get rid of this foolery
    loss, ps, reads, writes, adds, erases = fprop(self.W)
    deltas = bprop(self.W)

    return loss, deltas, ps, reads, writes, adds, erases
