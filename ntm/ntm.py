"""
This module implements a neural turing machine.
"""
import math
import autograd.numpy as np
from autograd import grad
from util.util import rando, sigmoid, softmax, softplus, unwrap, sigmoid_prime, tanh_prime, compare_deltas, dKdu, softmax_grads
import memory
import addressing
from addressing import cosine_sim
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
    self.W['berases'] = rando(self.M,1)

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
        erases[t] = sigmoid(np.dot(W['oerases'], os[t]) + W['berases'])

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
        a = np.multiply(ts[t] , np.log(ps[t] + epsilon))
        b = np.multiply(one - ts[t], np.log(one-ps[t] + epsilon))
        loss = loss - (a + b)

        # read from the memory
        rs[t] = memory.read(mems[t-1],w_rs[t])

        # write into the memory
        mems[t] = memory.write(mems[t-1],w_ws[t],erases[t],adds[t])

      self.stats = [loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs,
                    w_ws, adds, erases, k_rs, k_ws]
      return np.sum(loss)

    def manual_grads(params):
      """
      Compute the gradient of the loss WRT the parameters
      Ordering of the operations is reverse of that in fprop()
      """
      deltas = {}
      for key, val in params.iteritems():
        deltas[key] = np.zeros_like(val)

      loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs, w_ws, adds, erases, k_rs, k_ws = self.stats
      dd = {}
      drs = {}
      dzh = {}
      dmem = {} # might not need this, since we have dmemtilde
      dmemtilde = {}
      du_r = {}
      du_w = {}
      for t in reversed(xrange(len(targets))):

        dy = np.copy(ps[t])
        dy -= targets[t].T # backprop into y

        deltas['oy'] += np.dot(dy, os[t].T)
        deltas['by'] += dy

        if t < len(targets) - 1:
          # r[t] affects cost through zh[t+1] via Wrh
          drs[t] = np.dot(self.W['rh'].T, dzh[t + 1])

          # right now, mems[t] influences cost through rs[t+1], via w_rs[t+1]
          dmem[t] = np.dot( w_rs[t + 1], drs[t + 1].reshape((self.M,1)).T )
          # and also through mems at next step
          W = np.reshape(w_ws[t+1], (w_ws[t+1].shape[0], 1))
          E = np.reshape(erases[t+1], (erases[t+1].shape[0], 1))
          WTE = np.dot(W, E.T)
          KEEP = np.ones(mems[0].shape) - WTE
          dmem[t] += np.multiply(dmemtilde[t+1], KEEP)
          # and also through its influence on the content weighting next step
          dmem[t] += du_r[t+1] + du_w[t+1]

          dmemtilde[t] = dmem[t]

          # erases[t] affects cost through mems[t], via w_ws[t]
          derase = np.dot(np.multiply(dmemtilde[t], -mems[t-1]).T, w_ws[t])

          # zerase affects just erases through a sigmoid
          dzerase = derase * (erases[t] * (1 - erases[t]))

          # adds[t] affects costs through mems[t], via w_ws
          dadd = np.dot(dmem[t].T, w_ws[t])

          # zadds affects just adds through a tanh
          dzadd = dadd * (1 - adds[t] * adds[t])

          # dbadds is just dzadds
          deltas['badds'] += dzadd

          deltas['oadds'] += np.dot(dzadd, os[t].T)

          deltas['berases'] += dzerase

          deltas['oerases'] += np.dot(dzerase, os[t].T)

          # for now, content weighting affects this read through mems[t-1]
          dwc_r = np.zeros(w_rs[0].shape)
          # for every element of the weighting
          for i in range(self.N):
            # for every value in the read
            for j in range(self.M):
              dwc_r[i] += drs[t][j] * mems[t-1][i,j]

          dwc_w = np.zeros(w_ws[0].shape)
          # for every element of the weighting
          for i in range(self.N):
            # for every column in the memory
            for j in range(self.M):
              # write weights affect memtilde through erasing
              dwc_w[i] += dmemtilde[t][i,j] * (-erases[t][j] * mems[t-1][i,j])
              # and they affect mem through adding
              dwc_w[i] += dmem[t][i,j] * adds[t][j]

          # dwdK_r will be an N by M that holds in the i,j-th spot
          # the deriv of wc_r[j] w.r.t. K_r[i] (there are N Ks)
          """
          We need dw/dK
          now w has N elts and K has N elts
          and we want, for every elt of W, the grad of that elt w.r.t. each
          of the N elts of K. that gives us N * N things
          """
          # first, we must build up the K values (should be taken from fprop)
          K_rs = []
          K_ws = []
          for i in range(self.N):
            K_rs.append(cosine_sim(mems[t-1][i, :], k_rs[t]))
            K_ws.append(cosine_sim(mems[t-1][i, :], k_ws[t]))

          # then, we populate the grads
          dwdK_r = np.zeros((self.N, self.N))
          dwdK_w = np.zeros((self.N, self.N))
          # for every row in the memory
          for i in range(self.N):
            # for every element in the weighting
            for j in range(self.N):
              dwdK_r[i,j] += softmax_grads(K_rs, i, j)
              dwdK_w[i,j] += softmax_grads(K_ws, i, j)

          # compute dK for all i in N
          # K is the evaluated cosine similarity for the i-th row of mem matrix
          dK_r = np.zeros_like(w_rs[0])
          dK_w = np.zeros_like(w_rs[0])

          # for all i in N (for every row that we've simmed)
          for i in range(self.N):
            # for every j in N (for every elt of the weighting)
            for j in range(self.N):
              dK_r += dwc_r[j] * dwdK_r[i,j] # TODO: is order of i and j right?
              dK_w += dwc_w[j] * dwdK_w[i,j]

          """
          dK_r_dk_rs is a list of N things
          each elt of the list corresponds to grads of K_idx
          w.r.t. the key k_t
          so it should be a length N list of M by 1 vectors
          """

          dK_r_dk_rs = []
          dK_r_dmem = []
          for i in range(self.N):
            # let k_rs be u, Mem[i] be v
            u = np.reshape(k_rs[t], (self.M,))
            v = mems[t-1][i, :]
            dK_r_dk_rs.append( dKdu(u,v) )
            dK_r_dmem.append( dKdu(v,u))

          dK_w_dk_ws = []
          dK_w_dmem = []
          for i in range(self.N):
            # let k_ws be u, Mem[i] be v
            u = np.reshape(k_ws[t], (self.M,))
            v = mems[t-1][i, :]
            dK_w_dk_ws.append( dKdu(u,v) )
            dK_w_dmem.append( dKdu(v,u))

          # compute delta for keys
          dk_r = np.zeros_like(k_rs[0])
          dk_w = np.zeros_like(k_ws[0])
          # for every one of M elt of dk_r
          for i in range(self.M):
            # for every one of the N Ks
            for j in range(self.N):
              # add delta K_r[j] * dK_r[j] / dk_r[i]
              # add influence on through K_r[j]
              dk_r[i] += dK_r[j] * dK_r_dk_rs[j][i]
              dk_w[i] += dK_w[j] * dK_w_dk_ws[j][i]

          # these represent influence of mem on next K
          du_r[t] = np.zeros_like(mems[0])
          du_w[t] = np.zeros_like(mems[0])
          # for every row in mems[t-1]
          for i in range(self.N):
            # for every elt of this row (one of M)
            for j in range(self.M):
              du_r[t][i,:] = dK_r[i] * dK_r_dmem[i][j]
              du_w[t][i,:] = dK_w[i] * dK_w_dmem[i][j]

          # key values are activated as tanh
          dzk_r = dk_r * (1 - k_rs[t] * k_rs[t])
          dzk_w = dk_w * (1 - k_ws[t] * k_ws[t])

          deltas['ok_r'] += np.dot(dzk_r, os[t].T)
          deltas['ok_w'] += np.dot(dzk_w, os[t].T)

          deltas['bk_r'] += dzk_r
          deltas['bk_w'] += dzk_w

        else:
          drs[t] = np.zeros_like(rs[0])
          dmemtilde[t] = np.zeros_like(mems[0])
          du_r[t] = np.zeros_like(mems[0])
          du_w[t] = np.zeros_like(mems[0])

        # o affects y through Woy
        do = np.dot(params['oy'].T, dy)
        if t < len(targets) - 1:
          # and also zadd through Woadds
          do += np.dot(params['oadds'].T, dzadd)
          do += np.dot(params['oerases'].T, dzerase)
          # and also through the keys
          do += np.dot(params['ok_r'].T, dzk_r)
          do += np.dot(params['ok_w'].T, dzk_w)


        # compute deriv w.r.t. pre-activation of o
        dzo = do * (1 - os[t] * os[t])

        deltas['ho'] += np.dot(dzo, hs[t].T)
        deltas['bo'] += dzo

        # compute hidden dh
        dh = np.dot(params['ho'].T, dzo)

        # compute deriv w.r.t. pre-activation of h
        dzh[t] = dh * (1 - hs[t] * hs[t])

        deltas['xh'] += np.dot(dzh[t], xs[t].T)
        deltas['bh'] += dzh[t]

        # Wrh affects zh via rs[t-1]
        deltas['rh'] += np.dot(dzh[t], rs[t-1].reshape((self.M, 1)).T)

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
