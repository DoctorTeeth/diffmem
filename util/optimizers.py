"""
This module implements gradient-based optimizers.
"""
import autograd.numpy as np

def l2(x):
  """
  Hacky l2-norm computation to be used for tracking update magnitude.
  """
  return np.sqrt(np.sum(np.multiply(x, x)))

class RMSProp(object):

  """
  This class stores RMSProp state and generates RMSProp updates.
  """

  def __init__(self, W, learning_rate=10e-5, decay=0.95, blend=0.95):
    """
    This is the Alex Graves RMSProp variant from
    Generating Sequences with Recurrent Neural Networks.

    It scales parameter updates by a running estimate of the variance
    of the parameter rather than just a running estimate of the magnitude.

    decay governs how fast the momentum falls off.

    blend governs the extent to which we take the current parameter value
      into account when updating our estimate of variance.
    """
    self.lr = learning_rate
    self.d = decay
    self.b = blend

    self.ns  = {} # store the mean of the square
    self.gs  = {} # store the mean, which will later be squared
    self.ms  = {} # momentum
    self.qs  = {} # update norm over param norm - ideally this stays around 10e-3
    for k, v in W.iteritems():
      self.ns[k]  = np.zeros_like(v)
      self.gs[k]  = np.zeros_like(v)
      self.ms[k]  = np.zeros_like(v)
      self.qs[k]  = self.lr

  def update_weights(self, params, dparams):
    """
    params: the parameters of the model as they currently exist.
    dparams: the grad of the cost w.r.t. the parameters of the model.

    We update params based on dparams.
    """
    for k in params.keys():
      p = params[k]
      d = dparams[k]

      d = np.clip(d,-10,10)
      self.ns[k] = self.b * self.ns[k] + (1 - self.b) * (d*d)
      self.gs[k] = self.b * self.gs[k] + (1 - self.b) * d
      n = self.ns[k]
      g = self.gs[k]
      self.ms[k] = self.d * self.ms[k] - self.lr * (d / (np.sqrt(n - g*g + 1e-8)))
      self.qs[k] = self.b * self.qs[k] + (1 - self.b) * (l2(self.ms[k]) / l2(p))
      p += self.ms[k]
