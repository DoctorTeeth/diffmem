import autograd.numpy as np

def l2(x):
  return np.sqrt(np.sum(np.multiply(x,x)))

class RMSProp(object):

  def __init__(self, W, learning_rate=10e-5, decay=0.95, blend=0.95):
    """
    learning rate governs how much we use the computed grad
    decay governs how quickly accumulated momentum drops off
    blend governs how the running estimate of E(x)^2 and E(x)
      are updated
    """
    self.lr = learning_rate
    self.d = decay
    self.b = blend

    self.ns  = {}
    self.gs  = {}
    self.ms  = {} # momentum
    self.qs  = {} # update norm over param norm - want this to stay around 10e-3
    for k, v in W.iteritems():
      self.ns[k]  = np.zeros_like(v)
      self.gs[k]  = np.zeros_like(v)
      self.ms[k]  = np.zeros_like(v)
      self.qs[k]  = self.lr

  def update_weights(self, params, dparams):
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
