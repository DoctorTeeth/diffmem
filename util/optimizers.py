import autograd.numpy as np

def l2(x):
  return np.sqrt(np.sum(np.multiply(x,x)))

class RMSProp(object):

  def __init__(self, weights, learning_rate=10e-5, decay=0.95, blend=0.95):
    """
    learning rate governs how much we use the computed grad
    decay governs how quickly accumulated momentum drops off
    blend governs how the running estimate of E(x)^2 and E(x)
      are updated
    """
    self.lr = learning_rate
    self.d = decay
    self.b = blend

    self.ns = []
    self.gs = []
    self.ms = [] # momentum
    self.qs = [] # update norm over param norm - want this to stay around 10e-3
    self.lrs = []
    for tensor in weights:
      self.ns.append(np.zeros_like(tensor))
      self.gs.append(np.zeros_like(tensor))
      self.ms.append(np.zeros_like(tensor))
      self.qs.append(self.lr)
      self.lrs.append(self.lr)

  def update_weights(self, params, dparams):
    for p, d, i in zip(params,
                       dparams,
                       range(0,len(params))):
      d = np.clip(d,-10,10)
      self.ns[i] = self.b * self.ns[i] + (1 - self.b) * (d*d)
      self.gs[i] = self.b * self.gs[i] + (1 - self.b) * d
      n = self.ns[i]
      g = self.gs[i]
      self.ms[i] = self.d * self.ms[i] - self.lrs[i] * (d / (np.sqrt(n - g*g + 1e-8)))
      self.qs[i] = self.b * self.qs[i] + (1 - self.b) * (l2(self.ms[i]) / l2(p))
      p += self.ms[i]
