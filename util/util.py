"""
Miscellaneous utility functions.

Includes finite-difference based gradient checking, some nonlinearities,
visualization and a bit of code for dealing with Autograd nodes.
"""
import autograd.numpy as np
import pickle

def gradCheck(model, deltas, inputs, targets, epsilon, tolerance):
  """
  Finite difference based gradient checking.
  """

  diffs = getDiffs(model, deltas, inputs, targets, epsilon)
  answer = True

  for diffTensor, name, delta in zip(diffs, model.W.keys(), deltas.values()):

    if np.abs(diffTensor.max()) >= tolerance:
      print "DIFF CHECK FAILS FOR TENSOR: ", name
      print "DIFF TENSOR: "
      print diffTensor
      print "NUMERICAL GRADIENTS: "
      # diff = grad - delta => diff+delta = grad
      print diffTensor + delta
      print "BPROP GRADIENTS: "
      print delta
      answer = False
    else:
      pass

  return answer

def getDiffs(model, deltas, inputs, targets, epsilon):
  """
  For every (weight,delta) combo in zip(weights, deltas):
    Add epsilon to that weight and compute the loss (first_loss)
    Remove epsilon from that weight and compute the loss (second_loss)
    Check how close (first loss - second loss) / 2h is to the delta from bprop
  """

  diff_tensors = []
  for D in deltas.values():
    diff_tensors.append(np.zeros_like(D))

  for W,D,diffs in zip(model.W.values(), deltas.values(), diff_tensors):
  # for each weight tensor in our model

    i = np.random.randint(W.shape[0])
    j = np.random.randint(W.shape[1])

    # compute f(x+h) for that weight
    W[i,j] += epsilon
    loss, _, _, _, _, _, _  = model.lossFun(inputs, targets, False)
    loss_plus = np.sum(loss)

    # compute f(x - h) for that weight
    W[i,j] -= epsilon*2
    loss, _, _, _, _, _, _ = model.lossFun(inputs, targets, False)
    loss_minus = np.sum(loss)

    # grad check must leave weights unchanged
    # so reset the weight that we changed
    W[i,j] += epsilon

    # compute the numerical grad w.r.t. this param
    grad = (loss_plus - loss_minus) / (2 * epsilon)
    diffs[i,j] = grad - D[i,j]

  return diff_tensors

def rando(out_size,in_size):
  """
  Initialization of weight tensors.
  """
  sigma = np.sqrt( 6.0 / (out_size + in_size))
  return np.random.uniform(-sigma, sigma, (out_size, in_size))

def sigmoid(ys):
  """
  Basic sigmoid nonlinearity.
  Used when output must lie in (0,1)
  """
  return 1 / (1 + np.exp(-ys))

def softmax(xs):
  """
  Initialization of weight tensors.
  Used when outputs must sum to 1.
  """
  n = np.exp(xs)
  d = np.sum(np.exp(xs))
  return n/d

def softplus(xs):
  """
  Softplus nonlinearity.
  Used for the beta values.
  """
  return np.log(1 + np.exp(xs))

def serialize(filename, data):
  """
  Save state of the model to disk.
  """
  print "serializing"
  f = open(filename, 'w')
  pickle.dump(data,f)
  f.close()

def deserialize(filename):
  """
  Read state of the model from disk.
  """
  print "deserializing"
  f = open(filename, 'r')
  result = pickle.load(f)
  f.close()
  return result

def toArray(dic,h,w):
  """
  Convert dicts of arrays to arrays for the visualize function.
  """
  outList = []
  for k in dic:
    if k != -1:
      outList.append(dic[k].tolist())
  outC = np.round(np.array(outList),2)
  outC = outC.T
  out = np.reshape(outC,(h,w))
  return out

def visualize(inputs, outputs, reads, writes, adds, erases):
  """
  Print out some summary of what the NTM did for a given sequence.
  """
  wi = inputs.shape[0]
  hi = outputs[0].shape[0]
  np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=150)
  out = toArray(outputs, hi, wi)
  ins = np.array(inputs.T,dtype='float')
  print "inputs: "
  print ins
  print "outputs: "
  print out
  print "reads"
  print reads
  print "writes"
  print writes
  print "adds"
  print adds
  print "erases"
  print erases

def unwrap(x):
  """
  Strip numpy values out of collections of Autograd values.
  """
  if isinstance(x,dict):
    r = {}
    for k, v in x.iteritems():
      if hasattr(v,'value'):
        r[k] = v.value
      else:
        r[k] = v
    return r
  elif all(isinstance(elt, dict) for elt in x):
    l = []
    for d in x:
      r = {}
      for k, v in d.iteritems():
        if hasattr(v,'value'):
          r[k] = v.value
        else:
          r[k] = v
      l.append(r)
    return l
  elif type(x) == np.numpy_extra.ArrayNode:
    return x.value
  else:
    return x

def sigmoid_prime(z):
    y = sigmoid(z)
    return y * (1 - y)

def tanh_prime(z):
    y = np.tanh(z)
    return 1 - y * y

def compare_deltas(baseline=None, candidate=None, abs_tol=1e-5, rel_tol=0.01):
  # TODO: maybe add relative tolerance check
  epsilon = 1e-25
  if baseline.shape != candidate.shape:
    return False
  diff_tensor = np.abs(baseline - candidate)
  rel_tensor1 = diff_tensor / (np.abs(baseline) + 1e-25)
  rel_tensor2 = diff_tensor / (np.abs(candidate) + 1e-25)
  max_error = np.max(diff_tensor)
  max_rel = max(np.max(rel_tensor1), np.max(rel_tensor2))
  if max_error > abs_tol and max_rel > rel_tol:
    return False
  else:
    return True

def cosine_sim(a_t, b_t):
    """
    Computes the cosine similarity of vectors a and b.
    Specifically \frac{u \cdot v}{||u|| \cdot ||v||}.
    """
    # numerator is the inner product
    num = np.dot(a_t, b_t)

    # denominator is the product of the norms
    anorm = np.sqrt(np.sum(a_t*a_t))
    bnorm = np.sqrt(np.sum(b_t*b_t))
    den2 = (anorm * bnorm) + 1e-20

    return num / den2

def dKdu(u, v):
  """
  compute the grads of a given K w.r.t. u
  you can just switch order of args to compute it for v
  """
  anorm = np.sqrt(np.sum(u*u))
  bnorm = np.sqrt(np.sum(v*v))
  den2 = (anorm * bnorm) + 1e-20 

  a = v / den2
  b = u / np.sum(np.square(u))
  c = cosine_sim(u,v)
  return a - b*c

def softmax_grads(Ks, beta, i, j):
  """
  return the grad of the ith element of weighting w.r.t. j-th element of Ks
  """
  if j == i:
    num = beta*np.exp(Ks[i]*beta) * (np.sum(np.exp(Ks*beta)) - np.exp(Ks[i]*beta))
  else:
    num = -beta*np.exp(Ks[i]*beta + Ks[j]*beta)
  den1 = np.sum(np.exp(Ks*beta))
  return num / (den1 * den1)

def beta_grads(Ks, beta, i):
  Karr = np.array(Ks)
  anum = Ks[i] * np.exp(Ks[i] * beta)
  aden = np.sum(np.exp(beta * Karr))
  a = anum / aden

  bnum = np.exp(Ks[i] * beta) * (np.sum(np.multiply(Karr, np.exp(Karr * beta))))
  bden = aden * aden
  b = bnum / bden
  return a - b

def K_focus(Ks, b_t):
    """
    The content-addressing method described in 3.3.1.
    Specifically, this is equations (5) and (6).
    k_t is the similarity key vector.
    b_t is the similarity key strength.
    memObject is a ref to our NTM memory object.
    """
    def F(K):
        """
        Given the key vector k_t, compute our sim
        function between k_t and u and exponentiate.
        """
        return np.exp(b_t * K)

    # Apply above function to every row in the matrix
    # This is surely much slower than it needs to be
    l = []
    for K in Ks:
        l.append(F(K))

    # Return the normalized similarity weights
    # This is essentially a softmax over the similarities
        # with an extra degree of freedom parametrized by b_t
    sims = np.array(l)

    n = sims
    d = np.sum(sims)
    return n/d
