import autograd.numpy as np
import pickle
import pdb

def gradCheck(model, deltas, inputs, targets, epsilon, tolerance):

  diffs = getDiffs(model, deltas, inputs, targets, epsilon)
  answer = True

  for diffTensor, name, delta in zip(diffs, model.names, deltas):

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
  for D in deltas:
    diff_tensors.append(np.zeros_like(D))

  for W,D,N,diffs in zip(model.weights, deltas, model.names, diff_tensors):
  # for each weight tensor in our model

    for i in range(W.shape[0]):
      for j in range(W.shape[1]):
        # for each weight in that tensor

        # compute f(x+h) for that weight
        W[i,j] += epsilon
        loss, ds, os, _, _, _, _  = model.lossFun(inputs, targets, False)
        loss_plus = np.sum(loss)

        # compute f(x - h) for that weight
        W[i,j] -= epsilon*2
        loss, ds, os, _, _, _, _ = model.lossFun(inputs, targets, False)
        loss_minus = np.sum(loss)

        # grad check must leave weights unchanged
        # so reset the weight that we changed
        W[i,j] += epsilon

        # compute the numerical grad w.r.t. this param
        grad = (loss_plus - loss_minus) / (2 * epsilon) 
        diffs[i,j] = grad - D[i,j] 

  return diff_tensors 

def rando(out_size,in_size):
  sigma = np.sqrt( 6.0 / (out_size + in_size))
  return np.random.uniform(-sigma, sigma, (out_size, in_size))

def sigmoid(ys):
  return 1 / (1 + np.exp(-ys))

def softmax(xs):
  n = np.exp(xs)
  d = np.sum(np.exp(xs))
  return n/d

def softplus(xs):
  return np.log(1 + np.exp(xs))

def serialize(filename, data):
  print "serializing"
  f = open(filename, 'w')
  pickle.dump(data,f)
  f.close()

def deserialize(filename):
  print "deserializing"
  f = open(filename, 'r')
  result = pickle.load(f)
  f.close()
  return result

def toArray(dic,h,w):
  outList = []
  for k in dic:
    if k != -1:
      outList.append(dic[k].tolist())
  outC = np.round(np.array(outList),2)
  outC = outC.T
  out = np.reshape(outC,(h,w))
  return out

# TODO: make this work on the new heads setup
def visualize(inputs, outputs, reads, writes, adds, erases):
  wi = inputs.shape[0]
  hi = outputs[0].shape[0]
  np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, linewidth=150)
  # pdb.set_trace()
  out = toArray(outputs, hi, wi)
  ins = np.array(inputs.T,dtype='float')
  heads = len(reads)
  r,w,a,e = {},{},{},{}
  for idx in range(heads):
    r[idx] = toArray(reads[idx]  , reads[0][0].shape[0] , wi)
    w[idx] = toArray(writes[idx] , writes[0][0].shape[0] , wi)
    a[idx] = toArray(adds[idx]   , adds[0][0].shape[0] , wi)
    e[idx] = toArray(erases[idx] , erases[0][0].shape[0] , wi)
  print "inputs: "
  print ins
  print "outputs: "
  print out
  for idx in range(heads):
    print "reads" + str(idx)
    print r[idx]
    print "writes" + str(idx)
    print w[idx]
    print "adds" + str(idx)
    print a[idx]
    print "erases" + str(idx)
    print e[idx]

def unwrap(x):
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
  else:
    return x.value

