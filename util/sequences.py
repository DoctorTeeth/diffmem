"""
Generate random instances of various artificial tasks.
Right now, only a bit-vector copy task is implemented.
"""
import autograd.numpy as np
import pdb

def copy_sequence(seq_len, vec_size):
  """
  Returns inputs, outputs
  where inputs is a length 2 * seq_len + 2 sequence of vec_size + 2 vecs
  and outputs is a length 2 * seq_len + 2 sequence of vec_size vecs
  inputs starts with a start bit, then the seq to be copied, then and end bit, then 0s 
  outputs is 0s until after inputs has the end bit, then it's the first sequence, but without 
  the extra bits
  """
  input_size = vec_size + 2
  length  = seq_len * 2 + 2
  inputs  = np.zeros((length,input_size),dtype=np.uint8)
  outputs = np.zeros((length,vec_size),dtype=np.uint8)

  in_sequence = np.random.randint(2, size=(seq_len, input_size))
  in_sequence[:,-2:] = 0
  out_sequence = in_sequence[:,:-2]

  # set start bit in inputs
  start_vec = np.zeros(input_size)
  start_vec[-2] = 1
  inputs[0] = start_vec

  # set the pattern bits in inputs
  inputs[1:seq_len+1] = in_sequence

  # set stop bit in inputs
  stop_vec = np.zeros(input_size)
  stop_vec[-1] = 1
  inputs[seq_len+1] = stop_vec

  # set all the bits in outputs
  outputs[seq_len+2:] = out_sequence
  return inputs, outputs, seq_len

def easy_copy(seq_len, vec_size):
  """
  Returns inputs, outputs
  where inputs is a length 2 * seq_len + 2 sequence of vec_size + 2 vecs
  and outputs is a length 2 * seq_len + 2 sequence of vec_size vecs

  This task is the same as the copy task, but either the input bit or the output
  bit is always on, so that the NTM doesn't have to remember whether it needs to
  be reading or writing. Thus, this task is strictly easier than the copy task.
  """
  inputs, outputs = copy_sequence(seq_len, vec_size)
  inputs[1:seq_len,-2] = 1
  inputs[seq_len+2:-1,-1] = 1
  return inputs, outputs, seq_len

def repeat_copy(seq_len, vec_size, repeats):
  """
  Returns inputs, outputs

  Inputs consists of a sequence of length seq-length,
  followed by a scalar repeat count in another channel.

  After the input sequence, we get a scalar on the scalar channel,
  and we need to copy the sequence that number of times and emit the end marker.
  """
  r = repeats
  input_size  = vec_size + 2
  output_size = vec_size + 1
  length  = seq_len * (r+1) + 3
  inputs  = np.zeros((length,input_size),dtype=np.float32)
  outputs = np.zeros((length,output_size),dtype=np.float32)

  in_sequence = np.random.randint(2, size=(seq_len, input_size))
  in_sequence[:,-2:] = 0

  out_sequence = in_sequence[:,:-2]

  # set start bit in inputs
  start_vec = np.zeros(input_size)
  start_vec[-2] = 1
  inputs[0] = start_vec

  # set the pattern bits in inputs
  inputs[1:seq_len+1] = in_sequence

  # set repeat value
  stop_vec = np.zeros(input_size)
  stop_vec[-1] = r
  inputs[seq_len+1] = stop_vec

  # set all the bits in outputs
  for i in range(0,r):
    a = (i+1)*seq_len + 2
    b = a + seq_len
    outputs[a:b,0:vec_size] = out_sequence

  # set the finished bit in the target section
  finish_vec = np.zeros(output_size)
  finish_vec[-1] = 1
  outputs[b] = finish_vec
  return inputs, outputs, seq_len

def associative_recall(seq_len, vec_size, item_size):
  """
  We show between 2 and 6 items (seq len), each of which is 3 six bit binary vectors
  For i in range(seq_len):
    show a start bit, then item_size vectors
  then show a fetch bit, then item_size vectors
  then a final fetch bit
  then we expect item_size vectors to be reproduced

  so our total length is (seq_len+1)*(item_size+1) + 1 + seq_len
  """
  input_size  = vec_size + 2
  output_size = vec_size
  length  = (seq_len+1)*(item_size+1) + 1 + seq_len
  inputs  = np.zeros((length,input_size),dtype=np.float32)
  outputs = np.zeros((length,output_size),dtype=np.float32)

  start_bit = np.zeros((1,input_size))
  start_bit[0,-2] = 1
  fetch_bit = np.zeros((1,input_size))
  fetch_bit[0,-1] = 1

  # generate seq_len random items
  items = []
  for i in range(seq_len):
    items.append(np.random.randint(2, size=(item_size, vec_size)))
    a = i*(item_size+1)
    b = a + item_size
    inputs[a] = start_bit
    inputs[a+1:b+1,:-2] = items[i] 

  # pick an item at random that isn't the last item
  idx = np.random.randint(low=0, high=len(items) - 1) 
  fetch_item = items[idx]

  inputs[b+1] = fetch_bit
  inputs[b+2:b+2+item_size, :-2] = fetch_item
  inputs[b+2 + item_size] = fetch_bit
  # choose a random item to be the prompt

  outputs[-items[idx+1].shape[0]:inputs.shape[0]] = items[idx+1]

  return inputs, outputs, seq_len

def priority_sort(seq_len, vec_size):
  """
  We show seq_len vectors of vec_size,
  each with a scalar priority between -1 and 1
  the expected output then is the sorted sequence of vectors
  so the total length is 1 bit for start bit, 1 for stop, then
  2x the seq_len
  """
  input_size  = vec_size + 1
  output_size = vec_size
  length  = 2*seq_len + 2
  inputs  = np.zeros((length,input_size),dtype=np.float32)
  outputs = np.zeros((length,output_size),dtype=np.float32)

  start_bit = np.zeros((1,input_size))
  start_bit[0,-2] = 1
  stop_bit = np.zeros((1,input_size))
  stop_bit[0,-1] = 1

  inputs[0] = start_bit
  inputs[seq_len + 1] = stop_bit
  # now randomly generate seq_len vectors with their
  # priorities
  items = []
  for i in range(0,seq_len):
    seq = np.random.randint(2, size=(1, vec_size))
    priority = np.random.uniform(low=-1, high=1)
    items.append((seq, priority))
    inputs[i+1,:-1] = seq
    inputs[i+1,-1] = priority

  items.sort(key=lambda x: x[1])

  # fill in the outputs
  for i in range(0,seq_len):
    seq, _ = items[i]
    outputs[seq_len + 2 + i] = seq

  return inputs, outputs, seq_len

def ngrams(seq_len, vec_size):
  """
  this is the dynamic n-grams task
  """
  input_size  = vec_size + 1
  output_size = vec_size
  length  = 2*seq_len + 2
  inputs  = np.zeros((length,input_size),dtype=np.float32)
  outputs = np.zeros((length,output_size),dtype=np.float32)

  start_bit = np.zeros((1,input_size))
  start_bit[0,-2] = 1
  stop_bit = np.zeros((1,input_size))
  stop_bit[0,-1] = 1

  inputs[0] = start_bit
  inputs[seq_len + 1] = stop_bit
  # now randomly generate seq_len vectors with their
  # priorities
  items = []
  for i in range(0,seq_len):
    seq = np.random.randint(2, size=(1, vec_size))
    priority = np.random.uniform(low=-1, high=1)
    items.append((seq, priority))
    inputs[i+1,:-1] = seq
    inputs[i+1,-1] = priority

  items.sort(key=lambda x: x[1])

  # fill in the outputs
  for i in range(0,seq_len):
    seq, _ = items[i]
    outputs[seq_len + 2 + i] = seq

  return inputs, outputs, seq_len

class SequenceGen(object):

  # TODO: add last 3 tasks to this
  def __init__(self, sequenceType, vec_size, hi, lo):
    if sequenceType == 'copy':
      self.out_size = vec_size
      self.in_size  = vec_size + 2
      def make():
        seq_len = np.random.randint(lo, hi + 1)
        return copy_sequence(seq_len, vec_size)
      self.make = make
    elif sequenceType == 'repeat_copy':
      self.out_size = vec_size + 1
      self.in_size  = vec_size + 2
      def make():
        seq_len = np.random.randint(lo, hi + 1)
        repeats = np.random.randint(lo, hi + 1)
        return repeat_copy(seq_len, vec_size, repeats)
      self.make = make
    elif sequenceType == 'associative_recall':
      self.out_size = vec_size
      self.in_size  = vec_size + 2
      item_size = 3 # this is hardcoded for now
      def make():
        seq_len = np.random.randint(lo, hi + 1)
        return associative_recall(seq_len, vec_size, item_size)
      self.make = make
    else:
      raise NotImplementedError

# i, t, l = ngrams(2, 3)
# print i
# print t
