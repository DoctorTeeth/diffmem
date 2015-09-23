"""
Generate random instances of various artificial tasks.
Right now, only a bit-vector copy task is implemented.
"""
import autograd.numpy as np

def copy_sequence(seq_length, vec_size):
  """
  Returns inputs, outputs
  where inputs is a length 2 * seq_length + 2 sequence of vec_size + 2 vecs
  and outputs is a length 2 * seq_length + 2 sequence of vec_size vecs
  inputs starts with a start bit, then the seq to be copied, then and end bit, then 0s 
  outputs is 0s until after inputs has the end bit, then it's the first sequence, but without 
  the extra bits
  """
  input_size = vec_size + 2
  length  = seq_length * 2 + 2
  inputs  = np.zeros((length,input_size),dtype=np.uint8)
  outputs = np.zeros((length,vec_size),dtype=np.uint8)

  in_sequence = np.random.randint(2, size=(seq_length, input_size))
  in_sequence[:,-2:] = 0
  out_sequence = in_sequence[:,:-2]

  # set start bit in inputs
  start_vec = np.zeros(input_size)
  start_vec[-2] = 1 
  inputs[0] = start_vec

  # set the pattern bits in inputs
  inputs[1:seq_length+1] = in_sequence

  # set stop bit in inputs
  stop_vec = np.zeros(input_size)
  stop_vec[-1] = 1 
  inputs[seq_length+1] = stop_vec

  # set all the bits in outputs
  outputs[seq_length+2:] = out_sequence
  return inputs, outputs

def easy_copy(seq_length, vec_size):
  """
  Returns inputs, outputs
  where inputs is a length 2 * seq_length + 2 sequence of vec_size + 2 vecs
  and outputs is a length 2 * seq_length + 2 sequence of vec_size vecs

  This task is the same as the copy task, but either the input bit or the output
  bit is always on, so that the NTM doesn't have to remember whether it needs to
  be reading or writing. Thus, this task is strictly easier than the copy task.
  """
  inputs, outputs = copy_sequence(seq_length, vec_size)
  inputs[1:seq_length,-2] = 1
  inputs[seq_length+2:-1,-1] = 1
  return inputs, outputs
