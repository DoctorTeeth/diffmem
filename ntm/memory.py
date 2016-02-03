"""
Implements functions for reading and writing from and to the memory bank
via weight vectors as in section 3.
"""

import autograd.numpy as np

def read(mem, weight_vector):
    """
    The reading procedure as described in 3.1.
    weight_vector (w_t below) is a weighting over the rows such that:
        1. \sum_i w_t(i) = 1
        2. 0 \leq w_t(i) \leq 1, \forall i
    We return \sum_i w_t(i) M_t(i); M_t(i) is the i-th row of memory.
    """

    return np.dot(weight_vector.T, mem)

def write(mem, w_t, e_t, a_t):
    """
    The writing procedure as described in 3.2.
    w_t is a length N weighting over the rows as above.
    e_t (the erase vector) is length M with elements all in (0,1).
    a_t (the add vector) is length M with no such restrictions.
    We first multiply the memory matrix pointwise by [1-w_t(i)e_t]
    Then we do M_t(i) <- w_t(i)a_t.
    According to the paper, the erase/add decomposition was
        inspired by the forget/input gates in LSTM.
    """
    # Perform erasure on the existing memory, parametrized by e_t and w_t
    W = np.reshape(w_t, (w_t.shape[0], 1))
    E = np.reshape(e_t, (e_t.shape[0], 1))

    # Transpose W so we can create WTE, a matrix whose i,j-th element
        # represents the extent to which we will erase M_t[i,j]
    WTE = np.dot(W, E.T)

    # KEEP is such that KEEP[i,j] represents the extent to which we
        # will keep M_t[i,j]
    KEEP = np.ones(mem.shape) - WTE

    # To complete erasure, multiply memory pointwise by KEEP
    newmem = np.multiply(mem, KEEP)

    # Perform addition on the newly erased memory
    # Convert add vector to a matrix
    A = np.reshape(a_t, (a_t.shape[0], 1))

    # Add is the add vector weighted by w_t, which is added pointwise to
        # the existing memory, finishing the write sequence.
    ADD = np.dot(W, A.T)
    newmem = newmem + ADD

    return newmem
