import math
import autograd.numpy as np
from autograd import grad
import memory
import addressing
import sys

"""
Testing our hand-computed derivatives

compute the grad of the Ks w.r.t. each of the outputs of the softmax
"""

def softmax_0(Ks):
    return np.exp(Ks[0]) / np.sum(np.exp(Ks))


if __name__ == "__main__":
    length = 5
    Ks = np.random.uniform(high=1, low=-1, size=(length,1))
    print softmax_0(Ks)

    # compute deltas automatically
    sm_grad = grad(softmax_0)
    auto_deltas = sm_grad(Ks)
    # compute the deltas manually
    manual_deltas = np.zeros_like(auto_deltas)

    this_i = 0
    for j in range(length):
        if j == this_i:
            num = np.exp(Ks[this_i]) * (np.sum(np.exp(Ks)) - np.exp(Ks[this_i]))
        else:
            num = -np.exp(Ks[this_i] + Ks[j])
        den1 = np.sum(np.exp(Ks))
        manual_deltas[j] = num / (den1 * den1)

    print "auto deltas"
    print auto_deltas
    print "manual deltas"
    print manual_deltas
