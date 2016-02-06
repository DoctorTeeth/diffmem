import math
import autograd.numpy as np
from autograd import grad
import memory
import addressing
import sys

"""
Testing our hand-computed derivatives for cosine sim

cos_sim(u,k) where k is the similarity key and u is an elt of the matrix
we want the grad of cosine sim w.r.t an element of u, though grad
w.r.t. an element of k is computed the same regardless of which elt of k,
and actually it is the same for u as well, since cos_sim is symmetric about
its arguments
"""

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
    den2 = (anorm * bnorm) + 1e-5

    return num / den2

if __name__ == "__main__":
    M = 5
    u = np.random.uniform(high=1, low=-1, size=(M,))
    v = np.random.uniform(high=1, low=-1, size=(M,))
    print cosine_sim(u,v)

    # compute deltas automatically
    # just with respect to u
    cs_grad = grad(cosine_sim, argnum=0)
    auto_deltas = cs_grad(u,v)

    # compute deltas manually
    manual_deltas = np.zeros_like(auto_deltas)

    # compute the denominator
    anorm = np.sqrt(np.sum(u*u))
    bnorm = np.sqrt(np.sum(v*v))
    den2 = (anorm * bnorm) + 1e-5

    a = v / den2
    b = u / np.sum(np.square(u))
    c = cosine_sim(u,v)
    manual_deltas = a - b*c
    

    print "auto deltas"
    print auto_deltas
    print "manual deltas"
    print manual_deltas

    """
    manual_deltas gives us dk_i / dK_j
    """

