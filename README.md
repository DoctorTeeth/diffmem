# Differentiable Memory

Reference implementations of various differentiable memory schemes for neural networks, in pure numpy.
These are meant to serve as correctness checks against which higher performance batched GPU implementations can be evaluated.
Very much a WIP.

## Usage

You will need to install the autograd library. 
I'm using autograd to do the backprop, since the architecture of the model is still in flux, but I found (at least when benchmarked using a GRU I have) that it's about 8x slower than doing backprop manually.

## Models

### NTM

Currently the only implemented model.
The code is awful right now, but there is a working NTM. 
It converges consistently on the copy task.
It has nice, easily interpretable memory access patterns, which you can view if you run the code.
It (often) generalizes well to sequences longer than those it was trained on, though sometimes it makes "off by one" mistakes that are close to a correct copy but have high CE errors, as mentioned in the paper.
It's also very slow, but if I had a fast version already then I wouldn't need this one.
