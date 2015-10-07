# Differentiable Memory

Reference implementations of various differentiable memory schemes for neural networks, in pure numpy.
These are meant to serve as correctness checks against which higher performance batched GPU implementations can be evaluated.

## Setup

You will need to install the autograd library (`pip install autograd`). 
I'm using autograd to do the backprop, since the architecture of the models is still in flux, but I found (at least when benchmarked using a GRU I have) that it's about 8x slower than doing backprop manually, so once I'm sure that everything is working properly I'll probably get rid of that dependency.

## Models

Currently, only the Neural Turing Machine is implemented.
I'd also like to implement some of the stuff from [Learning to Transduce with Unbounded Memory](http://arxiv.org/pdf/1506.02516v1.pdf), but it's unclear whether I'll have time (or whether it will be worth doing, given the generality of NTMs).

### Neural Turing Machine

This is an instance of the model [from this paper](http://arxiv.org/pdf/1410.5401v2.pdf).
The paper was relatively vague about the architecture of the model, so I had to make some stuff up.
It's likely that some of the stuff I made up is dumb, and I'd appreciate feedback to that effect.

All 5 of the tasks from the paper are implemented. There's more detail in the below sections about performance on individual tasks, but broadly speaking the model seems to be working; it generalizes to sequences longer than those it was trained on (e.g. it learns "compact internal programs")

#### Usage

The script `run_model.py` handles both training and testing.
Execute `./run_model.py --help` to get a list of arguments and their descriptions.
The saved_models directory contains pickled models for every task - at present, only some of them are good.
To test a copy model on 

#### Tasks

#### Miscellaneous Notes

* The controller is just a single FF layer with a tanh activation. All heads take input straight from the output of the controller layer.

* Restricting the range of allowed integer shifts to [-1,0,1] seems totally essential for length-generalization in the copy task. It's possible that this restriction (which currently I've hardcoded into the model), is harming performance on some of the other tasks.

* Gradients are clipped to (-10,10), as advised in the paper.

* The initial state of the memory and the initial state of the last read, the last write weight, and the last read weight are actually learnable parameters of the model, randomly intialized as are the rest of the weights. This modification greatly improved training performance.

* The initialization is currently being done wrong. Where multiple inputs feed into an activation, we should be concatenating the weight vectors that generate those inputs before intializing. Fixing this is high on my todo-list, as I expect it's seriously holding back multi-head performance.

* Performance of the code is poor, and this is mostly my fault. The focus of this project is on providing a reliable, deterministic reference implementation, so I'm not going to put a huge amount of effort into speeding up this code. That being said, I'm sure some of the stuff I'm doing is clearly (to some people), ridiculous from a performance perspective - please tell me if you see anything like this.
