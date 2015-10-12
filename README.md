# Differentiable Memory

Reference implementations of various differentiable memory schemes for neural networks, in pure numpy.
These are meant to serve as correctness checks against which higher performance batched GPU implementations can be evaluated.

## Code Structure

The code is structured as follows.
The ntm directory has 3 files of note: ntm.py, addressing.py, and memory.py.

* `addressing.py` deals with creating the memory read and write weights from the head parameters.
* `memory.py` deals with how the memory is addressed given the read and write weights.
* `ntm.py` implements fprop and bprop for the ntm model.

The utils directory also has 3 files of note.

* `optimizers.py` implements RMSprop
* `util.py` is a grab-bag of helper functions.
* `sequences.py` generates syntetic test and training data for the tasks from the paper.

The root directory contains dotfiles and `run_model.py`, the latter of which allows for training and testing of the ntm model on various tasks depending on the arguments passed to it.

## Setup

You will need to install the autograd library (`pip install autograd`). 
I'm using autograd to do the backprop, since the architecture of the models is still in flux, but I found (at least when benchmarked using a GRU I have) that it's about 8x slower than doing backprop manually (maybe this is operator error?), so once I'm sure that everything is working properly I'll probably get rid of that dependency.
The code is Python 2.7.

## Models

Currently, only the Neural Turing Machine is implemented.
I'd also like to implement some of the stuff from [Learning to Transduce with Unbounded Memory](http://arxiv.org/pdf/1506.02516v1.pdf), but it's unclear whether I'll have time (or whether it will be worth doing, given the generality of NTMs).

### Neural Turing Machine

This is an instance of the model from [this paper](http://arxiv.org/pdf/1410.5401v2.pdf).
The paper was relatively vague about the architecture of the model, so I had to make some stuff up.
It's likely that some of the stuff I made up is dumb, and I'd appreciate feedback to that effect.

All 5 of the tasks from the paper are implemented. There's more detail in the below sections about performance on individual tasks, but broadly speaking the model seems to be working; it generalizes to sequences longer than those it was trained on (i.e. it learns "compact internal programs").

#### Usage

The script `run_model.py` handles both training and testing.
Execute `./run_model.py --help` to get a list of arguments and their descriptions.
The saved_models directory contains pickled models for every task.
To test the saved copy model on sequences of length 10, do `./run_model.py --model=saved_models/copy.pkl --lo=10 --hi=10`.
Serialized models include information about about which input size and output size and number of hidden units etc were used, so you don't need to remember that.

#### Tasks

In some cases, a task as described by the paper wasn't completely well-defined.
In those cases I took some creative liberty - I think that the spirit of the tasks is the same.

##### Copy

In this task, we feed the model a start bit, then some bit vectors, then a stop bit, and ask it to reproduce all the bit vectors in the same order in which it saw them.
The model performs very well on this task, acheiving negligible error on the training set and generalizing well to sequences longer than those seen during training.
Below is logging output from a model that was trained on sequences of up to length 5 copying a sequence of length 10.
Time flows left to right - the memory of this model has 15 locations, each of size 7. It uses only one read head and one write head.
The 4th row in the input is the channel containing the start bit. The 5th row contains the stop bit. The first 3 rows constitute the bit vectors to be copied.

    inputs:
    [ 0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    outputs:
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  1.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  1.0]
    reads-0
    [ 0.0  0.1  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.1  0.1  0.4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.1  0.4  0.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.2  0.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.1  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0]
    [ 0.1  0.1  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0]
    [ 0.1  0.0  0.0  0.1  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.1  0.0  0.9  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.1  0.0  0.7  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0]
    writes-0
    [ 0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.3  0.2]
    [ 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.4  0.7  0.5]
    [ 0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.6  0.0  0.3]
    [ 0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.9  0.4  0.5  0.0  0.0  0.0]
    [ 0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.8  0.0  0.5  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.5  0.9  0.0  0.1  0.0  0.0  0.0  0.0  0.0]
    [ 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.5  0.0  0.1  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    adds-0
    [-1.0  0.5  0.9 -0.8 -1.0  0.5  0.9  0.1  0.1  0.1  0.9 -1.0  0.4  0.1  0.1  0.0  0.4  0.1  1.0  1.0  1.0  0.1]
    [-0.9 -1.0 -0.9  1.0  0.9 -1.0 -0.9 -1.0 -1.0  1.0 -0.9 -1.0  1.0  1.0 -1.0  0.2  1.0  1.0  1.0  1.0  1.0  1.0]
    [-1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0 -1.0 -0.8  1.0  1.0 -1.0 -1.0 -1.0  1.0 -1.0]
    [ 1.0 -0.8  1.0  1.0 -0.7 -0.9  1.0  1.0  1.0 -0.9  1.0  1.0 -1.0 -1.0 -1.0  1.0 -1.0 -1.0 -1.0 -1.0 -0.5 -1.0]
    [-1.0 -1.0 -1.0  0.6  0.8 -1.0 -1.0 -1.0 -1.0  0.9 -1.0  1.0  1.0  1.0 -0.7 -1.0  1.0  1.0  1.0  1.0  1.0  1.0]
    [-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0  1.0  0.9 -1.0 -1.0  1.0  1.0  1.0 -1.0  1.0]
    [ 1.0  1.0  1.0 -0.7 -0.8  1.0  1.0 -0.9 -0.9  1.0  1.0  1.0 -1.0 -1.0 -0.1  1.0 -1.0 -1.0 -0.4 -0.3 -1.0 -1.0]
    erases-0
    [ 1.0  1.0  1.0  1.0  0.9  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.8  1.0  0.0  0.0  0.0  0.0  0.1  0.0]
    [ 1.0  0.9  0.9  0.1  0.5  0.9  0.9  0.9  0.9  0.3  0.9  0.9  0.0  0.0  0.5  0.9  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.7  0.9  0.9  0.0  0.0  0.9  0.9  0.1  0.1  0.8  0.9]
    [ 1.0  1.0  1.0  0.6  0.8  1.0  1.0  0.9  0.9  0.9  1.0  0.6  0.1  0.0  0.1  0.9  0.1  0.0  0.0  0.1  0.1  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  1.0  0.0  0.0  0.5  1.0  0.2  0.2  0.0  1.0]
    [ 0.0  1.0  1.0  0.7  0.1  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.6  0.0  0.4  1.0  0.6  0.0  0.4  0.5  1.0  0.0]
    [ 0.5  0.0  0.0  0.3  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.4  0.8  0.9  0.0  0.0  0.8  0.9  0.1  0.1  0.8  0.9]

There are a few cool things to note about this example.
The first cool thing is that the error is so low. 
I've rounded the numbers to 1 digit after the decimal, but the actual BPC for this sequence was less than 2e-8.
This is especially interesting because we never trained the model on sequences of this length.
In fact, this particular model will perform well up to length 13, after which it starts running into trouble.
My best guess is that the NTM needs a few extra rows of memory for some reason, but I don't feel very strongly about that hypothesis.

The second cool thing is the memory access pattern.
The NTM starts writing the bit vectors at memory location 6 (counting from 1), and shifts backwards by one location every time step until it sees the stop bit.

Once the stop bit is read, the NTM then starts reading from the corresponding locations, after (presumably), doing a content-based shift back to the beginning of the sequence in memory.

Note that read and write shifts just wrap around - the memory is treated as circular.

##### Repeat Copy

This task is the same as the copy task, but instead of a stop bit, the model sees a scalar, which represents the number of copies to be made.
After the required number of copies is made, the model is also supposed to output a stop bit on a separate channel.
I don't normalize the repeat-scalar, though it's normalized in the paper.
I expect that this is seriously hurting generalization performance; fixing it is on my todo-list.

Here is the logging output of a small example model:

    inputs:
    [ 0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  2.0  0.0  0.0  0.0  0.0  0.0]
    outputs:
    [ 0.0  0.0  0.0  0.0  0.0  1.0  0.0  1.0  0.0]
    [ 0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0]
    [ 0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0]
    reads-0
    [ 0.2  0.0  0.0  0.0  0.0  0.2  0.0  0.2  0.1]
    [ 0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.1  0.2]
    [ 0.3  0.6  1.0  0.0  0.0  0.1  0.0  0.3  0.2]
    [ 0.2  0.2  0.0  1.0  0.0  0.7  0.0  0.5  0.2]
    [ 0.1  0.0  0.0  0.0  1.0  0.0  0.9  0.0  0.3]
    writes-0
    [ 0.0  0.0  0.1  0.4  0.1  0.2  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.3  0.4  0.3  0.2  0.2  0.1]
    [ 0.0  0.0  0.0  0.0  0.3  0.3  0.3  0.2  0.4]
    [ 0.9  0.6  0.0  0.2  0.0  0.1  0.3  0.3  0.1]
    [ 0.1  0.4  0.9  0.1  0.2  0.1  0.1  0.2  0.3]
    adds-0
    [ 1.0  1.0  0.8  1.0 -1.0 -1.0 -1.0 -1.0 -1.0]
    [-1.0 -1.0 -1.0 -1.0  1.0  1.0  1.0  1.0 -0.4]
    [-1.0 -1.0  0.9  0.5 -0.9  0.2 -1.0  0.3 -0.9]
    [ 1.0  0.9  1.0  1.0  1.0  1.0  1.0  0.7 -1.0]
    [-1.0  0.9  1.0  1.0  1.0  1.0  0.8  1.0 -1.0]
    [-1.0  1.0  1.0 -1.0 -1.0  0.8 -1.0  1.0  1.0]
    [ 1.0 -1.0 -0.3  1.0  1.0  0.2  1.0  0.3  0.6]
    erases-0
    [ 0.6  0.2  0.8  0.5  0.9  0.2  0.9  0.2  0.5]
    [ 0.6  0.1  0.1  0.1  0.6  0.7  0.5  0.6  0.5]
    [ 0.5  0.9  1.0  0.0  1.0  0.6  1.0  0.9  1.0]
    [ 0.3  1.0  1.0  0.0  1.0  0.9  1.0  1.0  1.0]
    [ 1.0  1.0  1.0  0.0  1.0  0.6  1.0  0.6  1.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.7  0.0  1.0  0.0]
    [ 0.7  0.9  1.0  0.9  0.9  0.6  0.9  0.6  0.9]

Again, note the memory access patterns.
They're not as clean as in the simple copy example (I'm trying to figure out exactly why), but it's still relatively clear what's going on:
The NTM writes the first bit vector to the 4th region of memory, then writes the second bit vector to the 5th region.
Then, once the repeat scalar is seen, the NTM reads out locations 4,5,4 and then 5, after which it does whatever, since it will now ignore the read values.
It outputs a stop bit in the correct location.

##### Associative Recall
In this task, we feed the model groups of bit vectors delimited by start bits, then a query vector delimited by end bits.
The goal is for the model to output the vector that it saw immediateley following the query vector.

    inputs:
    [ 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0]
    [ 1.0  0.0  1.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  1.0  0.0]
    outputs:
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.3]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    reads-0
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.5  0.5  0.4  0.4  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.0]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1]
    writes-0
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.1  0.1  0.6]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.1  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.0  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.1  0.1  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.0  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.1  0.0  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.0  0.1  0.0]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.2  0.0  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.3  0.0]
    adds-0
    [ 1.0  1.0  1.0 -1.0 -1.0 -1.0  1.0  1.0  1.0  1.0]
    [-1.0 -1.0 -1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0]
    [ 1.0  1.0  1.0 -1.0 -1.0  1.0  1.0 -1.0  1.0  1.0]
    [ 1.0  1.0  1.0 -1.0  1.0 -1.0  1.0  1.0  1.0  1.0]
    [-1.0 -1.0 -1.0  1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0]
    [-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0]
    [ 1.0  1.0  1.0 -1.0 -1.0 -0.9  1.0  1.0  1.0  1.0]
    erases-0
    [ 0.0  0.0  1.0  0.0  1.0  0.0  0.0  0.9  0.0  0.0]
    [ 1.0  1.0  0.5  0.0  0.0  0.1  1.0  1.0  1.0  0.6]
    [ 1.0  0.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0]
    [ 1.0  1.0  0.9  0.9  0.0  0.9  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    [ 1.0  0.0  0.9  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 1.0  1.0  1.0  1.0  1.0  0.8  0.0  1.0  0.0  0.0]

In the above sequence, the key could have been 0,0,1 or 0,0,0.
It was 0,0,1 (8th time step), and so the model outputs (or mostly outputs, in this case), the item it saw after that, which was 0,0,0.

##### Dynamic n-Grams

In this task, we fix a natural number n and then generate a probability table that encodes the likelihood that the next bit in a sequence will be a 1, having observed the last n-1 bits. We generate a new table for each training sequence, and train the model to predict the next bit at all times.

It's hard to generate an engaging visualization of this task, but if you look at the following it's possible you can get a sense of what's going on.

    inputs:
    [ 1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0  1.0  0.0  1.0  0.0  1.0  0.0  1.0  0.0  1.0  0.0]
    outputs:
    [ 0.5  0.8  0.5  0.2  0.2  0.1  0.2  0.5  0.4  0.0  0.2  0.2  0.1  0.1  0.1  0.0  0.1  0.1  0.1  0.1]
    reads-0
    [ 0.1  0.1  0.2  0.1  0.1  0.1  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.1  0.1  0.1]
    [ 0.1  0.0  0.0  0.1  0.0  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.1  0.1  0.1  0.0]
    [ 0.1  0.0  0.1  0.0  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.3  0.0  0.0  0.0  0.1  0.1  0.1  0.0  0.0]
    [ 0.1  0.2  0.0  0.0  0.1  0.1  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.1  0.1  0.0  0.0  0.0  0.0]
    [ 0.1  0.1  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.3  0.0  0.0  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.3  0.0  0.0  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.4  0.0  0.0  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.0  0.2  0.1  0.0  0.0  0.2  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.2  0.1  0.1  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.1  0.1  0.0  0.0  0.2  0.1  0.1  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5]
    [ 0.1  0.1  0.0  0.3  0.1  0.1  0.1  0.2  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.0]
    [ 0.1  0.1  0.4  0.1  0.1  0.1  0.2  0.1  0.1  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0]
    [ 0.1  0.0  0.0  0.0  0.0  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.6  0.0  0.0  0.0]
    [ 0.1  0.1  0.0  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.1]
    [ 0.1  0.1  0.0  0.1  0.1  0.1  0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.1  0.1]
    writes-0
    [ 0.0  0.5  0.9  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.0  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.0  0.1  0.1  0.1]
    [ 0.1  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.2  0.2  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.1  0.0  0.1  0.1  0.1  0.0  0.1  0.1  0.1]
    [ 0.9  0.5  0.1  0.1  0.3  0.2  0.8  1.0  0.9  0.1  0.1  0.1  0.0  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.2  0.3  0.1  0.1  0.1  0.2  0.2  0.2  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.2  0.0  0.0  0.0  0.1  0.0  0.1  0.0  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.2  0.0  0.1  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.1  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.1  0.0  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.7  0.7  0.8  0.2  0.0  0.0  0.0  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    adds-0
    [ 1.0  1.0  1.0  1.0  1.0  1.0  0.9  1.0  1.0  0.2  0.9  1.0  0.9  0.7  0.6 -0.0  0.9  0.9  0.8  0.7]
    [-1.0  1.0 -0.9 -1.0 -1.0 -1.0 -1.0 -0.1 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0]
    [ 1.0 -1.0 -1.0 -0.8 -0.7 -0.3 -0.9 -1.0 -0.8  0.8 -0.7 -0.6 -0.5  0.8  0.9  0.9 -0.6 -0.2  0.3  0.8]
    [-0.9  1.0  1.0  1.0  1.0  0.9  0.9  1.0  1.0  0.7  0.9  1.0  0.9  0.8  0.2  0.6  0.9  0.9  0.8  0.8]
    [ 1.0  0.6  0.6  0.9  0.9  0.9  0.9  0.9  0.8  1.0  1.0  0.9  1.0  1.0  1.0  1.0  0.9  0.9  1.0  1.0]
    [-1.0 -0.9 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0]
    [-1.0 -1.0 -1.0 -0.9 -1.0 -0.9 -1.0 -1.0 -1.0  0.7 -0.9 -1.0 -0.9  0.1  0.4  0.8 -0.9 -0.9 -0.5 -0.1]
    erases-0
    [ 0.1  0.1  0.1  0.2  0.1  0.2  0.1  0.1  0.1  0.6  0.2  0.2  0.2  0.5  0.5  0.7  0.2  0.2  0.3  0.5]
    [ 0.1  0.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.2  0.6  0.1  0.1  0.1  0.1  0.2  0.3  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.8  0.8  0.1  0.0  0.0  0.0  0.1  0.3  0.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.2  0.7  0.0  0.0  0.0  0.0  0.0  0.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.1  0.1  0.2  0.0  0.0  0.0  0.1]
    [ 0.7  0.0  0.0  0.0  0.0  0.1  0.0  0.0  0.0  0.6  0.1  0.0  0.1  0.5  0.7  0.8  0.1  0.1  0.2  0.4]

##### Priority Sort

In this task, we feed the model a sequence of bit vectors, each with a real-valued priority that lies in (-1,1), which comes on a separate priority channel.
The desired output is that sequence of bit vectors, sorted by the scalar priority. 

Here is an example of an NTM sorting a bit vector of length 2.
The first 3 rows represent the bit vectors to be sorted, and the last row represents the scalar priority.
The fourth and fifth rows are channels just for the start and stop bits.

    inputs:
    [ 0.0  1.0  0.0  0.0  0.0  0.0]
    [ 0.0  1.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  1.0  0.0  0.0  0.0]
    [ 1.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  1.0  0.0  0.0]
    [ 0.0  0.7 -0.6  0.0  0.0  0.0]
    outputs:
    [ 0.0  0.0  0.0  0.0  0.0  1.0]
    [ 0.0  0.0  0.0  0.0  0.0  1.0]
    [ 0.0  0.0  0.0  0.0  1.0  0.0]
    reads-0
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    writes-0
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    [ 0.1  0.1  0.1  0.1  0.1  0.1]
    adds-0
    [-0.3 -0.7 -0.8  0.1 -0.8  0.2]
    [ 0.1  0.9  0.1  0.7 -0.1  0.2]
    [ 0.6  0.5  0.9  0.4 -0.2  0.4]
    [ 0.3 -0.4  0.7 -0.6  0.9 -0.1]
    [ 0.3  0.2 -0.3 -0.6  0.3  0.0]
    [-0.1 -0.2 -0.2 -0.1 -0.0 -0.1]
    [ 0.8  0.9  0.9 -0.1  0.9 -0.8]
    erases-0
    [ 0.3  0.6  0.9  0.5  0.9  0.5]
    [ 0.4  0.6  0.5  0.4  0.4  0.4]
    [ 0.7  0.5  0.9  0.7  0.8  0.6]
    [ 0.3  0.5  0.3  0.3  0.4  0.5]
    [ 0.7  0.7  0.3  0.8  0.4  0.7]
    [ 0.4  0.5  0.4  0.2  0.3  0.2]
    [ 0.6  0.7  0.7  0.4  0.6  0.4]
    reads-1
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.4  0.4  0.3  0.0  0.1]
    [ 0.0  0.0  0.0  0.0  0.3  0.0]
    [ 0.7  0.0  0.0  0.0  0.0  0.1]
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.3  0.4  0.5  0.0  0.1]
    [ 0.0  0.0  0.0  0.0  0.6  0.1]
    [ 0.0  0.2  0.2  0.0  0.0  0.2]
    [ 0.0  0.0  0.0  0.0  0.0  0.1]
    [ 0.0  0.0  0.0  0.1  0.0  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.0]
    [ 0.2  0.0  0.0  0.0  0.0  0.1]
    writes-1
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.0]
    [ 0.0  0.0  0.0  0.0  0.1  0.1]
    [ 0.7  0.0  0.7  0.1  0.2  0.1]
    [ 0.0  0.2  0.0  0.0  0.0  0.1]
    [ 0.0  0.0  0.0  0.1  0.1  0.0]
    [ 0.0  0.0  0.0  0.0  0.0  0.1]
    [ 0.0  0.0  0.0  0.1  0.1  0.1]
    [ 0.3  0.1  0.3  0.1  0.2  0.1]
    [ 0.0  0.2  0.0  0.1  0.0  0.1]
    [ 0.0  0.1  0.0  0.1  0.1  0.0]
    [ 0.0  0.2  0.0  0.1  0.0  0.1]
    [ 0.0  0.0  0.0  0.1  0.1  0.1]
    [ 0.0  0.0  0.0  0.0  0.1  0.1]
    [ 0.0  0.0  0.0  0.1  0.1  0.1]
    adds-1
    [ 1.0 -1.0  1.0  1.0  0.9 -1.0]
    [ 1.0  1.0  1.0  1.0  1.0  0.7]
    [-0.8 -1.0  0.8  1.0 -0.7  1.0]
    [-1.0  1.0 -0.9 -1.0 -0.9 -0.3]
    [ 1.0  0.5  0.6  1.0  0.9 -0.8]
    [ 1.0 -1.0  1.0  1.0 -0.4 -1.0]
    [ 1.0  1.0  0.5  1.0  0.6  0.5]
    erases-1
    [ 0.1  0.3  0.0  0.4  0.0  0.4]
    [ 0.9  0.7  0.1  0.8  0.4  0.7]
    [ 0.0  0.6  0.0  0.0  0.0  0.8]
    [ 0.0  0.9  0.0  0.0  0.0  0.7]
    [ 0.2  0.2  0.8  0.0  0.9  0.1]
    [ 0.0  0.1  0.1  0.0  0.2  0.9]
    [ 1.0  0.9  1.0  1.0  1.0  0.1]

Note that the NTM correctly sorts the example sequence.

In general, the model will fail to sort properly if the priorities are too close together.
This supports the hypothsis put forward in the paper, which is that the location of the writes is determined by the priorities, and that at read time the NTM simply linearly traverses the memory.

However, if this were all that was going on, it seems the model ought to converge examples this size using just one head, and I wasn't able to get it to converge on this task without using 2 heads. Moreover, the memory access patterns above don't exactly align with that interpretation.

#### Miscellaneous Notes

* The controller is just a single FF layer with a tanh activation. All heads take input straight from the output of the controller layer. I couple the number of read heads and write heads. If you pass --heads=8 to the model, there will be 8 read and 8 write heads. As mentioned in the paper, the order of the reads/writes doesn't matter.

* Restricting the range of allowed integer shifts to [-1,0,1] seems totally essential for length-generalization in the copy task. It's possible that this restriction (which currently I've hardcoded into the model), is harming performance on some of the other tasks.

* Gradients are clipped to (-10,10), as advised in the paper.

* The initial state of the memory and the initial state of the last read, the last write weight, and the last read weight are actually learnable parameters of the model, randomly intialized as are the rest of the weights. This modification greatly improved training performance.

* The initialization is currently suboptimal. Where multiple inputs feed into an activation, we should be concatenating the weight vectors that generate those inputs before intializing. Fixing this is high on my todo-list, as I expect it's seriously holding back multi-head performance.

* Performance of the code is poor. The focus of this project is on providing a reliable, deterministic reference implementation, so I'm not going to put a huge amount of effort into speeding up this code. That being said, I'm sure some of the stuff I'm doing is clearly (to some people) ridiculous from a performance perspective - please tell me if you see anything like this.

* It seems that starting with shorter sequences and gradually increasing sequence length as training error falls speeds up convergence, but so far it's been my experience that the NTM will converge if you just let it train for long enough.

* You might see negative BPC because of the way I clip the loss to avoid log(0).
