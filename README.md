UESMANN CPP
===========

This is the new C++ implementation of the UESMANN modulatory neural network
architecture, based on the original code used in my thesis. 
See [my University web site for more details and publications](http://users.aber.ac.uk/jcf12/research/uesmann/)
or the brief introduction below.

## What is UESMANN?

UESMANN is a very simple modification of a standard multilayer perceptron
(MLP) with a logistic sigmoid activation function, trained using stochastic
gradient descent. The modification consists of a modulatory factor *h*
on the weights, such that the weights have their nominal values at *h*=0
and double those values at *h*=1. The biases are unmodulated.

As such, the network is able to perform multiple functions at different
modulator levels, and is typically trained using examples of one
function at *h*=0 and another at *h*=1, where it typically performs well.
For example, a single network can be trained
to perform any possible pairing of binary boolean functions in the same
number of network parameters (weights and biases) required for a single
such function. It has also been tested in MNIST handwriting recognition
and line recognition tasks, and in a homeostatic robot control problem.

A rather more complete set of documentation, including a description
of the network and a Doxygen docs, can be found at

https://jimfinnis.github.io/uesmanncpp/html/index.html

## About the code

The code itself is very
simplistic, using scalar as opposed to matrix operations and no GPU
acceleration. This is to make it as clear as possible, as befits a
reference implementation, and also to match the implementation used in the
thesis. There are no dependencies on any libraries beyond those found in a
standard C++ install, and libboost-test for testing. You may find the code
somewhat lacking in modern C++ style because I'm an 80's coder.

Implementations of the other network types mentioned in the thesis
are also included:

* output blending (training two networks with identical architectures
to perform the two different functions and using the modulator to
linearly interpolate between their outputs);
* h-as-input (applying the modulator as an extra input to a standard MLP
and training accordingly)
* plain (a straightforward MLP with no modifications)



I originally intended to use Keras/Tensorflow,
but would have been limited to using the low-level Tensorflow operations
because of the somewhat peculiar nature of optimisation in UESMANN:
we descend the gradient relative to the weights for one function,
and the gradient relative to the weights times some constant for the other,
alternating between the two. This makes the standard optimisers (such as ADAM)
unsuitable. More investigation is planned, however, because 
a UESMANN layer may prove useful within a larger system.
**If you can help with this, please let me know.**

[![Build Status](https://travis-ci.com/jimfinnis/uesmanncpp.svg?branch=master)](https://travis-ci.com/jimfinnis/uesmanncpp)
