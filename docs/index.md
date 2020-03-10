Main Page   {#mainpage}
=========

## Introduction

This is the new C++ implementation of UESMANN, based on the original
code used in my thesis. That code was written as a ```.angso```
library for the Angort language, and has evolved to become
rather unwieldly (as well as being intended for use from a language
no-one but me uses).

The code is very simplistic, using scalar as opposed to matrix operations
and no GPU acceleration. This is to make the code as clear as possible,
as befits a reference implementation, and also to match the implementation
used in the thesis. There are no dependencies on any libraries beyond
those found in a standard C++ install, and libboost-test for testing.
You may find the code somewhat lacking in modern C++ style because I'm
an 80's coder.

I originally intended to write use Keras/Tensorflow,
but would have been limited to using the low-level Tensorflow operations
because of the somewhat peculiar nature of optimisation in UESMANN:
we descend the gradient relative to the weights for one function,
and the gradient relative to the weights times some constant for the other.
A Keras/Tensorflow implementation is planned.

Implementations of the other network types mentioned in the thesis
are also included.

The top level class is @ref Net, which is an virtual type describing the neural net interface
and performing some basic operations. 

## Tests

The files named *test* have various unit tests in them which can be useful to 
examine. The [Tests](@ref tests) page shows some of these tests in more detail.


