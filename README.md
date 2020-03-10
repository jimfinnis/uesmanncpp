UESMANN CPP
===========

---
**PROGRESS**
Current progress:
* UESMANN itself is not yet implemented
* load/save files not yet implemented

---


This is the new C++ implementation of UESMANN, based on the original
code used in my thesis. 
The code is very simplistic, using scalar as opposed to matrix operations
and no GPU acceleration. This is to make the code as clear as possible,
as befits a reference implementation, and also to match the implementation
used in the thesis. There are no dependencies on any libraries beyond
those found in a standard C++ install, and libboost-test for testing.
You may find the code somewhat lacking in modern C++ style because I'm
an 80's coder.

A rather more complete set of documentation, including a description
of the network and a Doxygen docs, can be found at

https://jimfinnis.github.io/uesmanncpp/html/index.html

I originally intended to write use Keras/Tensorflow,
but would have been limited to using the low-level Tensorflow operations
because of the somewhat peculiar nature of optimisation in UESMANN:
we descend the gradient relative to the weights for one function,
and the gradient relative to the weights times some constant for the other.
A Keras/Tensorflow implementation is planned.

Implementations of the other network types mentioned in the thesis
are also included.


[![Build Status](https://travis-ci.com/jimfinnis/uesmanncpp.svg?branch=master)](https://travis-ci.com/jimfinnis/uesmanncpp)
